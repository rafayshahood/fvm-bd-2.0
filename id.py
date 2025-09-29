# id.py — shared analyzers for front and back sides of ID
# Front: brightness → ID-in-ROI → overlap/size/area/ar → FACE-on-ID → OCR
# Back:  brightness → ID-in-ROI → overlap/size/area/ar → QR CODE (no OCR, no face)

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import os, sys, re

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image  # enhancement fallback for FRONT only
from rapidfuzz import fuzz
import easyocr

# ---- Optional GFPGAN import (used by FRONT enhancement helper only) ----
ENHANCER_AVAILABLE = False
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "GFPGAN")))
    from gfpgan import GFPGANer  # type: ignore
    ENHANCER_AVAILABLE = True
except Exception as _e:
    print("⚠️ GFPGAN not available; will use OpenCV fallback for enhancement.", _e)

# -----------------------------------------------------------------------------
# Device & Models
# -----------------------------------------------------------------------------
def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE: str = _select_device()
print(f"🖥️ Using device: {DEVICE.upper()}")

FACE_MODEL = YOLO("yolov8n-face-lindevs.pt")
ID_MODEL   = YOLO("iddetection.pt")

# -----------------------------------------------------------------------------
# Config — thresholds / gates
# -----------------------------------------------------------------------------
CONF, IOU, MAX_DETS = 0.50, 0.45, 3
FACE_CONF = 0.50

EXPECTED_AR: float      = float(os.getenv("ID_EXPECTED_AR", "1.586"))
AR_TOL: float           = float(os.getenv("ID_AR_TOL", "0.18"))
MIN_AREA_FRAC: float    = float(os.getenv("ID_MIN_AREA_FRAC", "0.02"))

OVERLAP_MIN: float           = float(os.getenv("ID_OVERLAP_MIN", "0.60"))
OCR_BOX_KEEP_PCT: float      = float(os.getenv("ID_OCR_KEEP_PCT", "0.90"))
MIN_GUIDE_COVER_FRAC: float  = float(os.getenv("ID_MIN_GUIDE_COVER", "0.30"))

# OCR thresholds (for FRONT only now)
OCR_REQUIRED_HITS: int  = int(os.getenv("ID_OCR_HITS", "1"))
OCR_MIN_CONF: float     = float(os.getenv("ID_OCR_MIN_CONF", "0.45"))
FUZZY: int              = int(os.getenv("ID_OCR_FUZZY", "70"))

# Brightness thresholds
BRIGHT_MIN: int = int(os.getenv("ID_BRIGHT_MIN", "60"))
BRIGHT_MAX: int = int(os.getenv("ID_BRIGHT_MAX", "220"))

ID_IMG_SIZE: int = 640
LB_COLOR = (114,114,114)

# Guide rectangle (must match FE)
RECT_W_RATIO: float = 0.95
RECT_H_RATIO: float = 0.45

# EasyOCR with multi-language support (FRONT only)
_EASYOCR_USE_GPU = bool(torch.cuda.is_available())
reader_primary = easyocr.Reader(['en','es'], gpu=_EASYOCR_USE_GPU)
reader_urdu = easyocr.Reader(['en','ur'], gpu=_EASYOCR_USE_GPU)

# -----------------------------------------------------------------------------
# OCR keyword/regex sets (FRONT ONLY)
# -----------------------------------------------------------------------------
KW_FRONT = [
    "identidad","identificación","cédula","ciudadanía","república","colombia","nacional",
    "autoridad","expedición","vencimiento","sexo","nombre","apellidos","nuip",
    "identity","identification","id card","national","authority","republic","government","passport"
]
RGX_FRONT = [
    r"\b\d{6,}\b",
    r"\b(19|20)\d{2}[./\- ]\d{1,2}[./\- ]\d{1,2}\b",
    r"\b\d{1,2}\s*(ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic|"
    r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*(19|20)\d{2}\b",
    r"\b(NUIP|N\.U\.I\.P\.?)\b",
    r"\b\d{1,3}\.\d{3}\.\d{3}\.\d{1,4}\b"
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _center_rect_for_image(w: int, h: int, w_ratio: float = RECT_W_RATIO, h_ratio: float = RECT_H_RATIO):
    rw = w * float(w_ratio); rh = h * float(h_ratio)
    return (w - rw) / 2.0, (h - rh) / 2.0, rw, rh

def _letterbox_square(img: np.ndarray, size: int = 640, color=(114,114,114)):
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = size - new_w, size - new_h
    left, right = dw // 2, dw - dw // 2
    top,  bottom = dh // 2, dh - dh // 2
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, r, (left, top)

def _map_box_back(xyxy, r, pad, orig_w, orig_h):
    left, top = pad
    x1, y1, x2, y2 = xyxy
    x1 = (x1 - left) / r; y1 = (y1 - top) / r
    x2 = (x2 - left) / r; y2 = (y2 - top) / r
    x1 = max(0, min(orig_w - 1, x1)); y1 = max(0, min(orig_h - 1, y1))
    x2 = max(0, min(orig_w - 1, x2)); y2 = max(0, min(orig_h - 1, y2))
    return float(x1), float(y1), float(x2), float(y2)

def _rect_intersect(a_xyxy, b_xyxy):
    ax1, ay1, ax2, ay2 = a_xyxy; bx1, by1, bx2, by2 = b_xyxy
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None, 0.0, 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aarea = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    return (ix1, iy1, ix2, iy2), float(inter) / (aarea + 1e-6), float(inter)

def _aspect_ok(w: float, h: float, exp: float = EXPECTED_AR, tol: float = AR_TOL):
    if w <= 0 or h <= 0: return False, 0.0
    ar = w / float(h)
    return (exp*(1-tol) <= ar <= exp*(1+tol)), ar

def _area_frac_ok(x1: float, y1: float, x2: float, y2: float, W: int, H: int, minf: float = MIN_AREA_FRAC):
    return ((x2 - x1) * (y2 - y1)) / (W * H + 1e-6) >= minf

def _brightness_eval(img_bgr: np.ndarray):
    """Return (ok, mean, status) using Y (luma) from YCrCb."""
    y = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    mean = float(np.mean(y))
    if mean < BRIGHT_MIN:
        return False, mean, "dark"
    if mean > BRIGHT_MAX:
        return False, mean, "bright"
    return True, mean, "ok"

def _detect_id_card_in_roi(frame_bgr: np.ndarray, guide_xyxy: Tuple[int,int,int,int]):
    """Detect ONLY inside the guide ROI."""
    gx1, gy1, gx2, gy2 = guide_xyxy
    roi = frame_bgr[gy1:gy2, gx1:gx2]
    if roi.size == 0:
        return False, None, None
    H_roi, W_roi = roi.shape[:2]
    lb, r, pad = _letterbox_square(roi, size=ID_IMG_SIZE, color=LB_COLOR)
    det = ID_MODEL.predict(source=[lb], imgsz=ID_IMG_SIZE, conf=CONF, iou=IOU,
                           max_det=MAX_DETS, device=DEVICE, verbose=False)[0]
    best = None
    if det and det.boxes is not None and len(det.boxes) > 0:
        for b in det.boxes:
            cls = int(b.cls[0]); conf = float(b.conf[0])
            if cls != 0:
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            rx1, ry1, rx2, ry2 = _map_box_back((x1, y1, x2, y2), r, pad, W_roi, H_roi)
            fx1, fy1, fx2, fy2 = gx1 + rx1, gy1 + ry1, gx1 + rx2, gy1 + ry2
            if best is None or conf > best[1]:
                best = ((fx1, fy1, fx2, fy2), conf)
    if best is None:
        return False, None, None
    return True, best[0], best[1]

def _detect_face_in_id_crop(id_crop_bgr: np.ndarray, imgsz: int = 320):
    lb, r, pad = _letterbox_square(id_crop_bgr, size=imgsz, color=LB_COLOR)
    det = FACE_MODEL.predict(source=[lb], imgsz=imgsz, conf=FACE_CONF, iou=0.45,
                             max_det=3, device=DEVICE, verbose=False)[0]
    if not det or det.boxes is None or len(det.boxes) == 0:
        return False, None
    (Hc, Wc) = id_crop_bgr.shape[:2]
    crop_area = float(Hc * Wc) + 1e-6
    candidates = sorted(det.boxes,
                        key=lambda bb: float(bb.conf[0]) if getattr(bb, "conf", None) is not None else 1.0,
                        reverse=True)
    for b in candidates:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        fx1, fy1, fx2, fy2 = _map_box_back((x1, y1, x2, y2), r, pad, Wc, Hc)
        if ((fx2 - fx1) * (fy2 - fy1) / crop_area) >= 0.02:
            return True, (int(fx1), int(fy1), int(fx2), int(fy2))
    return False, None

def _ocr_verify_crop_inside_custom(
    crop_bgr: np.ndarray, ix1: int, iy1: int, guide_xyxy, kw_list: List[str], rgx_list: List[str],
    required_hits: int = OCR_REQUIRED_HITS, min_conf: float = OCR_MIN_CONF, fuzzy: int = FUZZY
):
    res = reader_primary.readtext(crop_bgr, detail=1, paragraph=False)
    if not res:
        try:
            res = reader_urdu.readtext(crop_bgr, detail=1, paragraph=False)
        except Exception:
            res = []
            
    if not res:
        return False, 0.0, 0, "", 0.0
    texts, confs, inside = [], [], []
    gx1, gy1, gx2, gy2 = guide_xyxy
    for (pts, txt, conf) in res:
        if not txt:
            continue
        cx = sum(p[0] for p in pts) / 4.0 + ix1
        cy = sum(p[1] for p in pts) / 4.0 + iy1
        inside.append(gx1 <= cx <= gx2 and gy1 <= cy <= gy2)
        texts.append(txt); confs.append(float(conf))
    if not texts:
        return False, 0.0, 0, "", 0.0

    inside_ratio = (sum(inside) / max(1, len(inside)))
    if inside_ratio < OCR_BOX_KEEP_PCT:
        return False, float(np.mean(confs)), 0, " ".join(texts).lower(), inside_ratio

    joined = " ".join(texts).lower()
    hits = 0
    for kw in kw_list:
        if fuzz.partial_ratio(kw, joined) >= fuzzy:
            hits += 1
    for rx in rgx_list:
        if re.search(rx, joined):
            hits += 1
    mean_conf = float(np.mean(confs))
    ok = (hits >= required_hits) and (mean_conf >= min_conf)
    return ok, mean_conf, hits, joined, inside_ratio

def _detect_qr_code(image_bgr: np.ndarray) -> bool:
    """Detect if image contains a QR code using OpenCV."""
    try:
        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(image_bgr)
        return bbox is not None and len(bbox) > 0
    except Exception as e:
        print(f"⚠️ QR detection error: {e}")
        return False

# -----------------------------------------------------------------------------
# Public analyzers
# -----------------------------------------------------------------------------
def analyze_id_frame(
    image_bgr: np.ndarray,
    rect_w_ratio: float = RECT_W_RATIO,
    rect_h_ratio: float = RECT_H_RATIO,
) -> Dict[str, Optional[object]]:
    """FRONT side analyzer (unchanged behavior)."""
    H, W = image_bgr.shape[:2]
    out: Dict[str, Optional[object]] = {
        "rect": None, "roi_xyxy": None,
        "brightness_ok": None, "brightness_status": None, "brightness_mean": None,
        "id_card_detected": False, "id_card_bbox": None, "id_card_conf": None,
        "id_frac_in": None, "id_overlap_ok": None,
        "id_size_ratio": None, "id_size_ok": None,
        "id_ar": None,
        "face_on_id": False, "largest_bbox": None,
        "ocr_ok": None, "ocr_inside_ratio": None, "ocr_hits": None, "ocr_mean_conf": None,
        "verified": False,
    }

    rx, ry, rw, rh = _center_rect_for_image(W, H, rect_w_ratio, rect_h_ratio)
    gx1, gy1 = int(rx), int(ry); gx2, gy2 = int(rx + rw), int(ry + rh)
    guide_xyxy = (gx1, gy1, gx2, gy2)
    out["rect"] = [float(rx), float(ry), float(rw), float(rh)]
    out["roi_xyxy"] = [gx1, gy1, gx2, gy2]

    b_ok, b_mean, b_status = _brightness_eval(image_bgr)
    out["brightness_ok"] = bool(b_ok)
    out["brightness_mean"] = float(b_mean)
    out["brightness_status"] = str(b_status)
    if not b_ok:
        return out

    id_ok, id_bbox, id_conf = _detect_id_card_in_roi(image_bgr, guide_xyxy)
    out["id_card_detected"] = bool(id_ok)
    out["id_card_bbox"] = list(id_bbox) if id_bbox else None
    out["id_card_conf"] = float(id_conf) if id_conf is not None else None
    if not id_ok:
        return out

    x1, y1, x2, y2 = id_bbox
    ar_ok, ar = _aspect_ok(x2 - x1, y2 - y1); out["id_ar"] = float(ar)
    inter, frac_in, inter_area = _rect_intersect(id_bbox, guide_xyxy); out["id_frac_in"] = float(frac_in)
    if inter is None or frac_in < OVERLAP_MIN: out["id_overlap_ok"] = False; return out
    out["id_overlap_ok"] = True
    guide_area = float((gx2 - gx1) * (gy2 - gy1)) + 1e-6
    size_ratio = inter_area / guide_area; out["id_size_ratio"] = float(size_ratio)
    out["id_size_ok"] = bool(size_ratio >= MIN_GUIDE_COVER_FRAC)
    if not out["id_size_ok"]: return out
    if not _area_frac_ok(x1, y1, x2, y2, W, H): return out
    if not ar_ok: return out

    ix1, iy1, ix2, iy2 = map(int, inter)
    id_crop = image_bgr[iy1:iy2, ix1:ix2]

    f_ok, f_box = _detect_face_in_id_crop(id_crop, imgsz=320)
    out["face_on_id"] = bool(f_ok)
    if f_ok and f_box is not None:
        fx1, fy1, fx2, fy2 = f_box
        out["largest_bbox"] = [ix1 + fx1, iy1 + fy1, ix1 + fx2, iy1 + fy2]

    ocr_ok, mean_conf, hits, _joined, inside_ratio = _ocr_verify_crop_inside_custom(
        id_crop, ix1, iy1, guide_xyxy, KW_FRONT, RGX_FRONT, OCR_REQUIRED_HITS, OCR_MIN_CONF, FUZZY
    )
    out["ocr_ok"] = bool(ocr_ok)
    out["ocr_inside_ratio"] = float(inside_ratio)
    out["ocr_hits"] = int(hits)
    out["ocr_mean_conf"] = float(mean_conf)

    out["verified"] = bool(out["id_overlap_ok"] and out["id_size_ok"] and out["face_on_id"] and out["ocr_ok"])
    return out


def analyze_id_back_frame(
    image_bgr: np.ndarray,
    rect_w_ratio: float = RECT_W_RATIO,
    rect_h_ratio: float = RECT_H_RATIO,
) -> Dict[str, Optional[object]]:
    """BACK side analyzer: brightness → ID detection → overlap/size → QR CODE only."""
    H, W = image_bgr.shape[:2]
    out: Dict[str, Optional[object]] = {
        "rect": None, "roi_xyxy": None,
        "brightness_ok": None, "brightness_status": None, "brightness_mean": None,
        "id_card_detected": False, "id_card_bbox": None, "id_card_conf": None,
        "id_frac_in": None, "id_overlap_ok": None,
        "id_size_ratio": None, "id_size_ok": None,
        "id_ar": None,
        "qr_detected": None,
        "verified": False,
    }

    rx, ry, rw, rh = _center_rect_for_image(W, H, rect_w_ratio, rect_h_ratio)
    gx1, gy1 = int(rx), int(ry); gx2, gy2 = int(rx + rw), int(ry + rh)
    guide_xyxy = (gx1, gy1, gx2, gy2)
    out["rect"] = [float(rx), float(ry), float(rw), float(rh)]
    out["roi_xyxy"] = [gx1, gy1, gx2, gy2]

    b_ok, b_mean, b_status = _brightness_eval(image_bgr)
    out["brightness_ok"] = bool(b_ok)
    out["brightness_mean"] = float(b_mean)
    out["brightness_status"] = str(b_status)
    if not b_ok:
        return out

    id_ok, id_bbox, id_conf = _detect_id_card_in_roi(image_bgr, guide_xyxy)
    out["id_card_detected"] = bool(id_ok)
    out["id_card_bbox"] = list(id_bbox) if id_bbox else None
    out["id_card_conf"] = float(id_conf) if id_conf is not None else None
    if not id_ok:
        return out

    x1, y1, x2, y2 = id_bbox
    ar_ok, ar = _aspect_ok(x2 - x1, y2 - y1); out["id_ar"] = float(ar)
    inter, frac_in, inter_area = _rect_intersect(id_bbox, guide_xyxy); out["id_frac_in"] = float(frac_in)
    if inter is None or frac_in < OVERLAP_MIN: out["id_overlap_ok"] = False; return out
    out["id_overlap_ok"] = True
    guide_area = float((gx2 - gx1) * (gy2 - gy1)) + 1e-6
    size_ratio = inter_area / guide_area; out["id_size_ratio"] = float(size_ratio)
    out["id_size_ok"] = bool(size_ratio >= MIN_GUIDE_COVER_FRAC)
    if not out["id_size_ok"]: return out
    if not _area_frac_ok(x1, y1, x2, y2, W, H): return out
    if not ar_ok: return out

    ix1, iy1, ix2, iy2 = map(int, inter)
    id_crop = image_bgr[iy1:iy2, ix1:ix2]

    qr_found = _detect_qr_code(id_crop)
    out["qr_detected"] = bool(qr_found)

    out["verified"] = bool(out["id_overlap_ok"] and out["id_size_ok"] and out["qr_detected"])
    return out

# -----------------------------------------------------------------------------
# Enhancement & face-crop-on-still (FRONT only; unchanged)
# -----------------------------------------------------------------------------
def enhance_id_image(input_path: str, output_path: str) -> bool:
    if ENHANCER_AVAILABLE:
        model_path = "GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth"
        try:
            restorer = GFPGANer(
                model_path=model_path, upscale=2, arch="clean",
                channel_multiplier=2, bg_upsampler=None
            )
            img = Image.open(input_path).convert("RGB")
            img_np = np.array(img)
            _, _, restored = restorer.enhance(
                img_np, has_aligned=False, only_center_face=True, paste_back=True
            )
            Image.fromarray(restored).save(output_path)
            print(f"✅ Enhanced (GFPGAN) saved to {output_path}")
            return True
        except Exception as e:
            print(f"⚠️ GFPGAN failed ({e}); using OpenCV fallback.")

    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        raise FileNotFoundError(f"ID image not found: {input_path}")

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    L2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
    img2 = cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(img2, (0, 0), 1.0)
    sharpen = cv2.addWeighted(img2, 1.5, blur, -0.5, 0)
    cv2.imwrite(output_path, sharpen)
    print(f"✅ Enhanced (OpenCV) saved to {output_path}")
    return True

def run_id_extraction(input_path: str, output_path: str) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"ID image not found at path: {input_path}")

    lb, r, pad = _letterbox_square(image, size=320, color=LB_COLOR)
    det = FACE_MODEL.predict(source=[lb], imgsz=320, conf=FACE_CONF,
                             device=DEVICE, verbose=False)[0]
    if not det or det.boxes is None or len(det.boxes) == 0:
        raise Exception("❌ No face detected in ID image.")

    best = None; best_area = -1.0
    for b in det.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        bx = _map_box_back((x1, y1, x2, y2), r, pad, image.shape[1], image.shape[0])
        area = max(0.0, (bx[2]-bx[0])) * max(0.0, (bx[3]-bx[1]))
        if area > best_area:
            best_area = area; best = bx

    x1_f, y1_f, x2_f, y2_f = map(int, best)
    CROP_PAD_X, CROP_PAD_Y = 0.50, 0.50
    h, w = image.shape[:2]
    pad_x = int((x2_f - x1_f) * CROP_PAD_X)
    pad_y = int((y2_f - y1_f) * CROP_PAD_Y)
    x1 = max(x1_f - pad_x, 0)
    y1 = max(y1_f - pad_y, 0)
    x2 = min(x2_f + pad_x, w - 1)
    y2 = min(y2_f + pad_y, h - 1)

    cropped_face = image[y1:y2, x1:x2]
    cv2.imwrite(str(out_path), cropped_face)
    print(f"✅ Cropped face saved to {output_path}")

__all__ = [
    "analyze_id_frame",
    "analyze_id_back_frame",
    "run_id_extraction",
    "enhance_id_image",
]
