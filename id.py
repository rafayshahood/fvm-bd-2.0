# id.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import os, sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image  # enhancement fallback

# ---- Optional GFPGAN import (safe) ----
ENHANCER_AVAILABLE = False
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "GFPGAN")))
    from gfpgan import GFPGANer  # type: ignore
    ENHANCER_AVAILABLE = True
except Exception as _e:
    print("⚠️ GFPGAN not available; will use OpenCV fallback for enhancement.", _e)

# -----------------------------------------------------------------------------
# Devices & Models
# -----------------------------------------------------------------------------
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {DEVICE.upper()}")

# Face model
FACE_MODEL = YOLO("yolov8n-face-lindevs.pt").to(DEVICE)

# ID-card model
ID_WEIGHTS_PATH = "iddetection.pt"   # put your trained weights here
ID_MODEL = YOLO(ID_WEIGHTS_PATH).to(DEVICE)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
# Frame brightness thresholds
BRIGHT_LOW: float  = 60.0
BRIGHT_HIGH: float = 180.0

# Face-ROI brightness (advisory only; no longer a hard gate)
FACE_BRIGHT_LOW: float  = 80.0
FACE_BRIGHT_HIGH: float = 175.0
FACE_PAD_RATIO_X: float = 0.15
FACE_PAD_RATIO_Y: float = 0.15

# Overlay rectangle (keep aligned with FE)
RECT_W_RATIO: float = 0.90
RECT_H_RATIO: float = 0.30

# Tolerance (accept small over/under)
RECT_TOL_FRAC: float = 0.08   # allow ~8% margin for "inside rectangle"
IN_ID_TOL_FRAC: float = 0.06  # allow ~6% margin for "face inside ID"

# Size gates (avoid “too small” faces/ID)
MIN_ID_W_RATIO: float = 0.1  # ID box must fill at least 55% of frame width
MIN_FACE_W_PX: int    = 10   # face bbox min width in pixels

# Output crop padding (still used when we save the ID face)
CROP_PAD_X: float = 0.20
CROP_PAD_Y: float = 0.20

# ID-card detector settings
ID_IMG_SIZE: int = 640
ID_CONF_THRESH: float = 0.75
ID_CARDS_CLASS_INDEX: int = 0   # "Cards" class
LB_COLOR = (114, 114, 114)

# -----------------------------------------------------------------------------
# Enhancement
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

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _brightness_status(bgr_img: np.ndarray) -> Tuple[str, float]:
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    if mean_val < BRIGHT_LOW:  return "too_dark", mean_val
    if mean_val > BRIGHT_HIGH: return "too_bright", mean_val
    return "ok", mean_val

def _largest_face_bbox(bgr_img: np.ndarray) -> Optional[np.ndarray]:
    det = FACE_MODEL.predict(source=[bgr_img], imgsz=640, device=DEVICE, verbose=False)[0]
    if det is None or len(det.boxes) == 0:
        return None
    boxes = det.boxes.xyxy.detach().cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return boxes[areas.argmax()].astype(float)

def _center_rect_for_image(w: int, h: int, w_ratio: float = RECT_W_RATIO, h_ratio: float = RECT_H_RATIO):
    rw = w * float(w_ratio); rh = h * float(h_ratio)
    # return x, y, w, h
    return (w - rw) / 2.0, (h - rh) / 2.0, rw, rh

def _expand_rect(rect_xywh, frac: float):
    rx, ry, rw, rh = rect_xywh
    ex = rw * frac; ey = rh * frac
    return (rx - ex, ry - ey, rw + 2*ex, rh + 2*ey)

def _rect_contains_bbox(rect_xywh, bbox_xyxy) -> bool:
    rx, ry, rw, rh = rect_xywh
    x1, y1, x2, y2 = bbox_xyxy
    def inside(px, py): return (rx <= px <= rx + rw) and (ry <= py <= ry + rh)
    return all(inside(px, py) for px, py in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)])

def _bbox_contains_bbox(outer_xyxy, inner_xyxy) -> bool:
    ox1, oy1, ox2, oy2 = outer_xyxy
    ix1, iy1, ix2, iy2 = inner_xyxy
    return (ox1 <= ix1) and (oy1 <= iy1) and (ox2 >= ix2) and (oy2 >= iy2)

def _bbox_expand_frac(bbox_xyxy, frac: float):
    x1, y1, x2, y2 = bbox_xyxy
    w, h = (x2 - x1), (y2 - y1)
    dx, dy = w * frac, h * frac
    return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)

def _roi_from_bbox(img: np.ndarray, bbox: np.ndarray,
                   pad_x_ratio: float = FACE_PAD_RATIO_X,
                   pad_y_ratio: float = FACE_PAD_RATIO_Y) -> np.ndarray:
    H, W = img.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    px = int((x2 - x1) * pad_x_ratio); py = int((y2 - y1) * pad_y_ratio)
    xi1 = max(0, x1 - px); yi1 = max(0, y1 - py)
    xi2 = min(W - 1, x2 + px); yi2 = min(H - 1, y2 + py)
    return img[yi1:yi2, xi1:xi2]

def _face_brightness_status(image_bgr: np.ndarray, bbox: np.ndarray) -> Tuple[str, float]:
    roi = _roi_from_bbox(image_bgr, bbox, FACE_PAD_RATIO_X, FACE_PAD_RATIO_Y)
    if roi is None or roi.size == 0:
        return "too_dark", 0.0
    mean_val = float(np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)))
    if mean_val < FACE_BRIGHT_LOW:  return "too_dark", mean_val
    if mean_val > FACE_BRIGHT_HIGH: return "too_bright", mean_val
    return "ok", mean_val

def _glare_flags(bgr_roi: np.ndarray) -> bool:
    if bgr_roi is None or bgr_roi.size == 0:
        return False
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)
    N = gray.size
    pct_clipped  = float((gray >= 250).sum()) / max(1, N)
    pct_specular = float(((V >= 240) & (S <= 40)).sum()) / max(1, N)
    mean_g = float(np.mean(gray)); std_g = float(np.std(gray))
    return (pct_clipped > 0.03) or (pct_specular > 0.08) or (std_g < 18 and mean_g > 180)

# --- Letterbox helpers for ID-card detector ---
def _letterbox_square(img: np.ndarray, size: int = 640, color=(114,114,114)):
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = size - new_w, size - new_h
    left, right = dw // 2, dw - dw // 2
    top, bottom = dh // 2, dh - dh // 2
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

def _detect_id_card(bgr_img: np.ndarray) -> Tuple[bool, Optional[Tuple[float,float,float,float]], Optional[float]]:
    """
    Returns: (detected, bbox_xyxy, conf) in ORIGINAL frame coords.
    Only accepts class == ID_CARDS_CLASS_INDEX and conf >= ID_CONF_THRESH.
    """
    H, W = bgr_img.shape[:2]
    lb, r, pad = _letterbox_square(bgr_img, size=ID_IMG_SIZE, color=LB_COLOR)
    res = ID_MODEL.predict(source=[lb], imgsz=ID_IMG_SIZE, conf=ID_CONF_THRESH,
                           device=DEVICE, verbose=False)[0]
    best = None
    if res and res.boxes is not None and len(res.boxes) > 0:
        for b in res.boxes:
            cls = int(b.cls[0]); conf = float(b.conf[0])
            if cls == ID_CARDS_CLASS_INDEX and conf >= ID_CONF_THRESH:
                if (best is None) or (conf > best[1]):
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    bx = _map_box_back((x1, y1, x2, y2), r, pad, W, H)
                    best = (bx, conf)
    if best is None:
        return False, None, None
    return True, best[0], best[1]

# -----------------------------------------------------------------------------
# Public API (NEW order)
# -----------------------------------------------------------------------------
def analyze_id_frame(
    image_bgr: np.ndarray,
    use_auto_rect: bool = True,
    rect_w_ratio: float = RECT_W_RATIO,
    rect_h_ratio: float = RECT_H_RATIO,
) -> Dict[str, Optional[object]]:
    """
    New strict order:
      1) overall frame brightness
      2) ID card present
      3) ID card inside the center rectangle (with tolerance)
      4) face present
      5) face inside ID-card bbox (with tolerance)
      6) size / proximity check (ID fill + min face px)
      7) glare (advisory gate at the end)
    """
    H, W = image_bgr.shape[:2]
    out: Dict[str, Optional[object]] = {
        "brightness_status": None,
        "brightness_mean": None,

        "id_card_detected": None,
        "id_card_bbox": None,
        "id_card_conf": None,
        "id_inside_rect": None,

        "face_detected": False,
        "largest_bbox": None,
        "face_inside_id": None,

        "id_fill_ratio": None,
        "id_fill_ok": None,
        "face_w_px": None,
        "face_size_ok": None,

        "face_brightness_status": None,  # advisory only
        "face_brightness_mean": None,

        "rect": None,
        "glare_detected": False,
    }

    # 1) Frame brightness
    b_status, b_mean = _brightness_status(image_bgr)
    out["brightness_status"] = b_status
    out["brightness_mean"] = float(b_mean)
    if b_status != "ok":
        return out

    # 2) ID card present
    id_ok, id_bbox, id_conf = _detect_id_card(image_bgr)
    out["id_card_detected"] = bool(id_ok)
    out["id_card_bbox"] = list(id_bbox) if id_bbox else None
    out["id_card_conf"] = float(id_conf) if id_conf is not None else None
    if not id_ok:
        return out

    # 3) ID inside center rectangle (with tolerance)
    if use_auto_rect:
        rect = _center_rect_for_image(W, H, rect_w_ratio, rect_h_ratio)
        out["rect"] = [float(r) for r in rect]
        rect_expanded = _expand_rect(rect, RECT_TOL_FRAC)
        out["id_inside_rect"] = _rect_contains_bbox(rect_expanded, id_bbox)
        if not out["id_inside_rect"]:
            return out

    # 4) Face present
    face_bbox = _largest_face_bbox(image_bgr)
    if face_bbox is None:
        return out
    out["face_detected"] = True
    out["largest_bbox"] = [float(v) for v in face_bbox]

    # 5) Face inside ID-card (with tolerance)
    id_expanded = _bbox_expand_frac(id_bbox, IN_ID_TOL_FRAC)
    out["face_inside_id"] = _bbox_contains_bbox(id_expanded, face_bbox)
    if not out["face_inside_id"]:
        return out

    # 6) Size / proximity checks
    id_w = float(id_bbox[2] - id_bbox[0])
    face_w = float(face_bbox[2] - face_bbox[0])
    id_fill_ratio = id_w / max(1.0, float(W))
    out["id_fill_ratio"] = round(id_fill_ratio, 4)
    out["id_fill_ok"] = bool(id_fill_ratio >= MIN_ID_W_RATIO)

    out["face_w_px"] = int(round(face_w))
    out["face_size_ok"] = bool(face_w >= MIN_FACE_W_PX)

    if not (out["id_fill_ok"] and out["face_size_ok"]):
        return out

    # 7) Advisory: face ROI brightness + glare
    f_status, f_mean = _face_brightness_status(image_bgr, face_bbox)
    out["face_brightness_status"] = f_status
    out["face_brightness_mean"] = float(f_mean)

    roi_for_glare = _roi_from_bbox(image_bgr, face_bbox, 0.15, 0.15)
    out["glare_detected"] = _glare_flags(roi_for_glare)

    return out

def run_id_extraction(input_path: str, output_path: str) -> None:
    """
    Detect largest face, pad, crop, and save to output_path.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"ID image not found at path: {input_path}")

    bbox = _largest_face_bbox(image)
    if bbox is None:
        raise Exception("❌ No face detected in ID image.")
    x1_f, y1_f, x2_f, y2_f = map(int, bbox)

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
    "run_id_extraction",
    "enhance_id_image",
]