# onlanders.video.py
# -------------------------------------------------
# Video analysis utilities
# -------------------------------------------------

import os
import sys
import subprocess
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ---- Feature flags (toggle checks) ----
CHECKS = {
    "face":       bool(int(os.getenv("CHECK_FACE", "1"))),
    "ellipse":    bool(int(os.getenv("CHECK_ELLIPSE", "1"))),
    "brightness": bool(int(os.getenv("CHECK_BRIGHTNESS", "1"))),
    "spoof":      bool(int(os.getenv("CHECK_SPOOF", "1"))),
    "glasses":    bool(int(os.getenv("CHECK_GLASSES", "1"))),
}

def get_checks() -> dict:
    return CHECKS.copy()

# ---- Anti-spoofing repo wiring (used only in LIVE analyze_frame) ----
sys.path.append(str(Path(__file__).resolve().parent / "Silent_Face_Anti_Spoofing"))
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

# ---- Environment paths for anti-spoofing ----
DETECTION_MODEL_PATH = Path(__file__).resolve().parent / "Silent_Face_Anti_Spoofing" / "resources" / "detection_model"
os.environ["DETECTION_MODEL_PATH"] = str(DETECTION_MODEL_PATH)
SPOOF_MODEL_DIR = str(Path(__file__).resolve().parent / "Silent_Face_Anti_Spoofing" / "resources" / "anti_spoof_models")

# ---- Auto device selection ----
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"🖥️ onlanders.video using device: {DEVICE}")

# ---- Models (loaded once; used by LIVE path) ----
MODEL = YOLO("yolov8n-face-lindevs.pt").to(DEVICE)
GLASSES_MODEL = YOLO("./glasses.pt").to(DEVICE)
GLASSES_CLASS_NAMES = ["glasses", "no glasses"]

spoof_model = AntiSpoofPredict(0)
image_cropper = CropImage()

# ---- Offline pipeline config ----
NUM_FRAMES_TO_SELECT = 15


# -------------------------------------------------
# Utility: normalize to mp4 (h264/aac) with validation
# -------------------------------------------------
def convert_to_mp4(input_path: str | Path, output_dir: str | Path) -> Path:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / (input_path.stem + ".mp4")

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")

    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_path),
         "-c:v", "libx264", "-preset", "ultrafast",
         "-c:a", "aac", str(out)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        # tidy up partial file and surface a clear error
        try:
            if out.exists():
                out.unlink()
        except Exception:
            pass
        raise RuntimeError(f"ffmpeg conversion failed for {input_path.name}: {proc.stderr[:800] if proc.stderr else 'no stderr'}")

    return out


# -------------------------------------------------
# LIVE path: per-frame checks for UX gating
# -------------------------------------------------
def analyze_frame(frame, ellipse_params=None) -> dict:
    result = {
        "checks": get_checks(),
        "face_detected": False,
        "num_faces": 0,
        "one_face": True,
        "largest_bbox": None,
        "inside_ellipse": False,
        "brightness_status": None,
        "glasses_detected": None,
        "spoof_is_real": None,
        "spoof_status": None,
    }

    # 1) Face detection
    if CHECKS["face"]:
        det = MODEL.predict(source=[frame], imgsz=640, device=DEVICE, verbose=False)[0]
        if not det or len(det.boxes) == 0:
            return result
        result["face_detected"] = True
        result["num_faces"] = len(det.boxes)
        result["one_face"] = (len(det.boxes) == 1)

        boxes = det.boxes.xyxy.detach().cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        x1, y1, x2, y2 = boxes[areas.argmax()].astype(float)
        result["largest_bbox"] = [float(x1), float(y1), float(x2), float(y2)]
    else:
        result["face_detected"] = True
        result["num_faces"] = 1
        result["one_face"] = True

    # 2) Inside ellipse
    if CHECKS["ellipse"]:
        if ellipse_params is None or result["largest_bbox"] is None:
            return result
        ex = float(ellipse_params["ellipseCx"])
        ey = float(ellipse_params["ellipseCy"])
        rx = float(ellipse_params["ellipseRx"])
        ry = float(ellipse_params["ellipseRy"])

        x1, y1, x2, y2 = result["largest_bbox"]
        w, h = (x2 - x1), (y2 - y1)
        x1_t = x1 + w * 0.12; x2_t = x2 - w * 0.12
        y1_t = y1;           y2_t = y2

        def inside(px, py):
            nx = (px - ex) / max(1e-6, rx)
            ny = (py - ey) / max(1e-6, ry)
            return (nx * nx + ny * ny) <= 1.0

        corners = [(x1_t, y1_t), (x2_t, y1_t), (x1_t, y2_t), (x2_t, y2_t)]
        result["inside_ellipse"] = all(inside(px, py) for px, py in corners)
        if not result["inside_ellipse"]:
            return result
    else:
        result["inside_ellipse"] = True

    # 3) Brightness
    if CHECKS["brightness"]:
        mean_b = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        if mean_b < 50:
            result["brightness_status"] = "too_dark"; return result
        if mean_b > 200:
            result["brightness_status"] = "too_bright"; return result
        result["brightness_status"] = "ok"
    else:
        result["brightness_status"] = "ok"

    # 4) Anti-spoof
    if CHECKS["spoof"]:
        try:
            image_bbox = spoof_model.get_bbox(frame)
            prediction = np.zeros((1, 3))
            for model_name in os.listdir(SPOOF_MODEL_DIR):
                h_in, w_in, model_type, scale = parse_model_name(model_name)
                param = {
                    "org_img": frame, "bbox": image_bbox, "scale": scale,
                    "out_w": w_in, "out_h": h_in, "crop": scale is not None,
                }
                img = image_cropper.crop(**param)
                prediction += spoof_model.predict(img, os.path.join(SPOOF_MODEL_DIR, model_name))
            label = int(np.argmax(prediction))  # 0=spoof, 1=real
            result["spoof_is_real"] = (label == 1)
            result["spoof_status"] = "ok"
        except Exception as e:
            print("⚠️ Spoof error:", e)
            result["spoof_is_real"] = None
            result["spoof_status"] = "error"
    else:
        result["spoof_is_real"] = None
        result["spoof_status"] = "disabled"

    # 5) Glasses
    if CHECKS["glasses"]:
        g_res = GLASSES_MODEL.predict([frame], device=DEVICE, verbose=False)
        has_glasses = any(int(box.cls[0]) == 0 for box in g_res[0].boxes)  # idx 0 -> "glasses"
        result["glasses_detected"] = has_glasses
    else:
        result["glasses_detected"] = False

    return result


# -------------------------------------------------
# OFFLINE pipeline: uniformly sample frames only
# -------------------------------------------------
def run_full_frame_pipeline(video_path: str, output_dir: str):
    video_path = Path(video_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous outputs
    for f in out_dir.iterdir():
        try:
            f.unlink()
        except Exception:
            pass

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"❌ Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = frame_count / fps if fps > 0 else 0.0
    print(f"🎞 FPS: {fps} | frames: {frame_count} | duration(s): {duration:.2f}")
    if frame_count <= 0:
        cap.release()
        raise ValueError("❌ Video has zero frames.")

    n = min(NUM_FRAMES_TO_SELECT, frame_count)
    indices = sorted(set(np.linspace(0, frame_count - 1, n, dtype=int)))

    saved = 0
    target_set = set(indices)
    cur_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if cur_idx in target_set:
            out_path = out_dir / f"frame_{saved + 1}.png"
            cv2.imwrite(str(out_path), frame)
            saved += 1
            if saved >= n:
                break

        cur_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    if saved == 0:
        raise ValueError("❌ Failed to sample frames from video.")

    print(f"✅ Saved {saved} uniformly spaced frames to: {out_dir}")