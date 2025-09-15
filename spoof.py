# test_antispoof.py
import argparse, os, sys, cv2, numpy as np
from pathlib import Path

# Add repo path (adjust if your structure differs)
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "Silent_Face_Anti_Spoofing"))

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

# Env for detection model inside anti-spoof repo
os.environ["DETECTION_MODEL_PATH"] = str(ROOT / "Silent_Face_Anti_Spoofing" / "resources" / "detection_model")
SPOOF_MODEL_DIR = str(ROOT / "Silent_Face_Anti_Spoofing" / "resources" / "anti_spoof_models")

def check_one_frame(frame_bgr, spoof_model, cropper):
    """Return (is_real, score_real_minus_fake) from ensemble."""
    try:
        image_bbox = spoof_model.get_bbox(frame_bgr)
    except Exception:
        return None, None  # can't localize face region

    prediction = np.zeros((1,3))
    for model_name in os.listdir(SPOOF_MODEL_DIR):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": frame_bgr,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True if scale is not None else False,
        }
        img = cropper.crop(**param)
        prediction += spoof_model.predict(img, os.path.join(SPOOF_MODEL_DIR, model_name))

    label = int(np.argmax(prediction))  # 0=spoof, 1=real, 2=unknown (depends on model set)
    # Simple score heuristic: real prob - fake prob
    probs = prediction[0] / np.clip(np.sum(prediction[0]), 1e-6, None)
    score = float(probs[1] - probs[0])
    return (label == 1), score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0 for webcam, or path to video")
    ap.add_argument("--smooth", type=int, default=5, help="temporal smoothing window")
    args = ap.parse_args()

    spoof_model = AntiSpoofPredict(0)
    cropper = CropImage()

    cap = cv2.VideoCapture(0 if args.source=="0" else args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    recent = []  # store last N boolean results for smoothing
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        

        is_real, score = check_one_frame(frame, spoof_model, cropper)
        text = "NO FACE" if is_real is None else ("REAL" if is_real else "SPOOF")
        color = (0,255,0) if is_real else ((0,0,255) if is_real is not None else (128,128,128))

        # Temporal smoothing (optional)
        if is_real is not None:
            recent.append(1 if is_real else 0)
            if len(recent) > max(1, args.smooth):
                recent.pop(0)
            avg = sum(recent)/len(recent)
            text += f" | smooth={avg:.2f}"
        if score is not None:
            text += f" | score={score:+.2f}"

        vis = frame.copy()
        cv2.putText(vis, text, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.imshow("Anti-Spoof Test (q to quit)", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()