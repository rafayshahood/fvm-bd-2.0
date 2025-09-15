# Minimal YOLOv8 webcam tester (CPU) for ID-card detection
# Now with letterbox padding + coordinate mapping back to the original frame.

from pathlib import Path
import time, cv2, numpy as np
from ultralytics import YOLO

# ---------- CONFIG ----------
WEIGHTS_PATH = "./iddetection.pt"          # your trained weights (e.g., best.pt or last.pt)
CLASS_NAMES  = ["Cards", "Lanyard"]
CARDS_IDX    = 0                  # only draw "Cards"
CONF_THRESH  = 0.75               # confidence threshold
IMG_SIZE     = 640                # letterbox square size
CAM_INDEX    = 0                  # webcam index
CAP_WIDTH    = 1280               # capture hint
CAP_HEIGHT   = 720                # capture hint
SAVE_DIR     = Path("snapshots")  # where 's' snapshots go
WINDOW_NAME  = "ID Detector (CPU, letterbox)"
LB_COLOR     = (114, 114, 114)    # padding color like YOLOv8
# ----------------------------------------------

def letterbox_square(img, size=640, color=(114, 114, 114)):
    """
    Resize and pad image to a square 'size' while keeping aspect ratio.
    Returns:
      out_img: (size x size) BGR
      r:       scale ratio used
      pad:     (pad_left, pad_top)
    """
    h, w = img.shape[:2]
    r = min(size / w, size / h)  # scale
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw, dh = size - new_w, size - new_h
    left, right = dw // 2, dw - dw // 2
    top, bottom = dh // 2, dh - dh // 2

    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             borderType=cv2.BORDER_CONSTANT, value=color)
    return out, r, (left, top)

def map_box_back(xyxy, r, pad, orig_w, orig_h):
    """
    Map bbox from letterboxed space back to original frame.
    xyxy in letterbox (size x size). r is scale, pad=(left, top).
    """
    left, top = pad
    x1, y1, x2, y2 = xyxy
    x1 = (x1 - left) / r
    y1 = (y1 - top) / r
    x2 = (x2 - left) / r
    y2 = (y2 - top) / r
    # clip
    x1 = max(0, min(orig_w - 1, x1))
    y1 = max(0, min(orig_h - 1, y1))
    x2 = max(0, min(orig_w - 1, x2))
    y2 = max(0, min(orig_h - 1, y2))
    return int(x1), int(y1), int(x2), int(y2)

def draw_box(img, xyxy, color=(0, 255, 0), label=None):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        yb = max(0, y1 - th - 6)
        cv2.rectangle(img, (x1, yb), (x1 + tw + 6, yb + th + 6), color, -1)
        cv2.putText(img, label, (x1 + 3, yb + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

def main():
    model = YOLO(WEIGHTS_PATH)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {CAM_INDEX}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    SAVE_DIR.mkdir(exist_ok=True)

    t0, frames = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]

        # --- letterbox to square (IMG_SIZE) ---
        lb_img, r, pad = letterbox_square(frame, size=IMG_SIZE, color=LB_COLOR)

        # --- inference (CPU) on letterboxed image ---
        res = model.predict(
            source=lb_img,
            imgsz=IMG_SIZE,
            conf=CONF_THRESH,
            device="cpu",
            verbose=False
        )[0]

        # --- draw only 'Cards' mapped back to original frame ---
        if res.boxes is not None:
            for b in res.boxes:
                cls = int(b.cls[0])
                conf = float(b.conf[0])
                if cls == CARDS_IDX and conf >= CONF_THRESH:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    bx = map_box_back((x1, y1, x2, y2), r, pad, W, H)
                    label = f"{CLASS_NAMES[cls]} {conf:.2f}"
                    draw_box(frame, bx, (0, 255, 0), label)

        # FPS overlay
        frames += 1
        dt = time.time() - t0
        fps = frames / dt if dt > 0 else 0.0
        cv2.putText(frame, f"FPS: {fps:.1f}  conf>={CONF_THRESH}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            ts = time.strftime("%Y%m%d-%H%M%S")
            outp = SAVE_DIR / f"snap_{ts}.jpg"
            cv2.imwrite(str(outp), frame)
            print(f"Saved {outp}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()