# glass_cam_face_cls_single.py
# pip install ultralytics opencv-python numpy

import time, cv2, numpy as np
from ultralytics import YOLO

# ---------- CONFIG ----------
FACE_WEIGHTS = "yolov8n-face-lindevs.pt"   # face detector
CLS_WEIGHTS  = "glass-classification.pt"   # classifier (with_glasses / without_glasses)
IMG_SIZE_DET = 640                         # letterbox size for face detector (like your working script)
IMG_SIZE_CLS = 224                         # your classifier training size
CONF_FACE    = 0.40
CAM_INDEX    = 0
CAP_WIDTH    = 1280
CAP_HEIGHT   = 720
WIN_NAME     = "Face -> Glasses Classifier (CPU)"
PAD_COLOR    = (114,114,114)
DEVICE       = "cpu"                       # keep CPU for stability on macOS
# ----------------------------------

def letterbox_square(img, size=640, color=(114,114,114)):
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = size - new_w, size - new_h
    left, right = dw // 2, dw - dw // 2
    top, bottom = dh // 2, dh - dh // 2
    out = cv2.copyMakeBorder(resized, top, bottom, left, right,
                             borderType=cv2.BORDER_CONSTANT, value=color)
    return out, r, (left, top)

def map_box_back(xyxy, r, pad, orig_w, orig_h):
    left, top = pad
    x1, y1, x2, y2 = map(float, xyxy)
    x1 = (x1 - left) / r; y1 = (y1 - top) / r
    x2 = (x2 - left) / r; y2 = (y2 - top) / r
    x1 = max(0, min(orig_w - 1, x1)); y1 = max(0, min(orig_h - 1, y1))
    x2 = max(0, min(orig_w - 1, x2)); y2 = max(0, min(orig_h - 1, y2))
    return int(x1), int(y1), int(x2), int(y2)

def pad_to_square(img, pad_value=114):
    h, w = img.shape[:2]
    if h == w: return img
    if h > w:
        d = h - w; l = d // 2; r = d - l
        return cv2.copyMakeBorder(img, 0, 0, l, r, cv2.BORDER_CONSTANT, value=(pad_value,)*3)
    else:
        d = w - h; t = d // 2; b = d - t
        return cv2.copyMakeBorder(img, t, b, 0, 0, cv2.BORDER_CONSTANT, value=(pad_value,)*3)

def draw_box(img, xyxy, color=(0, 255, 0), label=None):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        yb = max(0, y1 - th - 6)
        cv2.rectangle(img, (x1, yb), (x1 + tw + 6, yb + th + 6), color, -1)
        cv2.putText(img, label, (x1 + 3, yb + th), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

def main():
    # load models
    face_model = YOLO(FACE_WEIGHTS)
    cls_model  = YOLO(CLS_WEIGHTS)
    print("Classifier classes:", cls_model.names)  # {0:'with_glasses', 1:'without_glasses'}

    # warmups (avoid first-call jit inside GUI loop)
    _ = face_model.predict(source=np.zeros((IMG_SIZE_DET, IMG_SIZE_DET, 3), dtype=np.uint8),
                           imgsz=IMG_SIZE_DET, device=DEVICE, verbose=False)
    _ = cls_model.predict(source=np.zeros((IMG_SIZE_CLS, IMG_SIZE_CLS, 3), dtype=np.uint8),
                          imgsz=IMG_SIZE_CLS, device=DEVICE, verbose=False)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {CAM_INDEX}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    print("q=quit")

    t0, frames = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]

        # --- DETECT FACES on letterboxed image (exactly like your working script style) ---
        lb_img, r, pad = letterbox_square(frame, size=IMG_SIZE_DET, color=PAD_COLOR)
        det = face_model.predict(source=lb_img, imgsz=IMG_SIZE_DET,
                                 conf=CONF_FACE, device=DEVICE, verbose=False)[0]

        if det.boxes is not None and len(det.boxes) > 0:
            # sort faces by area, largest first (optional)
            areas = []
            for b in det.boxes:
                x1l, y1l, x2l, y2l = b.xyxy[0].tolist()
                areas.append((x2l-x1l)*(y2l-y1l))
            order = np.argsort(areas)[::-1]

            for idx in order:
                b = det.boxes[int(idx)]
                x1l, y1l, x2l, y2l = b.xyxy[0].tolist()
                # map back to original frame coords
                x1, y1, x2, y2 = map_box_back((x1l, y1l, x2l, y2l), r, pad, W, H)

                # crop from original frame (no distortion), pad-square, resize to 224
                face_roi = frame[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
                if face_roi.size == 0:
                    continue
                face_sq = pad_to_square(face_roi, 114)
                face_224 = cv2.resize(face_sq, (IMG_SIZE_CLS, IMG_SIZE_CLS), interpolation=cv2.INTER_AREA)

                # --- CLASSIFY THIS ONE CROP (no batching) ---
                cls_res = cls_model.predict(source=face_224[:, :, ::-1],  # BGR->RGB
                                            imgsz=IMG_SIZE_CLS, device=DEVICE, verbose=False)[0]
                cls_id = int(cls_res.probs.top1)
                conf   = float(cls_res.probs.top1conf)
                label  = f"{cls_model.names.get(cls_id, cls_id)} {conf:.2f}"

                draw_box(frame, (x1, y1, x2, y2), (0, 255, 0), label)

        # FPS overlay
        frames += 1
        dt = time.time() - t0
        fps = frames / dt if dt > 0 else 0.0
        cv2.putText(frame, f"FPS: {fps:.1f}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(WIN_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()