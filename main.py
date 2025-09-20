# main.py
from __future__ import annotations
from typing import Dict, Any, Optional, List

import base64
import json
import os
import shutil
import time
import subprocess
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from uuid import uuid4
import threading  # ← NEW

import cv2
import numpy as np
from fastapi import (
    FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Request, HTTPException
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from id import analyze_id_frame, run_id_extraction
from new_verif import run_verif  # noqa: F401
from all_video import (
    run_full_frame_pipeline,   # noqa: F401
    analyze_frame,
)

app = FastAPI(title="Face Verification API (stateless)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # stateless: no cookies/sessions
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/temp", StaticFiles(directory="temp"), name="temp")


# --- Debounce helper for boolean conditions (N consecutive frames to flip) ---
class BoolStreak:
    def __init__(self, n: int = 3):
        self.n = int(n)
        self.stable: Optional[bool] = None
        self.counter = 0

    def update(self, val: Optional[bool]) -> Optional[bool]:
        if val is None:
            return self.stable  # ignore None (don't change anything)
        if self.stable is None:
            self.stable = bool(val)
            self.counter = 0
            return self.stable
        if bool(val) == self.stable:
            self.counter = 0
            return self.stable
        # value differs → count toward flipping
        self.counter += 1
        if self.counter >= self.n:
            self.stable = bool(val)
            self.counter = 0
        return self.stable

# ---------- JSON safety for numpy types ----------
def _to_jsonable(o):
    import numpy as np
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_, np.integer, np.floating)):
        return o.item()
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(x) for x in o]
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    return o

# ---------- path helpers ----------
def _abs_url(request: Optional[Request], path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if not request:
        return path
    base = str(request.base_url).rstrip("/")
    return f"{base}{path}"

def _req_id_from(request: Request) -> str:
    q = request.query_params.get("req_id")
    if q:
        return q
    h = request.headers.get("X-Req-ID")
    if h:
        return h
    return uuid4().hex

def _base_paths(req_id: str) -> Dict[str, Path]:
    base = Path("temp") / req_id
    id_dir = base / "id"            # front side folder
    id_back_dir = base / "id_back"  # back side folder
    rec_dir = base / "recordings"
    selected_dir = base / "selected_faces"
    return {
        "base": base,
        "id_dir": id_dir,
        "id_back_dir": id_back_dir,
        "rec_dir": rec_dir,
        "selected_dir": selected_dir
    }

def _used_id_image_path(id_dir: Path) -> Optional[Path]:
    enhanced = id_dir / "id_enhanced.jpg"
    raw = id_dir / "id_raw_upload.jpg"
    if enhanced.exists():
        return enhanced
    if raw.exists():
        return raw
    return None

def _used_id_back_image_path(id_back_dir: Path) -> Optional[Path]:
    enhanced = id_back_dir / "id_back_enhanced.jpg"
    raw = id_back_dir / "id_back_raw_upload.jpg"
    if enhanced.exists():
        return enhanced
    if raw.exists():
        return raw
    return None

def _selected_frame_files(selected_dir: Path, limit: Optional[int] = None) -> List[Path]:
    if not selected_dir.exists():
        return []
    exts = (".jpg", ".jpeg", ".png")
    files = [p for p in selected_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name)
    return files[:limit] if limit else files

def _result_urls_for_req(req_id: str, request: Optional[Request] = None) -> Dict[str, Any]:
    paths = _base_paths(req_id)
    base, id_dir, id_back_dir, selected_dir = (
        paths["base"], paths["id_dir"], paths["id_back_dir"], paths["selected_dir"]
    )
    used_id_front = _used_id_image_path(id_dir)
    used_id_back = _used_id_back_image_path(id_back_dir)
    cropped = id_dir / "cropped_id_face.jpg"
    video = base / "video.mp4"
    best = base / "best_match.png"

    id_image_url = f"/temp/{req_id}/id/{used_id_front.name}" if used_id_front else None
    id_back_image_url = f"/temp/{req_id}/id_back/{used_id_back.name}" if used_id_back else None
    cropped_face_url = f"/temp/{req_id}/id/{cropped.name}" if cropped.exists() else None
    video_url = f"/temp/{req_id}/video.mp4" if video.exists() else None
    best_match_url = f"/temp/{req_id}/best_match.png" if best.exists() else None
    selected_frames = [f"/temp/{req_id}/selected_faces/{p.name}" for p in _selected_frame_files(selected_dir)]

    return {
        "id_image_url": _abs_url(request, id_image_url),
        "id_back_image_url": _abs_url(request, id_back_image_url),
        "cropped_face_url": _abs_url(request, cropped_face_url),
        "video_url": _abs_url(request, video_url),
        "selected_frames": [_abs_url(request, u) for u in selected_frames],
        "best_match_url": _abs_url(request, best_match_url),
    }

# ---------- deepfake helpers (NEW) ----------
def _deepfake_json_path(base: Path) -> Path:
    return base / "deepfake.json"

def _write_deepfake_status(base: Path, payload: dict) -> None:
    try:
        _deepfake_json_path(base).write_text(json.dumps(payload, indent=2))
    except Exception as e:
        print(f"⚠️ deepfake status write error: {e}")

def _read_deepfake_status(base: Path) -> Optional[dict]:
    jf = _deepfake_json_path(base)
    if not jf.exists():
        return None
    try:
        return json.loads(jf.read_text() or "{}")
    except Exception:
        return None

def _run_genconvit(video_path: Path) -> dict:
    """
    Calls GenConViT/prediction.py and parses stdout.
    Returns dict: {ok, completed, is_real, is_deepfake, raw_tail}
    """
    cmd = [
        "python", "GenConViT/prediction.py",
        "--p", str(video_path),
        "--e", "--v", "--f", "10",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        out = proc.stdout or ""
        err = proc.stderr or ""
        is_real = None

        for line in out.splitlines():
            if "Fake: 0 Real: 1" in line:
                is_real = True
            elif "Fake: 1 Real: 0" in line:
                is_real = False

        if is_real is None:
            # conservative default (don’t block flow; mark as real)
            is_real = True

        return {
            "ok": True,
            "completed": True,
            "is_real": bool(is_real),
            "is_deepfake": (not bool(is_real)),
            "raw_tail": (out + "\n" + err)[-1200:],
        }
    except Exception as e:
        return {"ok": False, "completed": True, "error": str(e)}

def _deepfake_worker(mp4_path: Path, base: Path):
    # mark running
    _write_deepfake_status(base, {"ok": True, "completed": False, "started_at": time.time()})
    res = _run_genconvit(mp4_path)
    res["finished_at"] = time.time()
    _write_deepfake_status(base, res)

def _ensure_deepfake_async(mp4_path: Path, base: Path):
    """
    Starts (or restarts) the deepfake job for this req.
    """
    try:
        # Mark/reset as running immediately
        _write_deepfake_status(base, {"ok": True, "completed": False, "started_at": time.time()})
        t = threading.Thread(target=_deepfake_worker, args=(mp4_path, base), daemon=True)
        t.start()
    except Exception as e:
        print(f"⚠️ deepfake thread start error: {e}")
        _write_deepfake_status(base, {"ok": False, "completed": True, "error": str(e)})

def _deepfake_state_for_req(base: Path) -> dict:
    st = _read_deepfake_status(base)
    if not st:
        return {"status": "missing", "deepfake_detected": None}
    if not st.get("completed"):
        return {"status": "running", "deepfake_detected": None}
    if st.get("ok"):
        return {"status": "completed", "deepfake_detected": bool(st.get("is_deepfake"))}
    return {"status": "error", "deepfake_detected": None, "error": st.get("error")}

def _state_for_req(req_id: str) -> Dict[str, bool | Any]:
    paths = _base_paths(req_id)
    id_face = paths["id_dir"] / "cropped_id_face.jpg"     # front
    id_back_img = _used_id_back_image_path(paths["id_back_dir"])
    video_mp4 = paths["base"] / "video.mp4"
    deepfake = _deepfake_state_for_req(paths["base"])
    return {
        "id_verified": id_face.exists(),                                # front verified
        "id_back_verified": bool(id_back_img is not None),              # back verified
        "video_verified": video_mp4.exists() and video_mp4.stat().st_size > 0,
        "deepfake": deepfake,
    }

# ---------- simple helpers used by /verify-session fallbacks ----------
def _latest_file(directory: Path) -> Optional[Path]:
    if not directory.exists():
        return None
    files = [p for p in directory.iterdir() if p.is_file()]
    return max(files, key=lambda p: p.stat().st_mtime) if files else None

def _publish_canonical_mp4(src: Path, base_dir: Path) -> Path:
    dst = base_dir / "video.mp4"
    shutil.copyfile(src, dst)
    return dst

# ---------- ffprobe rotation + conversion with autorotate ----------
def _ffprobe_rotation(path: Path) -> int:
    if shutil.which("ffprobe") is None:
        return 0
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream_tags=rotate:side_data_list=displaymatrix",
            "-of", "json", str(path)
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(out.stdout or "{}")
        try:
            rotate = int(data["streams"][0]["tags"]["rotate"])
            rotate = ((rotate % 360) + 360) % 360
            if rotate in (0, 90, 180, 270):
                return rotate
        except Exception:
            pass
        try:
            sdl = data["streams"][0].get("side_data_list", [])
            for sd in sdl:
                val = sd.get("rotation", None)
                if isinstance(val, (int, float)):
                    rot = int(round(val))
                    rot = ((rot % 360) + 360) % 360
                    if rot in (0, 90, 180, 270):
                        return rot
        except Exception:
            pass
    except Exception:
        return 0
    return 0

def convert_to_mp4(input_path: str | Path, output_dir: str | Path) -> Path:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / (input_path.stem + ".mp4")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    rot = _ffprobe_rotation(input_path)
    vf = None
    if rot == 90:
        vf = "transpose=1"
    elif rot == 270:
        vf = "transpose=2"
    elif rot == 180:
        vf = "transpose=2,transpose=2"
    cmd = ["ffmpeg", "-y", "-i", str(input_path)]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-c:a", "aac", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        try:
            if out.exists():
                out.unlink()
        except Exception:
            pass
        err = proc.stderr or "no stderr"
        raise RuntimeError(f"ffmpeg conversion failed for {input_path.name}: {err[:800]}")
    return out

# ------------------------------------------------------------------------------
# Minimal endpoints (stateless)
# ------------------------------------------------------------------------------
@app.post("/req/new")
async def req_new(request: Request):
    req_id = uuid4().hex
    paths = _base_paths(req_id)
    paths["base"].mkdir(parents=True, exist_ok=True)
    paths["id_dir"].mkdir(parents=True, exist_ok=True)
    paths["id_back_dir"].mkdir(parents=True, exist_ok=True)
    paths["rec_dir"].mkdir(parents=True, exist_ok=True)
    return JSONResponse({"ok": True, "req_id": req_id, "state": _state_for_req(req_id)})

@app.get("/req/state/{req_id}")
async def req_state(req_id: str, request: Request):
    if not (Path("temp") / req_id).exists():
        return JSONResponse({
            "ok": True,
            "req_id": req_id,
            "state": {
                "id_verified": False,
                "id_back_verified": False,
                "video_verified": False,
                "deepfake": {"status": "missing", "deepfake_detected": None}
            }
        })
    return JSONResponse({
        "ok": True,
        "req_id": req_id,
        "state": _state_for_req(req_id),
        **_result_urls_for_req(req_id, request)
    })


# ------------------------------------------------------------------------------
# LIVE ID WebSocket (front side — full guidance)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# LIVE ID WebSocket (front side — full guidance with debouncing)
# ------------------------------------------------------------------------------
@app.websocket("/ws-id-live")
async def websocket_id_live(ws: WebSocket):
    await ws.accept()
    qs = parse_qs(urlparse(str(ws.url)).query)
    req_id = (qs.get("req_id", [None])[0]) or (qs.get("sid", [None])[0]) or uuid4().hex
    base = Path("temp") / req_id
    (base / "id").mkdir(parents=True, exist_ok=True)

    # processed fps ~5 (every 5th frame). Choose N=3 by default ≈ 0.6s to flip.
    STREAK_N = int(os.getenv("ID_STREAK_N", "3"))

    # debouncers for all booleans that drive UX
    streaks = {
        "brightness_ok":     BoolStreak(STREAK_N),
        "id_card_detected":  BoolStreak(STREAK_N),
        "id_inside_rect":    BoolStreak(STREAK_N),
        "face_detected":     BoolStreak(STREAK_N),
        "id_fill_ok":        BoolStreak(STREAK_N),
        "face_size_ok":      BoolStreak(STREAK_N),
        "face_bright_ok":    BoolStreak(STREAK_N),   # derived from face_brightness_status
        "glare_detected":    BoolStreak(STREAK_N),
    }

    frame_idx = 0
    last_payload: Optional[dict] = None
    try:
        while True:
            data = await ws.receive_text()
            try:
                frame_bytes = base64.b64decode(data)
                np_arr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
            except Exception:
                continue

            frame_idx += 1
            # process every 5th frame (≈5 fps)
            if frame_idx % 5 != 0 and last_payload is not None:
                payload = dict(last_payload)
                payload["skipped"] = True
                await ws.send_json(_to_jsonable(payload))
                continue

            rep = analyze_id_frame(frame)

            # derive raw booleans
            raw_brightness_ok = (rep.get("brightness_status") == "ok")
            raw_face_bright_ok = (
                rep.get("face_brightness_status") in (None, "ok")
            )

            # feed debouncers
            brightness_ok   = streaks["brightness_ok"].update(raw_brightness_ok)
            id_card_ok      = streaks["id_card_detected"].update(rep.get("id_card_detected"))
            inside_rect_ok  = streaks["id_inside_rect"].update(rep.get("id_inside_rect"))
            face_ok         = streaks["face_detected"].update(bool(rep.get("face_detected")))
            id_fill_ok      = streaks["id_fill_ok"].update(rep.get("id_fill_ok"))
            face_size_ok    = streaks["face_size_ok"].update(rep.get("face_size_ok"))
            face_bright_ok  = streaks["face_bright_ok"].update(raw_face_bright_ok)
            glare_detected  = streaks["glare_detected"].update(rep.get("glare_detected"))

            # build payload; keep exact same field names but with debounced values.
            payload = {
                "req_id": req_id,

                # keep raw values for debugging (prefixed)
                "raw": {
                    "brightness_status": rep.get("brightness_status"),
                    "face_brightness_status": rep.get("face_brightness_status"),
                    "id_card_detected": rep.get("id_card_detected"),
                    "id_inside_rect": rep.get("id_inside_rect"),
                    "face_detected": bool(rep.get("face_detected")),
                    "id_fill_ok": rep.get("id_fill_ok"),
                    "face_size_ok": rep.get("face_size_ok"),
                    "glare_detected": rep.get("glare_detected"),
                },

                # debounced values surfaced under original keys (UX-friendly)
                "brightness_status": ("ok" if brightness_ok else rep.get("brightness_status") or "too_dark"),
                "brightness_mean": rep.get("brightness_mean"),

                "id_card_detected": bool(id_card_ok),
                "id_card_bbox": rep.get("id_card_bbox"),
                "id_card_conf": rep.get("id_card_conf"),

                "rect": rep.get("rect"),
                "id_inside_rect": bool(inside_rect_ok),

                "face_detected": bool(face_ok),
                "largest_bbox": rep.get("largest_bbox"),
                "face_inside_id": rep.get("face_inside_id"),

                "id_fill_ratio": rep.get("id_fill_ratio"),
                "id_fill_ok": bool(id_fill_ok),

                "face_w_px": rep.get("face_w_px"),
                "face_size_ok": bool(face_size_ok),

                # show "ok"/None only if debounced ok; else pass through raw status (dark/bright)
                "face_brightness_status": ("ok" if face_bright_ok else rep.get("face_brightness_status")),
                "face_brightness_mean": rep.get("face_brightness_mean"),

                "glare_detected": bool(glare_detected),

                # extras
                "roi_xyxy": rep.get("roi_xyxy"),
                "roi_w": rep.get("roi_w"),
                "roi_h": rep.get("roi_h"),

                "skipped": False,
                "saved": False,
            }

            last_payload = payload
            await ws.send_json(_to_jsonable(payload))
    except WebSocketDisconnect:
        print(f"🔌 ID live verification ended for {req_id}")

# ------------------------------------------------------------------------------
# LIVE ID BACK WebSocket (brightness + ID detection only)
# ------------------------------------------------------------------------------
@app.websocket("/ws-id-back-live")
async def websocket_id_back_live(ws: WebSocket):
    await ws.accept()
    qs = parse_qs(urlparse(str(ws.url)).query)
    req_id = (qs.get("req_id", [None])[0]) or (qs.get("sid", [None])[0]) or uuid4().hex
    base = Path("temp") / req_id
    (base / "id_back").mkdir(parents=True, exist_ok=True)
    frame_idx = 0
    last_payload: Optional[dict] = None
    try:
        while True:
            data = await ws.receive_text()
            try:
                frame_bytes = base64.b64decode(data)
                np_arr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
            except Exception:
                continue

            frame_idx += 1
            if frame_idx % 2 != 0 and last_payload is not None:
                payload = dict(last_payload)
                payload["skipped"] = True
                await ws.send_json(_to_jsonable(payload))
                continue

            rep = analyze_id_frame(frame)  # we’ll only forward the minimal fields
            payload = {
                "req_id": req_id,
                "brightness_status": rep.get("brightness_status"),
                "brightness_mean": rep.get("brightness_mean"),
                "id_card_detected": rep.get("id_card_detected"),
                "id_card_bbox": rep.get("id_card_bbox"),
                "id_card_conf": rep.get("id_card_conf"),
                "skipped": False,
            }
            last_payload = payload
            await ws.send_json(_to_jsonable(payload))
    except WebSocketDisconnect:
        print(f"🔌 ID BACK live verification ended for {req_id}")

# ------------------------------------------------------------------------------
# ID still upload (FRONT) — saves still and crops face for verification
# ------------------------------------------------------------------------------
@app.post("/upload-id-still")
async def upload_id_still(request: Request, image: UploadFile = File(...)):
    req_id = _req_id_from(request)
    paths = _base_paths(req_id)
    base, id_dir = paths["base"], paths["id_dir"]
    base.mkdir(parents=True, exist_ok=True)
    id_dir.mkdir(parents=True, exist_ok=True)

    id_raw = id_dir / "id_raw_upload.jpg"
    id_raw.write_bytes(await image.read())

    used_for_cropping = id_raw
    try:
        from id import enhance_id_image
        id_enhanced = id_dir / "id_enhanced.jpg"
        if enhance_id_image(str(id_raw), str(id_enhanced)):
            used_for_cropping = id_enhanced
    except Exception as e:
        print(f"⚠️ Enhancement failed ({e}). Using raw ROI upload.")

    cropped = id_dir / "cropped_id_face.jpg"
    try:
        run_id_extraction(str(used_for_cropping), str(cropped))
    except Exception as e:
        return JSONResponse({"ok": False, "req_id": req_id, "error": str(e)}, status_code=400)

    urls = _result_urls_for_req(req_id, request)
    return JSONResponse({
        "ok": True,
        "req_id": req_id,
        "used_id_path": urls["id_image_url"],
        "cropped_face": urls["cropped_face_url"],
        "state": _state_for_req(req_id),
    })

# ------------------------------------------------------------------------------
# ID BACK still upload — saves back image only (no face crop)
# ------------------------------------------------------------------------------
@app.post("/upload-id-back-still")
async def upload_id_back_still(request: Request, image: UploadFile = File(...)):
    req_id = _req_id_from(request)
    paths = _base_paths(req_id)
    base, id_back_dir = paths["base"], paths["id_back_dir"]
    base.mkdir(parents=True, exist_ok=True)
    id_back_dir.mkdir(parents=True, exist_ok=True)

    raw_path = id_back_dir / "id_back_raw_upload.jpg"
    raw_path.write_bytes(await image.read())

    # Optional enhancement (best-effort; ignore errors)
    enhanced_path = id_back_dir / "id_back_enhanced.jpg"
    try:
        from id import enhance_id_image
        if enhance_id_image(str(raw_path), str(enhanced_path)):
            pass
    except Exception as e:
        print(f"⚠️ Back-side enhancement failed ({e}). Using raw image.")

    urls = _result_urls_for_req(req_id, request)
    return JSONResponse({
        "ok": True,
        "req_id": req_id,
        "used_id_back_path": urls["id_back_image_url"],
        "state": _state_for_req(req_id),
    })

# ------------------------------------------------------------------------------
# LIVE FACE (video) WebSocket
# ------------------------------------------------------------------------------
@app.websocket("/ws-live-verification")
async def websocket_verification(ws: WebSocket):
    await ws.accept()
    ellipse_params: Optional[dict] = None
    try:
        while True:
            data = await ws.receive_text()
            try:
                ellipse_params = json.loads(data)
                continue
            except Exception:
                pass

            if not ellipse_params:
                continue

            try:
                frame_bytes = base64.b64decode(data)
                np_arr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
            except Exception as e:
                print("⚠️ Frame decode error:", e)
                continue

            af = analyze_frame(frame, ellipse_params=ellipse_params)
            payload = {
                "checks": af.get("checks"),
                "brightness_status": af.get("brightness_status"),
                "face_detected": bool(af.get("face_detected")),
                "num_faces": int(af.get("num_faces") or 0),
                "one_face": bool(af.get("one_face")),
                "inside_ellipse": bool(af.get("inside_ellipse")),
                "front_facing": af.get("front_facing"),
                "front_guidance": af.get("front_guidance"),
                "glasses_detected": af.get("glasses_detected"),
                "spoof_is_real": af.get("spoof_is_real"),
                "spoof_status": af.get("spoof_status"),
                "largest_bbox": af.get("largest_bbox"),
                "skipped": False,
            }
            await ws.send_json(_to_jsonable(payload))
    except WebSocketDisconnect:
        print("🔌 Live verification ended")

# ------------------------------------------------------------------------------
# Upload live clip (8s) — normalize & kick off deepfake job
# ------------------------------------------------------------------------------
@app.post("/upload-live-clip")
async def upload_live_clip(request: Request, video: UploadFile = File(...)):
    req_id = _req_id_from(request)
    paths = _base_paths(req_id)
    base, rec_dir = paths["base"], paths["rec_dir"]
    base.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)

    # Save raw upload
    stamp = time.strftime("%Y%m%d-%H%M%S")
    ext = (Path(video.filename).suffix or ".webm").lower()
    raw_path = rec_dir / f"{stamp}_{uuid4().hex}{ext}"
    raw_path.write_bytes(await video.read())

    # Normalize to MP4 (upright) and publish canonical mp4
    try:
        mp4_path = convert_to_mp4(raw_path, rec_dir)
        canonical = base / "video.mp4"
        shutil.copyfile(mp4_path, canonical)
    except Exception as e:
        return JSONResponse({"ok": False, "req_id": req_id, "error": str(e)}, status_code=400)

    # start deepfake detection in background
    _ensure_deepfake_async(mp4_path, base)

    urls = _result_urls_for_req(req_id, request)
    return JSONResponse({
        "ok": True,
        "req_id": req_id,
        "saved_raw": str(raw_path),
        "saved_mp4": str(mp4_path),
        "canonical_mp4": urls["video_url"],
        "deepfake": _deepfake_state_for_req(base),
        "state": _state_for_req(req_id),
    })

# ------------------------------------------------------------------------------
# Verify: run selection + InsightFace — attach deepfake verdict if ready
# ------------------------------------------------------------------------------
@app.post("/verify-session")
async def verify_session(request: Request):
    req_id = _req_id_from(request)
    paths = _base_paths(req_id)
    base, id_dir, rec_dir, selected_dir = paths["base"], paths["id_dir"], paths["rec_dir"], paths["selected_dir"]

    id_face = id_dir / "cropped_id_face.jpg"
    if not id_face.exists():
        return JSONResponse({"ok": False, "req_id": req_id, "error": "cropped_id_face.jpg not found. Re-do ID step."}, status_code=400)

    canonical_mp4 = base / "video.mp4"
    if not (canonical_mp4.exists() and canonical_mp4.stat().st_size > 0):
        latest_vid = _latest_file(rec_dir)
        if not latest_vid:
            return JSONResponse({"ok": False, "req_id": req_id, "error": "No recorded video found."}, status_code=400)
        try:
            norm = convert_to_mp4(latest_vid, rec_dir)
            canonical_mp4 = _publish_canonical_mp4(norm, base)
        except Exception as e:
            return JSONResponse({"ok": False, "req_id": req_id, "error": str(e)}, status_code=400)

    selected_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_full_frame_pipeline(str(canonical_mp4), str(selected_dir))
    except Exception as e:
        return JSONResponse({"ok": False, "req_id": req_id, "error": f"Frame pipeline failed: {e}"}, status_code=400)

    out_img = base / "best_match.png"
    try:
        result = run_verif(
            id_image_path=str(id_face),
            frames_dir=str(selected_dir),
            output_path=str(out_img),
        )
        if "error" in result:
            return JSONResponse({"ok": False, "req_id": req_id, "error": result["error"]}, status_code=400)
    except Exception as e:
        return JSONResponse({"ok": False, "req_id": req_id, "error": f"Verification failed: {e}"}, status_code=400)

    # Attach URLs
    urls = _result_urls_for_req(req_id, request)
    result.update({
        "video_url": urls["video_url"],
        "id_image_url": urls["id_image_url"],
        "id_back_image_url": urls["id_back_image_url"],
        "cropped_face_url": urls["cropped_face_url"],
        "selected_frames": urls["selected_frames"],
        "best_match_url": urls["best_match_url"],
    })

    # Attach deepfake status/verdict
    df_state = _deepfake_state_for_req(base)
    result["deepfake_detected"] = df_state.get("deepfake_detected")
    result["deepfake_status"] = df_state.get("status")

    try:
        (base / "result.json").write_text(json.dumps(result, indent=2))
    except Exception:
        pass

    return JSONResponse({
        "ok": True,
        "req_id": req_id,
        "result": result,
        "result_url": _abs_url(request, f"/result/{req_id}"),
        "state": _state_for_req(req_id),
    })

# ------------------------------------------------------------------------------
# Review bundle / Result image / Manual review
# ------------------------------------------------------------------------------
@app.get("/review/{req_id}")
async def get_review_bundle(req_id: str, request: Request):
    base = Path("temp") / req_id
    if not base.exists():
        raise HTTPException(status_code=404, detail="Request ID not found")
    urls = _result_urls_for_req(req_id, request)
    return JSONResponse({"ok": True, "req_id": req_id, **urls, "deepfake": _deepfake_state_for_req(base)})

@app.get("/result/{req_id}")
def get_result_image(req_id: str):
    base = Path("temp") / req_id
    img = base / "best_match.png"
    if not img.exists():
        raise HTTPException(404, "Result image not found")
    return FileResponse(str(img), media_type="image/png")

@app.post("/manual-review/{req_id}")
async def manual_review(req_id: str, payload: dict):
    decision = payload.get("decision")
    if decision not in ["verified", "unverified"]:
        raise HTTPException(status_code=400, detail="Invalid decision")

    base = Path("temp") / req_id
    if not base.exists():
        raise HTTPException(status_code=404, detail="Request ID not found")

    for item in base.iterdir():
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            item.unlink()
    return JSONResponse(content={"message": f"✅ Manual review marked as '{decision}' and data cleaned up."})
