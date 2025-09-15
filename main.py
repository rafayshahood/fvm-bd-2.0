# main.py
from __future__ import annotations
from typing import Dict, Any, Optional, List

import base64
import json
import os
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from uuid import uuid4

import cv2
import numpy as np
from fastapi import (
    FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from id import analyze_id_frame, run_id_extraction
from new_verif import run_verif  # noqa: F401  (used elsewhere in your project)
from all_video import (
    run_full_frame_pipeline,   # noqa: F401
    convert_to_mp4,
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
    id_dir = base / "id"
    rec_dir = base / "recordings"
    selected_dir = base / "selected_faces"
    return {"base": base, "id_dir": id_dir, "rec_dir": rec_dir, "selected_dir": selected_dir}

def _used_id_image_path(id_dir: Path) -> Optional[Path]:
    enhanced = id_dir / "id_enhanced.jpg"
    raw = id_dir / "id_raw_upload.jpg"
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
    base, id_dir, selected_dir = paths["base"], paths["id_dir"], paths["selected_dir"]
    used_id = _used_id_image_path(id_dir)
    cropped = id_dir / "cropped_id_face.jpg"
    video = base / "video.mp4"
    best = base / "best_match.png"

    id_image_url = f"/temp/{req_id}/id/{used_id.name}" if used_id else None
    cropped_face_url = f"/temp/{req_id}/id/{cropped.name}" if cropped.exists() else None
    video_url = f"/temp/{req_id}/video.mp4" if video.exists() else None
    best_match_url = f"/temp/{req_id}/best_match.png" if best.exists() else None
    selected_frames = [f"/temp/{req_id}/selected_faces/{p.name}" for p in _selected_frame_files(selected_dir)]

    return {
        "id_image_url": _abs_url(request, id_image_url),
        "cropped_face_url": _abs_url(request, cropped_face_url),
        "video_url": _abs_url(request, video_url),
        "selected_frames": [_abs_url(request, u) for u in selected_frames],
        "best_match_url": _abs_url(request, best_match_url),
    }

def _state_for_req(req_id: str) -> Dict[str, bool]:
    paths = _base_paths(req_id)
    id_face = paths["id_dir"] / "cropped_id_face.jpg"
    video_mp4 = paths["base"] / "video.mp4"
    return {
        "id_verified": id_face.exists(),
        "video_verified": video_mp4.exists() and video_mp4.stat().st_size > 0,
    }

# ------------------------------------------------------------------------------
# Minimal endpoints (stateless)
# ------------------------------------------------------------------------------
@app.post("/req/new")
async def req_new(request: Request):
    req_id = uuid4().hex
    paths = _base_paths(req_id)
    paths["base"].mkdir(parents=True, exist_ok=True)
    paths["id_dir"].mkdir(parents=True, exist_ok=True)
    paths["rec_dir"].mkdir(parents=True, exist_ok=True)
    return JSONResponse({"ok": True, "req_id": req_id, "state": _state_for_req(req_id)})

@app.get("/req/state/{req_id}")
async def req_state(req_id: str, request: Request):
    if not (Path("temp") / req_id).exists():
        return JSONResponse({"ok": True, "req_id": req_id, "state": {"id_verified": False, "video_verified": False}})
    return JSONResponse({"ok": True, "req_id": req_id, "state": _state_for_req(req_id), **_result_urls_for_req(req_id, request)})

# ------------------------------------------------------------------------------
# LIVE ID WebSocket (guidance)
# ------------------------------------------------------------------------------
@app.websocket("/ws-id-live")
async def websocket_id_live(ws: WebSocket):
    await ws.accept()

    qs = parse_qs(urlparse(str(ws.url)).query)
    req_id = (qs.get("req_id", [None])[0]) or (qs.get("sid", [None])[0]) or uuid4().hex

    base = Path("temp") / req_id
    (base / "id").mkdir(parents=True, exist_ok=True)

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
            if frame_idx % 5 != 0 and last_payload is not None:
                payload = dict(last_payload)
                payload["skipped"] = True
                await ws.send_json(_to_jsonable(payload))
                continue

            rep = analyze_id_frame(frame)
            payload = {
                "req_id": req_id,
                "brightness_status": rep.get("brightness_status"),
                "brightness_mean": rep.get("brightness_mean"),
                "id_card_detected": rep.get("id_card_detected"),
                "id_card_bbox": rep.get("id_card_bbox"),
                "id_card_conf": rep.get("id_card_conf"),
                "rect": rep.get("rect"),
                "id_inside_rect": rep.get("id_inside_rect"),
                "face_detected": bool(rep.get("face_detected")),
                "largest_bbox": rep.get("largest_bbox"),
                "face_inside_id": rep.get("face_inside_id"),
                "id_fill_ratio": rep.get("id_fill_ratio"),
                "id_fill_ok": rep.get("id_fill_ok"),
                "face_w_px": rep.get("face_w_px"),
                "face_size_ok": rep.get("face_size_ok"),
                "face_brightness_status": rep.get("face_brightness_status"),
                "face_brightness_mean": rep.get("face_brightness_mean"),
                "glare_detected": rep.get("glare_detected"),
                "skipped": False,
                "saved": False,
            }

            last_payload = payload
            await ws.send_json(_to_jsonable(payload))

    except WebSocketDisconnect:
        print(f"🔌 ID live verification ended for {req_id}")

# ------------------------------------------------------------------------------
# ID still upload
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
        print(f"⚠️ Enhancement failed ({e}). Using raw upload.")

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
# LIVE FACE (video) WebSocket — FE throttles frames; we process all received
# ------------------------------------------------------------------------------
@app.websocket("/ws-live-verification")
async def websocket_verification(ws: WebSocket):
    await ws.accept()
    ellipse_params: Optional[dict] = None
    try:
        while True:
            data = await ws.receive_text()
            # 1) ellipse JSON
            try:
                ellipse_params = json.loads(data)
                continue
            except Exception:
                pass  # not JSON → treat as image

            if not ellipse_params:
                continue

            # 2) frame
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
# Upload live clip (8s) — keep ALL clips in recordings/, plus publish canonical MP4
# ------------------------------------------------------------------------------
@app.post("/upload-live-clip")
async def upload_live_clip(request: Request, video: UploadFile = File(...)):
    req_id = _req_id_from(request)
    paths = _base_paths(req_id)
    base, rec_dir = paths["base"], paths["rec_dir"]
    base.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)

    # Save the raw upload in recordings/
    stamp = time.strftime("%Y%m%d-%H%M%S")
    ext = (Path(video.filename).suffix or ".webm").lower()
    raw_path = rec_dir / f"{stamp}_{uuid4().hex}{ext}"
    raw_path.write_bytes(await video.read())

    # Convert to MP4 (in recordings/) when needed; if already MP4, reuse it
    try:
        if ext not in (".mp4",):
            mp4_path = convert_to_mp4(raw_path, rec_dir)  # out: recordings/<stem>.mp4
        else:
            mp4_path = raw_path

        # Also publish/overwrite canonical temp/<req_id>/video.mp4 (used by homepage/state)
        canonical = base / "video.mp4"
        shutil.copyfile(mp4_path, canonical)

    except Exception as e:
        return JSONResponse({"ok": False, "req_id": req_id, "error": str(e)}, status_code=400)

    urls = _result_urls_for_req(req_id, request)
    return JSONResponse({
        "ok": True,
        "req_id": req_id,
        "saved_raw": str(raw_path),
        "saved_mp4": str(mp4_path),
        "canonical_mp4": urls["video_url"],
        "state": _state_for_req(req_id),
    })

# ------------------------------------------------------------------------------
# Verify: run selection + InsightFace
# ------------------------------------------------------------------------------
@app.post("/verify-session")
async def verify_session(request: Request):
    req_id = _req_id_from(request)
    paths = _base_paths(req_id)
    base, id_dir, rec_dir, selected_dir = paths["base"], paths["id_dir"], paths["rec_dir"], paths["selected_dir"]

    id_face = id_dir / "cropped_id_face.jpg"
    if not id_face.exists():
        return JSONResponse({"ok": False, "req_id": req_id, "error": "cropped_id_face.jpg not found. Re-do ID step."}, status_code=400)

    # Prefer existing canonical video.mp4 (already created at upload)
    canonical_mp4 = base / "video.mp4"
    if not (canonical_mp4.exists() and canonical_mp4.stat().st_size > 0):
        latest_vid = _latest_file(rec_dir)
        if not latest_vid:
            return JSONResponse({"ok": False, "req_id": req_id, "error": "No recorded video found."}, status_code=400)
        try:
            canonical_mp4 = _publish_canonical_mp4(latest_vid, base)
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

    urls = _result_urls_for_req(req_id, request)
    result.update({
        "video_url": urls["video_url"],
        "id_image_url": urls["id_image_url"],
        "cropped_face_url": urls["cropped_face_url"],
        "selected_frames": urls["selected_frames"],
        "best_match_url": urls["best_match_url"],
    })

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
    return JSONResponse({"ok": True, "req_id": req_id, **urls})

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