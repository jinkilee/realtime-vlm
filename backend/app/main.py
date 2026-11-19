"""realtime-vlm app: stream management REST + WebSocket results + MJPEG preview."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import httpx
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.inference.preprocess import resize_long_edge
from app.events.store import EventStore
from app.profiling import prof
from app.schemas import StreamCreate, StreamUpdate
from app.streams.manager import StreamError, StreamManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

CFG = config.load()
prof.enabled = bool(CFG.get("profiling", {}).get("enabled", False))
ROOT = config.ROOT
VIDEO_DIR = ROOT / "videos"
UPLOAD_DIR = VIDEO_DIR / "uploads"
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts"}
SERVING_URL = config.serving_url(CFG)
EVENT_DIR = ROOT / CFG["events"]["dir"]


# ---- WebSocket fan-out (thread-safe publish from worker threads) ---------------------------
class Hub:
    def __init__(self):
        self.clients: set[asyncio.Queue] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def publish(self, msg: dict) -> None:
        if self.loop is None:
            return
        data = json.dumps(msg, ensure_ascii=False)
        self.loop.call_soon_threadsafe(self._fanout, data)

    def _fanout(self, data: str) -> None:
        for q in list(self.clients):
            if q.full():
                try:
                    q.get_nowait()  # drop oldest for slow clients
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(data)


hub = Hub()
events = EventStore(EVENT_DIR, CFG["events"]["cooldown_sec"], CFG["events"]["max_in_memory"], ROOT, hub.publish)
manager = StreamManager(CFG, ROOT, SERVING_URL, hub.publish, events)


def _chown_like_workspace(path: Path) -> None:
    """Files created by the (root) container get the owner of the mounted project dir."""
    try:
        st = ROOT.stat()
        os.chown(path, st.st_uid, st.st_gid)
    except OSError:
        pass


async def _status_loop():
    while True:
        t = time.perf_counter()
        await asyncio.sleep(1.0)
        prof.add("loop.lag", (time.perf_counter() - t - 1.0) * 1000)  # event loop / GIL stalls (WS, REST)
        if hub.clients:
            hub.publish({"type": "status", "ts": time.time(),
                         "streams": [s.info() for s in manager.streams.values()]})


@asynccontextmanager
async def lifespan(_app: FastAPI):
    hub.loop = asyncio.get_running_loop()
    task = asyncio.create_task(_status_loop())
    log.info("app ready: serving=%s model=%s strategy=%s", SERVING_URL, CFG["serving"]["model_id"],
             CFG["inference"]["strategy"])
    yield
    task.cancel()
    manager.shutdown()


app = FastAPI(title="realtime-vlm", lifespan=lifespan)


@app.exception_handler(StreamError)
async def _stream_error(_req, exc: StreamError):
    return JSONResponse(status_code=exc.status, content={"detail": str(exc)})


# ---- health / config ------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    serving_ok = False
    try:
        async with httpx.AsyncClient(timeout=2) as c:
            serving_ok = (await c.get(f"{SERVING_URL}/health")).status_code == 200
    except httpx.HTTPError:
        pass
    return {
        "app": "ok",
        "serving": {"backend": CFG["serving"]["backend"], "ok": serving_ok,
                    "model": Path(CFG["serving"]["model_id"]).name},
        "inference": {k: CFG["inference"][k] for k in ("strategy", "num_frames", "window_sec", "judge_threshold")},
        "streams": len(manager.streams),
        "max_streams": CFG["stream"]["max_streams"],
    }


# ---- streams --------------------------------------------------------------------------------
@app.get("/api/streams")
def list_streams():
    return [s.info() for s in manager.streams.values()]


@app.post("/api/streams", status_code=201)
def add_stream(body: StreamCreate):
    s = manager.add(body.url, body.prompt, body.name, num_frames=body.num_frames, window_sec=body.window_sec,
                    interval_ms=body.interval_ms, judge_threshold=body.judge_threshold)
    return s.info()


@app.get("/api/streams/{sid}")
def get_stream(sid: str):
    return manager.get(sid).info()


@app.patch("/api/streams/{sid}")
def update_stream(sid: str, body: StreamUpdate):
    return manager.update(sid, **body.model_dump()).info()


@app.delete("/api/streams/{sid}", status_code=204)
def delete_stream(sid: str):
    manager.remove(sid)


@app.get("/api/streams/{sid}/mjpeg")
def mjpeg(sid: str, overlay: bool = Query(False, description="draw the last boxes server-side")):
    stream = manager.get(sid)
    stc = CFG["stream"]
    period = 1.0 / stc["mjpeg_fps"]
    boundary = "frame"

    def gen():
        last_seq = -1
        while sid in manager.streams:
            t0 = time.monotonic()
            frame, _ts, seq = stream.source.latest()
            if frame is not None and seq != last_seq:
                last_seq = seq
                t_m = time.perf_counter()
                img = resize_long_edge(frame, stc["mjpeg_max_side"])
                if overlay:
                    img = img.copy()
                    h, w = img.shape[:2]
                    for b in stream.worker.last.boxes:
                        x1, y1, x2, y2 = b.to_pixels(w, h)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, stc["mjpeg_quality"]])
                prof.add("mjpeg.encode", (time.perf_counter() - t_m) * 1000)  # resize + overlay + JPEG
                if ok:
                    yield (f"--{boundary}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(buf)}\r\n\r\n"
                           .encode() + buf.tobytes() + b"\r\n")
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}",
                             headers={"Cache-Control": "no-store"})


@app.get("/api/streams/{sid}/snapshot")
def snapshot(sid: str):
    frame, _ts, _ = manager.get(sid).source.latest()
    if frame is None:
        raise HTTPException(404, "no frame yet")
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return StreamingResponse(iter([buf.tobytes()]), media_type="image/jpeg")


# ---- profiling (see app/profiling.py) ----------------------------------------------------------
@app.get("/api/profile")
def get_profile():
    return prof.summary()


@app.post("/api/profile")
def set_profile(enabled: bool = True, reset: bool = True):
    """Turn stage timing on/off at runtime; reset=true clears what was collected."""
    prof.enabled = enabled
    if reset:
        prof.reset()
    return prof.summary()


# ---- videos ---------------------------------------------------------------------------------
@app.get("/api/videos")
def list_videos():
    if not VIDEO_DIR.exists():
        return []
    files = sorted(p for p in VIDEO_DIR.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    return [{"path": str(p.relative_to(ROOT)), "size": p.stat().st_size} for p in files]


@app.post("/api/upload", status_code=201)
async def upload(file: UploadFile = File(...)):
    name = Path(file.filename or "upload.mp4").name
    if Path(name).suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(400, f"unsupported extension; allowed: {sorted(VIDEO_EXTS)}")
    name = re.sub(r"[^\w.\-]", "_", name)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _chown_like_workspace(UPLOAD_DIR)
    dest = UPLOAD_DIR / name
    stem, suffix, n = dest.stem, dest.suffix, 1
    while dest.exists():  # never overwrite: name_1.mp4, name_2.mp4, ...
        dest = UPLOAD_DIR / f"{stem}_{n}{suffix}"
        n += 1
    with open(dest, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    _chown_like_workspace(dest)
    return {"path": str(dest.relative_to(ROOT)), "size": dest.stat().st_size}


# ---- events ---------------------------------------------------------------------------------
@app.get("/api/events")
def list_events(stream_id: str | None = None, limit: int = Query(100, ge=1, le=1000)):
    return events.list(stream_id, limit)


@app.delete("/api/events")
def clear_events():
    return {"deleted": events.clear()}


app.mount("/event-files", StaticFiles(directory=EVENT_DIR), name="event-files")


# ---- websocket ------------------------------------------------------------------------------
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    hub.clients.add(q)
    try:
        await websocket.send_text(json.dumps({"type": "status", "ts": time.time(),
                                              "streams": [s.info() for s in manager.streams.values()]}))
        while True:
            await websocket.send_text(await q.get())
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.clients.discard(q)


@app.get("/")
def index():
    # the UI is served by the frontend container (http://127.0.0.1:8000); this is the API only
    return RedirectResponse("/docs")
