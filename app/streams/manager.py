"""Owns up to `stream.max_streams` (source + worker) pairs."""
from __future__ import annotations

import dataclasses
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.streams.source import FrameSource, is_network_url
from app.streams.worker import InferenceWorker, StreamSettings

log = logging.getLogger(__name__)


class StreamError(Exception):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status = status


@dataclass
class Stream:
    id: str
    name: str
    url: str
    created: float
    source: FrameSource
    worker: InferenceWorker

    def info(self) -> dict:
        st, w = self.source.status, self.worker
        return {
            "id": self.id, "name": self.name, "url": self.url, "created": self.created,
            "settings": dataclasses.asdict(w.settings),
            "status": {
                "state": st.state, "error": w.error or st.error, "src_fps": st.src_fps,
                "width": st.width, "height": st.height, "reconnects": st.reconnects,
                "infer_fps": w.infer_fps, "infer_count": w.infer_count,
                "latency_ms": round(w.last.latency_ms), "detected": w.last.detected,
                "last_result_ts": w.last.frame_ts,
            },
        }


class StreamManager:
    def __init__(self, cfg: dict, root: Path, server_url: str, publish: Callable[[dict], None],
                 events=None):
        self.cfg = cfg
        self.root = root.resolve()
        self.server_url = server_url
        self.publish = publish
        self.events = events  # EventStore | None
        self.streams: dict[str, Stream] = {}
        self._lock = threading.Lock()

    # ---- helpers ---------------------------------------------------------------------
    def resolve_path(self, url: str) -> str | None:
        """None for network URLs; absolute path for files inside the project. Raises on invalid."""
        if is_network_url(url):
            return None
        p = Path(url)
        p = (p if p.is_absolute() else self.root / p).resolve()
        if self.root not in p.parents:
            raise StreamError(400, f"file must be inside the project directory: {url}")
        if not p.is_file():
            raise StreamError(400, f"file not found: {url}")
        return str(p)

    def _settings(self, base: StreamSettings | None = None, **kw) -> StreamSettings:
        inf, stc = self.cfg["inference"], self.cfg["stream"]
        s = base or StreamSettings(prompt="", num_frames=inf["num_frames"], window_sec=inf["window_sec"],
                                   interval_ms=stc["interval_ms"], judge_threshold=inf["judge_threshold"])
        s = dataclasses.replace(s, **{k: v for k, v in kw.items() if v is not None})
        if not s.prompt.strip():
            raise StreamError(400, "prompt is required")
        if not 1 <= s.num_frames <= self.cfg["serving"]["max_images_per_request"]:
            raise StreamError(400, f"num_frames must be 1..{self.cfg['serving']['max_images_per_request']}")
        if not 0 < s.window_sec <= stc["buffer_sec"]:
            raise StreamError(400, f"window_sec must be in (0, {stc['buffer_sec']}]")
        if s.interval_ms < 0 or not 0 <= s.judge_threshold <= 1:
            raise StreamError(400, "interval_ms must be >= 0 and judge_threshold in [0, 1]")
        return s

    # ---- API -------------------------------------------------------------------------
    def add(self, url: str, prompt: str, name: str | None = None, **kw) -> Stream:
        url = url.strip()
        path = self.resolve_path(url)
        settings = self._settings(prompt=prompt, **kw)
        with self._lock:
            if len(self.streams) >= self.cfg["stream"]["max_streams"]:
                raise StreamError(409, f"max {self.cfg['stream']['max_streams']} streams")
            sid = uuid.uuid4().hex[:8]
            stc, pp = self.cfg["stream"], self.cfg["preprocess"]
            source = FrameSource(url, path, stc["buffer_max_side"], stc["buffer_fps"], stc["buffer_sec"],
                                 stc["reconnect_max_sec"])
            stream_name = name or Path(url).name or url
            on_det = None
            if self.events is not None:
                def on_det(prompt, ts, score, boxes, frame, _sid=sid):
                    st = self.streams.get(_sid)
                    self.events.on_detection(_sid, st.name if st else stream_name, prompt, ts, score, boxes, frame)
            worker = InferenceWorker(sid, source, settings, self.cfg, self.server_url, self.publish, on_det)
            stream = Stream(sid, stream_name, url, time.time(), source, worker)
            self.streams[sid] = stream
        source.start()
        worker.start()
        log.info("stream added %s %s", sid, url)
        return stream

    def get(self, sid: str) -> Stream:
        s = self.streams.get(sid)
        if s is None:
            raise StreamError(404, f"stream {sid} not found")
        return s

    def update(self, sid: str, name: str | None = None, **kw) -> Stream:
        s = self.get(sid)
        s.worker.settings = self._settings(s.worker.settings, **kw)
        if name:
            s.name = name
        return s

    def remove(self, sid: str) -> None:
        with self._lock:
            s = self.streams.pop(sid, None)
        if s is None:
            raise StreamError(404, f"stream {sid} not found")
        threading.Thread(target=lambda: (s.worker.stop(), s.source.stop()), daemon=True).start()
        log.info("stream removed %s", sid)

    def shutdown(self) -> None:
        for sid in list(self.streams):
            s = self.streams.pop(sid)
            s.worker._stop.set()
            s.source.stop()
