"""Per-stream inference loop: sample N frames from the ring buffer -> VLM -> publish result."""
from __future__ import annotations

import copy
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import httpx

from app.inference.parser import Box
from app.inference.vllm_client import VLLMClient
from app.streams.source import FrameSource

log = logging.getLogger(__name__)


@dataclass
class StreamSettings:
    prompt: str
    num_frames: int
    window_sec: float
    interval_ms: int
    judge_threshold: float


@dataclass
class LastResult:
    frame_ts: float = 0.0
    boxes: list[Box] = field(default_factory=list)
    score: float | None = None
    detected: bool = False
    latency_ms: float = 0.0


class InferenceWorker:
    def __init__(self, stream_id: str, source: FrameSource, settings: StreamSettings, cfg: dict,
                 server_url: str, publish: Callable[[dict], None]):
        self.stream_id = stream_id
        self.source = source
        self.settings = settings
        self.cfg = copy.deepcopy(cfg)
        self.client = VLLMClient(self.cfg, server_url)
        self.publish = publish
        self.last = LastResult()
        self.error = ""
        self.infer_count = 0
        self._done: deque[float] = deque(maxlen=20)   # completion times, for infer fps
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"infer:{stream_id}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.cfg["inference"]["timeout_sec"] + 2)

    @property
    def infer_fps(self) -> float:
        if len(self._done) < 2 or time.time() - self._done[-1] > 5:
            return 0.0
        return round((len(self._done) - 1) / (self._done[-1] - self._done[0]), 2)

    def _run(self) -> None:
        last_ts = 0.0
        while not self._stop.is_set():
            s = self.settings  # replaced atomically by PATCH
            t0 = time.monotonic()
            sampled = self.source.sample(s.num_frames, s.window_sec)
            if sampled is None or sampled[1][-1] == last_ts:
                self._stop.wait(0.05)
                continue
            frames, times = sampled
            last_ts = times[-1]
            self.cfg["inference"]["judge_threshold"] = s.judge_threshold
            try:
                r = self.client.infer(frames, s.prompt, s.window_sec)
            except (httpx.HTTPError, ValueError) as e:
                self.error = f"inference failed: {e}"
                log.warning("[%s] %s", self.stream_id, self.error)
                self.publish({"type": "error", "stream_id": self.stream_id, "error": self.error})
                self._stop.wait(1.0)
                continue
            self.error = ""
            self.infer_count += 1
            self._done.append(time.time())
            self.last = LastResult(last_ts, r.boxes, r.score, r.detected, r.latency_ms)
            self.publish({
                "type": "result",
                "stream_id": self.stream_id,
                "frame_ts": last_ts,
                "frame_times": times,
                "prompt": s.prompt,
                "detected": r.detected,
                "score": r.score,
                "boxes": [b.to_dict() for b in r.boxes],
                "latency_ms": round(r.latency_ms),
                "infer_fps": self.infer_fps,
            })
            wait = s.interval_ms / 1000 - (time.monotonic() - t0)
            if wait > 0:
                self._stop.wait(wait)
