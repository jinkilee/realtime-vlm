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
from app.profiling import prof
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
                 server_url: str, publish: Callable[[dict], None],
                 on_detection: Callable[..., None] | None = None):
        """on_detection(prompt, frame_ts, score, boxes, frame) is called for every detected result."""
        self.stream_id = stream_id
        self.on_detection = on_detection
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
            t0_perf = time.perf_counter()
            with prof.span("worker.sample"):
                sampled = self.source.sample(s.num_frames, s.window_sec)
            if sampled is None or sampled[1][-1] == last_ts:
                self._stop.wait(0.05)
                continue
            frames, times = sampled
            last_ts = times[-1]
            self.cfg["inference"]["judge_threshold"] = s.judge_threshold
            def on_judge(score: float, detected: bool, elapsed_ms: float, _ts=last_ts, _p=s.prompt) -> None:
                # sent before grounding so the UI can alert ~0.3 s after the frame instead of ~1-2 s
                prof.add("age.judge", (time.time() - _ts) * 1000, start=t0_perf)  # newest frame captured -> judge published
                self.publish({"type": "judge", "stream_id": self.stream_id, "frame_ts": _ts, "prompt": _p,
                              "score": score, "detected": detected, "latency_ms": round(elapsed_ms)})

            prof.add("age.sampled", (time.time() - last_ts) * 1000)  # newest frame age when the cycle starts
            t_inf = time.perf_counter()
            try:
                r = self.client.infer(frames, s.prompt, s.window_sec, on_judge=on_judge)
            except (httpx.HTTPError, ValueError) as e:
                self.error = f"inference failed: {e}"
                log.warning("[%s] %s", self.stream_id, self.error)
                self.publish({"type": "error", "stream_id": self.stream_id, "error": self.error})
                self._stop.wait(1.0)
                continue
            prof.add("worker.infer" + (".2call" if r.calls == 2 else ".1call"), (time.perf_counter() - t_inf) * 1000,
                     start=t0_perf)
            self.error = ""
            self.infer_count += 1
            self._done.append(time.time())
            self.last = LastResult(last_ts, r.boxes, r.score, r.detected, r.latency_ms)
            t_pub = time.perf_counter()
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
            prof.add("worker.publish", (time.perf_counter() - t_pub) * 1000, start=t0_perf)
            prof.add("age.result", (time.time() - last_ts) * 1000, start=t0_perf)  # newest frame captured -> result published
            if r.detected and self.on_detection:
                try:
                    with prof.span("worker.event_store"):
                        self.on_detection(s.prompt, last_ts, r.score, r.boxes, frames[-1])
                except Exception:  # never let event storage kill the inference loop
                    log.exception("[%s] event store failed", self.stream_id)
            wait = s.interval_ms / 1000 - (time.monotonic() - t0)
            prof.add("worker.cycle", (time.monotonic() - t0) * 1000, start=t0_perf)  # work only, no interval wait
            prof.add("worker.idle", max(wait, 0) * 1000, start=t0_perf)
            if wait > 0:
                self._stop.wait(wait)
