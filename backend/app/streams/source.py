"""Video sources (MP4 file / RTSP / HTTP) decoded in a background thread.

Keeps two things:
  - latest full-resolution frame (for MJPEG preview and event snapshots)
  - a time-stamped ring buffer of downscaled frames at `buffer_fps` (for inference sampling)
"""
from __future__ import annotations

import bisect
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass

# RTSP over TCP + socket timeout (us). Must be set before VideoCapture is created.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000|timeout;5000000")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.inference.preprocess import resize_long_edge  # noqa: E402
from app.profiling import prof  # noqa: E402

log = logging.getLogger(__name__)

NETWORK_SCHEMES = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://")


def is_network_url(url: str) -> bool:
    return url.lower().startswith(NETWORK_SCHEMES)


@dataclass
class SourceStatus:
    state: str = "connecting"        # connecting | running | reconnecting | error | stopped
    src_fps: float = 0.0             # measured decode rate
    width: int = 0
    height: int = 0
    frames_read: int = 0
    reconnects: int = 0
    error: str = ""


class FrameSource:
    def __init__(self, url: str, path: str | None, max_side: int, buffer_fps: float, buffer_sec: float,
                 reconnect_max_sec: float = 10.0):
        """url: user-facing string; path: resolved local file path (None for network streams)."""
        self.url = url
        self.path = path
        self.is_file = path is not None
        self.max_side = max_side
        self.buffer_period = 1.0 / buffer_fps
        self.buffer_sec = buffer_sec
        self.reconnect_max_sec = reconnect_max_sec

        self.status = SourceStatus()
        self._lock = threading.Lock()
        self._times: deque[float] = deque()
        self._frames: deque[np.ndarray] = deque()
        self._latest: np.ndarray | None = None
        self._latest_ts = 0.0
        self._latest_seq = 0
        self._next_buffer_t = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"src:{url}", daemon=True)

    # ---- lifecycle -----------------------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self.status.state = "stopped"

    # ---- consumers -----------------------------------------------------------------
    def latest(self) -> tuple[np.ndarray | None, float, int]:
        """(full-res frame, wall-clock ts, sequence number)."""
        with self._lock:
            return self._latest, self._latest_ts, self._latest_seq

    def sample(self, n: int, window_sec: float) -> tuple[list[np.ndarray], list[float]] | None:
        """n frames evenly spaced over the last `window_sec` (the last one = newest). None if empty."""
        with self._lock:
            if not self._times:
                return None
            times = list(self._times)
            frames = list(self._frames)
        end = times[-1]
        start = max(times[0], end - window_sec)
        if n <= 1 or end == start:
            return [frames[-1]] * max(n, 1), [end] * max(n, 1)
        picked_f, picked_t = [], []
        for i in range(n):
            target = start + (end - start) * i / (n - 1)
            j = min(bisect.bisect_left(times, target), len(times) - 1)
            if j > 0 and abs(times[j - 1] - target) < abs(times[j] - target):
                j -= 1
            picked_f.append(frames[j])
            picked_t.append(times[j])
        return picked_f, picked_t

    # ---- decode loop ---------------------------------------------------------------
    def _open(self) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(self.path if self.is_file else self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            return None
        self.status.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.status.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cap

    def _push(self, frame: np.ndarray, now: float) -> None:
        with self._lock:
            self._latest = frame
            self._latest_ts = now
            self._latest_seq += 1
            if now >= self._next_buffer_t:
                # fixed slots (not "period since last push") so 30 fps input gives exactly buffer_fps
                self._next_buffer_t = max(self._next_buffer_t + self.buffer_period, now - self.buffer_period)
                self._times.append(now)
                with prof.span("decode.buffer_resize"):  # under the lock: sample() waits for this
                    self._frames.append(resize_long_edge(frame, self.max_side))
                while self._times and now - self._times[0] > self.buffer_sec:
                    self._times.popleft()
                    self._frames.popleft()

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                self.status.state = "error" if self.is_file else "reconnecting"
                self.status.error = f"cannot open {self.url}"
                if self.is_file:
                    log.error("cannot open file %s", self.path)
                    return
                self.status.reconnects += 1
                self._stop.wait(backoff)
                backoff = min(backoff * 2, self.reconnect_max_sec)
                continue

            self.status.state, self.status.error, backoff = "running", "", 1.0
            file_fps = cap.get(cv2.CAP_PROP_FPS) if self.is_file else 0
            period = 1.0 / file_fps if file_fps and file_fps < 240 else 1 / 30
            next_t = time.monotonic()
            fps_t0, fps_n = time.monotonic(), 0

            while not self._stop.is_set():
                with prof.span("decode.read"):
                    ok, frame = cap.read()
                if not ok:
                    if self.is_file:  # loop the file
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                        if not ok:
                            self.status.state, self.status.error = "error", "file has no readable frames"
                            cap.release()
                            return
                    else:
                        log.warning("stream lost: %s", self.url)
                        self.status.state, self.status.error = "reconnecting", "stream lost"
                        self.status.reconnects += 1
                        break
                with prof.span("decode.push"):
                    self._push(frame, time.time())
                self.status.frames_read += 1
                fps_n += 1
                now = time.monotonic()
                if now - fps_t0 >= 2.0:
                    self.status.src_fps = round(fps_n / (now - fps_t0), 1)
                    fps_t0, fps_n = now, 0
                if self.is_file:  # play at native speed
                    next_t += period
                    delay = next_t - time.monotonic()
                    if delay > 0:
                        self._stop.wait(delay)
                    else:
                        next_t = time.monotonic()
            cap.release()
            if not self._stop.is_set():
                self._stop.wait(backoff)
