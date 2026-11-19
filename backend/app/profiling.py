"""Lightweight stage timing for the whole backend (decode -> sample -> preprocess -> HTTP -> publish).

Off by default (config: profiling.enabled, or toggled at runtime via POST /api/profile).
When off, `span()` / `add()` cost one attribute check.

    from app.profiling import prof
    with prof.span("infer.prep"):
        ...
    prof.add("frame_age.result", ms)

GET /api/profile returns per-stage count, rate, mean / p50 / p90 / p99 / max (ms) and
busy = total time spent in the stage per wall-clock second since the last reset
(e.g. decode.read busy 0.35 = decode threads spent 0.35 s per second reading frames, summed over streams),
plus CPU use per thread (from /proc, Linux only): 1.0 = one core fully busy. Python threads are named
(src:<url> = decode, infer:<id> = inference, MainThread = asyncio loop / REST / WS); "native" is the rest
of the process (ffmpeg decoder threads, OpenCV worker threads, ...).
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Iterator

MAX_SAMPLES = 50_000  # per stage; 4 streams x 30 fps x 60 s = 7200
TICK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def _cpu_sec(stat_path: str) -> float | None:
    """utime + stime (s) from a /proc/.../stat file."""
    try:
        with open(stat_path) as f:
            fields = f.read().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / TICK
    except (OSError, IndexError, ValueError):
        return None


def _cpu_snapshot() -> dict[str, float]:
    """{thread name: cpu sec} for live Python threads, plus "__process__" for the whole process."""
    out: dict[str, float] = {}
    for t in threading.enumerate():
        c = _cpu_sec(f"/proc/self/task/{t.native_id}/stat") if t.native_id else None
        if c is not None:
            out[t.name] = out.get(t.name, 0.0) + c
    proc = _cpu_sec("/proc/self/stat")
    if proc is not None:
        out["__process__"] = proc
    return out


class Profiler:
    def __init__(self) -> None:
        self.enabled = False
        self._lock = threading.Lock()
        self._data: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=MAX_SAMPLES))
        self._t0 = time.monotonic()
        self._reset_perf = time.perf_counter()
        self._cpu0 = _cpu_snapshot()

    def add(self, stage: str, ms: float, start: float | None = None) -> None:
        """start: perf_counter() when the measured interval began; intervals that began before the
        last reset are dropped, so a window never mixes a whole cycle with only part of its sub-stages."""
        if not self.enabled or (start is not None and start < self._reset_perf):
            return
        with self._lock:
            self._data[stage].append(ms)

    @contextmanager
    def span(self, stage: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(stage, (time.perf_counter() - t0) * 1000, start=t0)

    def reset(self) -> None:
        with self._lock:
            self._data.clear()
            self._t0 = time.monotonic()
            self._reset_perf = time.perf_counter()
            self._cpu0 = _cpu_snapshot()

    def summary(self) -> dict:
        with self._lock:
            data = {k: sorted(v) for k, v in self._data.items()}
            wall = max(time.monotonic() - self._t0, 1e-9)

        def pct(xs: list[float], p: float) -> float:
            return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]

        stages = {}
        for k in sorted(data):
            xs = data[k]
            if not xs:
                continue
            total = sum(xs)
            stages[k] = {
                "n": len(xs), "rate": round(len(xs) / wall, 2),
                "mean": round(total / len(xs), 2), "p50": round(pct(xs, 50), 2), "p90": round(pct(xs, 90), 2),
                "p99": round(pct(xs, 99), 2), "max": round(xs[-1], 2), "busy": round(total / 1000 / wall, 3),
            }
        return {"enabled": self.enabled, "wall_sec": round(wall, 1), "stages": stages,
                "cpu": self._cpu_usage(wall)}

    def _cpu_usage(self, wall: float) -> dict[str, float]:
        """cores used per thread since reset; threads started after reset count from 0."""
        now = _cpu_snapshot()
        proc = now.pop("__process__", None)
        proc0 = self._cpu0.get("__process__")
        threads = {k: round((v - self._cpu0.get(k, 0.0)) / wall, 3) for k, v in now.items()}
        out = dict(sorted(threads.items(), key=lambda kv: -kv[1]))
        if proc is not None and proc0 is not None:
            out["__process__"] = round((proc - proc0) / wall, 3)
            # threads that already exited count toward native too; fine for a steady-state window
            out["__native__"] = round(max(out["__process__"] - sum(threads.values()), 0.0), 3)
        return out


prof = Profiler()
