"""Detection event log.

- A detection on a stream opens an event; further detections within `cooldown_sec` of the last one
  are merged into it (count / last_seen / max score updated) instead of creating new events.
- Each change appends the full event as one JSON line to events.jsonl (last line per id wins on load).
- The opening detection's frame is saved with boxes drawn as <dir>/<YYYYMMDD>/<id>.jpg.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from app.inference.parser import Box

log = logging.getLogger(__name__)
COLORS = [(0, 0, 255), (0, 200, 0), (255, 128, 0), (0, 200, 255), (255, 0, 255)]


def draw_boxes(frame: np.ndarray, boxes: list[Box]) -> np.ndarray:
    img = frame.copy()
    h, w = img.shape[:2]
    t = max(2, round(max(h, w) / 300))
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b.to_pixels(w, h)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[i % len(COLORS)], t)
    return img


class EventStore:
    def __init__(self, directory: Path, cooldown_sec: float, max_in_memory: int, owner_of: Path,
                 publish: Callable[[dict], None]):
        self.dir = directory
        self.jsonl = directory / "events.jsonl"
        self.cooldown = cooldown_sec
        self.max_in_memory = max_in_memory
        self.publish = publish
        self._owner = owner_of.stat()
        self._lock = threading.Lock()
        self.events: OrderedDict[str, dict] = OrderedDict()  # oldest -> newest
        self._open: dict[str, str] = {}                      # stream_id -> open event id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._chown(self.dir)
        self._load()

    # ---- persistence ------------------------------------------------------------------------
    def _chown(self, p: Path) -> None:
        try:
            os.chown(p, self._owner.st_uid, self._owner.st_gid)
        except OSError:
            pass

    def _load(self) -> None:
        if not self.jsonl.exists():
            return
        latest: dict[str, dict] = {}
        with open(self.jsonl, encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                    latest[e["id"]] = e
                except (json.JSONDecodeError, KeyError):
                    continue
        for e in sorted(latest.values(), key=lambda e: e["ts_start"])[-self.max_in_memory:]:
            self.events[e["id"]] = e
        log.info("loaded %d events from %s", len(self.events), self.jsonl)

    def _append(self, e: dict) -> None:
        new = not self.jsonl.exists()
        with open(self.jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
        if new:
            self._chown(self.jsonl)

    def _save_snapshot(self, eid: str, ts: float, frame: np.ndarray, boxes: list[Box]) -> str:
        day = datetime.fromtimestamp(ts).strftime("%Y%m%d")
        d = self.dir / day
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            self._chown(d)
        p = d / f"{eid}.jpg"
        cv2.imwrite(str(p), draw_boxes(frame, boxes), [cv2.IMWRITE_JPEG_QUALITY, 88])
        self._chown(p)
        return f"{day}/{eid}.jpg"

    # ---- API ----------------------------------------------------------------------------------
    def on_detection(self, stream_id: str, stream_name: str, prompt: str, ts: float, score: float | None,
                     boxes: list[Box], frame: np.ndarray | None) -> dict:
        with self._lock:
            eid = self._open.get(stream_id)
            e = self.events.get(eid) if eid else None
            if e is not None and ts - e["last_seen"] <= self.cooldown and e["prompt"] == prompt:
                e["last_seen"] = ts
                e["count"] += 1
                if score is not None:
                    e["max_score"] = max(e["max_score"] or 0.0, score)
                if not e["snapshot"] and frame is not None and boxes:   # first frame that had boxes
                    e["snapshot"] = self._save_snapshot(e["id"], ts, frame, boxes)
                    e["boxes"] = [b.to_dict() for b in boxes]
                update = True
            else:
                eid = uuid.uuid4().hex[:12]
                e = {
                    "id": eid, "stream_id": stream_id, "stream_name": stream_name, "prompt": prompt,
                    "ts_start": ts, "last_seen": ts, "count": 1, "max_score": score,
                    "boxes": [b.to_dict() for b in boxes],
                    "snapshot": self._save_snapshot(eid, ts, frame, boxes) if frame is not None else "",
                }
                self.events[eid] = e
                self._open[stream_id] = eid
                while len(self.events) > self.max_in_memory:
                    self.events.popitem(last=False)
                update = False
            self._append(e)
            snapshot = dict(e)
        self.publish({"type": "event", "update": update, "event": snapshot})
        return snapshot

    def list(self, stream_id: str | None = None, limit: int = 100) -> list[dict]:
        with self._lock:
            evs = [e for e in reversed(self.events.values()) if not stream_id or e["stream_id"] == stream_id]
        return [dict(e) for e in evs[:limit]]

    def clear(self) -> int:
        """Forget all events and delete the log + snapshots."""
        with self._lock:
            n = len(self.events)
            self.events.clear()
            self._open.clear()
            if self.jsonl.exists():
                self.jsonl.unlink()
            for d in self.dir.iterdir():
                if d.is_dir() and d.name.isdigit():
                    for f in d.glob("*.jpg"):
                        f.unlink()
                    try:
                        d.rmdir()
                    except OSError:
                        pass
        self.publish({"type": "events_cleared", "ts": time.time()})
        return n
