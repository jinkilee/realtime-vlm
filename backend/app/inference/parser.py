"""Parse VLM grounding output into normalized boxes.

Qwen3-VL emits bbox_2d in a 0..1000 relative coordinate system, often wrapped in ```json fences.
Output boxes are normalized to 0..1 (x1, y1, x2, y2) relative to the (last) input image.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict

COORD_SCALE = 1000.0  # Qwen3-VL relative coordinates

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_BOX = re.compile(r'"bbox_2d"\s*:\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]'
                  r'(?:\s*,\s*"label"\s*:\s*"([^"]*)")?')


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_pixels(self, w: int, h: int) -> tuple[int, int, int, int]:
        return round(self.x1 * w), round(self.y1 * h), round(self.x2 * w), round(self.y2 * h)


def _make_box(coords, label: str, scale: float) -> Box | None:
    try:
        x1, y1, x2, y2 = (float(c) / scale for c in coords)
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)))
    y1, y2 = sorted((min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)))
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return Box(x1, y1, x2, y2, label or "")


def parse(text: str, scale: float = COORD_SCALE) -> list[Box]:
    m = _FENCE.search(text)
    body = (m.group(1) if m else text).strip()

    try:
        data = json.loads(body)
        if isinstance(data, dict):
            data = data.get("objects", [data])
        boxes = []
        for item in data if isinstance(data, list) else []:
            if isinstance(item, dict) and "bbox_2d" in item:
                b = _make_box(item["bbox_2d"], str(item.get("label", "")), scale)
                if b:
                    boxes.append(b)
        return boxes
    except json.JSONDecodeError:
        pass

    # Fallback for truncated / slightly malformed JSON
    boxes = []
    for g in _BOX.finditer(body):
        b = _make_box(g.groups()[:4], g.group(5) or "", scale)
        if b:
            boxes.append(b)
    return boxes
