"""Frame preprocessing before sending to the VLM server."""
from __future__ import annotations

import base64

import cv2
import numpy as np


def resize_long_edge(frame: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so that max(h, w) <= max_side, keeping aspect ratio. Never upscales."""
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def to_data_url(frame_bgr: np.ndarray, jpeg_quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def prepare(frames_bgr: list[np.ndarray], max_side: int, jpeg_quality: int) -> list[str]:
    return [to_data_url(resize_long_edge(f, max_side), jpeg_quality) for f in frames_bgr]
