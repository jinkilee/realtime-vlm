"""Client for the vLLM OpenAI-compatible server."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx
import numpy as np

from app.inference import parser, preprocess, prompts


@dataclass
class InferResult:
    boxes: list[parser.Box]
    raw: str
    latency_ms: float
    usage: dict = field(default_factory=dict)
    image_size: tuple[int, int] = (0, 0)  # (w, h) actually sent


class VLLMClient:
    def __init__(self, cfg: dict, base_url: str):
        self.cfg = cfg
        self.model = cfg["serving"]["served_model_name"]
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.http = httpx.Client(timeout=cfg["inference"]["timeout_sec"])

    def build_request(self, frames_bgr: list[np.ndarray], query: str, window_sec: float) -> dict:
        pp, inf = self.cfg["preprocess"], self.cfg["inference"]
        content = [{"type": "image_url", "image_url": {"url": u}}
                   for u in preprocess.prepare(frames_bgr, pp["max_side"], pp["jpeg_quality"])]
        content.append({"type": "text", "text": prompts.build(query, len(frames_bgr), window_sec)})
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": inf["max_tokens"],
            "temperature": inf["temperature"],
        }

    def infer(self, frames_bgr: list[np.ndarray], query: str, window_sec: float | None = None) -> InferResult:
        window_sec = self.cfg["inference"]["window_sec"] if window_sec is None else window_sec
        body = self.build_request(frames_bgr, query, window_sec)
        t0 = time.perf_counter()
        r = self.http.post(self.url, json=body)
        r.raise_for_status()
        latency = (time.perf_counter() - t0) * 1000
        data = r.json()
        text = data["choices"][0]["message"]["content"] or ""
        h, w = preprocess.resize_long_edge(frames_bgr[-1], self.cfg["preprocess"]["max_side"]).shape[:2]
        return InferResult(parser.parse(text), text, latency, data.get("usage", {}), (w, h))
