"""Client for the vLLM OpenAI-compatible server.

Strategies (config: inference.strategy):
  multi_image  : N frames -> boxes on the last frame in one call
  two_stage    : N frames -> yes/no judge (P(yes) = score); if score >= judge_threshold,
                 ground boxes on the last frame alone (second call)
  single_frame : last frame only -> boxes (no temporal context)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import httpx
import numpy as np

from app.inference import parser, preprocess, prompts

STRATEGIES = ("multi_image", "two_stage", "single_frame")


@dataclass
class InferResult:
    boxes: list[parser.Box]
    raw: str
    latency_ms: float
    usage: dict = field(default_factory=dict)
    image_size: tuple[int, int] = (0, 0)  # (w, h) of the last frame actually sent
    score: float | None = None            # two_stage: P(yes) from the judge
    detected: bool = False
    calls: int = 1
    parse_error: bool = False             # model answered something but no valid box was parsed


def _add_usage(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}


class VLLMClient:
    def __init__(self, cfg: dict, base_url: str):
        self.cfg = cfg
        self.model = cfg["serving"]["served_model_name"]
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.http = httpx.Client(timeout=cfg["inference"]["timeout_sec"])

    # ---- low level --------------------------------------------------------------------
    def _chat(self, frames_bgr: list[np.ndarray], text: str, max_tokens: int, logprobs: bool = False) -> dict:
        pp, inf = self.cfg["preprocess"], self.cfg["inference"]
        content = [{"type": "image_url", "image_url": {"url": u}}
                   for u in preprocess.prepare(frames_bgr, pp["max_side"], pp["jpeg_quality"])]
        content.append({"type": "text", "text": text})
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": inf["temperature"],
        }
        if logprobs:
            body.update(logprobs=True, top_logprobs=10)
        r = self.http.post(self.url, json=body)
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _yes_prob(data: dict) -> float:
        """P(yes) from the first generated token's top logprobs (normalized over yes/no)."""
        try:
            top = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError):
            text = (data["choices"][0]["message"]["content"] or "").strip().lower()
            return 1.0 if text.startswith("yes") else 0.0
        p_yes = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "yes")
        p_no = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "no")
        return p_yes / (p_yes + p_no) if p_yes + p_no > 0 else 0.0

    def _ground(self, frames: list[np.ndarray], query: str, window: float) -> tuple[list, str, dict]:
        data = self._chat(frames, prompts.ground(query, len(frames), window), self.cfg["inference"]["max_tokens"])
        text = data["choices"][0]["message"]["content"] or ""
        return parser.parse(text), text, data.get("usage", {})

    # ---- public -----------------------------------------------------------------------
    def infer(self, frames_bgr: list[np.ndarray], query: str, window_sec: float | None = None,
              strategy: str | None = None) -> InferResult:
        inf = self.cfg["inference"]
        window = inf["window_sec"] if window_sec is None else window_sec
        strategy = strategy or inf.get("strategy", "multi_image")
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}, expected one of {STRATEGIES}")

        t0 = time.perf_counter()
        score, calls = None, 1
        if strategy == "single_frame":
            boxes, raw, usage = self._ground(frames_bgr[-1:], query, window)
        elif strategy == "multi_image":
            boxes, raw, usage = self._ground(frames_bgr, query, window)
        else:  # two_stage
            data = self._chat(frames_bgr, prompts.judge(query, len(frames_bgr), window), 1, logprobs=True)
            score = self._yes_prob(data)
            usage = data.get("usage", {})
            raw = f"judge P(yes)={score:.3f}"
            boxes = []
            if score >= inf.get("judge_threshold", 0.5):
                boxes, graw, gusage = self._ground(frames_bgr[-1:], query, window)
                raw += f" | ground: {graw}"
                usage = _add_usage(usage, gusage)
                calls = 2
        latency = (time.perf_counter() - t0) * 1000

        h, w = preprocess.resize_long_edge(frames_bgr[-1], self.cfg["preprocess"]["max_side"]).shape[:2]
        answered_boxes = "bbox_2d" in raw
        return InferResult(
            boxes=boxes, raw=raw, latency_ms=latency, usage=usage, image_size=(w, h), score=score,
            detected=bool(boxes), calls=calls, parse_error=answered_boxes and not boxes,
        )
