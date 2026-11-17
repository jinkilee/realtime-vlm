"""Prompt templates. The user query (free text, any language) is embedded verbatim."""
from __future__ import annotations

# --- grounding: boxes on a single image -------------------------------------------------
GROUND_SINGLE = (
    # Qwen3-VL's native grounding phrasing; the custom "answer ONLY with JSON" variant produced
    # malformed 9-number boxes on lying-person frames with the 2B model.
    'Locate every instance that belongs to the following categories: "{query}". '
    "Report bbox coordinates in JSON format."
)

# --- grounding: boxes on the last of N frames (strategy=multi_image) --------------------
GROUND_MULTI = (
    "The {n} images are consecutive frames from the same camera, in time order "
    "(image 1 = oldest, image {n} = most recent, spanning about {window:.1f} seconds).\n"
    'Use the motion across frames to find every target that matches the description: "{query}".\n'
    "Give the bounding boxes in the coordinates of the LAST image (image {n}).\n"
    "Answer ONLY with a JSON list, no other text:\n"
    '[{{"bbox_2d": [x1, y1, x2, y2], "label": "<short label>"}}]\n'
    "If nothing matches, answer []."
)

# --- judge: yes/no over N frames (strategy=two_stage, stage 1) --------------------------
JUDGE_MULTI = (
    "The {n} images are consecutive frames from the same camera, in time order "
    "(image 1 = oldest, image {n} = most recent, spanning about {window:.1f} seconds).\n"
    'Considering the motion across the frames, does the description "{query}" '
    "apply to anything in the most recent frame?\n"
    "Answer with a single word: yes or no."
)

JUDGE_SINGLE = (
    'Does the description "{query}" apply to anything in this image?\n'
    "Answer with a single word: yes or no."
)


def ground(query: str, num_frames: int, window_sec: float) -> str:
    if num_frames <= 1:
        return GROUND_SINGLE.format(query=query)
    return GROUND_MULTI.format(query=query, n=num_frames, window=window_sec)


def judge(query: str, num_frames: int, window_sec: float) -> str:
    if num_frames <= 1:
        return JUDGE_SINGLE.format(query=query)
    return JUDGE_MULTI.format(query=query, n=num_frames, window=window_sec)
