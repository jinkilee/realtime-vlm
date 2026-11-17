"""Grounding prompt templates. The user query (free text, any language) is embedded verbatim."""
from __future__ import annotations

SINGLE_FRAME = (
    'Find every target in the image that matches the description: "{query}".\n'
    "Answer ONLY with a JSON list, no other text:\n"
    '[{{"bbox_2d": [x1, y1, x2, y2], "label": "<short label>"}}]\n'
    "If nothing matches, answer []."
)

MULTI_FRAME = (
    "The {n} images are consecutive frames from the same camera, in time order "
    "(image 1 = oldest, image {n} = most recent, spanning about {window:.1f} seconds).\n"
    'Use the motion across frames to find every target that matches the description: "{query}".\n'
    "Give the bounding boxes in the coordinates of the LAST image (image {n}).\n"
    "Answer ONLY with a JSON list, no other text:\n"
    '[{{"bbox_2d": [x1, y1, x2, y2], "label": "<short label>"}}]\n'
    "If nothing matches, answer []."
)


def build(query: str, num_frames: int, window_sec: float) -> str:
    if num_frames <= 1:
        return SINGLE_FRAME.format(query=query)
    return MULTI_FRAME.format(query=query, n=num_frames, window=window_sec)
