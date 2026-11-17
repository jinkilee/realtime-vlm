from __future__ import annotations

from pydantic import BaseModel, Field


class StreamCreate(BaseModel):
    url: str = Field(..., description="rtsp://... | http(s)://... | path inside the project, e.g. videos/x.mp4")
    prompt: str = Field(..., description='free-text query, e.g. "넘어지는 사람"')
    name: str | None = None
    num_frames: int | None = None
    window_sec: float | None = None
    interval_ms: int | None = None
    judge_threshold: float | None = None


class StreamUpdate(BaseModel):
    name: str | None = None
    prompt: str | None = None
    num_frames: int | None = None
    window_sec: float | None = None
    interval_ms: int | None = None
    judge_threshold: float | None = None
