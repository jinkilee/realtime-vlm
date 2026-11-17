export type Box = { x1: number; y1: number; x2: number; y2: number; label: string };

export type StreamSettings = {
  prompt: string;
  num_frames: number;
  window_sec: number;
  interval_ms: number;
  judge_threshold: number;
};

export type StreamStatus = {
  state: "connecting" | "running" | "reconnecting" | "error" | "stopped" | string;
  error: string;
  src_fps: number;
  width: number;
  height: number;
  reconnects: number;
  infer_fps: number;
  infer_count: number;
  latency_ms: number;
  detected: boolean;
  last_result_ts: number;
};

export type StreamInfo = {
  id: string;
  name: string;
  url: string;
  created: number;
  settings: StreamSettings;
  status: StreamStatus;
};

export type DetEvent = {
  id: string;
  stream_id: string;
  stream_name: string;
  prompt: string;
  ts_start: number;
  last_seen: number;
  count: number;
  max_score: number | null;
  boxes: Box[];
  snapshot: string;
};

export type Health = {
  app: string;
  serving: { backend: string; ok: boolean; model: string };
  inference: { strategy: string; num_frames: number; window_sec: number; judge_threshold: number };
  streams: number;
  max_streams: number;
};

/** Latest inference output per stream, stamped with local receive time (ms). */
export type LiveResult = {
  frame_ts: number;
  boxes: Box[];
  score: number | null;
  detected: boolean;
  latency_ms: number;
  recv: number;
};

export type LiveJudge = { frame_ts: number; score: number; detected: boolean; latency_ms: number; recv: number };

export type Live = { result?: LiveResult; judge?: LiveJudge };

export type VideoFile = { path: string; size: number };
