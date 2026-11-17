"""Test inference against the running VLM server (same preprocess/prompt/parser as the app).

Examples (run through backend/examples/run_vlm_inference.sh, paths relative to the project root):
  # images given in time order (last one is the target frame)
  run_vlm_inference.sh --image a.jpg b.jpg c.jpg d.jpg --prompt "넘어지는 사람"
  # N frames sampled from the window ending at --at seconds
  run_vlm_inference.sh --video videos/samples/urfd-fall-01-rgb.mp4 --at 3.0 --prompt "넘어지는 사람"
  # slide over the whole video every --sweep seconds
  run_vlm_inference.sh --video videos/samples/urfd-fall-01-rgb.mp4 --sweep 0.5 --prompt "넘어지는 사람"
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from app import config
from app.inference.vllm_client import InferResult, VLLMClient

COLORS = [(0, 0, 255), (0, 200, 0), (255, 128, 0), (0, 200, 255), (255, 0, 255)]


def read_frames(cap: cv2.VideoCapture, indices: list[int]) -> list[np.ndarray]:
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {idx}")
        frames.append(f)
    return frames


def sample_indices(at_sec: float, window_sec: float, n: int, fps: float, total: int) -> list[int]:
    end = min(round(at_sec * fps), total - 1)
    if n <= 1:
        return [end]
    start = max(0, end - round(window_sec * fps))
    return [round(start + (end - start) * i / (n - 1)) for i in range(n)]


def render(frames: list[np.ndarray], res: InferResult, title: str) -> np.ndarray:
    """Top: input frames as thumbnails. Bottom: last frame (original size) with boxes."""
    last = frames[-1].copy()
    h, w = last.shape[:2]
    for i, b in enumerate(res.boxes):
        x1, y1, x2, y2 = b.to_pixels(w, h)
        cv2.rectangle(last, (x1, y1), (x2, y2), COLORS[i % len(COLORS)], 2)
        cv2.putText(last, f"#{i}", (x1 + 2, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    COLORS[i % len(COLORS)], 1, cv2.LINE_AA)
    tw = w // max(len(frames), 1)
    th = round(h * tw / w)
    strip = np.hstack([cv2.resize(f, (tw, th)) for f in frames])
    strip = cv2.copyMakeBorder(strip, 0, 0, 0, w - strip.shape[1], cv2.BORDER_CONSTANT)
    header = np.zeros((22, w, 3), np.uint8)
    cv2.putText(header, title, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([header, strip, last])


def report(tag: str, res: InferResult, show_raw: bool) -> None:
    boxes = [{"label": b.label, "bbox": [round(v, 3) for v in (b.x1, b.y1, b.x2, b.y2)]} for b in res.boxes]
    print(f"{tag:>10} | {res.latency_ms:6.0f} ms | sent {res.image_size[0]}x{res.image_size[1]} "
          f"| prompt_tok {res.usage.get('prompt_tokens', '?'):>5} "
          f"| score {'-' if res.score is None else f'{res.score:.2f}':>4} | {len(boxes)} box(es) "
          f"{json.dumps(boxes, ensure_ascii=False)}")
    if show_raw:
        print("           raw:", res.raw.replace("\n", " "))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", nargs="+", help="image file(s) in time order")
    src.add_argument("--video", help="video file")
    ap.add_argument("--prompt", required=True, help='free-text query, e.g. "넘어지는 사람"')
    ap.add_argument("--at", type=float, help="video: timestamp (s) of the last frame (default: middle)")
    ap.add_argument("--sweep", type=float, help="video: run every N seconds over the whole video")
    ap.add_argument("--strategy", help="override inference.strategy (multi_image|two_stage|single_frame)")
    ap.add_argument("--num-frames", type=int, help="override inference.num_frames")
    ap.add_argument("--window-sec", type=float, help="override inference.window_sec")
    ap.add_argument("--max-side", type=int,
                    help="override preprocess.max_side (client side only; server cap is fixed at startup)")
    ap.add_argument("--server", help="server base url (default: from config)")
    ap.add_argument("--out", default="data/inference", help="output directory for rendered images")
    ap.add_argument("--raw", action="store_true", help="print raw model output")
    args = ap.parse_args()

    cfg = copy.deepcopy(config.load())
    if args.strategy:
        cfg["inference"]["strategy"] = args.strategy
    if args.num_frames:
        cfg["inference"]["num_frames"] = args.num_frames
    if args.window_sec:
        cfg["inference"]["window_sec"] = args.window_sec
    if args.max_side:
        cfg["preprocess"]["max_side"] = args.max_side
    n, window = cfg["inference"]["num_frames"], cfg["inference"]["window_sec"]

    client = VLLMClient(cfg, args.server or config.serving_url(cfg))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"server={client.url} model={client.model} strategy={cfg['inference']['strategy']} prompt={args.prompt!r} "
          f"max_side={cfg['preprocess']['max_side']}")

    if args.image:
        frames = [cv2.imread(p) for p in args.image]
        if any(f is None for f in frames):
            print("cannot read one of the images", file=sys.stderr)
            return 1
        res = client.infer(frames, args.prompt, window)
        report("images", res, args.raw)
        path = out / f"{Path(args.image[-1]).stem}_result.jpg"
        cv2.imwrite(str(path), render(frames, res, f"{args.prompt} | {res.latency_ms:.0f} ms"))
        print(f"saved {path}")
        return 0

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"cannot open {args.video}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    print(f"video={args.video} fps={fps:.1f} frames={total} duration={duration:.2f}s "
          f"num_frames={n} window={window}s")

    if args.sweep:
        times = list(np.arange(min(window, duration), duration, args.sweep)) + [duration]
    else:
        times = [args.at if args.at is not None else duration / 2]

    stem = Path(args.video).stem
    latencies = []
    for t in times:
        idx = sample_indices(t, window, n, fps, total)
        frames = read_frames(cap, idx)
        res = client.infer(frames, args.prompt, window)
        latencies.append(res.latency_ms)
        report(f"t={t:5.2f}s", res, args.raw)
        title = f"t={t:.2f}s frames={idx} | {args.prompt} | {res.latency_ms:.0f} ms | {len(res.boxes)} box"
        cv2.imwrite(str(out / f"{stem}_t{t:06.2f}.jpg"), render(frames, res, title))
    if len(latencies) > 1:
        print(f"latency ms: mean {np.mean(latencies):.0f}, p50 {np.median(latencies):.0f}, "
              f"max {np.max(latencies):.0f} (first call includes warmup)")
    print(f"saved {len(times)} image(s) to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
