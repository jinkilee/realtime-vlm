"""Compare inference strategies on labeled videos (detection + localization + latency).

  python3 examples/eval_strategies.py --labels examples/eval/urfd_fall.yaml \
      --strategies multi_image two_stage single_frame --step 0.25

Detection is scored per timestamp (prediction = at least one box). Localization: IoU of the best
predicted box against a pseudo-GT person box (single-frame grounding of `person_query`) on TP frames.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from app import config
from app.inference.vllm_client import VLLMClient

sys.path.insert(0, str(Path(__file__).parent))
from run_vlm_inference import read_frames, render, sample_indices  # noqa: E402


def in_ranges(t: float, ranges: list) -> bool:
    return any(a <= t < b for a, b in ranges)


def iou(a, b) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="examples/eval/urfd_fall.yaml")
    ap.add_argument("--strategies", nargs="+", default=["multi_image", "two_stage", "single_frame"])
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--start", type=float, default=0.5, help="first timestamp (s)")
    ap.add_argument("--num-frames", type=int)
    ap.add_argument("--window-sec", type=float)
    ap.add_argument("--query", help="override query from labels file")
    ap.add_argument("--out", default="data/eval")
    ap.add_argument("--save-images", action="store_true")
    args = ap.parse_args()

    labels = yaml.safe_load(open(args.labels, encoding="utf-8"))
    query = args.query or labels["query"]
    cfg = copy.deepcopy(config.load())
    if args.num_frames:
        cfg["inference"]["num_frames"] = args.num_frames
    if args.window_sec:
        cfg["inference"]["window_sec"] = args.window_sec
    n, window = cfg["inference"]["num_frames"], cfg["inference"]["window_sec"]
    client = VLLMClient(cfg, config.serving_url(cfg))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"model={cfg['serving']['model_id']} query={query!r} num_frames={n} window={window}s step={args.step}")

    stats = {s: dict(tp=0, fp=0, fn=0, tn=0, parse_err=0, ious=[], lat=[], calls=0) for s in args.strategies}
    rows = []
    for v in labels["videos"]:
        cap = cv2.VideoCapture(v["path"])
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stem = Path(v["path"]).stem
        line = {s: "" for s in args.strategies}
        gt_line = ""
        for t in np.arange(args.start, total / fps, args.step):
            t = float(t)
            if in_ranges(t, v.get("ignore", [])):
                gt_line += "."
                for s in args.strategies:
                    line[s] += "."
                continue
            gt = in_ranges(t, v.get("positive", []))
            gt_line += "1" if gt else "0"
            frames = read_frames(cap, sample_indices(t, window, n, fps, total))
            person = client.infer(frames, labels["person_query"], strategy="single_frame").boxes if gt else []
            for s in args.strategies:
                r = client.infer(frames, query, window, strategy=s)
                st = stats[s]
                st["lat"].append(r.latency_ms)
                st["calls"] += r.calls
                st["parse_err"] += int(r.parse_error)
                pred = r.detected
                if gt and pred:
                    st["tp"] += 1
                    if person:
                        st["ious"].append(max(iou(b, p) for b in r.boxes for p in person))
                elif pred:
                    st["fp"] += 1
                elif gt:
                    st["fn"] += 1
                else:
                    st["tn"] += 1
                line[s] += ("1" if pred else "0") if not r.parse_error else "x"
                rows.append(dict(video=stem, t=round(t, 2), gt=gt, strategy=s, pred=pred, score=r.score,
                                 parse_error=r.parse_error, latency_ms=round(r.latency_ms),
                                 boxes=[b.to_dict() for b in r.boxes],
                                 person=[b.to_dict() for b in person], raw=r.raw))
                if args.save_images:
                    title = f"{s} t={t:.2f} gt={int(gt)} pred={int(pred)} {r.latency_ms:.0f}ms"
                    d = out / s
                    d.mkdir(exist_ok=True)
                    cv2.imwrite(str(d / f"{stem}_t{t:05.2f}.jpg"), render(frames, r, title))
        print(f"\n{stem}  (1=positive 0=negative x=parse error .=ignored, every {args.step}s from {args.start}s)")
        print(f"  {'GT':<13} {gt_line}")
        for s in args.strategies:
            print(f"  {s:<13} {line[s]}")

    print(f"\n{'strategy':<13} {'prec':>5} {'recall':>6} {'F1':>5} {'FP':>3} {'FN':>3} {'parse_err':>9} "
          f"{'IoU(TP)':>8} {'lat ms p50':>10} {'calls/req':>9}")
    summary = {}
    for s, st in stats.items():
        p = st["tp"] / (st["tp"] + st["fp"]) if st["tp"] + st["fp"] else 0.0
        rc = st["tp"] / (st["tp"] + st["fn"]) if st["tp"] + st["fn"] else 0.0
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
        miou = float(np.mean(st["ious"])) if st["ious"] else 0.0
        lat = float(np.median(st["lat"]))
        cpr = st["calls"] / len(st["lat"])
        summary[s] = dict(precision=p, recall=rc, f1=f1, fp=st["fp"], fn=st["fn"], parse_err=st["parse_err"],
                          iou=miou, latency_p50=lat, calls_per_req=cpr)
        print(f"{s:<13} {p:5.2f} {rc:6.2f} {f1:5.2f} {st['fp']:3d} {st['fn']:3d} {st['parse_err']:9d} "
              f"{miou:8.2f} {lat:10.0f} {cpr:9.2f}")

    tag = f"n{n}_w{window}"
    (out / f"results_{tag}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    (out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out}/results_{tag}.jsonl, summary_{tag}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
