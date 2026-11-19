#!/usr/bin/env bash
# vLLM-only benchmark with `vllm bench serve` (no backend, no videos): sends this service's two request
# shapes at the same time, as Poisson traffic, and prints TTFT / e2e latency percentiles for each.
#   judge : 4 images 448x252 + ~88 text tokens -> 1 output token   (every cycle of every stream)
#   ground: 1 image  448x252 + ~34 text tokens -> 44 output tokens (only after a detection)
# Images are random noise, so vLLM's multimodal caches never hit (the real service reuses frames:
# slightly pessimistic numbers). Run on the host from the project root; stop the UI streams first.
#
#   sh backend/examples/bench_vllm.sh <label> [judge_rps] [ground_rps] [sec]
#   sh backend/examples/bench_vllm.sh base 3.2 0.6 60     # ~ 4 streams, interval 1000 ms, falls
#   sh backend/examples/bench_vllm.sh solo 1 0 30         # one judge at a time (latency floor)
#
# Result: data/metrics/bench/<label>.{judge,ground}.json      (full vllm bench output, per-request details)
#         data/metrics/bench/<label>.{judge,ground}.hist.json (latency histogram: e2e + TTFT, bins + raw samples)
#         summary on stdout. Plot: python3 backend/examples/plot_latency_hist.py <label> [<label> ...]
set -eu
LABEL=$1; JUDGE_RPS=${2:-3.2}; GROUND_RPS=${3:-0.6}; SEC=${4:-60}
C=realtime-vlm-serving
OUT=data/metrics/bench
MODEL=$(docker exec "$C" sh -c 'eval "$(python3 serving/gen_env.py)"; echo "$MODEL_ID"')
IMG='{(252, 448, 1): 1.0}'   # (height, width, frames) = what preprocess.max_side 448 sends for 16:9 cameras
mkdir -p "$OUT"
docker exec "$C" rm -rf /tmp/bench && docker exec "$C" mkdir -p /tmp/bench

run() {  # kind rps n_images text_tokens out_tokens seed
  local n; n=$(python3 -c "import math;print(max(1, math.ceil($2 * $SEC)))")
  docker exec "$C" vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
    --base-url http://127.0.0.1:9000 --model "$MODEL" --served-model-name vlm \
    --dataset-name random-mm --random-input-len "$4" --random-output-len "$5" --random-range-ratio 0 \
    --random-mm-base-items-per-request "$3" --random-mm-num-mm-items-range-ratio 0 \
    --random-mm-limit-mm-per-prompt "{\"image\": $3, \"video\": 0}" --random-mm-bucket-config "$IMG" \
    --num-prompts "$n" --request-rate "$2" --ignore-eos --seed "$6" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --save-detailed --result-dir /tmp/bench --result-filename "$LABEL.$1.json" --label "$LABEL-$1" \
    > "$OUT/$LABEL.$1.log" 2>&1
}

echo "[$LABEL] judge ${JUDGE_RPS}/s + ground ${GROUND_RPS}/s for ${SEC}s"
python3 -c "import sys; sys.exit(0 if float('$JUDGE_RPS') > 0 else 1)" && run judge "$JUDGE_RPS" 4 88 1 0 &
python3 -c "import sys; sys.exit(0 if float('$GROUND_RPS') > 0 else 1)" && run ground "$GROUND_RPS" 1 34 44 1 &
wait
docker cp "$C:/tmp/bench/." "$OUT/"

python3 - "$OUT" "$LABEL" "$JUDGE_RPS" "$GROUND_RPS" "$SEC" <<'EOF'
import json, math, sys
from pathlib import Path
out, label = Path(sys.argv[1]), sys.argv[2]
rps = {"judge": float(sys.argv[3]), "ground": float(sys.argv[4])}
BIN_MS = {"judge": 50, "ground": 250}  # e2e bin per kind (fixed, so runs compare bin by bin)
TTFT_BIN_MS = 50


def pct(xs, p):  # linear interpolation, same as numpy / vllm bench
    if not xs:
        return None
    k = p / 100 * (len(xs) - 1)
    lo = int(k)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (k - lo)


def hist(xs, w):
    n = int(math.ceil(max(xs) / w)) if xs else 0
    counts = [0] * n
    for x in xs:
        counts[min(int(x // w), n - 1)] += 1
    return [{"lo": i * w, "hi": (i + 1) * w, "count": c} for i, c in enumerate(counts)]


def summary(xs):
    xs = sorted(xs)
    return {"n": len(xs), "mean": round(sum(xs) / len(xs), 1) if xs else None,
            **{f"p{p}": round(pct(xs, p), 1) if xs else None for p in (50, 90, 99)},
            "max": round(xs[-1], 1) if xs else None}


for kind in ("judge", "ground"):
    f = out / f"{label}.{kind}.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text())
    e2e = [round(x * 1000, 1) for x in d["latencies"]]
    ttft = [round(x * 1000, 1) for x in d.get("ttfts", []) if x]
    w = BIN_MS[kind]
    (out / f"{label}.{kind}.hist.json").write_text(json.dumps({
        "label": label, "kind": kind, "date": d.get("date"),
        "request_rate": {"judge": rps["judge"], "ground": rps["ground"]}, "duration_sec": d["duration"],
        "unit": "ms",
        "e2e": {**summary(e2e), "bin_ms": w, "bins": hist(e2e, w), "samples": e2e},
        "ttft": {**summary(ttft), "bin_ms": TTFT_BIN_MS, "bins": hist(ttft, TTFT_BIN_MS), "samples": ttft},
    }, indent=1))
print(f"{'':7} {'n':>5} {'req/s':>6} {'TTFT p50':>9} {'p90':>7} {'p99':>7} {'e2e mean':>9} {'p50':>7} {'p90':>7} {'p99':>7} {'TPOT':>6}  (ms)")
for kind in ("judge", "ground"):
    f = out / f"{label}.{kind}.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text())
    print(f"{kind:7} {d['completed']:5} {d['request_throughput']:6.2f} {d['p50_ttft_ms']:9.0f} {d['p90_ttft_ms']:7.0f} "
          f"{d['p99_ttft_ms']:7.0f} {d['mean_e2el_ms']:9.0f} {d['p50_e2el_ms']:7.0f} {d['p90_e2el_ms']:7.0f} "
          f"{d['p99_e2el_ms']:7.0f} {d['mean_tpot_ms']:6.1f}" + (f"  failed {d['failed']}" if d.get("failed") else ""))
EOF
