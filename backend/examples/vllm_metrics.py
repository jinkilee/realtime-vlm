"""Collect vLLM /metrics over a time window and print where request time goes.

Takes a snapshot of the Prometheus /metrics endpoint before and after the window, so every
histogram / counter below covers only requests that finished inside the window. Gauges
(running / waiting requests, KV cache usage) and GPU utilization are sampled during it.

Standard library only: runs on the host or inside a container.

  # on the host (serving publishes 127.0.0.1:9000)
  python3 backend/examples/vllm_metrics.py --sec 60
  python3 backend/examples/vllm_metrics.py --sec 60 --label 4stream-1000ms --out data/metrics/4s-1000.json
  # inside the backend container
  docker exec realtime-vlm-backend python3 examples/vllm_metrics.py --url http://serving:9000 --sec 60

The mean is exact (histogram sum / count). Percentiles are only upper bounds ("<=") because
vLLM's buckets are coarse (e.g. queue / prefill / decode: 0.3, 0.5, 0.8, 1.0, 1.5 s).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request

LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)$')
LABEL = re.compile(r'(\w+)="([^"]*)"')

# (metric, description) - per-request time histograms, in seconds
TIME_HISTS = [
    ("e2e_request_latency_seconds", "end-to-end (server side)"),
    ("request_queue_time_seconds", "  queue (waiting to be scheduled)"),
    ("request_prefill_time_seconds", "  prefill (vision encoder + prompt)"),
    ("request_decode_time_seconds", "  decode (output tokens)"),
    ("time_to_first_token_seconds", "time to first token"),
    ("inter_token_latency_seconds", "inter-token latency (per token)"),
]
TOKEN_HISTS = [
    ("request_prompt_tokens", "prompt tokens / request"),
    ("request_generation_tokens", "output tokens / request"),
    ("iteration_tokens_total", "tokens / engine step (batch size)"),
]
COUNTERS = ["request_success_total", "prompt_tokens_total", "generation_tokens_total",
            "num_preemptions_total", "prefix_cache_queries_total", "prefix_cache_hits_total",
            "mm_cache_queries_total", "mm_cache_hits_total"]
GAUGES = ["num_requests_running", "num_requests_waiting", "kv_cache_usage_perc"]


def scrape(url: str) -> dict:
    """{name: {labels_without_engine_and_model: value}}"""
    with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=5) as r:
        text = r.read().decode()
    out: dict = collections.defaultdict(dict)
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        m = LINE.match(line)
        if not m:
            continue
        name, labels, value = m.group(1)[5:], m.group(2) or "", float(m.group(3))
        lab = tuple(sorted((k, v) for k, v in LABEL.findall(labels) if k not in ("engine", "model_name")))
        out[name][lab] = out[name].get(lab, 0.0) + value
    return out


def delta(a: dict, b: dict, name: str) -> dict:
    return {k: v - a.get(name, {}).get(k, 0.0) for k, v in b.get(name, {}).items()}


def hist(a: dict, b: dict, name: str) -> dict | None:
    """Histogram stats over the window: n, mean, p50, p90, p99 (bucket-interpolated)."""
    n = sum(delta(a, b, name + "_count").values())
    if n <= 0:
        return None
    s = sum(delta(a, b, name + "_sum").values())
    buckets = sorted(((math.inf if dict(k)["le"] == "+Inf" else float(dict(k)["le"])), v)
                     for k, v in delta(a, b, name + "_bucket").items())

    def pct(q: float) -> float:
        # upper bound of the bucket holding the q-quantile (honest with coarse buckets)
        for le, c in buckets:
            if c >= q * n:
                return le
        return math.inf

    return {"n": int(n), "mean": s / n, "p50": pct(.5), "p90": pct(.9), "p99": pct(.99),
            "buckets": [[le if not math.isinf(le) else "+Inf", int(c)] for le, c in buckets]}



class Sampler(threading.Thread):
    """Samples gauges from /metrics and GPU utilization from nvidia-smi."""

    def __init__(self, url: str, period: float):
        super().__init__(daemon=True)
        self.url, self.period = url, period
        self.samples: dict = collections.defaultdict(list)
        self.stop = threading.Event()
        self.smi = shutil.which("nvidia-smi")

    def run(self) -> None:
        while not self.stop.wait(self.period):
            try:
                m = scrape(self.url)
                for g in GAUGES:
                    self.samples[g].append(sum(m.get(g, {}).values()))
            except OSError:
                pass
            if self.smi:
                try:
                    o = subprocess.run([self.smi, "--query-gpu=utilization.gpu,memory.used",
                                        "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=3).stdout
                    util, mem = (float(x) for x in o.splitlines()[0].split(","))
                    self.samples["gpu_util_pct"].append(util)
                    self.samples["gpu_mem_mib"].append(mem)
                except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                    self.smi = None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("VLLM_URL", "http://127.0.0.1:9000"))
    ap.add_argument("--sec", type=float, default=60, help="collection window (s)")
    ap.add_argument("--sample", type=float, default=0.5, help="gauge sampling period (s)")
    ap.add_argument("--label", default="", help="free text stored in the JSON (scenario name)")
    ap.add_argument("--out", help="also save the full result as JSON")
    ap.add_argument("--buckets", action="store_true", help="print raw histogram buckets")
    a = ap.parse_args()

    sampler = Sampler(a.url, a.sample)
    before = scrape(a.url)
    t0 = time.time()
    sampler.start()
    print(f"collecting {a.url}/metrics for {a.sec:.0f}s ...", flush=True)
    try:
        time.sleep(a.sec)
    except KeyboardInterrupt:
        print("interrupted, reporting what was collected")
    sampler.stop.set()
    after = scrape(a.url)
    elapsed = time.time() - t0

    res: dict = {"label": a.label, "url": a.url, "elapsed_sec": round(elapsed, 1),
                 "time": time.strftime("%Y-%m-%d %H:%M:%S"), "hist": {}, "counters": {}, "gauges": {}}
    for name, _ in TIME_HISTS + TOKEN_HISTS:
        res["hist"][name] = hist(before, after, name)
    for name in COUNTERS:
        res["counters"][name] = sum(delta(before, after, name).values())
    for name, vals in sampler.samples.items():
        if vals:
            res["gauges"][name] = {"avg": sum(vals) / len(vals), "max": max(vals), "n": len(vals)}

    c = res["counters"]
    nreq = c["request_success_total"]
    print(f"\n== {a.label or 'vLLM metrics'}  ({elapsed:.1f}s, {nreq:.0f} requests, {nreq / elapsed:.2f} req/s)")
    print(f"{'time per request':40} {'n':>5} {'mean':>8} {'p50<=':>8} {'p90<=':>8} {'p99<=':>8}")
    for name, desc in TIME_HISTS:
        h = res["hist"][name]
        if h:
            print(f"{desc:40} {h['n']:5} " + " ".join(f"{h[k] * 1000:6.0f}ms" for k in ("mean", "p50", "p90", "p99")))
    print(f"\n{'tokens':40} {'n':>5} {'mean':>8} {'p50<=':>8} {'p90<=':>8} {'p99<=':>8}")
    for name, desc in TOKEN_HISTS:
        h = res["hist"][name]
        if h:
            print(f"{desc:40} {h['n']:5} " + " ".join(f"{h[k]:8.1f}" for k in ("mean", "p50", "p90", "p99")))

    def rate(hit: str, q: str) -> str:
        return f"{100 * c[hit] / c[q]:.0f}%" if c[q] else "-"

    print(f"\nthroughput: prompt {c['prompt_tokens_total'] / elapsed:.0f} tok/s, "
          f"output {c['generation_tokens_total'] / elapsed:.1f} tok/s, preemptions {c['num_preemptions_total']:.0f}")
    print(f"cache hit: prefix {rate('prefix_cache_hits_total', 'prefix_cache_queries_total')}, "
          f"multimodal {rate('mm_cache_hits_total', 'mm_cache_queries_total')}")
    g = res["gauges"]
    print("sampled: " + ", ".join(
        f"{k} avg {v['avg']:.2f} max {v['max']:.2f}" for k, v in g.items()))
    if a.buckets:
        for name, h in res["hist"].items():
            if h:
                print(f"\n{name}: " + " ".join(f"<={le}:{n}" for le, n in h["buckets"]))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
