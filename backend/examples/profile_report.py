"""Combined report for one metrics_scenario.sh run: backend stages + vLLM server side + CPU + py-spy.

  python3 backend/examples/profile_report.py data/metrics/<label>

Reads whichever of these exist: <label>.profile.json (GET /api/profile), <label>.json (vllm_metrics.py),
<label>.pyspy.json (py-spy speedscope, --threads --native), <label>.gil.txt (py-spy raw, --gil).
Standard library only.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

GIL_RATE = 100  # samples/s used for the --gil recording in metrics_scenario.sh


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def stage_table(st: dict) -> None:
    print(f"{'stage':32} {'n':>6} {'/s':>6} {'mean':>8} {'p50':>8} {'p90':>8} {'p99':>8} {'max':>8} {'busy':>6}")
    for k, v in st.items():
        unit = "KB" if k.endswith("_kb") else "ms"
        busy = "" if unit == "KB" else f"{v['busy']:6.2f}"
        print(f"{k:32} {v['n']:6} {v['rate']:6.1f} " + " ".join(f"{v[c]:8.1f}" for c in ("mean", "p50", "p90", "p99", "max"))
              + f" {busy}" + ("  (KB)" if unit == "KB" else ""))


def cycle_breakdown(st: dict, vllm: dict | None) -> None:
    def m(k):  # total ms per worker cycle spent in stage k
        return st[k]["mean"] * st[k]["n"] if k in st else 0.0

    cycles = sum(st[k]["n"] for k in ("worker.infer.1call", "worker.infer.2call") if k in st)
    if not cycles:
        return
    print(f"\n== where one inference cycle goes (mean over {cycles} cycles, ms) ==")
    rows = [("sample frames from ring buffer", m("worker.sample") / max(st.get("worker.sample", {}).get("n", 1), 1))]
    for kind in ("judge", "ground"):
        for part, desc in (("prep", "resize + JPEG + base64"), ("encode", "JSON encode"),
                           ("http", "HTTP round trip (network + vLLM)"), ("decode", "JSON decode")):
            rows.append((f"{kind}: {desc}", m(f"infer.{kind}.{part}") / cycles))
    rows += [("publish result (json + WS hand-off)", m("worker.publish") / cycles),
             ("event store (detections only)", m("worker.event_store") / cycles)]
    infer_total = (m("worker.infer.1call") + m("worker.infer.2call")) / cycles
    accounted = sum(r[1] for r in rows[1:9])
    rows.append(("infer: other (parse, logprobs, callbacks)", infer_total - accounted))
    total = sum(r[1] for r in rows)
    for name, v in rows:
        print(f"  {name:44} {v:8.1f}  {100 * v / total:5.1f}%")
    print(f"  {'= cycle (without interval wait)':44} {total:8.1f}   (measured worker.cycle mean "
          f"{st.get('worker.cycle', {}).get('mean', 0):.1f})")

    if vllm:
        http_n = sum(st[f"infer.{k}.http"]["n"] for k in ("judge", "ground") if f"infer.{k}.http" in st)
        http_ms = sum(m(f"infer.{k}.http") for k in ("judge", "ground")) / max(http_n, 1)
        h = vllm["hist"]
        e2e = h["e2e_request_latency_seconds"]["mean"] * 1000
        print("\n== HTTP round trip vs vLLM engine (mean per request, ms) ==")
        print(f"  client HTTP (backend)              {http_ms:8.1f}   n={http_n}")
        print(f"  vLLM e2e (engine metric)           {e2e:8.1f}   n={h['e2e_request_latency_seconds']['n']}")
        for k, d in (("request_queue_time_seconds", "queue"), ("request_prefill_time_seconds", "prefill"),
                     ("request_decode_time_seconds", "decode")):
            print(f"    {d:32} {h[k]['mean'] * 1000:8.1f}")
        print(f"  outside the engine metric          {http_ms - e2e:8.1f}   "
              "(network + vLLM API server: request parse, image decode/process, tokenize, detokenize)")
        g = vllm.get("gauges", {})
        if "gpu_util_pct" in g:
            print(f"  GPU util avg {g['gpu_util_pct']['avg']:.0f}%  running reqs avg "
                  f"{g['num_requests_running']['avg']:.2f}  waiting avg {g['num_requests_waiting']['avg']:.2f}")


def cpu_table(cpu: dict) -> None:
    print("\n== backend CPU by thread (cores; 1.0 = one core busy) ==")
    groups: dict[str, float] = collections.defaultdict(float)
    for k, v in cpu.items():
        if k.startswith("__"):
            continue
        g = "decode (src:*)" if k.startswith("src:") else "inference (infer:*)" if k.startswith("infer:") else \
            "MainThread (asyncio: REST/WS)" if k == "MainThread" else "other python (" + re.sub(r"[-_:].*", "", k) + ")"
        groups[g] += v
    groups["native (ffmpeg / OpenCV threads)"] = cpu.get("__native__", 0.0)
    for g, v in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"  {g:40} {v:6.2f}")
    print(f"  {'= process total':40} {cpu.get('__process__', 0):6.2f}")


# leaf symbols that mean "blocked", which py-spy --native still reports as active
WAIT_LEAFS = ("PyThread_acquire_lock_timed", "pthread_cond_wait", "pthread_cond_timedwait", "epoll_wait",
              "epoll_pwait", "poll", "select", "futex", "nanosleep", "clock_nanosleep", "sem_wait", "recv", "read")


def thread_group(name: str) -> str:
    n = name.lower()
    for key, g in (("src:", "decode"), ("infer:", "inference"), ("mainthread", "main/asyncio")):
        if key in n:
            return g
    return "other"


def pyspy_report(sp: dict, top: int = 10) -> None:
    frames = sp["shared"]["frames"]
    groups: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    incl: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    totals: collections.Counter = collections.Counter()
    waits: collections.Counter = collections.Counter()
    for prof in sp["profiles"]:
        g = thread_group(prof["name"])
        for stack, w in zip(prof["samples"], prof["weights"]):
            if not stack:
                continue
            # deepest frame with a symbol (unresolved native frames show up as bare addresses)
            leaf = next((frames[i] for i in reversed(stack) if not frames[i]["name"].startswith("0x")),
                        frames[stack[-1]])
            if leaf["name"] in WAIT_LEAFS:
                waits[g] += w
                continue
            totals[g] += w
            groups[g][f"{leaf['name']}  [{Path(leaf.get('file') or '?').name}]"] += w
            # first python frame from the leaf up = which of our functions is responsible
            for i in reversed(stack):
                f = frames[i]
                if "/app/" in (f.get("file") or ""):
                    incl[g][f"{f['name']}  [{Path(f['file']).name}:{f.get('line', '')}]"] += w
                    break
    all_n = sum(totals.values()) or 1e-9
    print(f"\n== py-spy CPU time by thread group (total {all_n:.1f} thread-s; blocked-in-syscall/lock "
          f"samples excluded: {sum(waits.values()):.1f} s) ==")
    print("  threads: " + ", ".join(sorted({p['name'] for p in sp['profiles']}))[:400])
    for g, n in totals.most_common():
        print(f"\n  [{g}] {n:.1f} s ({100 * n / all_n:.0f}%)")
        print("    hottest leaf frames (self time, incl. native):")
        for k, v in groups[g].most_common(top):
            print(f"      {100 * v / n:5.1f}%  {k[:110]}")
        if incl[g]:
            print("    nearest app/ function (inclusive):")
            for k, v in incl[g].most_common(6):
                print(f"      {100 * v / n:5.1f}%  {k[:110]}")


def gil_report(path: Path, sec: float, top: int = 12) -> None:
    leafs: collections.Counter = collections.Counter()
    owners: collections.Counter = collections.Counter()
    n = 0
    for line in path.read_text().splitlines():
        stack, _, cnt = line.rpartition(" ")
        if not cnt.isdigit():
            continue
        c = int(cnt)
        n += c
        fr = stack.split(";")
        leafs[re.sub(r"\s+\(/.*/", " (", fr[-1])] += c
        app = [f for f in fr if "/app/" in f]
        owners[re.sub(r"\s+\(/.*/", " (", app[-1]) if app else "(no app/ frame)"] += c
    print(f"\n== GIL holders (py-spy --gil, {n} samples ~ {n / (GIL_RATE * sec):.0%} of the time someone holds it) ==")
    for k, v in leafs.most_common(top):
        print(f"  {100 * v / max(n, 1):5.1f}%  {k[:120]}")
    print("  by nearest app/ function:")
    for k, v in owners.most_common(8):
        print(f"  {100 * v / max(n, 1):5.1f}%  {k[:120]}")


def main() -> None:
    prefix = Path(sys.argv[1])
    base = prefix.parent / prefix.name
    prof = load(base.with_name(base.name + ".profile.json"))
    vllm = load(base.with_name(base.name + ".json"))
    sec = prof["wall_sec"] if prof else 60
    if prof:
        print(f"\n== backend stage timing ({prof['wall_sec']} s window, ms unless noted) ==")
        stage_table(prof["stages"])
        cycle_breakdown(prof["stages"], vllm)
        cpu_table(prof["cpu"])
    sp = base.with_name(base.name + ".pyspy.json")
    if sp.exists() and sp.stat().st_size:
        pyspy_report(json.loads(sp.read_text()))
    gil = base.with_name(base.name + ".gil.txt")
    if gil.exists() and gil.stat().st_size:
        gil_report(gil, sec)


if __name__ == "__main__":
    main()
