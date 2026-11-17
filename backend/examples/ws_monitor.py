"""Watch /ws for N seconds and print per-stream stats (results, detection rate, latency, state changes).

  docker exec -it realtime-vlm-backend python3 examples/ws_monitor.py --sec 20
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import time

import websockets


async def main(url: str, sec: float, verbose: bool) -> None:
    async with websockets.connect(url) as ws:
        t0 = time.time()
        res = collections.defaultdict(list)
        names, states = {}, collections.defaultdict(list)
        errors = 0
        while (remaining := sec - (time.time() - t0)) > 0:
            try:
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=remaining))
            except asyncio.TimeoutError:
                break
            if m["type"] == "result":
                res[m["stream_id"]].append(m)
                if verbose:
                    print(f"{time.time() - t0:6.2f}s {m['stream_id']} det={int(m['detected'])} "
                          f"score={m['score']} {m['latency_ms']}ms boxes={len(m['boxes'])}")
            elif m["type"] == "error":
                errors += 1
                print(f"{time.time() - t0:6.2f}s ERROR {m['stream_id']}: {m['error']}")
            elif m["type"] == "status":
                for s in m["streams"]:
                    names[s["id"]] = s["name"]
                    st = s["status"]["state"]
                    if not states[s["id"]] or states[s["id"]][-1][1] != st:
                        states[s["id"]].append((round(time.time() - t0), st))
        elapsed = time.time() - t0
        print(f"{'stream':26} {'n':>4} {'det%':>5} {'lat p50':>8} {'p90':>6} {'det p90':>8} {'req/s':>6}  states")
        total = 0
        for sid, rs in res.items():
            lat = sorted(r["latency_ms"] for r in rs)
            dlat = sorted(r["latency_ms"] for r in rs if r["detected"]) or [0]
            total += len(rs)
            print(f"{names.get(sid, sid):26} {len(rs):4} {100 * sum(r['detected'] for r in rs) / len(rs):4.0f}% "
                  f"{lat[len(lat) // 2]:6}ms {lat[int(len(lat) * .9)]:4}ms {dlat[int(len(dlat) * .9)]:6}ms "
                  f"{len(rs) / elapsed:6.2f}  {states.get(sid)}")
        print(f"total req/s {total / elapsed:.2f}, inference errors {errors}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    ap.add_argument("--sec", type=float, default=20)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    asyncio.run(main(a.url, a.sec, a.verbose))
