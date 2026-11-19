"""Plot latency histograms from bench_vllm.sh (<label>.{judge,ground}.hist.json) as one SVG.

Two panels (judge | ground). Each run is a step-line histogram, y = share of requests (%), so runs
with different request counts compare directly. p50 / p99 are marked on each line and listed in the
table under each panel. Standard library only.

  python3 backend/examples/plot_latency_hist.py 1s:"1 stream" 4s:"4 streams"
  python3 backend/examples/plot_latency_hist.py 1s 4s --metric ttft --out data/metrics/bench/ttft.svg
"""
from __future__ import annotations

import argparse
import json
import math
from html import escape
from pathlib import Path

# reference palette (dataviz skill), light mode: categorical slots in fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, TEXT, TEXT2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#8a8984"
FONT = "font-family='system-ui, -apple-system, Segoe UI, sans-serif'"
KINDS = {"judge": "judge: 4 images → 1 token", "ground": "ground (boxes): 1 image → 44 tokens"}

PW, PH = 460, 260               # plot area of one panel
ML, MR, MT, MB = 64, 24, 92, 46  # panel margins (top: legend row + panel title)
TABLE_ROW = 20


def nice_step(span: float, target: int = 5) -> float:
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    return next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)


def fmt_ms(v: float) -> str:
    return f"{v / 1000:.1f} s" if v >= 1000 else f"{v:.0f} ms"


def panel(x0: float, kind: str, runs: list[tuple[str, dict]], metric: str) -> tuple[list[str], float]:
    out = []
    data = [(name, h[metric]) for name, h in runs]
    w = data[0][1]["bin_ms"]
    xmax_raw = max(d["max"] for _, d in data if d["n"])
    xstep = nice_step(xmax_raw)
    xmax = math.ceil(xmax_raw / xstep) * xstep
    shares = [[100 * b["count"] / d["n"] for b in d["bins"]] for _, d in data]
    ymax_raw = max(max(s) for s in shares if s)
    ystep = nice_step(ymax_raw, 4)
    ymax = math.ceil(ymax_raw / ystep) * ystep
    px = lambda v: x0 + ML + PW * v / xmax  # noqa: E731
    py = lambda v: MT + PH - PH * v / ymax  # noqa: E731

    out.append(f"<text x='{x0 + ML}' y='{MT - 36}' {FONT} font-size='15' font-weight='600' fill='{TEXT}'>"
               f"{escape(KINDS.get(kind, kind))}</text>")
    # grid + axes (recessive)
    y = 0.0
    while y <= ymax + 1e-9:
        out.append(f"<line x1='{x0 + ML}' x2='{x0 + ML + PW}' y1='{py(y):.1f}' y2='{py(y):.1f}' stroke='{GRID}'/>")
        out.append(f"<text x='{x0 + ML - 8}' y='{py(y) + 4:.1f}' {FONT} font-size='11' fill='{TEXT2}' "
                   f"text-anchor='end'>{y:g}%</text>")
        y += ystep
    x = 0.0
    while x <= xmax + 1e-9:
        out.append(f"<line x1='{px(x):.1f}' x2='{px(x):.1f}' y1='{MT + PH}' y2='{MT + PH + 4}' stroke='{AXIS}'/>")
        out.append(f"<text x='{px(x):.1f}' y='{MT + PH + 18}' {FONT} font-size='11' fill='{TEXT2}' "
                   f"text-anchor='middle'>{fmt_ms(x) if x else '0'}</text>")
        x += xstep
    out.append(f"<line x1='{x0 + ML}' x2='{x0 + ML + PW}' y1='{MT + PH}' y2='{MT + PH}' stroke='{AXIS}'/>")
    out.append(f"<text x='{x0 + ML + PW / 2}' y='{MT + PH + 38}' {FONT} font-size='12' fill='{TEXT2}' "
               f"text-anchor='middle'>{'end-to-end latency' if metric == 'e2e' else 'time to first token'}"
               f" (bin {w} ms)</text>")
    out.append(f"<text transform='translate({x0 + 16},{MT + PH / 2}) rotate(-90)' {FONT} font-size='12' "
               f"fill='{TEXT2}' text-anchor='middle'>share of requests</text>")

    # step lines, 2 px; p50 / p99 as >=8 px markers on the line
    for i, ((name, d), s) in enumerate(zip(data, shares)):
        color = SERIES[i % len(SERIES)]
        pts = [(px(0), py(0))]
        for b, v in zip(d["bins"], s):
            pts += [(px(b["lo"]), py(v)), (px(b["hi"]), py(v))]
        pts.append((pts[-1][0], py(0)))
        path = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
        out.append(f"<polyline points='{path}' fill='{color}' fill-opacity='0.10' stroke='{color}' "
                   f"stroke-width='2' stroke-linejoin='round'/>")
        for p in ("p50", "p99"):
            v = d[p]
            # interpolated percentiles can fall into an empty bin: sit on the bin of the next real sample
            nxt = min((x for x in d["samples"] if x >= v), default=v)
            k = min(int(nxt // w), len(s) - 1)
            cx, cy = px(v), py(s[k])
            out.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='4.5' fill='{color}' stroke='{SURFACE}' "
                       f"stroke-width='2'/>")
            out.append(f"<text x='{cx + 7:.1f}' y='{cy - 7:.1f}' {FONT} font-size='11' fill='{TEXT}'>"
                       f"{p} {fmt_ms(v)}</text>")

    # table view under the plot (identity never by color alone)
    ty = MT + PH + 66
    cols = [("run", 0, "start"), ("n", 150, "end"), ("mean", 215, "end"), ("p50", 280, "end"),
            ("p90", 345, "end"), ("p99", 410, "end"), ("max", 470, "end")]
    for label, dx, anchor in cols:
        out.append(f"<text x='{x0 + ML + dx}' y='{ty}' {FONT} font-size='11' fill='{TEXT2}' "
                   f"text-anchor='{anchor}'>{label}</text>")
    for i, (name, d) in enumerate(data):
        ry = ty + TABLE_ROW * (i + 1)
        color = SERIES[i % len(SERIES)]
        out.append(f"<rect x='{x0 + ML}' y='{ry - 9}' width='10' height='10' rx='2' fill='{color}'/>")
        vals = [escape(name), str(d["n"])] + [fmt_ms(d[k]) for k in ("mean", "p50", "p90", "p99", "max")]
        for (label, dx, anchor), v in zip(cols, vals):
            out.append(f"<text x='{x0 + ML + dx + (16 if label == 'run' else 0)}' y='{ry}' {FONT} font-size='12' "
                       f"fill='{TEXT}' text-anchor='{anchor}'>{v}</text>")
    return out, ty + TABLE_ROW * len(data) + 16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label or label:'display name' (data/metrics/bench/<label>.*.hist.json)")
    ap.add_argument("--dir", default="data/metrics/bench")
    ap.add_argument("--metric", choices=("e2e", "ttft"), default="e2e")
    ap.add_argument("--out", default=None, help="default: <dir>/latency_hist_<labels>_<metric>.svg")
    a = ap.parse_args()

    runs = [(r.split(":", 1)[0], r.split(":", 1)[1] if ":" in r else r) for r in a.runs]
    d = Path(a.dir)
    parts, height = [], 0.0
    pw = ML + PW + MR
    for i, kind in enumerate(("judge", "ground")):
        loaded = [(name, json.loads((d / f"{label}.{kind}.hist.json").read_text()))
                  for label, name in runs if (d / f"{label}.{kind}.hist.json").exists()]
        if not loaded:
            continue
        p, h = panel(i * pw, kind, loaded, a.metric)
        parts += p
        height = max(height, h)
    # legend, one row on top (series identity)
    legend = []
    lx = ML
    for i, (_, name) in enumerate(runs):
        legend.append(f"<line x1='{lx}' x2='{lx + 18}' y1='20' y2='20' stroke='{SERIES[i % len(SERIES)]}' "
                      f"stroke-width='2'/><text x='{lx + 24}' y='24' {FONT} font-size='12' fill='{TEXT}'>"
                      f"{escape(name)}</text>")
        lx += 30 + 8 * len(name) + 24
    width = 2 * pw
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height:.0f}' "
           f"viewBox='0 0 {width} {height:.0f}'><rect width='100%' height='100%' fill='{SURFACE}'/>"
           + "".join(legend) + "".join(parts) + "</svg>\n")
    out = Path(a.out) if a.out else d / f"latency_hist_{'_'.join(l for l, _ in runs)}_{a.metric}.svg"
    out.write_text(svg)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
