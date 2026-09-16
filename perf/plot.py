#!/usr/bin/env python3
"""Turns the benchmark CSVs in perf/data into the SVG charts in img/perf.

Standard library only. Regenerate the data first with, e.g.
    ./build/bin/cis5650_stream_compaction_test --bench scan 26 > perf/data/scan.csv
then run: python3 perf/plot.py
"""

import csv
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "perf", "data")
OUT = os.path.join(ROOT, "img", "perf")

# Reference categorical palette, light mode, in its fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MARKERS = ["circle", "square", "triangle", "diamond", "triangle-down", "hexagon", "pentagon", "circle"]

# An implementation keeps the same color and marker in every chart it appears in.
SLOT = {
    "cpu": 0, "cpu-without-scan": 0, "cpu-std-sort": 0,
    "naive": 1, "cpu-with-scan": 1,
    "efficient-unoptimized": 2, "radix": 2,
    "efficient": 3,
    "thrust": 4, "thrust-sort": 4,
    "shared-naive": 5,
    "shared-efficient": 6,
    "efficient-compact": 7,
}


def read(name):
    with open(os.path.join(DATA, name)) as f:
        return [r for r in csv.DictReader(f) if r["experiment"] != "occupancy"]


def size_label(n):
    for unit, div in (("M", 1 << 20), ("K", 1 << 10)):
        if n >= div:
            v = n / div
            return f"{v:g}{unit}"
    return str(n)


def marker(kind, x, y, color, r=4.5):
    ring = f'stroke="{SURFACE}" stroke-width="2" fill="{color}"'
    if kind == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" {ring}/>'
    if kind == "square":
        s = r * 1.7
        return f'<rect x="{x - s / 2:.1f}" y="{y - s / 2:.1f}" width="{s:.1f}" height="{s:.1f}" rx="1.5" {ring}/>'
    pts = []
    sides, rot, rad = {
        "triangle": (3, -90, r * 1.25),
        "triangle-down": (3, 90, r * 1.25),
        "diamond": (4, 0, r * 1.3),
        "hexagon": (6, 0, r * 1.15),
        "pentagon": (5, -90, r * 1.2),
    }[kind]
    for i in range(sides):
        a = math.radians(rot + 360 * i / sides)
        pts.append(f"{x + rad * math.cos(a):.1f},{y + rad * math.sin(a):.1f}")
    return f'<polygon points="{" ".join(pts)}" {ring}/>'


def line_chart(path, title, subtitle, series, x_label, y_label, x_log2=True, x_ticks=None, x_tick_label=str,
               y_log=True):
    """series: list of (name, [(x, y), ...]). Y is log10 scaled unless y_log is False (then linear from 0)."""
    width, height = 760, 470
    left, right, top = 72, 24, 70
    names = [name for name, _ in series]

    # Legend rows, wrapped to the plot width.
    legend_rows, row, row_w = [], [], 0
    for name in names:
        w = 30 + 7.2 * len(name) + 18
        if row and row_w + w > width - left - right:
            legend_rows.append(row)
            row, row_w = [], 0
        row.append((name, w))
        row_w += w
    legend_rows.append(row)
    top += 22 * len(legend_rows)
    bottom = 58
    pw, ph = width - left - right, height + 22 * len(legend_rows) - top - bottom
    height = top + ph + bottom

    xs = [x for _, pts in series for x, _ in pts]
    ys = [y for _, pts in series for _, y in pts if y > 0]
    fx = (lambda v: math.log2(v)) if x_log2 else (lambda v: v)
    x0, x1 = fx(min(xs)), fx(max(xs))
    if y_log:
        y0 = math.floor(math.log10(min(ys)))
        y1 = math.ceil(math.log10(max(ys)))
        if y1 == y0:
            y1 += 1
        fy = math.log10
        y_ticks = [(10 ** e, f"{10 ** e:g}" if e >= -3 else f"1e{e}") for e in range(y0, y1 + 1)]
    else:
        step = 0.25
        while max(ys) / step > 8:
            step *= 2
        y0, y1 = 0.0, math.ceil(max(ys) / step) * step
        fy = lambda v: v
        y_ticks = [(i * step, f"{i * step:g}") for i in range(int(round(y1 / step)) + 1)]

    def px(v):
        return left + (fx(v) - x0) / (x1 - x0) * pw

    def py(v):
        return top + ph - (fy(v) - y0) / (y1 - y0) * ph

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="{FONT}">',
           f'<rect width="{width}" height="{height}" rx="8" fill="{SURFACE}"/>',
           f'<text x="{left}" y="28" font-size="16" font-weight="600" fill="{INK}">{title}</text>',
           f'<text x="{left}" y="48" font-size="12" fill="{INK_2}">{subtitle}</text>']

    ly = 72
    for row in legend_rows:
        lx = left
        for i, (name, w) in enumerate(row):
            idx = SLOT.get(name, names.index(name))
            out.append(f'<line x1="{lx}" y1="{ly - 4}" x2="{lx + 22}" y2="{ly - 4}" stroke="{SERIES[idx]}" stroke-width="2"/>')
            out.append(marker(MARKERS[idx], lx + 11, ly - 4, SERIES[idx], 4))
            out.append(f'<text x="{lx + 30}" y="{ly}" font-size="12" fill="{INK_2}">{name}</text>')
            lx += w
        ly += 22

    for value, label in y_ticks:
        y = py(value) if value > 0 or not y_log else top + ph
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + pw}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{left - 8}" y="{y + 4:.1f}" font-size="11" fill="{MUTED}" text-anchor="end">{label}</text>')
    out.append(f'<line x1="{left}" y1="{top + ph}" x2="{left + pw}" y2="{top + ph}" stroke="{AXIS}" stroke-width="1"/>')

    for t in x_ticks or sorted(set(xs)):
        x = px(t)
        out.append(f'<line x1="{x:.1f}" y1="{top + ph}" x2="{x:.1f}" y2="{top + ph + 4}" stroke="{AXIS}" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{top + ph + 18}" font-size="11" fill="{MUTED}" text-anchor="middle">{x_tick_label(t)}</text>')

    out.append(f'<text x="{left + pw / 2}" y="{height - 12}" font-size="12" fill="{INK_2}" text-anchor="middle">{x_label}</text>')
    out.append(f'<text transform="translate(16 {top + ph / 2}) rotate(-90)" font-size="12" fill="{INK_2}" text-anchor="middle">{y_label}</text>')

    for position, (name, pts) in enumerate(series):
        idx = SLOT.get(name, position)
        pts = sorted(p for p in pts if p[1] > 0)
        d = " ".join(f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(y):.1f}" for i, (x, y) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="{SERIES[idx]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in pts:
            out.append(marker(MARKERS[idx], px(x), py(y), SERIES[idx]))
            out.append(f'<title>{name}: {x_tick_label(x)} -&gt; {y:.3f} ms</title>')

    out.append("</svg>")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, path), "w") as f:
        f.write("\n".join(out))
    print("wrote", os.path.join("img", "perf", path))


def by_impl(rows, keep, key="n"):
    groups = defaultdict(list)
    for r in rows:
        if keep(r):
            groups[r["impl"]].append((int(r[key]), float(r["median_ms"])))
    return groups


def pot(r):
    n = int(r["n"])
    return n & (n - 1) == 0


def size_ticks(rows):
    ns = sorted({int(r["n"]) for r in rows if pot(r)})
    return [n for n in ns if int(math.log2(n)) % 2 == 0]


def main():
    scan = read("scan.csv")
    g = by_impl(scan, pot)
    order = ["cpu", "naive", "efficient-unoptimized", "efficient", "thrust", "shared-naive", "shared-efficient"]
    line_chart("scan.svg", "Scan: time vs. array size",
               "Power-of-two sizes, median of 7 runs, memory transfers excluded (Tesla T4)",
               [(k, g[k]) for k in order if k in g], "Array size (elements)", "Time (ms, log scale)",
               x_ticks=size_ticks(scan), x_tick_label=size_label)

    compact = read("compact.csv")
    g = by_impl(compact, pot)
    order = ["cpu-without-scan", "cpu-with-scan", "efficient", "thrust"]
    line_chart("compact.svg", "Stream compaction: time vs. array size",
               "Power-of-two sizes, values in [0, 4), median of 7 runs (Tesla T4)",
               [(k, g[k]) for k in order if k in g], "Array size (elements)", "Time (ms, log scale)",
               x_ticks=size_ticks(compact), x_tick_label=size_label)

    sort = read("sort.csv")
    for dist, label in (("0", "values in [0, 1000)"), ("1", "full signed int range")):
        g = by_impl(sort, lambda r: pot(r) and r["param"] == dist)
        order = ["cpu-std-sort", "radix", "thrust-sort"]
        line_chart(f"sort_{'small' if dist == '0' else 'full'}.svg", f"Sorting: {label}",
                   "Power-of-two sizes, median of 7 runs (Tesla T4)",
                   [(k, g[k]) for k in order if k in g], "Array size (elements)", "Time (ms, log scale)",
                   x_ticks=size_ticks(sort), x_tick_label=size_label)

    npot = read("npot.csv")
    g = by_impl(npot, lambda r: True)
    order = ["cpu", "efficient-unoptimized", "efficient", "thrust", "shared-efficient"]
    line_chart("npot.svg", "Non-power-of-two sizes: the padding stair-step",
               "Sizes on a linear grid. Separate session on a slower VM: compare shapes, not absolute values",
               [(k, g[k]) for k in order if k in g], "Array size (elements)", "Time (ms, log scale)",
               x_ticks=[1 << 20, 1 << 21, 1 << 22, 1 << 23, 10000000],
               x_tick_label=lambda n: size_label(n) if n & (n - 1) == 0 else "10M")

    blocks = read("blocksize.csv")
    g = by_impl(blocks, lambda r: True, key="blockSize")
    order = ["naive", "efficient-unoptimized", "efficient", "efficient-compact", "shared-naive", "shared-efficient"]
    n = int(blocks[0]["n"])
    line_chart("blocksize.svg", "Block size tuning",
               f"n = {size_label(n)} elements, median of 7 runs (Tesla T4)",
               [(k, g[k]) for k in order if k in g], "Threads per block", "Time (ms, log scale)")

    banks = read("banks.csv")
    g = defaultdict(list)
    for r in banks:
        g[f"{r['blockSize']} threads"].append((int(r["param"]), float(r["median_ms"])))
    names = sorted(g, key=lambda k: int(k.split()[0]))
    line_chart("banks.svg", "Shared memory work-efficient scan: bank count",
               f"n = {size_label(int(banks[0]['n']))}, 0 = original Example 39-2 layout, k = 2^k banks (Tesla T4)",
               [(k, g[k]) for k in names], "log2(number of banks)", "Time (ms)", x_log2=False,
               x_ticks=list(range(0, 8)), y_log=False)


if __name__ == "__main__":
    main()
