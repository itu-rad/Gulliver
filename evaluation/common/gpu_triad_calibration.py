#!/usr/bin/env python3
"""Calibrate the GB10's Nsight L2-miss byte counter against known GPU memory traffic.

Runs idle, paced and unpaced STREAM triad, and an L2-resident matmul control while
gb10_samplers.py samples nsys; fits a multiplier from counted to known bytes/s.

    python evaluation/common/gpu_triad_calibration.py --out evaluation/collocation/calibration/gb10_l2miss
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLERS = HERE / "gb10_samplers.py"


def triad_step(x, y, z, target_gbps: float | None, seconds: float) -> dict:
    import torch
    nbytes = 3 * x.numel() * x.element_size()  # read x, read y, write z
    t0 = time.time(); moved = 0; n = 0
    torch.cuda.synchronize()
    while time.time() - t0 < seconds:
        it0 = time.time()
        torch.add(x, y, alpha=3.0, out=z)
        torch.cuda.synchronize()
        moved += nbytes; n += 1
        if target_gbps:
            want = nbytes / (target_gbps * 1e9)
            rest = want - (time.time() - it0)
            if rest > 0:
                time.sleep(rest)
    t1 = time.time()
    return {"t0": t0, "t1": t1, "bytes": moved, "iters": n, "known_gbps": moved / 1e9 / (t1 - t0)}


def matmul_step(seconds: float) -> dict:
    import torch
    a = torch.ones(4096, 4096, dtype=torch.bfloat16, device="cuda")
    t0 = time.time(); n = 0
    torch.cuda.synchronize()
    while time.time() - t0 < seconds:
        for _ in range(20):
            torch.mm(a, a)
        torch.cuda.synchronize(); n += 20
    t1 = time.time()
    del a
    return {"t0": t0, "t1": t1, "matmuls": n, "known_gbps": None}


def idle_step(seconds: float) -> dict:
    t0 = time.time(); time.sleep(seconds); return {"t0": t0, "t1": time.time(), "known_gbps": 0.0}


def window_means(csv_path: Path, t0: float, t1: float, cols: list[str]) -> dict:
    vals = {c: [] for c in cols}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ts = float(row["wall_ts"])
            if t0 + 0.5 <= ts <= t1 - 0.5:      # trim edges: half a second each side
                for c in cols:
                    v = row.get(c)
                    if v not in (None, ""):
                        vals[c].append(float(v))
    return {c: (sum(v) / len(v) if v else None) for c, v in vals.items()} | {"n": len(vals[cols[0]])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output stem (writes <out>.json and <out>_nsys.csv)")
    ap.add_argument("--targets", default="25,50,100,150", help="paced triad targets, GB/s")
    ap.add_argument("--seconds", type=float, default=20.0, help="seconds per step")
    ap.add_argument("--hz", type=int, default=100)
    ap.add_argument("--size-mib", type=int, default=1024, help="per-tensor size (fp32)")
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    nsys_csv = out.parent / f"{out.name}_nsys.csv"

    import torch
    n = (a.size_mib << 20) // 4
    x = torch.ones(n, dtype=torch.float32, device="cuda")
    y = torch.empty_like(x); z = torch.empty_like(x)
    torch.cuda.synchronize()

    samp = subprocess.Popen([sys.executable, str(SAMPLERS), "nsys", "--out", str(nsys_csv),
                             "--hz", str(a.hz), "--cap", str(int(a.seconds * 12 + 120))])
    time.sleep(4)
    steps = []
    try:
        steps.append({"step": "idle"} | idle_step(10))
        for tg in [float(t) for t in a.targets.split(",") if t]:
            steps.append({"step": f"triad_{tg:g}"} | triad_step(x, y, z, tg, a.seconds))
            print(f"  {steps[-1]['step']}: delivered {steps[-1]['known_gbps']:.1f} GB/s", flush=True)
        steps.append({"step": "triad_unpaced"} | triad_step(x, y, z, None, a.seconds))
        print(f"  triad_unpaced: delivered {steps[-1]['known_gbps']:.1f} GB/s (roof)", flush=True)
        steps.append({"step": "matmul"} | matmul_step(a.seconds))
        steps.append({"step": "idle_end"} | idle_step(5))
    finally:
        samp.send_signal(signal.SIGTERM)
        samp.wait(timeout=400)

    if not nsys_csv.exists():
        print("no nsys CSV produced; calibration aborted", file=sys.stderr)
        return 1
    with open(nsys_csv) as f:
        header = next(csv.reader(f))
    miss_col = next((c for c in header if "misses_from_l1tex" in c and c.endswith("_bytes")), None)
    tot_col = next((c for c in header if "total_from_l1tex" in c and c.endswith("_bytes")), None)
    sm_col = next((c for c in header if c.startswith("sm_throughput")), None)
    gpc_cyc = next((c for c in header if c.startswith("gpc_frequency") and c.endswith("cycles")), None)
    cols = [c for c in (miss_col, tot_col, sm_col, gpc_cyc) if c]
    if not miss_col:
        print(f"miss-bytes column not found in {header}", file=sys.stderr)
        return 1
    period = 1.0 / a.hz
    rows = []
    for s in steps:
        m = window_means(nsys_csv, s["t0"], s["t1"], cols)
        counted = (m[miss_col] or 0.0) / period / 1e9      # bytes per sample -> GB/s (one slice)
        rows.append({**s, "samples": m["n"], "counted_gbps_raw": counted,
                     "l2_total_gbps_raw": (m[tot_col] or 0.0) / period / 1e9 if tot_col else None,
                     "sm_throughput_pct": m.get(sm_col), "gpc_mhz": (m.get(gpc_cyc) or 0) / period / 1e6 if gpc_cyc else None})
    fit = [(r["known_gbps"], r["counted_gbps_raw"]) for r in rows
           if r["step"].startswith("triad") and r["known_gbps"] and r["counted_gbps_raw"] > 0]
    mult = (sum(k * c for k, c in fit) / sum(c * c for _, c in fit)) if fit else None
    print("\n| step | known GB/s | counted (raw) | x mult | resid % | L2 total (raw) | SM % | GPC MHz | n |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        k = r["known_gbps"]; c = r["counted_gbps_raw"]; sc = c * mult if mult else None
        resid = (sc - k) / k * 100 if (k and sc) else None
        print(f"| {r['step']} | {k if k is not None else '—':>10} | {c:.3f} | "
              f"{sc:.1f} | {'' if resid is None else f'{resid:+.1f}'} | "
              f"{r['l2_total_gbps_raw'] or 0:.3f} | {r['sm_throughput_pct'] or 0:.1f} | "
              f"{r['gpc_mhz'] or 0:.0f} | {r['samples']} |".replace("| None", "| —"))
    print(f"\nfitted slice multiplier (through origin, triad steps): {mult:.2f}" if mult else "\nno fit")
    out.with_suffix(".json").write_text(json.dumps({"steps": rows, "multiplier": mult, "hz": a.hz,
                                                    "miss_col": miss_col, "size_mib": a.size_mib,
                                                    "recorded": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    print(f"wrote {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
