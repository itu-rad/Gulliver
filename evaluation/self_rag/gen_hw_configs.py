#!/usr/bin/env python3
"""Generate the GPU-metrics variants of the serial CUDA configs.

Adds the `gpumetrics` listener and `--shutdown-timeout 60` to every vLLM server,
so the engine is not killed under an active Nsight session.

    python evaluation/self_rag/gen_hw_configs.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "configs")

CELLS = {
    "decomposed_serial0": "factoid_decomposed_shared_serial0_cuda.yml",
    "decomposed_serial1024nf4": "factoid_decomposed_shared_serialq4_cuda.yml",
    "decomposed_serial1024": "factoid_decomposed_shared_serial1024_cuda.yml",
    "decomposed_engine_serial1024mxfp4": "factoid_decomposed_shared_server_serialq4_cuda.yml",
    "decomposed_engine_serial1024": "factoid_decomposed_shared_server_serial1024_cuda.yml",
}


def patch(doc: dict) -> dict:
    ls = list(doc.get("listeners") or [])
    if "gpumetrics" not in ls:
        ls.append("gpumetrics")
    doc["listeners"] = ls
    for p in doc["pipelines"]:
        for s in p["stages"]:
            srv = (s.get("config") or {}).get("server")
            if srv:
                la = srv.setdefault("launch_args", [])
                if "--shutdown-timeout" not in la:
                    la += ["--shutdown-timeout", "60"]
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "configs_hw"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for cell, base in CELLS.items():
        doc = patch(yaml.safe_load(open(os.path.join(SRC, base))))
        with open(os.path.join(args.out_dir, cell + ".yml"), "w") as f:
            yaml.safe_dump(doc, f, sort_keys=False)
    print(f"{len(CELLS)} configs -> {args.out_dir}")


if __name__ == "__main__":
    main()
