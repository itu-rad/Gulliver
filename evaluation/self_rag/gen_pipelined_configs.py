#!/usr/bin/env python3
"""Generate pipelined (concurrent, Poisson-arrival) Self-RAG configs from the base configs.

Arrival rate is rho / serial service time per (task, device), or one absolute --rate.
For the monolith_4b and decomposed_shared arms it also emits a `_server` twin whose
LLM stages call one vLLM server.

    python gen_pipelined_configs.py --budget 256 --queries 24 --rho 0.7 \\
        --service '{"factoid/mlx": 45, "factoid/cuda": 40}'
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP = ("ollama", "vllm", "_0.8b_", "_2b_", "_27b_", "serial", "_acc_", "_pipe_",
        "quantest", "e4smoke")
LISTENERS = {"mlx": ["macmon"], "cuda": ["dcgmi", "top"]}
SERVER_MODEL = "Qwen/Qwen3.5-4B"     # same weights as the HF/MLX bf16 arms
TOKENIZER = "Qwen/Qwen3.5-4B"
# Server arms use vLLM on CUDA and vllm-metal on Apple silicon, each in its own
# environment (VLLM_EXECUTABLE). vllm-metal sizes its cache itself, so it gets
# no --gpu-memory-utilization.
SERVER_ENGINES = {"cuda": "vllm", "mlx": "vllm"}
GPU_MEM_UTIL = {"cuda": 0.5, "mlx": None}
# 4-bit server settings: the MLX-4bit checkpoint on mlx (0.45 leaves memory for the
# harness), MXFP4 quantisation at load on cuda.
Q4_SERVER = {"mlx": {"model": "mlx-community/Qwen3.5-4B-MLX-4bit", "util": 0.45, "dtype": None,
                     "launch": []},
             "cuda": {"model": "Qwen/Qwen3.5-4B", "util": 0.5, "dtype": None,
                      "launch": ["--quantization", "mxfp4"]}}


def serverise(cfg: dict, slots: int, dev: str, precision: str = "bf16") -> dict:
    """Swap every LLM stage of a pipeline for the server backend.

    The first LLM stage owns the server; the others depend on it. The per-stage
    `concurrency` is how many requests each stage keeps in flight; the engine
    batches whatever arrives together.
    """
    cfg = copy.deepcopy(cfg)
    owner = None
    for st in cfg["pipelines"][0]["stages"]:
        comp = st.get("component", "")
        if comp.startswith("stages.llm_mlx") or comp.startswith("stages.llm_huggingface"):
            old = st["config"]
            gk = dict(old["model"].get("gen_kwargs", {}))
            ans = gk.get("max_tokens", gk.get("max_new_tokens", 256))
            st["component"] = "stages.llm_server.Inference"
            new = {"concurrency": slots,
                   "model": {"gen_kwargs": {"max_new_tokens": ans, "do_sample": False},
                             "thinking_budget": old["model"].get("thinking_budget", 0)}}
            if owner is None:
                owner = st["id"]
                new["server"] = {"engine": "vllm", "model": SERVER_MODEL, "port": 0,
                                 "dtype": "bfloat16", "max_model_len": 8192,
                                 "launch_args": ["--max-num-seqs", str(slots)]}
                if GPU_MEM_UTIL[dev] is not None:
                    new["server"]["gpu_memory_utilization"] = GPU_MEM_UTIL[dev]
                if precision == "q4":
                    q = Q4_SERVER[dev]
                    new["server"]["model"] = q["model"]
                    new["server"]["gpu_memory_utilization"] = q["util"]
                    new["server"]["launch_args"] += q["launch"]
                    if q["dtype"] is None:
                        new["server"].pop("dtype", None)
                new["tokenizer_name"] = TOKENIZER
            else:
                new["depends_on_id"] = owner
            st["config"] = new
        # formatters fetch the tokenizer from an LLM stage id; the owner has it
        if st.get("config", {}).get("tokenizer_stage_id") is not None and owner is not None:
            st["config"]["tokenizer_stage_id"] = owner
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=int, required=True, help="thinking budget (tokens)")
    ap.add_argument("--queries", type=int, default=24)
    ap.add_argument("--rho", type=float, default=0.7,
                    help="offered load relative to the shared arm's serial service time")
    ap.add_argument("--service", required=True,
                    help='JSON: {"<task>/<engine>": seconds} shared-arm serial service time '
                         'at this budget, e.g. \'{"factoid/mlx": 176}\'')
    ap.add_argument("--slots", type=int, default=8, help="server slots / per-stage concurrency")
    ap.add_argument("--rate", type=float, default=None,
                    help="one absolute arrival rate (q/s) for every device and arm, overriding rho/service")
    ap.add_argument("--suffix", default="pipe",
                    help="pass token in the filename/label; use pipe<budget> when collecting "
                         "more than one budget so the runs do not overwrite each other")
    ap.add_argument("--precision", choices=["bf16", "q4"], default="bf16",
                    help="q4 = the 4-bit lane (see Q4_SERVER); 'q4' is appended to the suffix")
    ap.add_argument("--stage-budgets", default=None,
                    help="per-ROLE thinking budgets for the decomposed graph, e.g. "
                         "'grade=0,generate=1024,hallucination=0,rewrite=0' (see gen_serialized_configs)")
    ap.add_argument("--arm-suffix", default=None, help="token appended to the arm in the output name, e.g. 'mix'")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "configs"))
    args = ap.parse_args()
    service = json.loads(args.service)
    stage_budgets = {}
    if args.stage_budgets:
        for kv in args.stage_budgets.split(","):
            k, v = kv.split("="); stage_budgets[k.strip().lower()] = int(v)
    if args.precision == "q4" and "q4" not in args.suffix:
        args.suffix += "q4"

    n = 0
    for src in sorted(glob.glob(os.path.join(HERE, "configs", "*_mlx.yml"))
                      + glob.glob(os.path.join(HERE, "configs", "*_cuda.yml"))):
        base = os.path.basename(src)
        if any(s in base for s in SKIP) or re.search(r"_(pipe|acc|serial)\d*(q4m?)?(s\d+)?(r\d+)?_", base):
            continue
        stem, dev = base[:-4].rsplit("_", 1)
        task, arm = stem.split("_", 1)
        key = f"{task}/{dev}"
        if key not in service:
            print(f"skip {base}: no service time for {key}")
            continue
        rate = args.rate if args.rate else args.rho / float(service[key])
        cfg = yaml.safe_load(open(src, "r", encoding="utf-8"))
        cfg = copy.deepcopy(cfg)
        cfg["listeners"] = LISTENERS[dev]
        pipe = cfg["pipelines"][0]
        pipe["serialize_queries"] = False
        pipe["loadgen"] = {"component": "loadgen.PoissonLoadScheduler",
                           "queue_depth": args.queries, "max_queries": args.queries,
                           "timeout": 3600000, "config": {"rate": round(rate, 6), "seed": 0}}
        for st in pipe["stages"]:
            m = st.get("config", {}).get("model")
            if isinstance(m, dict) and "thinking_budget" in m:
                m["thinking_budget"] = args.budget
            # budget 0 = thinking off: the formatters must drop the <think> block too,
            # or the model reasons unbounded inside the answer budget
            if args.budget == 0 and "tokenizer_stage_id" in st.get("config", {}):
                st["config"]["enable_thinking"] = False
        if args.precision == "q4":
            from gen_serialized_configs import apply_precision
            apply_precision(cfg, dev, "q4")
        if stage_budgets:
            import re as _re
            stages = pipe["stages"]; role_of = {}
            for st in stages:
                m_ = _re.search(r"LLM \((\w+)\)", st.get("name", ""))
                if m_:
                    role_of[st["id"]] = m_.group(1).lower()
            for st in stages:
                role = role_of.get(st["id"])
                if role and role in stage_budgets:
                    st["config"]["model"]["thinking_budget"] = stage_budgets[role]
            fmt_role = {"grader formatter": "grade", "retrieval grader": "grade", "answer generator": "generate",
                        "hallucination grader": "hallucination", "rewrite": "rewrite", "rewriter": "rewrite"}
            for st in stages:
                nm = st.get("name", "").lower()
                if "formatter" in nm:
                    role = next((r for k, r in fmt_role.items() if k in nm), None)
                    if role in stage_budgets:
                        st["config"]["enable_thinking"] = stage_budgets[role] > 0
        base_arm = arm
        arm_out = f"{arm}_{args.arm_suffix}" if args.arm_suffix else arm
        variants = [(arm_out, cfg)]
        if base_arm in ("monolith_4b", "decomposed_shared") and dev in SERVER_ENGINES:
            sv = serverise(cfg, args.slots, dev, args.precision)
            sv["listeners"] = list(LISTENERS[dev]) + ["vllm"]     # engine-side gauges too
            variants.append((f"{arm_out}_server", sv))
        for name, c in variants:
            out = os.path.join(args.out_dir, f"{task}_{name}_{args.suffix}_{dev}.yml")
            with open(out, "w", encoding="utf-8") as f:
                yaml.safe_dump(c, f, sort_keys=False)
            n += 1
            print(f"wrote {os.path.basename(out)}  rate={rate:.5f}/s (1 per {1/rate:.0f} s)")
    print(f"wrote {n} pipelined configs (budget {args.budget}, {args.queries} queries, rho {args.rho})")


if __name__ == "__main__":
    main()
