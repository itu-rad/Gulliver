#!/usr/bin/env python3
"""Generate serialized Self-RAG configs (one request in flight, closed loop) from the base configs.

    python gen_serialized_configs.py [--queries 30] [--budget N] [--precision q4] [--server]
"""
import argparse
import copy
import glob
import os
import re

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
# Precision "q4": uniform 4-bit MLX checkpoints on mlx; on cuda the stage's own
# bitsandbytes NF4 (`quantize: true`).
Q4_MLX = {"Qwen3.5-4B": "mlx-community/Qwen3.5-4B-MLX-4bit",
          "Qwen3.5-9B": "mlx-community/Qwen3.5-9B-MLX-4bit"}


def apply_precision(cfg: dict, dev: str, precision: str) -> None:
    """Rewrite every in-process LLM stage of `cfg` for the requested precision."""
    if precision != "q4":
        return
    for st_ in cfg["pipelines"][0]["stages"]:
        comp = st_.get("component", "")
        m = st_.get("config", {}).get("model")
        if not isinstance(m, dict):
            continue
        if comp.startswith("stages.llm_mlx"):
            for fam, q in Q4_MLX.items():
                if fam in m.get("name", ""):
                    m["name"] = q
        elif comp.startswith("stages.llm_huggingface"):
            m["quantize"] = True
SKIP = ("ollama", "vllm", "_0.8b_", "_2b_", "_27b_", "serial", "_acc_", "_pipe_", "quantest", "e4smoke")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=30)
    ap.add_argument("--budget", type=int, default=None,
                    help="thinking budget to set on every LLM stage; 0 = thinking OFF "
                         "(formatters render the empty think block, no reasoning is generated)")
    ap.add_argument("--only", default=None,
                    help="glob on the base config stem to restrict the arms, e.g. "
                         "'factoid_{monolith_4b,decomposed_shared}' (fnmatch, braces expanded)")
    ap.add_argument("--suffix", default="serial",
                    help="serial = latency pass (small n, R>1); acc = accuracy pass "
                         "(large n, R=1; greedy + fixed budget is deterministic per machine)")
    ap.add_argument("--precision", choices=["bf16", "q4"], default="bf16",
                    help="q4 = the 4-bit lane: MLX-4bit checkpoints on mlx, bitsandbytes NF4 "
                         "on cuda; 'q4' is appended to the suffix so files never collide")
    ap.add_argument("--server", action="store_true",
                    help="also emit the server-arm twins (monolith_4b, decomposed_shared) "
                         "serialised the same way, via gen_pipelined_configs.serverise")
    ap.add_argument("--stage-budgets", default=None,
                    help="per-ROLE thinking budgets for the decomposed graph, e.g. "
                         "'grade=0,generate=1024,hallucination=0,rewrite=0'; roles are matched "
                         "against the LLM stage names ('... LLM (grade)') and their formatters "
                         "get enable_thinking False where the budget is 0. Overrides --budget "
                         "for the named roles.")
    ap.add_argument("--arm-suffix", default=None,
                    help="token appended to the arm in the output name (e.g. 'mix' -> "
                         "factoid_decomposed_shared_mix_<suffix>_<dev>.yml) so the analyzer "
                         "keys the cell as its own arm")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "configs"))
    args = ap.parse_args()
    stage_budgets = {}
    if args.stage_budgets:
        for kv in args.stage_budgets.split(","):
            k, v = kv.split("="); stage_budgets[k.strip().lower()] = int(v)
    if args.precision == "q4" and not args.suffix.endswith("q4"):
        args.suffix += "q4"

    n = 0
    for src in sorted(glob.glob(os.path.join(HERE, "configs", "*_mlx.yml"))
                      + glob.glob(os.path.join(HERE, "configs", "*_cuda.yml"))):
        base = os.path.basename(src)
        # derive only from base configs, never from generated pass configs
        if any(s in base for s in SKIP) or re.search(r"_(pipe|acc|serial)\d*(q4m?)?(s\d+)?(r\d+)?_", base):
            continue
        if args.only:
            import fnmatch, itertools
            pats = [args.only]
            if "{" in args.only:
                pre, rest = args.only.split("{", 1); alts, post = rest.split("}", 1)
                pats = [pre + a + post for a in alts.split(",")]
            stem0 = base[:-4].rsplit("_", 1)[0]
            if not any(fnmatch.fnmatch(stem0, pt) for pt in pats):
                continue
        cfg = yaml.safe_load(open(src, "r", encoding="utf-8"))
        cfg = copy.deepcopy(cfg)
        if args.budget is not None:
            for st_ in cfg["pipelines"][0]["stages"]:
                m = st_.get("config", {}).get("model")
                if isinstance(m, dict) and "thinking_budget" in m:
                    m["thinking_budget"] = args.budget
                if args.budget == 0 and "tokenizer_stage_id" in st_.get("config", {}):
                    st_["config"]["enable_thinking"] = False     # formatters: no <think> block
        if stage_budgets:
            # an LLM stage's role is the word in its parentheses; its formatter's name
            # carries the same role word
            import re as _re
            stages = cfg["pipelines"][0]["stages"]
            role_of = {}
            for st_ in stages:
                m_ = _re.search(r"LLM \((\w+)\)", st_.get("name", ""))
                if m_:
                    role_of[st_["id"]] = m_.group(1).lower()
            for st_ in stages:
                role = role_of.get(st_["id"])
                if role and role in stage_budgets:
                    st_["config"]["model"]["thinking_budget"] = stage_budgets[role]
            fmt_role = {"grader formatter": "grade", "retrieval grader": "grade", "answer generator": "generate",
                        "hallucination grader": "hallucination", "rewrite": "rewrite", "rewriter": "rewrite"}
            for st_ in stages:
                nm = st_.get("name", "").lower()
                if "formatter" not in nm:
                    continue
                role = next((r for k, r in fmt_role.items() if k in nm), None)
                if role in stage_budgets:
                    st_["config"]["enable_thinking"] = stage_budgets[role] > 0
        for pipe in cfg["pipelines"]:
            pipe["serialize_queries"] = True
            lg = pipe["loadgen"]
            lg["queue_depth"] = 1
            lg["max_queries"] = args.queries
            # closed loop: the next request is issued when the previous one finishes
            lg["component"] = "loadgen.OfflineLoadScheduler"
            lg["config"] = {"rate": 0}
            lg["timeout"] = 3600000
        # Output name <task>_<arm>_<suffix>_<device>.yml keeps the device last for collect.sh.
        stem, dev = base[:-4].rsplit("_", 1)
        if args.arm_suffix:
            stem = f"{stem}_{args.arm_suffix}"
        cfg["listeners"] = {"mlx": ["macmon"], "cuda": ["dcgmi", "top"]}[dev]
        apply_precision(cfg, dev, args.precision)
        variants = [(stem, cfg)]
        if args.server and stem.split("_", 1)[1] in ("monolith_4b", "decomposed_shared"):
            from gen_pipelined_configs import serverise
            variants.append((f"{stem}_server", (lambda c: (c.__setitem__('listeners', list(c.get('listeners') or []) + ['vllm']) or c))(serverise(cfg, 8, dev, args.precision))))
        for name, c in variants:
            out = os.path.join(args.out_dir, f"{name}_{args.suffix}_{dev}.yml")
            with open(out, "w", encoding="utf-8") as f:
                yaml.safe_dump(c, f, sort_keys=False)
            n += 1
    print(f"wrote {n} {args.suffix} configs ({args.queries} queries, queue_depth 1)")


if __name__ == "__main__":
    main()
