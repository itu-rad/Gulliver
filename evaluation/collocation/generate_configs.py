#!/usr/bin/env python3
"""Generate the collocation configs: a decomposed Self-RAG foreground on vLLM
beside a serial background job (CLIP image encoder or e5-base indexer), and the
background jobs alone.

    python evaluation/collocation/generate_configs.py real --tag e5s --sizes 4b,9b,27b \
        --service '{"4b":129,"9b":248.6,"27b":707.9}' --cells baseline,bg_encode_mps,bg_encode_ts
    python evaluation/collocation/generate_configs.py alone --tag e5s --sizes 4b --cells bg_encode_mps
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FG_BASE = ROOT / "evaluation/self_rag/configs/factoid_decomposed_shared_server_pipe1024r32_cuda.yml"

SIZES = {  # model, vLLM gpu_memory_utilization
    "4b": ("Qwen/Qwen3.5-4B", 0.5),
    "9b": ("Qwen/Qwen3.5-9B", 0.5),
    "27b": ("Qwen/Qwen3.5-27B", 0.7),
}
ENGINE_LOAD = 4.0          # arrival rate = ENGINE_LOAD / batch-1 service time
FG_QUERIES = 24
LISTENERS_FG = ["dcgmi", "top", "vllm"]
LISTENERS_BG = ["dcgmi", "top"]

BG_POINTS = {
    "index_e5base_32": {"kind": "index", "embedder": "intfloat/e5-base-v2", "docs_per_query": 32},
    "encode_1": {"kind": "encode", "batch_size": 1},
    "encode_1s": {"kind": "encode", "batch_size": 1, "split": True},
}
CELLS = {  # cell -> (background point, MPS)
    "baseline": (None, False),
    "bg_index_mps": ("index_e5base_32", True),
    "bg_index_ts": ("index_e5base_32", False),
    "bg_encode_mps": ("encode_1", True),
    "bg_encode_ts": ("encode_1", False),
    "bg_encodes_mps": ("encode_1s", True),
    "bg_encodes_ts": ("encode_1s", False),
}
VQA = {"name": "Multimodal-Fatima/OK-VQA_train", "split": "train", "image_column": "image",
       "question_column": "question", "answers_column": "answers", "max_samples": 100}
CLIP = {"name": "openai/clip-vit-large-patch14"}


def foreground(size: str, service_s: float, queries: int) -> tuple[dict, float]:
    doc = copy.deepcopy(yaml.safe_load(FG_BASE.read_text(encoding="utf-8")))
    model, util = SIZES[size]
    for st in doc["pipelines"][0]["stages"]:
        cfg = st.get("config") or {}
        if "server" in cfg:
            cfg["server"]["model"] = model
            cfg["server"]["gpu_memory_utilization"] = util
            la = [x for x in cfg["server"].get("launch_args", []) if x not in ("--quantization", "mxfp4")]
            # Let the engine exit on SIGTERM rather than be killed under an Nsight session.
            cfg["server"]["launch_args"] = la + ["--shutdown-timeout", "60"]
            cfg["tokenizer_name"] = model
    doc["pipelines"][0]["name"] = f"Self-RAG Decomposed-Shared {size}"
    rate = ENGINE_LOAD / service_s
    doc["pipelines"][0]["loadgen"] = {"component": "loadgen.PoissonLoadScheduler", "queue_depth": queries,
                                      "max_queries": queries, "timeout": 3600000,
                                      "config": {"rate": round(rate, 6), "seed": 0}}
    doc["listeners"] = LISTENERS_FG
    return doc, rate


def background(name: str, cell_tag: str) -> dict:
    """A background pipeline that issues its next query when the previous one finishes."""
    point = BG_POINTS[name]
    loadgen = {"component": "loadgen.OfflineLoadScheduler", "queue_depth": 1,
               "max_queries": 1000000, "timeout": 10800, "config": {"rate": 0}}
    if point["kind"] == "index":
        stages = [
            {"name": "Corpus chunk loader", "id": 0, "outputs": [1],
             "component": "stages.rag_indexing.CorpusChunkLoader",
             "config": {"dataset": {"name": "rag-datasets/rag-mini-wikipedia", "subset": "text-corpus",
                                    "split": "passages", "text_column": "passage"},
                        "docs_per_query": point["docs_per_query"], "shard": {"index": 0, "count": 1}}},
            {"name": "Embed passages (e5base)", "id": 1, "outputs": [2],
             "component": "stages.rag_indexing.EmbedStage",
             "config": {"model": {"name": point["embedder"]}, "device": "cuda", "max_length": 256}},
            {"name": "Chroma indexer", "id": 2, "component": "stages.rag_indexing.ChromaIndexer",
             "config": {"collection_name": f"bg_{name}_{cell_tag}"}},
        ]
        return {"name": f"BG indexer {name}", "inputs": [0], "outputs": [2], "dataset_stage_id": 0,
                "loadgen": loadgen, "stages": stages}
    stages = [{"name": "VQA dataloader", "id": 0, "outputs": [1],
               "component": "stages.multimodal_vqa.VQADataLoader",
               "config": {"dataset": dict(VQA), "batch_size": point["batch_size"]}}]
    if point.get("split"):
        # CPU preprocessing as its own stage, so the encoder stage covers only GPU work.
        stages.append({"name": "CLIP preprocess", "id": 1, "outputs": [2],
                       "component": "stages.multimodal_vqa.CLIPImagePreprocess",
                       "config": {"model": dict(CLIP)}})
    stages.append({"name": "CLIP vision encoder", "id": len(stages),
                   "component": "stages.multimodal_vqa.CLIPVisionEncoder",
                   "config": {"model": dict(CLIP), "device": "cuda"}})
    stages[0]["outputs"] = [1]
    return {"name": f"BG encoder {name}", "inputs": [0], "outputs": [len(stages) - 1],
            "dataset_stage_id": 0, "loadgen": loadgen, "stages": stages}


def dump(doc: dict, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{stem}_cuda.yml", "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["real", "alone"])
    ap.add_argument("--tag", required=True, help="config/label prefix")
    ap.add_argument("--sizes", default="4b", help="comma list of " + ",".join(SIZES))
    ap.add_argument("--service", default='{"4b": 129}', help="JSON size -> batch-1 service seconds")
    ap.add_argument("--cells", required=True, help="comma list of " + ",".join(CELLS))
    ap.add_argument("--queries", type=int, default=FG_QUERIES, help="foreground queries per cell")
    ap.add_argument("--out-dir", type=Path, default=HERE / "configs")
    a = ap.parse_args()
    sizes = [s for s in a.sizes.split(",") if s]
    service = json.loads(a.service)
    for size in sizes:
        for cell in a.cells.split(","):
            point, mps = CELLS[cell]
            tag = f"{size}_{cell}"
            if a.mode == "alone":
                bg = background(point, tag)
                bg["loadgen"]["timeout"] = 600
                stem = f"fp_serial_{point}"
                dump({"name": f"serial background alone ({stem}_cuda)", "listeners": LISTENERS_BG,
                      "pipelines": [bg], "collocation": ""}, a.out_dir, stem)
                continue
            doc, rate = foreground(size, float(service[size]), a.queries)
            doc["collocation"] = "mps" if mps else ""
            if point:
                bg = background(point, tag)
                bg["stop_with"] = doc["pipelines"][0]["name"]
                doc["pipelines"].append(bg)
            doc["name"] = f"Collocation {size} {cell}"
            dump(doc, a.out_dir, f"{a.tag}_{size}_{cell}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
