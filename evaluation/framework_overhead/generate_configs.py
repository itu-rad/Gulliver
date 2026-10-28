"""Write the NoOp chain configs: depths 1-128 without payload, and depth 10
with payloads of 1 KiB, 1 MiB and 10 MiB. Per-stage CSV rows are off.

    python generate_configs.py [--out-dir DIR]
"""
import argparse
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DEPTHS = [1, 2, 4, 8, 16, 32, 64, 128]
PAYLOAD_DEPTH, PAYLOAD_SIZES = 10, [0, 1024, 1048576, 10485760]


def noop_chain(depth: int, payload_size: int, max_queries: int = 101) -> dict:
    stages = []
    for i in range(depth):
        component, config = "stages.stage.Stage", {}
        if i == 0 and payload_size > 0:
            component = "stages.evaluation.data_injector.DataInjector"
            config = {"payload_size": payload_size}
        stages.append({
            "name": f"Ref Stage {i}",
            "id": i,
            "component": component,
            "outputs": [i + 1] if i < depth - 1 else [],
            "polling_policy": "utils.queues.polling.SingleQueuePolicy",
            "disable_logs": True,
            "config": config,
        })
    pipeline = {
        "name": f"NoOp Chain Depth {depth} Size {payload_size} Mode ref nolog",
        "inputs": [0],
        "outputs": [depth - 1],
        "dataset_stage_id": 0,
        "loadgen": {
            "component": "loadgen.OfflineLoadScheduler",
            "queue_depth": 100,
            "max_queries": max_queries,
            "timeout": 60000,
            "config": {"rate": 0},
        },
        "stages": stages,
    }
    return {"name": f"noop_benchmark_depth_{depth}_size_{payload_size}_mode_ref (logs off)",
            "pipelines": [pipeline]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(HERE, "configs"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cells = [(d, 0) for d in DEPTHS] + [(PAYLOAD_DEPTH, s) for s in PAYLOAD_SIZES]
    for depth, size in cells:
        path = os.path.join(args.out_dir, f"noop_depth_{depth}_size_{size}_mode_ref_nolog.yml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(noop_chain(depth, size), f, sort_keys=False)
    print(f"wrote {len(cells)} configs to {args.out_dir}")


if __name__ == "__main__":
    main()
