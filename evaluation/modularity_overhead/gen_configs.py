#!/usr/bin/env python3
"""Generate one modularity-overhead config per (sweep cell, device).

Reads the cell manifest (configs/scale_sweep.yml) and the pipeline template
(configs/torchvision_training.yml); baseline_finetune.py takes the same
(model, weights, batch) triple on its command line.

    python gen_configs.py [--devices mps cuda] [--out-dir configs/generated]
"""

import argparse
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "configs", "torchvision_training.yml")
CELLS = os.path.join(HERE, "configs", "scale_sweep.yml")
NUM_WORKERS = 0

# The torch device string goes inside the config; filenames carry the machine name.
DEVICE_NAME = {"mps": "m3pro", "cuda": "gb10"}


def cell_name(cell):
    """Stable per-cell label: m<tag>_b<batch> (matches the CSV suffix scheme)."""
    return f"m{cell.get('tag', 'canonical')}_b{cell['batch']}"


def build(cell, device, template):
    cfg = yaml.safe_load(open(template, "r", encoding="utf-8"))
    # No listeners: these cells run via `main.py -p 0` without radt.
    cfg["listeners"] = []
    # Measurement comes from spans; the synchronous per-query CSV write would land
    # inside the measured interval, so pipeline logs are off.
    cfg["name"] = f"Modularity overhead {cell['model']} batch {cell['batch']} ({device})"
    pipe = cfg["pipelines"][0]
    pipe["disable_logs"] = True
    pipe["loadgen"]["max_queries"] = int(cell.get("max_batches", 300))
    for stage in pipe["stages"]:
        comp = stage.get("component", "")
        # Per-stage CSV rows off for the same reason; the breakdown comes from spans.
        stage["disable_logs"] = True
        if "TorchVisionClassification" in comp:
            stage["config"]["device"] = device
            stage["config"]["model"]["component"] = f"torchvision.models.{cell['model']}"
        if "TorchVisionDataLoader" in comp:
            stage["config"]["num_workers"] = NUM_WORKERS
            stage["config"]["batch_size"] = cell["batch"]
            # dataset.weights sets the preprocessing transform (input resolution)
            stage["config"]["dataset"]["weights"] = cell["weights"]
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="+", default=["mps", "cuda"])
    ap.add_argument("--out-dir", default=os.path.join(HERE, "configs", "generated"))
    ap.add_argument("--cells", default=CELLS)
    ap.add_argument("--template", default=TEMPLATE)
    args = ap.parse_args()

    cells = yaml.safe_load(open(args.cells, "r", encoding="utf-8"))["cells"]
    os.makedirs(args.out_dir, exist_ok=True)
    n = 0
    for device in args.devices:
        for cell in cells:
            cfg = build(cell, device, args.template)
            machine = DEVICE_NAME.get(device, device)
            path = os.path.join(args.out_dir,
                                f"mod_{cell_name(cell)}_{machine}.yml")
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            print(f"wrote {os.path.relpath(path, HERE)}")
            n += 1
    print(f"\n{n} configs ({len(cells)} cells x {len(args.devices)} devices), "
          f"num_workers={NUM_WORKERS}")


if __name__ == "__main__":
    main()
