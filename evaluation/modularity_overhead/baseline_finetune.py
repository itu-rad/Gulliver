"""Hand-written PyTorch fine-tuning baseline (no framework) for the modularity-overhead runs.

    python baseline_finetune.py --model efficientnet_v2_s --batch-size 8 --max-batches 1100 [--no-radt]
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models
from torch.utils.data import DataLoader
import os
import time
import logging
import radt

# Same trace column layout as utils/logger.py, inlined so this script runs standalone.
PERF_FORMAT = "%(created)f, %(message)s, %(perf)d"


def _install_perf_clock():
    _orig_factory = logging.getLogRecordFactory()

    def _factory(*args, **kwargs):
        record = _orig_factory(*args, **kwargs)
        record.perf = time.perf_counter_ns()
        return record

    logging.setLogRecordFactory(_factory)


def setup_logging(label="baseline_finetune"):
    log_dir = os.path.abspath(os.environ.get("SUITE_OUTPUT_DIR", "results"))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{label}.csv")

    # Same logging setup as main.py's -p 0 path (one synchronous FileHandler on the
    # `benchmark` logger), except mode='w' so re-runs do not append to stale files.
    _install_perf_clock()
    formatter = logging.Formatter(PERF_FORMAT)
    file_handler = logging.FileHandler(filename=log_file, mode='w')
    file_handler.setFormatter(formatter)

    logger = logging.getLogger("benchmark")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger, file_handler


def parse_args():
    ap = argparse.ArgumentParser(
        description="Hand-written PyTorch baseline for the modularity-overhead "
        "runs: a torchvision Imagenette transfer-learning fine-tune without the framework."
    )
    ap.add_argument("--device", choices=["cuda", "mps", "cpu", "auto"],
                    default="auto", help="compute device (auto: cuda>mps>cpu)")
    ap.add_argument("--model", default="efficientnet_v2_s",
                    help="torchvision.models factory name (e.g. efficientnet_v2_s, "
                         "efficientnet_v2_m, efficientnet_v2_l). "
                         "Must match the framework config's model.component.")
    ap.add_argument("--weights", default="EfficientNet_V2_S_Weights.IMAGENET1K_V1",
                    help="torchvision weights enum name for the preprocessing "
                         "transform (input resolution). Must match the framework "
                         "config's dataloader dataset.weights.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader worker processes (match the framework config)")
    ap.add_argument("--max-batches", type=int, default=1000,
                    help="number of training steps to run")
    ap.add_argument("--label", default="baseline_finetune",
                    help="output CSV basename in $SUITE_OUTPUT_DIR (default results/)")
    ap.add_argument("--seed", type=int, default=None,
                    help="deterministic shuffle: epoch e (1-based) uses torch.manual_seed(seed + e), "
                         "as the framework's TorchVisionDataLoader does")
    ap.add_argument("--no-radt", action="store_true",
                    help="skip the RADTBenchmark telemetry wrapper")
    return ap.parse_args()


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def replace_classifier_head(model, num_classes):
    """Replace the model's final module with Linear(in_features, num_classes).

    Mirrors stages/torchvision_classification/classification.py. Returns the
    replaced module's dotted name, used to select the trainable params.
    """
    *_, (last_name, last_mod) = model.named_modules()
    new_head = nn.Linear(last_mod.in_features, num_classes)
    parts = last_name.split(".")
    if len(parts) == 1:
        setattr(model, parts[0], new_head)
    else:
        model.__getattr__(".".join(parts[:-1]))[int(parts[-1])] = new_head
    return last_name


def order_fingerprint(batch, seed, epoch, n=16):
    """Batch-order fingerprint, identical to stages/torchvision_classification/dataloader.py."""
    labels = batch[1] if isinstance(batch, (list, tuple)) else batch
    try:
        labs = [int(v) for v in labels[:n].tolist()]
    except Exception:
        labs = []
    return f"epoch{epoch}_seed{'none' if seed is None else seed}_" + "-".join(map(str, labs))


def multi_epoch(loader, seed=None, logger=None):
    """Yield batches across as many epochs as needed; the caller bounds the count."""
    epoch = 0
    while True:
        epoch += 1
        if seed is not None:
            # Seed immediately before building the iterator: RandomSampler draws
            # its permutation seed from the global RNG at __iter__.
            torch.manual_seed(seed + epoch)
        first = True
        for batch in loader:
            if first and logger is not None:
                logger.info("baseline_finetune, dataset, order, %s",
                            order_fingerprint(batch, seed, epoch))
                first = False
            yield batch


def run_training(args, logger):
    batch_size = args.batch_size
    num_classes = 10
    lr = 0.001
    max_batches = args.max_batches

    device = resolve_device(args.device)

    logger.info(f"baseline_finetune, system, setup, start, device={device}")

    weights = models.get_weight(args.weights)
    preprocess = weights.transforms()

    # Same location as TorchVisionDataLoader, so both share one downloaded copy.
    dataset_path = os.path.join(os.getcwd(), "tmp", "torchvision_dataset", "Imagenette")
    os.makedirs(dataset_path, exist_ok=True)
    already = os.path.isdir(os.path.join(dataset_path, "imagenette2"))

    try:
        train_dataset = datasets.Imagenette(
            root=dataset_path,
            split="train",
            size="full",
            download=not already,
            transform=preprocess,
        )
    except Exception as e:
        logger.error(f"baseline_finetune, system, error, {str(e)}")
        return

    # drop_last=True to match TorchVisionDataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    model = models.get_model(args.model, weights=None)

    # Freeze everything, then unfreeze only the replaced head, as the stage does.
    for param in model.parameters():
        param.requires_grad = False

    last_name = replace_classifier_head(model, num_classes)
    params_to_update = []
    for name, param in model.named_parameters():
        if last_name in name:
            param.requires_grad = True
            params_to_update.append(param)
    model = model.to(device)

    optimizer = optim.Adam(params_to_update, lr=lr)
    criterion = nn.CrossEntropyLoss()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"baseline_finetune, model, info, trainable_params={trainable_params}, total_params={total_params}"
    )
    print(f"Number of trainable parameters: {trainable_params}")
    print(f"Number of total parameters: {total_params}")

    logger.info("baseline_finetune, system, setup, end")

    batch_count = 0

    logger.info(f"baseline_finetune, training_loop, run, start")

    try:
        for inputs, labels in multi_epoch(train_loader, args.seed, logger):
            if batch_count >= max_batches:
                break

            logger.info("baseline_finetune, training_step, run, start")

            model.train()
            with torch.set_grad_enabled(True):
                inputs = inputs.to(device)
                # LongTensor cast matches stages/torchvision_classification/classification.py
                labels = labels.type(torch.LongTensor)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # synchronize so the step timing includes GPU execution
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()

            logger.info("baseline_finetune, training_step, run, end")

            batch_count += 1

    except KeyboardInterrupt:
        logger.info("baseline_finetune, training_loop, run, interrupted")

    logger.info(f"baseline_finetune, training_loop, run, end")


def main():
    args = parse_args()
    logger, file_handler = setup_logging(args.label)

    try:
        if args.no_radt:
            run_training(args, logger)
        else:
            with radt.run.RADTBenchmark() as run:
                run_training(args, logger)
    finally:
        # flush explicitly so the last rows survive a hard teardown
        file_handler.flush()
        file_handler.close()


if __name__ == "__main__":
    main()
