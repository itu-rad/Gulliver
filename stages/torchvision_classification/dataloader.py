import logging
import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.models import get_weight
from torchvision.datasets import VisionDataset, wrap_dataset_for_transforms_v2

from stages.stage import Stage, log_phase
from utils.schemas import Query, StageModel, PipelineModel
from utils.component import get_component


class PreloadedDataset(VisionDataset):
    def __init__(self, dataset, transform=None):
        self.transform = transform
        self.data = []
        for data_point in dataset:
            self.data.append(data_point)

    def __getitem__(self, index):
        x, y = self.data[index]
        return self.transform(x), y

    def __len__(self):
        return len(self.data)


def transform_openimages(split):
    if split == "train":
        return v2.Compose(
            [
                v2.Resize((800, 800)),
                v2.RandomHorizontalFlip(),
                v2.ToTensor(),
            ]
        )
    else:
        return v2.Compose(
            [
                v2.RGB(),
                v2.Resize((800, 800)),
                v2.ToTensor(),
            ]
        )


def order_fingerprint(batch, seed, epoch, n=16):
    """`epoch<e>_seed<s>_<labels>` for the first batch of an epoch: the first n
    labels, joined by '-'. Comma-free, and enough to tell two shuffles apart."""
    labels = batch[1] if isinstance(batch, (list, tuple)) else batch
    try:
        labs = [int(v) for v in labels[:n].tolist()]
    except Exception:
        labs = []
    return f"epoch{epoch}_seed{'none' if seed is None else seed}_" + "-".join(map(str, labs))


class TorchVisionDataLoader(Stage):
    """This stage contains the Torch dataloading functionality with the possibility
    of prefetching and preprocessing the data.
    """

    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        """Initialize the stage by parsing the stage configuration.

        Args:
            stage_config (dict): Stage configuration, such as batch size and number of workers.
        """
        super().__init__(stage_config, pipeline_config)

        dataset_cls = get_component(self.extra_config["dataset"]["component"])

        self._splits = self.extra_config["dataset"].get("split", ["val"])
        dataset_name = self.extra_config["dataset"]["component"].split(".")[-1]
        dataset_root = self.extra_config["dataset"].get(
            "root",
            os.path.join(os.getcwd(), "tmp", "torchvision_dataset", dataset_name),
        )
        dataset_downloaded = os.path.isdir(dataset_root) and os.listdir(dataset_root)

        weights_name = self.extra_config["dataset"].get("weights", None)
        if weights_name is None:
            transform = v2.ToTensor()
        else:
            weights = get_weight(weights_name)
            transform = weights.transforms()
            print(f"Transform: {transform}")

        transform_cls = self.extra_config["dataset"].get("transform", None)

        self._datasets: dict[str, VisionDataset] = {
            x: dataset_cls(
                root=dataset_root,
                **({"split": x} if dataset_name.lower() != "cocodetection" else {}),
                **(
                    {"download": not dataset_downloaded}
                    if dataset_name.lower() not in ["imagenet", "cocodetection"]
                    else {}
                ),
                **(
                    {"annFile": self.extra_config["dataset"]["annFile"]}
                    if dataset_name.lower() == "cocodetection"
                    else {}
                ),
                transform=(
                    transform if not transform_cls else get_component(transform_cls)(x)
                ),
            )
            for x in self._splits
        }

        self._batch_size = self.extra_config.get("batch_size", 1)
        self._dataloaders = {}
        self._dataloader_iter = None
        # Optional deterministic shuffle: epoch e seeds torch with seed + e just
        # before building the iterator, so any process seeding the same way
        # loads the same batches in the same order. SUITE_DATA_SEED overrides
        # the config key.
        env_seed = os.environ.get("SUITE_DATA_SEED")
        self._seed = (int(env_seed) if env_seed not in (None, "")
                      else self.extra_config.get("seed", None))
        if self._seed is not None:
            self._seed = int(self._seed)

    def get_dataset_splits(self) -> dict[str, int]:
        """Get the number of batches for each dataset split.

        Returns:
            dict[str, int]: Dictionary with number of batches for each dataset split
        """
        return {k: len(v) // self._batch_size for (k, v) in self._datasets.items()}

    def _preload_dataset(self) -> None:
        for key in self._datasets.keys():
            # Cache the untransformed samples and apply the transform on access.
            transform = self._datasets[key].transform
            self._datasets[key].transform = None
            self._datasets[key] = PreloadedDataset(self._datasets[key], transform)

    @staticmethod
    def _custom_collate_fn(data):
        """Custom collate identity function, allowing us
        to perform preprocessing in a separate stage.
        """
        return data

    @log_phase
    def prepare(self):
        """Build the dataloaders."""
        super().prepare()

        if self.extra_config.get("preload", False):
            self._preload_dataset()

        num_workers = self.extra_config.get("num_workers", 0)

        if self.extra_config.get("preprocess", True):
            self._dataloaders = {
                k: DataLoader(
                    v,
                    self._batch_size,
                    num_workers=num_workers,
                    shuffle=True,
                    drop_last=True,
                )
                for (k, v) in self._datasets.items()
            }
        else:
            self._dataloaders = {
                k: DataLoader(
                    v,
                    self._batch_size,
                    num_workers=num_workers,
                    shuffle=True,
                    drop_last=True,
                    collate_fn=TorchVisionDataLoader._custom_collate_fn,
                )
                for (k, v) in self._datasets.items()
            }

    def run(self, query: Query) -> dict[int, Query]:
        if query.batch == 0:
            if self._seed is not None:
                torch.manual_seed(self._seed + query.epoch)
            self._dataloader_iter = iter(self._dataloaders[query.split])
        next_batch = next(self._dataloader_iter)
        if query.batch == 0:
            # Log the epoch's order fingerprint regardless of disable_logs, so
            # runs can be checked for identical data order.
            fp = order_fingerprint(next_batch, self._seed, query.epoch)
            logging.getLogger("benchmark").info(
                "%s, %s, order, %s", self.parent_name, self.name, fp)
        query.data = next_batch
        output = {idx: query for idx in self.output_queues}
        return output
