from torch import ne
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
)
from accelerate import Accelerator

from stages.stage import Stage, log_phase
from utils.schemas import Query, StageModel, PipelineModel


class HuggingFaceDataLoader(Stage):

    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        """
        Initializes the DataLoader with the given stage and pipeline configurations.
        Args:
            stage_config (StageModel): Configuration specific to the current stage.
            pipeline_config (PipelineModel): Configuration for the entire pipeline.
        """

        super().__init__(stage_config, pipeline_config)

        self._batch_size = self.extra_config.get("batch_size", 1)
        self._split = self.extra_config.get("split", ["train"])

        dataset_config = self.extra_config.get("dataset", {})
        dataset_name = dataset_config["name"]
        dataset_subset = dataset_config.get("subset", None)
        self._datasets = load_dataset(
            dataset_name,
            **({"name": dataset_subset} if dataset_subset else {}),
            split=self._split,
        )

        self._datasets = {
            self._split[idx]: split for idx, split in enumerate(self._datasets)
        }

        self._dataset_length = {k: len(v) for (k, v) in self._datasets.items()}
        self._datasets = {
            k: v.to_iterable_dataset(num_shards=self._dataset_length[k])
            for (k, v) in self._datasets.items()
        }
        if self.extra_config.get("shuffle", True):
            self._datasets = {k: v.shuffle() for (k, v) in self._datasets.items()}

        self._system_column_name = dataset_config.get("system_column_name")
        self._user_column_name = dataset_config.get("user_column_name")
        self._assistant_column_name = dataset_config.get("assistant_column_name")

        self._instruction = dataset_config.get("instruction", None)

        self._tokenizer = None
        self._dataloaders = {}
        self._dataloader_iter = None

    def get_batch_size(self) -> int:
        """
        Retrieve the batch size for the data loader.

        Returns:
            int: The batch size currently set for the data loader.
        """
        return self._batch_size

    def get_dataset_splits(self) -> dict[str, int]:
        """Get the number of batches for each dataset split.

        Returns:
            dict[str, int]: Dictionary with number of batches for each dataset split
        """
        return {
            split: self._dataset_length[split] // self._batch_size
            for split in self._split
        }

    def _concat(self, is_train: bool):
        def _concat_train(sample):
            if sample[self._user_column_name] != "":
                sample["concated_text"] = (
                    "Below is an instruction that describes a task, paired with an input that provides further context. "
                    "Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{self._instruction if self._instruction else sample[self._system_column_name]}\n\n### Input:\n{sample[self._user_column_name]}\n\n"
                    f"### Response:\n{sample[self._assistant_column_name]}\n\n"
                )
            else:
                sample["concated_text"] = (
                    "Below is an instruction that describes a task. "
                    "Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{sample[self._system_column_name]}\n\n"
                    f"### Response:\n{sample[self._assistant_column_name]}\n\n"
                )
            return sample

        def _concat_eval(sample):
            if sample[self._user_column_name] != "":
                sample["concated_text"] = (
                    "Below is an instruction that describes a task, paired with an input that provides further context. "
                    "Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{self._instruction if self._instruction else sample[self._system_column_name]}\n\n### Input:\n{sample[self._user_column_name]}\n\n"
                    f"### Response:\n"
                )
            else:
                sample["concated_text"] = (
                    "Below is an instruction that describes a task. "
                    "Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{sample[self._system_column_name]}\n\n"
                    f"### Response:\n"
                )
            return sample

        return _concat_train if is_train else _concat_eval

    def _tokenize(self, samples):
        inputs = self._tokenizer(
            samples["concated_text"],
            padding=True,
            truncation=True,
        )
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs

    @log_phase
    def prepare(self):
        """Build the dataloaders."""
        super().prepare()

        self._tokenizer = self.dispatch_call(
            self.extra_config.get("tokenizer_stage_id", 0), "get_tokenizer"
        )
        self._datasets = {
            k: v.map(self._concat(k == "train")) for (k, v) in self._datasets.items()
        }
        self._datasets = {
            k: v.map(self._tokenize, batched=True) for (k, v) in self._datasets.items()
        }
        self._datasets = {
            k: v.remove_columns(
                [
                    self._system_column_name,
                    self._user_column_name,
                    self._assistant_column_name,
                    "concated_text",
                ]
            )
            for (k, v) in self._datasets.items()
        }
        self._datasets = {
            k: v.with_format("torch") for (k, v) in self._datasets.items()
        }
        self._dataloaders = {
            k: DataLoader(
                v,
                batch_size=self._batch_size,
                collate_fn=DataCollatorForLanguageModeling(
                    tokenizer=self._tokenizer, mlm=False
                ),
            )
            for (k, v) in self._datasets.items()
        }

        accelerator: Accelerator | None = self.dispatch_call(
            self.extra_config.get("accelerator_stage_id", 0), "get_accelerator"
        )

        if accelerator:
            self._dataloaders = {
                k: accelerator.prepare(v) for (k, v) in self._dataloaders.items()
            }

    def run(self, query: Query) -> dict[int, Query]:
        """Poll for incoming data in the queues,
        load the next batch of data and pass it onto the output queues."""

        if query.batch == 0:
            self._dataloader_iter = iter(self._dataloaders[query.split])
        next_batch = next(self._dataloader_iter)
        print(next_batch)
        query.data = next_batch
        output = {idx: query for idx in self.output_queues}
        return output
