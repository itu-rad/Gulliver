from datasets import load_dataset
from torch.utils.data import DataLoader

from stages.stage import Stage, log_phase
from utils.schemas import Query


class SelfRAGDataLoader(Stage):
    """Configurable question dataloader for the Self-RAG pipelines.

    Loads a HuggingFace dataset specified in YAML config and serves
    questions in batches. Fully parametrized — no hardcoded dataset paths.

    YAML config example:
        config:
          batch_size: 1
          dataset:
            name: RUC-NLPIR/FlashRAG_datasets
            subset: web_questions
            split: test
            question_column: question
            # Optional sampling over the split; a stride spreads a small
            # budget over a split that is ordered by topic.
            offset: 0
            stride: 30
    """

    def __init__(self, stage_config, pipeline_config):
        super().__init__(stage_config, pipeline_config)

        self._batch_size = self.extra_config.get("batch_size", 1)

        dataset_config = self.extra_config["dataset"]
        self._dataset_name = dataset_config["name"]
        self._dataset_subset = dataset_config.get("subset", None)
        self._dataset_split = dataset_config.get("split", "test")
        self._question_column = dataset_config.get("question_column", "question")
        self._answers_column = dataset_config.get("answers_column", None)
        self._offset = int(dataset_config.get("offset", 0))
        self._stride = int(dataset_config.get("stride", 1))
        if self._stride < 1:
            raise ValueError(f"dataset.stride must be >= 1 (got {self._stride})")

        self._dataset = None
        self._data_index = 0

    def get_batch_size(self):
        return self._batch_size

    def get_dataset_splits(self) -> dict[str, int]:
        """Get the number of batches for each dataset split.

        Returns:
            dict[str, int]: Dictionary with number of batches for each dataset split
        """
        return {"val": len(self._dataset) // self._batch_size}

    @log_phase
    def prepare(self):
        """Load the dataset from HuggingFace."""
        super().prepare()

        if self._dataset_subset:
            raw = load_dataset(self._dataset_name, self._dataset_subset)[
                self._dataset_split
            ]
        else:
            raw = load_dataset(self._dataset_name)[self._dataset_split]

        if self._offset or self._stride > 1:
            raw = raw.select(range(self._offset, len(raw), self._stride))

        self._dataset = raw
        print(
            f"SelfRAGDataLoader: loaded {len(self._dataset)} samples "
            f"from {self._dataset_name}/{self._dataset_subset or ''} "
            f"(offset={self._offset}, stride={self._stride})"
        )

    def run(self, query: Query) -> dict[int, Query]:
        """Serve the next batch of questions."""

        if query.batch == 0:
            self._data_index = 0

        batch_questions = []
        batch_answers = []
        for _ in range(self._batch_size):
            idx = self._data_index % len(self._dataset)
            sample = self._dataset[idx]
            batch_questions.append(sample[self._question_column])
            if self._answers_column is not None:
                raw = sample.get(self._answers_column)
                if isinstance(raw, str):
                    batch_answers.append([raw])
                elif raw is None:
                    batch_answers.append([])
                else:
                    batch_answers.append(list(raw))
            self._data_index += 1

        # For batch_size=1, pass the question as a single string for
        # downstream compatibility with formatter stages.
        if self._batch_size == 1:
            query.data = batch_questions[0]
        else:
            query.data = batch_questions

        # "question" is the stable record of the originally-submitted query;
        # "original_query" can be overwritten by ChromaRetriever on retry.
        query.context = {"original_query": query.data, "question": query.data}
        if self._answers_column is not None:
            query.context["golden_answers"] = (
                batch_answers[0] if self._batch_size == 1 else batch_answers
            )

        output = {idx: query for idx in self.output_queues}
        return output
