from datasets import load_dataset

from stages.stage import Stage, log_phase
from utils.schemas import Query


class CorpusChunkLoader(Stage):
    """Dataset stage for a background indexing pipeline: emits one batch of
    corpus passages per query.

    Supports disjoint sharding so several indexers do equal-sized but
    independent work: shard i of N takes ``docs[i::N]``.

    YAML config example:
        component: stages.rag_indexing.CorpusChunkLoader
        config:
          dataset:
            name: rag-datasets/rag-mini-wikipedia
            subset: text-corpus
            split: passages
            text_column: passage
            max_docs: 3200
          docs_per_query: 32
          shard: {index: 0, count: 1}
    """

    def __init__(self, stage_config, pipeline_config):
        super().__init__(stage_config, pipeline_config)
        ds = self.extra_config["dataset"]
        self._name = ds["name"]
        self._subset = ds.get("subset")
        self._split = ds.get("split", "train")
        self._text_column = ds.get("text_column", "text")
        self._max_docs = int(ds.get("max_docs", 0)) or None
        self._docs_per_query = int(self.extra_config.get("docs_per_query", 32))
        shard = self.extra_config.get("shard") or {}
        self._shard_index = int(shard.get("index", 0))
        self._shard_count = int(shard.get("count", 1))
        self._docs: list[str] = []

    @log_phase
    def prepare(self):
        if self._subset:
            raw = load_dataset(self._name, self._subset)[self._split]
        else:
            raw = load_dataset(self._name)[self._split]
        docs = [str(r[self._text_column]) for r in raw]
        docs = docs[self._shard_index::self._shard_count]  # disjoint shard
        if self._max_docs:
            docs = docs[:self._max_docs]
        self._docs = docs
        super().prepare()

    def get_batch_size(self):
        return self._docs_per_query

    def get_dataset_splits(self) -> dict[str, int]:
        n = (len(self._docs) + self._docs_per_query - 1) // self._docs_per_query
        return {self._split: max(n, 1)}

    def run(self, query: Query) -> dict[int, Query]:
        i = query.batch * self._docs_per_query
        # Wrap around so a run can exceed one corpus pass; the indexer upserts
        # by deterministic ids, so the index does not grow.
        if i >= len(self._docs):
            i = i % max(len(self._docs), 1)
        query.data = self._docs[i:i + self._docs_per_query]
        query.context["chunk_offset"] = i
        return {idx: query for idx in self.output_queues}
