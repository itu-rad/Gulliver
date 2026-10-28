import chromadb

from stages.stage import Stage, log_phase
from utils.schemas import Query


class ChromaIndexer(Stage):
    """Insert-only ChromaDB sink for a background indexing pipeline.

    Adds pre-computed embeddings (from EmbedStage) into an in-process
    collection that no other pipeline reads.

    Ids are deterministic (``<collection>-<chunk_offset>-<i>``), so a loader
    that wraps around the corpus upserts instead of growing the collection.

    YAML config example:
        component: stages.rag_indexing.ChromaIndexer
        config:
          collection_name: bg_index_shard0
    """

    def __init__(self, stage_config, pipeline_config):
        super().__init__(stage_config, pipeline_config)
        self._collection_name = self.extra_config.get(
            "collection_name", "bg_index")
        self._client = None
        self._collection = None

    @log_phase
    def prepare(self):
        self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            self._collection_name)
        super().prepare()

    def run(self, query: Query) -> dict[int, Query]:
        texts, embeddings = query.data
        if len(texts):
            offset = query.context.get("chunk_offset", 0)
            ids = [f"{self._collection_name}-{offset}-{i}"
                   for i in range(len(texts))]
            self._collection.upsert(
                ids=ids,
                documents=list(texts),
                embeddings=embeddings.tolist(),
            )
        query.data = None
        return {idx: query for idx in self.output_queues}
