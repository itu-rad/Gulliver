import faiss
import numpy as np
from datasets import load_dataset

from stages.stage import Stage, log_phase
from utils.schemas import Query


class FAISSImageRetriever(Stage):
    """FAISS-based retriever that indexes text embeddings and retrieves
    documents using image embeddings from CLIP.

    YAML config example:
        config:
          clip_stage_id: 1
          corpus:
            name: wikimedia/wikipedia
            subset: 20220301.en
            split: train
            text_column: text
            max_docs: 1000
          top_k: 3
    """

    def __init__(self, stage_config, pipeline_config):
        """Initialize the FAISS retriever stage.

        Args:
            stage_config: Stage configuration from YAML.
            pipeline_config: Pipeline configuration from YAML.
        """
        super().__init__(stage_config, pipeline_config)

        self._clip_stage_id = self.extra_config["clip_stage_id"]

        corpus_config = self.extra_config["corpus"]
        self._corpus_name = corpus_config["name"]
        self._corpus_subset = corpus_config.get("subset", None)
        self._corpus_split = corpus_config.get("split", "train")
        self._corpus_text_column = corpus_config.get(
            "text_column", "text"
        )
        self._corpus_max_docs = corpus_config.get("max_docs", 1000)

        self._top_k = self.extra_config.get("top_k", 3)

        self._index = None
        self._corpus_texts = None

    @log_phase
    def prepare(self):
        """Load corpus, encode with CLIP, and build FAISS index."""
        super().prepare()

        print(
            f"FAISSImageRetriever: loading corpus from "
            f"{self._corpus_name}"
        )

        if self._corpus_subset:
            raw = load_dataset(
                self._corpus_name, self._corpus_subset
            )[self._corpus_split]
        else:
            raw = load_dataset(self._corpus_name)[self._corpus_split]

        if len(raw) > self._corpus_max_docs:
            raw = raw.select(range(self._corpus_max_docs))

        # Extract texts. Some caption datasets (COCO, Flickr) store a list
        # of captions per row; flatten those into individual documents.
        raw_column = raw[self._corpus_text_column]
        flat_texts = []
        for entry in raw_column:
            if entry is None:
                continue
            if isinstance(entry, (list, tuple)):
                for sub in entry:
                    if isinstance(sub, str) and sub.strip():
                        flat_texts.append(sub)
            elif isinstance(entry, str) and entry.strip():
                flat_texts.append(entry)
            if len(flat_texts) >= self._corpus_max_docs:
                break

        self._corpus_texts = flat_texts[: self._corpus_max_docs]

        # Truncate long docs to avoid tokenizer overflow.
        self._corpus_texts = [doc[:512] for doc in self._corpus_texts]

        print(
            f"FAISSImageRetriever: encoding {len(self._corpus_texts)} "
            f"documents via CLIP stage {self._clip_stage_id}"
        )

        text_embeddings = self.dispatch_call(
            self._clip_stage_id, "encode_texts", self._corpus_texts
        )
        text_embeddings = text_embeddings.astype(np.float32)

        # L2-normalised vectors make inner product equal to cosine similarity.
        faiss.normalize_L2(text_embeddings)

        embed_dim = text_embeddings.shape[1]
        self._index = faiss.IndexFlatIP(embed_dim)
        self._index.add(text_embeddings)

        print(
            f"FAISSImageRetriever: indexed {self._index.ntotal} "
            f"documents, top_k={self._top_k}"
        )

    def run(self, query: Query) -> dict[int, Query]:
        """Retrieve top-k documents using image embedding similarity.

        Args:
            query: Query with data=image_embedding (numpy, shape [1, D]).

        Returns:
            dict[int, Query]: Query with data=[doc_str_1, ...].
        """
        embedding = query.data.astype(np.float32)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        faiss.normalize_L2(embedding)
        _, indices = self._index.search(embedding, self._top_k)

        retrieved = [
            self._corpus_texts[i]
            for i in indices[0]
            if i < len(self._corpus_texts)
        ]

        query.context["retrieved_documents"] = retrieved
        query.data = retrieved

        print(
            f"FAISSImageRetriever: retrieved {len(retrieved)} docs "
            f"for query {query.query_id}"
        )

        output = {idx: query for idx in self.output_queues}
        return output
