import chromadb
from datasets import load_dataset

from stages.stage import Stage, log_phase
from utils.schemas import Query


class ChromaRetriever(Stage):
    """In-process vector search retriever using ChromaDB.

    Builds a ChromaDB collection from a HuggingFace dataset corpus during
    prepare(), then retrieves top_k documents for each incoming query.

    YAML config example:
        config:
          corpus_dataset:
            name: yahma/alpaca-cleaned
            split: train
            text_column: output
            max_docs: 5000
          top_k: 3
          collection_name: self_rag_corpus
    """

    def __init__(self, stage_config, pipeline_config):
        super().__init__(stage_config, pipeline_config)

        corpus_config = self.extra_config["corpus_dataset"]
        self._corpus_dataset_name = corpus_config["name"]
        self._corpus_subset = corpus_config.get("subset", None)
        self._corpus_split = corpus_config.get("split", "train")
        self._corpus_text_column = corpus_config.get("text_column", "output")
        self._corpus_max_docs = corpus_config.get("max_docs", 5000)

        # Optional: build the corpus from a HotpotQA-style nested context
        # field (passages shipped with each question) instead of a flat
        # text column.
        self._corpus_context_column = corpus_config.get("context_column", None)
        self._corpus_max_context_rows = corpus_config.get(
            "max_context_rows", self._corpus_max_docs
        )
        # Rows that contribute their context; must match the question
        # loader's offset/stride.
        self._corpus_context_offset = int(corpus_config.get("context_row_offset", 0))
        self._corpus_context_stride = int(corpus_config.get("context_row_stride", 1))

        self._top_k = self.extra_config.get("top_k", 3)
        self._collection_name = self.extra_config.get(
            "collection_name", "self_rag_corpus"
        )

        # Optional sentence-transformer embedder instead of ChromaDB's default.
        # Embeddings are computed here so asymmetric query/passage prefixes
        # (e5-style models) can be applied, which ChromaDB's built-in
        # embedding function cannot do.
        self._embedding_model_name = self.extra_config.get("embedding_model", None)
        self._query_prefix = self.extra_config.get("query_prefix", "")
        self._passage_prefix = self.extra_config.get("passage_prefix", "")
        self._embedder = None

        self._collection = None
        self._client = None

    def _build_context_corpus(self, raw_dataset) -> list[str]:
        """Build a corpus from a HotpotQA-style nested context field.

        Each row's ``<context_column>["context"]`` holds
        ``{"title": [...], "sentences": [[...], ...]}``; flatten each entry
        into a ``"title\\n<joined sentences>"`` passage and dedup across rows
        so the gold supporting passages sit alongside many distractors.
        """
        rows = range(
            self._corpus_context_offset, len(raw_dataset), self._corpus_context_stride
        )[: self._corpus_max_context_rows]
        seen: set[str] = set()
        documents: list[str] = []
        for i in rows:
            meta = raw_dataset[i][self._corpus_context_column]
            # Two layouts: FlashRAG nests it as metadata["context"]; hotpot_qa
            # (`distractor` config) stores the {"title", "sentences"} dict
            # directly in the `context` column.
            if isinstance(meta, dict) and "title" in meta and "sentences" in meta:
                ctx = meta
            else:
                ctx = meta.get("context") if isinstance(meta, dict) else None
            if not ctx:
                continue
            titles = ctx.get("title", [])
            sentences = ctx.get("sentences", [])
            for title, sents in zip(titles, sentences):
                passage = f"{title}\n{''.join(sents)}".strip()
                if passage and passage not in seen:
                    seen.add(passage)
                    documents.append(passage)
        return documents[: self._corpus_max_docs]

    @log_phase
    def prepare(self):
        """Load corpus from HuggingFace, build ChromaDB collection."""
        super().prepare()

        print(
            f"ChromaRetriever: loading corpus from {self._corpus_dataset_name} "
            f"(split={self._corpus_split}, column={self._corpus_text_column})"
        )

        if self._corpus_subset:
            raw_dataset = load_dataset(
                self._corpus_dataset_name, self._corpus_subset
            )[self._corpus_split]
        else:
            raw_dataset = load_dataset(self._corpus_dataset_name)[self._corpus_split]

        # Flat corpora only; context corpora select rows by offset/stride.
        if not self._corpus_context_column and len(raw_dataset) > self._corpus_max_docs:
            raw_dataset = raw_dataset.select(range(self._corpus_max_docs))

        if self._corpus_context_column:
            documents = self._build_context_corpus(raw_dataset)
        else:
            documents = [
                doc
                for doc in raw_dataset[self._corpus_text_column]
                if doc and len(doc.strip()) > 0
            ]

        print(f"ChromaRetriever: indexing {len(documents)} documents into ChromaDB")

        self._client = chromadb.Client()
        self._collection = self._client.create_collection(
            name=self._collection_name,
        )

        passage_embs = None
        if self._embedding_model_name:
            from sentence_transformers import SentenceTransformer

            print(
                f"ChromaRetriever: embedding corpus with "
                f"{self._embedding_model_name} (prefix={self._passage_prefix!r})"
            )
            self._embedder = SentenceTransformer(self._embedding_model_name)
            passage_embs = self._embedder.encode(
                [self._passage_prefix + d for d in documents],
                batch_size=64,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        # ChromaDB requires unique ids.
        batch_size = 500
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            ids = [f"doc_{j}" for j in range(i, i + len(batch))]
            if passage_embs is not None:
                self._collection.add(
                    documents=batch,
                    embeddings=passage_embs[i : i + len(batch)].tolist(),
                    ids=ids,
                )
            else:
                self._collection.add(documents=batch, ids=ids)

        print(
            f"ChromaRetriever: indexed {self._collection.count()} documents, "
            f"retrieving top_k={self._top_k}"
        )

    def run(self, query: Query) -> dict[int, Query]:
        """Retrieve top_k documents for the incoming query."""

        # On a retry, query.data holds the rewriter's follow-up query. It is
        # used for retrieval only; the user's question stays in the context.
        question = query.data
        if isinstance(question, list):
            question = question[0]
        query.context.pop("is_retry", None)

        if self._embedder is not None:
            q_emb = self._embedder.encode(
                [self._query_prefix + question],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            results = self._collection.query(
                query_embeddings=q_emb.tolist(),
                n_results=self._top_k,
            )
        else:
            results = self._collection.query(
                query_texts=[question],
                n_results=self._top_k,
            )
        new_docs = results["documents"][0] if results["documents"] else []

        # Accumulate documents across hops (deduplicated, in order), so
        # multi-hop answers can draw on every hop.
        accumulated = list(query.context.get("retrieved_documents", []) or [])
        seen = set(accumulated)
        for doc in new_docs:
            if doc not in seen:
                seen.add(doc)
                accumulated.append(doc)

        query.context["retrieved_documents"] = accumulated
        query.data = accumulated

        print(
            f"ChromaRetriever: query='{question[:80]}...' "
            f"retrieved {len(new_docs)} new, {len(accumulated)} accumulated docs"
        )

        output = {idx: query for idx in self.output_queues}
        return output
