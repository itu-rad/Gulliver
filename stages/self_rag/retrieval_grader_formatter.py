from stages.stage import Stage, log_phase
from utils.chat import apply_chat_template_safe
from utils.schemas.query import Query


def format_documents(docs) -> str:
    """Render retrieved documents as numbered paragraphs; shared by every
    prompt that shows documents."""
    if isinstance(docs, list):
        return "\n\n".join(f"Document {i + 1}: {d}" for i, d in enumerate(docs))
    return str(docs)


class RetrievalGraderFormatter(Stage):
    """Single-pass, per-document relevance grading.

    One LLM call returns a relevance verdict for each retrieved document, so
    the router can drop irrelevant documents before generation.

    Output contract (parsed by RelevanceFilterRouter): a JSON object mapping
    document number to "yes"/"no", e.g. {"1": "yes", "2": "no", ...}.

    YAML config example:
        config:
          tokenizer_stage_id: 3
          enable_thinking: true
    """

    def __init__(self, stage_config, pipeline_config):
        super().__init__(stage_config, pipeline_config)

        self._tokenizer_stage_id = self.extra_config["tokenizer_stage_id"]
        self._enable_thinking = bool(self.extra_config.get("enable_thinking", True))

        self._tokenizer = None

    @log_phase
    def prepare(self):
        """Load the tokenizer from another stage"""
        super().prepare()
        self._tokenizer = self.dispatch_call(self._tokenizer_stage_id, "get_tokenizer")

    def run(self, query: Query) -> dict[int, Query]:
        retrieved_documents = query.data
        if not isinstance(retrieved_documents, list):
            retrieved_documents = [retrieved_documents]
        query.context["retrieved_documents"] = retrieved_documents
        orig_query = query.context["original_query"]
        n = len(retrieved_documents)

        example = ", ".join(f'"{i + 1}": "yes"' for i in range(min(n, 2)))
        chat = [
            {
                "role": "system",
                "content": (
                    "You are a grader assessing the relevance of retrieved documents "
                    "to a user question. Grade EACH document separately. A document is "
                    "relevant if it contains keyword(s) or semantic meaning related to "
                    "the user question; it does not need to be a stringent test, the "
                    "goal is to filter out erroneous retrievals. Output ONLY a JSON "
                    "object mapping each document number to \"yes\" or \"no\", for "
                    f"all {n} documents, e.g. {{{example}, ...}}. No other text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User question: {orig_query}\n\n"
                    f"Retrieved documents:\n{format_documents(retrieved_documents)}"
                ),
            },
        ]

        query.data = apply_chat_template_safe(
            self._tokenizer,
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self._enable_thinking,
        )

        output = {idx: query for idx in self.output_queues}
        return output
