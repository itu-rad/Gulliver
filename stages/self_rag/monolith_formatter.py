from stages.stage import Stage, log_phase
from stages.self_rag.prompt_style import SHORT_ANSWER_STYLE
from utils.chat import apply_chat_template_safe
from utils.schemas.query import Query


class MonolithFormatter(Stage):
    """Formats an all-in-one Self-RAG prompt for the monolithic topology.

    Creates a single prompt that instructs the LLM to:
    1. Grade retrieval relevance
    2. Generate an answer (if relevant)
    3. Self-check for hallucinations

    The LLM is instructed to output structured JSON.

    YAML config example:
        config:
          tokenizer_stage_id: 3
    """

    def __init__(self, stage_config, pipeline_config):
        super().__init__(stage_config, pipeline_config)

        self._tokenizer_stage_id = self.extra_config["tokenizer_stage_id"]
        self._enable_thinking = bool(self.extra_config.get("enable_thinking", True))
        # always_answer: always produce an answer, as the decomposed generator
        # does; False lets the model return an empty answer when not relevant.
        self._always_answer = bool(self.extra_config.get("always_answer", True))
        self._tokenizer = None

    @log_phase
    def prepare(self):
        """Load the tokenizer from the LLM inference stage."""
        super().prepare()
        self._tokenizer = self.dispatch_call(self._tokenizer_stage_id, "get_tokenizer")

    def run(self, query: Query) -> dict[int, Query]:
        """Format the all-in-one RAG prompt with retrieved documents."""

        retrieved_documents = query.context.get("retrieved_documents", [])
        orig_query = query.context["original_query"]

        if isinstance(retrieved_documents, list):
            docs_text = "\n\n".join(
                f"Document {i+1}: {doc}" for i, doc in enumerate(retrieved_documents)
            )
        else:
            docs_text = str(retrieved_documents)

        chat = [
            {
                "role": "system",
                "content": (
                    "You are an expert Self-RAG assistant. Given a user question "
                    "and retrieved documents, you must perform ALL of the "
                    "following steps in a single response:\n\n"
                    "1. **Relevance Grading:** Assess whether the retrieved "
                    "documents are relevant to the user question.\n"
                    "2. **Answer Generation:** "
                    + ("Generate your best answer to the question, grounded in "
                       "the retrieved documents where they support it. "
                       if self._always_answer else
                       "If relevant, generate an answer grounded in the "
                       "retrieved documents. If not relevant, set the answer "
                       "to an empty string. ")
                    + SHORT_ANSWER_STYLE + "\n"
                    "3. **Hallucination Check:** Verify whether your generated "
                    "answer is fully supported by the retrieved documents.\n\n"
                    "Output your response as a JSON object with exactly these "
                    "keys:\n"
                    '  {"relevance_grade": "yes" or "no", '
                    '"answer": "your answer text", '
                    '"hallucination_check": "yes" or "no"}\n\n'
                    '"hallucination_check" should be "yes" if the answer '
                    'contains unsupported claims, "no" if it is grounded.\n\n'
                    "Output ONLY the JSON object, nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User question: {orig_query}\n\n"
                    f"Retrieved documents:\n{docs_text}"
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
