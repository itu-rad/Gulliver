from stages.stage import Stage, log_phase
from utils.chat import apply_chat_template_safe
from utils.schemas.query import Query


class QuestionRewriterFormatter(Stage):
    def __init__(self, stage_config, pipeline_config):
        """Initialize the stage by parsing the stage configuration.

        Args:
            stage_config (dict): Stage configuration, such as batch size and number of workers.
        """
        super().__init__(stage_config, pipeline_config)

        self._tokenizer_stage_id = self.extra_config["tokenizer_stage_id"]
        # Characters of each gathered document shown to the rewriter.
        self._doc_chars = int(self.extra_config.get("doc_chars", 1500))
        self._enable_thinking = bool(self.extra_config.get("enable_thinking", True))

        self._tokenizer = None

    @log_phase
    def prepare(self):
        """Load the tokenizer from another stage"""
        super().prepare()
        self._tokenizer = self.dispatch_call(self._tokenizer_stage_id, "get_tokenizer")

    def run(self, query: Query) -> dict[int, Query]:
        """Poll for incoming data in the queues,
        format the incoming data into a new prompt and pass it onto the output queues.
        """

        print("QuestionRewriterFormatter: ", query)
        # Prefer the user's question; original_query may be changed upstream.
        orig_query = query.context.get("question") or query.context["original_query"]

        # Condition on the evidence so far, so multi-hop questions get a
        # follow-up query instead of a paraphrase.
        docs = query.context.get("retrieved_documents", []) or []
        if isinstance(docs, list):
            docs_text = (
                "\n\n".join(
                    f"Document {i+1}: {str(d)[:self._doc_chars]}"
                    for i, d in enumerate(docs)
                )
                or "(no documents retrieved yet)"
            )
        else:
            docs_text = str(docs)[: self._doc_chars * 5]

        # A retry from the hallucination grader carries the rejected answer;
        # including it makes the retry prompt differ from the previous one.
        rejected = query.context.get("rejected_answer")
        if isinstance(rejected, list):
            rejected = rejected[0] if rejected else None
        rejected_text = (
            f"\n\nA previous answer, \"{rejected}\", was judged NOT supported by "
            "these documents; find evidence that settles the question instead."
            if rejected else ""
        )

        chat = [
            {
                "role": "system",
                "content": (
                    "You help answer a possibly multi-hop question by issuing better "
                    "retrieval queries. Given the user's question and the documents "
                    "retrieved so far, identify what information is still MISSING to "
                    "fully answer the question, and write ONE focused search query to "
                    "retrieve that missing information. For a multi-hop question this is "
                    "usually the next entity or fact to look up (a bridge query). If the "
                    "documents look irrelevant, rephrase the question for better retrieval "
                    "instead. Output ONLY the search query on a single line — no quotation "
                    "marks, preamble, explanation, or lists."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {orig_query}\n\n"
                    f"Documents retrieved so far:\n{docs_text}"
                    f"{rejected_text}\n\n"
                    f"Next search query:"
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

        query.context["is_retry"] = True

        print("Final query", query)

        output = {idx: query for idx in self.output_queues}
        return output
