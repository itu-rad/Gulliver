from stages.stage import Stage, log_phase
from stages.self_rag.prompt_style import SHORT_ANSWER_STYLE
from stages.self_rag.retrieval_grader_formatter import format_documents
from utils.chat import apply_chat_template_safe
from utils.schemas.query import Query


class AnswerGeneratorFormatter(Stage):
    def __init__(self, stage_config, pipeline_config):
        """Initialize the stage by parsing the stage configuration.

        Args:
            stage_config (dict): Stage configuration, such as batch size and number of workers.
        """
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
        """Poll for incoming data in the queues,
        format the incoming data into a new prompt and pass it onto the output queues.
        """

        orig_query = query.context["original_query"]
        retrieved_documents = query.context["retrieved_documents"]

        chat = [
            {
                "role": "system",
                "content": (
                    "You are an answer generator. Given the retrieved documents "
                    "and the original query, output an answer to the query "
                    "supported by the retrieved documents.\n"
                    + SHORT_ANSWER_STYLE
                ),
            },
            {
                "role": "user",
                "content": f"User question: {orig_query}\n\nRetrieved documents:\n{format_documents(retrieved_documents)}",
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
