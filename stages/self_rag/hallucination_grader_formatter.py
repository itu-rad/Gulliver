from stages.stage import Stage, log_phase
from stages.self_rag.retrieval_grader_formatter import format_documents
from utils.chat import apply_chat_template_safe
from utils.schemas.query import Query


class HallucinationGraderFormatter(Stage):
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

        generated_answer = query.data
        if isinstance(generated_answer, list):
            generated_answer = generated_answer[0] if generated_answer else ""
        query.context["generated_answer"] = generated_answer
        retrieved_documents = query.context["retrieved_documents"]

        chat = [
            {
                "role": "system",
                "content": """
                You are a grader assessing whether an LLM generation contains hallucinations (claims not supported by a set of retrieved facts). \n
                Give a binary score 'yes' or 'no'. 'Yes' means the answer contains unsupported claims (i.e. is hallucinated). 'No' means the answer is fully grounded in the retrieved facts. Output nothing but the 'yes' or 'no' score. \n
                """,
            },
            {
                "role": "user",
                "content": f"Set of facts:\n{format_documents(retrieved_documents)}\n\nLLM generation: {generated_answer}",
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
