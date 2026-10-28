from stages.stage import Stage, log_phase
from utils.schemas import Query


class MockDataLoader(Stage):
    def __init__(self, stage_config, pipeline_config):
        """Initialize the stage by parsing the stage configuration.

        Args:
            stage_config (dict): Stage configuration, such as batch size and number of workers.
        """
        super().__init__(stage_config, pipeline_config)

        self._batch_size = self.extra_config.get("batch_size", 1)
        self._data = [
            "<system>\nYou are a grader assessing whether an answer is grounded in / supported by a set of facts. Give a binary score true or false score to indicate whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.\n<user>\n------\nHere are the documents: [messi is a football player]\n------\nHere is the answer:  In an LLM-powered autonomous agent system, the Large Language Model (LLM) functions as the agent's brain. The agent has key components including memory, planning, and reflection mechanisms. The memory component is a long-term memory module that records a comprehensive list of agents experience in natural language. It includes a memory stream, which is an external database for storing past experiences. The reflection mechanism synthesizes memories into higher-level inferences over time and guides the agent's future behavior.\n<assistant>\n",
            "<system>\nYou are a grader assessing whether an answer is grounded in / supported by a set of facts. Give a binary score true or false score to indicate whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.\n<user>\n------\nHere are the documents: [this is a document about multi-agent llm systems]\n------\nHere is the answer:  In an LLM-powered autonomous agent system, the Large Language Model (LLM) functions as the agent's brain. The agent has key components including memory, planning, and reflection mechanisms. The memory component is a long-term memory module that records a comprehensive list of agents experience in natural language. It includes a memory stream, which is an external database for storing past experiences. The reflection mechanism synthesizes memories into higher-level inferences over time and guides the agent's future behavior.\n<assistant>\n",
            "<system>\nYou are a grader assessing whether an answer is grounded in / supported by a set of facts. Give a binary score true or false score to indicate whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.\n<user>\n------\nHere are the documents: [this is not a document about multi-agent llm systems]\n------\nHere is the answer:  In an LLM-powered autonomous agent system, the Large Language Model (LLM) functions as the agent's brain. The agent has key components including memory, planning, and reflection mechanisms. The memory component is a long-term memory module that records a comprehensive list of agents experience in natural language. It includes a memory stream, which is an external database for storing past experiences. The reflection mechanism synthesizes memories into higher-level inferences over time and guides the agent's future behavior.\n<assistant>\n",
        ]

    def get_batch_size(self):
        return self._batch_size

    def get_dataset_splits(self) -> dict[str, int]:
        """Get the number of batches for each dataset split.

        Returns:
            dict[str, int]: Dictionary with number of batches for each dataset split
        """
        return {split: len(self._data) // self._batch_size for split in ["train"]}

    @log_phase
    def prepare(self):
        """Build the dataloaders."""
        super().prepare()

    def run(self, query: Query) -> dict[int, Query]:
        """Poll for incoming data in the queues,
        load the next batch of data and pass it onto the output queues."""

        next_batch = self._data[: self._batch_size]
        query.data = next_batch
        output = {idx: query for idx in self.output_queues}
        return output
