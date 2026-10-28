from stages.stage import Stage
from utils.schemas import Query
import copy

class CopyStage(Stage):
    """
    A stage that forces a deep copy of the query data.
    Useful for simulating the overhead of serialization or strict immutability.
    """
    def __init__(self, stage_config, pipeline_config):
        super().__init__(stage_config, pipeline_config)

    def run(self, query: Query) -> dict[int, Query]:
        if query.data is not None:
            query.data = copy.deepcopy(query.data)
        
        return {idx: query for idx in self.output_queues}
