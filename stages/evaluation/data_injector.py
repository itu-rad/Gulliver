from stages.stage import Stage
from utils.schemas import Query

class DataInjector(Stage):
    """
    A stage that injects a fixed-size payload into the query data.
    Useful for measuring the overhead of passing large objects through the pipeline.
    """
    def __init__(self, stage_config, pipeline_config):
        super().__init__(stage_config, pipeline_config)
        self.payload_size = self.extra_config.get("payload_size", 0)
        # Allocated once so that run() does not measure allocation time.
        self.payload = bytearray(self.payload_size)

    def run(self, query: Query) -> dict[int, Query]:
        if query.data is None:
            query.data = self.payload
        
        return {idx: query for idx in self.output_queues}
