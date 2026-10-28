from pydantic import BaseModel
from .loadgen import LoadGenModel
from .stage import StageModel


class PipelineModel(BaseModel):
    """
    Pipeline configuration parsed from yaml file.
    """

    inputs: list[int]
    outputs: list[int]
    dataset_stage_id: int
    loadgen: LoadGenModel
    stages: list[StageModel]
    name: str = "Unknown pipeline name"
    # Suppress the pipeline's per-query CSV rows (`pipeline - <split>, run,
    # start|end`); the pipeline-level counterpart of StageModel.disable_logs.
    # The per-run `prepare` rows are always written.
    disable_logs: bool = False
    # When True, the pipeline waits for the previous query to fully exit
    # before admitting the next one. Useful for isolating per-query
    # latency.
    serialize_queries: bool = False
    # Environment variables set in this pipeline's process before any stage is built
    # (and so before CUDA initialises), e.g. CUDA_MPS_ACTIVE_THREAD_PERCENTAGE to cap
    # an MPS client's share of the SMs. Only the pipeline's own process sees them.
    env: dict[str, str] = {}
    # Name of another pipeline in the config: stop generating load when it finishes.
    # --max-queries does not apply to such a pipeline.
    stop_with: str | None = None
