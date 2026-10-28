from pydantic import BaseModel
from typing import Literal
from .pipeline import PipelineModel

from radt.run.listeners import listeners as radt_listeners


class BenchmarkModel(BaseModel):
    """
    Benchmark configuration parsed from yaml file.
    """

    pipelines: list[PipelineModel]
    # radT listeners to run; macmon covers Apple silicon, override on other hosts.
    listeners: list[Literal[tuple([x.lower() for x in radt_listeners.keys()])]] = ["macmon"]
    # How the pipelines share the GPU (each runs in its own process):
    #   ""      time-sliced (default)
    #   "mps"   radT starts the CUDA MPS control daemon for the group
    #   "<n>g.<m>gb"  a MIG profile string, e.g. "1g.10gb"
    collocation: str = ""
    name: str = "Unknown benchmark name"
