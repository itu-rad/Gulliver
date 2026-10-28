from dataclasses import dataclass, field
from typing import Any, Optional
import uuid


@dataclass
class Query:
    """
    Structure of a query, which is passed to the pipeline and flows between the stages.
    """

    split: str
    batch: int
    query_submitted_timestamp: float
    epoch: int = 0
    # Factory so each Query gets a fresh id.
    query_id: uuid.UUID = field(default_factory=uuid.uuid4)
    data: Any = None
    # Always a dict: stages attach context to queries created by schedulers.
    context: dict = field(default_factory=dict)
    # Flow id linking consecutive tracing spans of this query: a uuid.UUID from
    # the load scheduler, a str (utils.trace_span.new_flow_id) afterwards.
    out_flow_id: Optional[uuid.UUID | str] = None
