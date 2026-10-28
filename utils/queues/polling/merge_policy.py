import threading
from typing import Any
from queue import Empty

from utils.queues import PeekableQueue
from utils.schemas import Query
from .polling_policy import PollingPolicy


class MergePolicy(PollingPolicy):
    """
    Merges the data from all input queues into a single query.

    Returns None only once every upstream has sent its terminator. Blocks on
    the shared condition signalled by ``PeekableQueue.put``.
    """

    def __init__(
        self,
        input_queues: dict[int, PeekableQueue],
        notify_condition: threading.Condition | None = None,
    ):
        super().__init__(input_queues, notify_condition)
        self._drained = {key: False for key in self.input_queues}

    def get_input_from_queues(self) -> Query | None:
        """
        Retrieve and merge queries from all input queues into a single Query object.

        Waits until every non-drained upstream has a query available, then
        merges them. Drains None terminators as they arrive without
        exiting until every upstream has terminated.

        Returns:
            Query | None: A Query object with merged data, or None once
            every upstream has terminated.
        """
        with self._cond:
            while True:
                if all(self._drained.values()):
                    return None

                queries: dict[int, Query] = {}
                ready = True
                for key, queue in self.input_queues.items():
                    if self._drained[key]:
                        continue
                    try:
                        query = queue.peek()
                    except Empty:
                        ready = False
                        continue
                    if not query:
                        queue.get()
                        self._drained[key] = True
                        continue
                    queries[key] = query

                if ready and queries:
                    # Pop the peeked queries.
                    for key in queries:
                        self.input_queues[key].get()

                    first_query = next(iter(queries.values()))
                    merged_query = Query(
                        query_id=first_query.query_id,
                        split=first_query.split,
                        batch=first_query.batch,
                        epoch=first_query.epoch,
                        query_submitted_timestamp=first_query.query_submitted_timestamp,
                    )
                    merged_data: dict[int, Any] = {
                        key: q.data for key, q in queries.items()
                    }
                    merged_query.data = merged_data
                    return merged_query

                if all(self._drained.values()):
                    return None

                # The timeout is only a safety net (see FirstSubmittedPolicy).
                self._cond.wait(timeout=1.0)
