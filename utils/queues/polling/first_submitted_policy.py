import threading
from utils.queues import PeekableQueue
from queue import Empty

from utils.schemas import Query
from .polling_policy import PollingPolicy


class FirstSubmittedPolicy(PollingPolicy):
    """
    Polls the first query submitted to the pipeline.

    Returns None only once every upstream has sent its terminator, so a None
    on one input never drops queries queued on another (e.g. in cycles).
    Blocks on the shared condition signalled by ``PeekableQueue.put``.
    """

    def __init__(
        self,
        input_queues: dict[int, PeekableQueue],
        notify_condition: threading.Condition | None = None,
    ):
        super().__init__(input_queues, notify_condition)
        # Track which upstreams have sent their None.
        self._drained = {key: False for key in self.input_queues}

    def get_input_from_queues(self) -> Query | None:
        """
        Peeks at all of the queues and returns the first query submitted to the pipeline.
        Drains None markers from terminating upstreams without exiting until
        every upstream has terminated.

        Returns:
            Query | None: The first query submitted to the pipeline or None
            once every upstream has sent its terminator.
        """

        with self._cond:
            while True:
                if all(self._drained.values()):
                    return None

                queries: dict[int, Query] = {}
                for key, queue in self.input_queues.items():
                    if self._drained[key]:
                        continue
                    try:
                        query = queue.peek()
                    except Empty:
                        continue
                    if not query:
                        # Consume the None terminator and mark this upstream
                        # as drained, but keep polling the others.
                        queue.get()
                        self._drained[key] = True
                        continue
                    queries[key] = query

                if queries:
                    first_submitted = sorted(
                        queries.items(), key=lambda x: x[1].query_submitted_timestamp
                    )[0]

                    # Sole consumer and already peeked, so get() does not block.
                    return self.input_queues[first_submitted[0]].get()

                # A terminator consumed above may have drained the last upstream.
                if all(self._drained.values()):
                    return None

                # Readiness is re-checked while holding self._cond, so a notify
                # from put() cannot be lost; the timeout is only a safety net.
                self._cond.wait(timeout=1.0)
