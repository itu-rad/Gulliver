import threading
import uuid
from time import time
from queue import Queue

import mlflow
from utils.trace_span import trace_span

from utils.schemas import Query
from .scheduler import LoadScheduler


class SaturatingOfflineScheduler(LoadScheduler):
    """Open-loop *saturating* load generator.

    Enqueues every query up front with no inter-arrival pacing and without
    waiting for the pipeline to finish the previous one. This reproduces MLPerf
    Inference's **Offline** scenario, whose reported metric is throughput
    (completed queries / makespan), not per-query latency.

    Note: ``OfflineLoadScheduler`` is a closed-loop, one-in-flight generator,
    which corresponds to MLPerf **SingleStream**, not Offline.
    """

    def generate(self, queue: Queue, event: threading.Event) -> None:
        """Enqueue every query as fast as the entry queue accepts it.

        Args:
            queue (queue.Queue): the pipeline's input queue.
            event (threading.Event): pipeline completion signal. Unused here —
                it is ``set()`` once so the pipeline thread starts draining.
        """
        self.timer.start()
        event.set()

        counter = 0
        try:
            while counter < self.max_queries:
                for split_name, split_batches in self.dataset_splits.items():
                    for batch_idx in range(split_batches):
                        if self.stop:
                            break

                        flow_id = uuid.uuid4()
                        with trace_span(
                            name="generate query",
                            attributes={
                                "out_flow_id": str(flow_id),
                                "thread_id": threading.get_ident(),
                                "epoch": counter,
                                "batch": batch_idx,
                                "split": split_name,
                            },
                        ):
                            # Blocking put: back-pressures if queue_depth is
                            # smaller than the total number of queries.
                            queue.put(
                                Query(
                                    split=split_name,
                                    batch=batch_idx,
                                    query_submitted_timestamp=time(),
                                    out_flow_id=flow_id,
                                )
                            )

                        counter += 1
                        if counter >= self.max_queries:
                            self.stop = True
                            break
                    if self.stop:
                        break
                if self.stop:
                    break
        finally:
            # Always send the terminator and cancel the timer, even on exception.
            queue.put(None)
            self.timer.cancel()
