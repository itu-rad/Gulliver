import uuid
from time import time
from queue import Queue
import threading

import mlflow
from utils.trace_span import trace_span

from utils.schemas import Query
from .scheduler import LoadScheduler


class OfflineLoadScheduler(LoadScheduler):
    """Load generation scheduler, which waits for processing of the previous query"""

    def generate(self, queue: Queue, event: threading.Event) -> None:
        """Generate load, which is synchronized with the pipeline execution.
        The generation is stopped when the max_queries is reached or timer elapses.

        Args:
            queue (queue.Queue): Pipeline's input queue.
            event (threading.Event): Conditional variable for synchronizing between the load generation and the pipeline's execution.
        """
        counter = 0
        self.timer.start()

        print("Starting load generation")
        print("Splits: ", self.dataset_splits)

        try:
            while counter < self.max_queries:
                for split_name, split_batches in self.dataset_splits.items():
                    for batch_idx in range(split_batches):
                        if self.stop:
                            break

                        # Wait for the previous query to complete.
                        if counter > 0:
                            event.wait()
                            event.clear()

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
                            queue.put_nowait(
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
