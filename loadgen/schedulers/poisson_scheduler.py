import threading
import uuid
from random import Random
from time import sleep, time

import mlflow
from utils.trace_span import trace_span

from utils.schemas import Query
from .arrival_log import ArrivalLog
from .scheduler import LoadScheduler


class PoissonLoadScheduler(LoadScheduler):
    """
    PoissonLoadScheduler is a load scheduler that generates load based on a Poisson distribution.
    """

    def __init__(self, max_queries, timeout, load_scheduler_config, dataset_splits):
        """
        Initializes the PoissonScheduler with the given parameters.

        Args:
            max_queries (int): The maximum number of queries to handle.
            timeout (int): The timeout value for the scheduler.
            load_scheduler_config (dict): Configuration settings for the load scheduler.
            dataset_splits (list): List of dataset splits to be used.
        """
        super().__init__(max_queries, timeout, load_scheduler_config, dataset_splits)

        self.rate = self.extra_config.get("rate", 3)
        # Seeded so that runs being compared see the same arrival sequence.
        self.seed = self.extra_config.get("seed", 0)
        self._rng = Random(self.seed)
        self.offsets = []

    def prepare(self) -> None:
        """
        Generate time offsets based on the Poisson distribution.

        This method initializes the `offsets` list with a starting value of 0 and
        then appends additional time offsets generated using the exponential
        distribution with the specified rate. The number of offsets generated
        is determined by `self.max_queries`.

        Returns:
            None
        """
        self.offsets = [0]
        for _ in range(self.max_queries - 1):
            self.offsets.append(self._rng.expovariate(self.rate))

    def generate(self, queue, event):
        """
        Generates queries and pushes them onto the provided queue at intervals defined by offsets.
        Args:
            queue (Queue): The queue to push generated queries onto.
            event (Event): An event to signal the start of query generation.
        The method will generate queries until the maximum number of queries (`self.max_queries`) is reached or a stop signal is received.
        It uses the dataset splits defined in `self.dataset_splits` to generate queries for each split and batch.
        The method also handles timeout and stop signals to ensure proper termination of the query generation process.
        """

        self.timer.start()
        event.set()

        counter = 0

        # Coordinated-omission check: a put() slower than this threshold means
        # the entry queue was full and arrivals were throttled.
        _block_threshold_s = 0.005
        _blocked_puts = 0
        _total_block_s = 0.0
        _max_block_s = 0.0

        # Intended arrival of query i is t0 + sum(offsets[:i+1]).
        _arrivals = ArrivalLog()
        _t0 = None
        _intended = 0.0

        try:
            while counter < self.max_queries:
                for split_name, split_batches in self.dataset_splits.items():
                    for batch_idx in range(split_batches):
                        if self.stop:
                            break

                        sleep(self.offsets[counter])

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
                            # Blocking put: back-pressure instead of queue.Full.
                            _t_put = time()
                            _submitted = time()
                            queue.put(
                                Query(
                                    split=split_name,
                                    batch=batch_idx,
                                    query_submitted_timestamp=_submitted,
                                    out_flow_id=flow_id,
                                )
                            )
                            _put_s = time() - _t_put

                        if _put_s > _block_threshold_s:
                            _blocked_puts += 1
                            _total_block_s += _put_s
                            _max_block_s = max(_max_block_s, _put_s)

                        if _t0 is None:
                            _t0 = _submitted
                        _intended += self.offsets[counter]
                        _arrivals.record(counter, _t0 + _intended, _submitted, _put_s)

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
            _arrivals.write()
            if _blocked_puts:
                print(
                    f"[loadgen] WARNING coordinated omission: {_blocked_puts}/{counter} "
                    f"submits blocked on a full entry queue "
                    f"(total {_total_block_s:.3f}s, max {_max_block_s * 1000:.1f}ms). "
                    f"The realized arrival process was throttled below the intended "
                    f"Poisson rate={self.rate}; raise queue_depth or lower the rate for a "
                    f"valid open-loop latency measurement."
                )
