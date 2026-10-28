import threading
import uuid
from time import monotonic, sleep, time
from queue import Queue

import mlflow
from utils.trace_span import trace_span

from utils.schemas import Query
from .arrival_log import ArrivalLog
from .scheduler import LoadScheduler


class MultiStreamScheduler(LoadScheduler):
    """Fixed-interval load generator (MLPerf Inference **MultiStream**).

    Issues one query every ``interval`` seconds, open-loop. Samples per query
    are set by the dataset stage's ``batch_size``. Pacing uses an absolute
    schedule (t0 + i*interval) so per-iteration overhead does not accumulate.

    YAML:
        loadgen:
          component: loadgen.MultiStreamScheduler
          config:
            interval: 0.125   # seconds between queries
    """

    def __init__(self, max_queries, timeout, load_scheduler_config, dataset_splits):
        super().__init__(max_queries, timeout, load_scheduler_config, dataset_splits)
        self.interval = float(self.extra_config["interval"])  # no silent default

    def generate(self, queue: Queue, event: threading.Event) -> None:
        self.timer.start()
        event.set()

        counter = 0
        _arrivals = ArrivalLog()
        _t0_wall = None
        _t0_mono = monotonic()

        try:
            while counter < self.max_queries:
                for split_name, split_batches in self.dataset_splits.items():
                    for batch_idx in range(split_batches):
                        if self.stop:
                            break

                        # Absolute-deadline pacing: sleep until t0 + i*interval.
                        deadline = _t0_mono + counter * self.interval
                        delay = deadline - monotonic()
                        if delay > 0:
                            sleep(delay)

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

                        if _t0_wall is None:
                            _t0_wall = _submitted
                        _arrivals.record(counter, _t0_wall + counter * self.interval,
                                         _submitted, _put_s)

                        counter += 1
                        if counter >= self.max_queries:
                            self.stop = True
                            break
                    if self.stop:
                        break
                if self.stop:
                    break
        finally:
            queue.put(None)
            self.timer.cancel()
            _arrivals.write()
