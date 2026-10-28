import logging
import os
import threading
from queue import Queue, Full, Empty
from threading import Condition, Event, Thread
from time import monotonic

import mlflow
from utils.trace_span import trace_span, SPANS_ENABLED, new_flow_id

from stages import Stage
from utils.component import get_stage_component
from utils.queues import PeekableQueue
from utils.schemas import PipelineModel, StageModel, Query


class Pipeline:
    """The pipeline holds the different stages and orchestrates the communication between them,
    as well as, the logging of execution times of the separate stages.
    """

    def __init__(self, pipeline_config: PipelineModel):
        """Initialize the pipeline by parsing the pipeline configuration.

        Args:
            pipeline_config (PipelineModel): The configuration of the pipeline.
        """
        self._logger = logging.getLogger("benchmark")
        self._pipeline_config = pipeline_config
        self.name = self._pipeline_config.name
        # Pipeline-level twin of the Stage flag; see PipelineModel.disable_logs.
        self._disable_logs = bool(getattr(pipeline_config, "disable_logs", False))
        stage_config = pipeline_config.stages
        stage_config_dict = {stage.id: stage for stage in stage_config}

        self._input_queues: list[Queue] = []
        self._output_queues: list[PeekableQueue] = []
        # One terminator flag per output queue -- see _next_result.
        self._outputs_drained: list[bool] = []

        # Signalled by PeekableQueue.put on every output queue; the result
        # collector waits on it instead of polling.
        self._output_cond = Condition()

        # Guards queries_processed and wakes _wait_for_drain. Separate from
        # _output_cond so the collector never holds the producers' condition.
        self._drain_cond = Condition()
        self.queries_processed = 0

        self.stages = self._create_stages(stage_config_dict)

        self._populate_stages_with_stage_dict()
        self._populate_queues()

    def __str__(self) -> str:
        s = "subgraph Stages\n"
        for stage in self.stages.values():
            s += str(stage)
        s += "end\n"
        for input_stage_id in self._pipeline_config.inputs:
            s += f"load_sched -->|queue depth:{self._pipeline_config.loadgen.queue_depth}|{input_stage_id}\n"
        for output_stage_id in self._pipeline_config.outputs:
            s += f"{output_stage_id} --> load_sched\n"
        return s

    def _create_stages(
        self, stage_config_dict: dict[int, StageModel]
    ) -> dict[int, Stage]:
        """Create a dictionary of stage objects from the given configuration.

        Args:
            stage_config_dict (dict[int, StageModel]): A dictionary where the keys are stage IDs and the values are the configurations of the stages.

        Returns:
            dict: A dictionary where the keys are stage IDs and the values are the corresponding stage objects.
        """
        stages = {}
        for stage_idx, stage_config in stage_config_dict.items():
            stage_obj = get_stage_component(stage_config, self._pipeline_config)
            stages[stage_idx] = stage_obj
        return stages

    def _populate_stages_with_stage_dict(self) -> None:
        """
        Populate all of the stages in the pipeline with the stage dictionary.
        """
        for stage in self.stages.values():
            stage.set_stage_dict(self.stages)

    def _populate_queues(self) -> None:
        """
        Populate all of the stages in the pipeline with their output queues.
        """

        for stage in self.stages.values():
            stage.set_output_queues()

    def get_dataset_splits(self) -> dict[str, int]:
        """
        Get the number of batches in each dataset split of the pipeline.

        Returns:
            dict[str, int]: A dictionary where the keys are the dataset split names and the values are the number of batches in each split.
        """
        stage_id = self._pipeline_config.dataset_stage_id
        return self.stages[stage_id].get_dataset_splits()

    def prepare(self) -> None:
        """
        Prepare all the stages in the pipeline.

        This includes creating input queues, output queues, as well as calling the prepare method of each stage.
        """
        self._logger.info("%s, pipeline, prepare, start", self.name)

        # get all the input queues of the pipeline
        for input_stage_idx in self._pipeline_config.inputs:
            self._input_queues.append(self.stages[input_stage_idx].get_input_queue(-1))

        # create output queue for each output stage of the pipeline and pass it to the output stage
        for output_stage_idx in self._pipeline_config.outputs:
            # PeekableQueue notifies _output_cond after each put. Unbounded:
            # an output stage must never block pushing a result.
            queue = PeekableQueue(notify_condition=self._output_cond)
            self._output_queues.append(queue)
            self.stages[output_stage_idx].set_output_queue(queue)
        self._outputs_drained = [False] * len(self._output_queues)

        # start the stage threads
        for stage in self.stages.values():
            stage.prepare()

        self._logger.info("%s, pipeline, prepare, end", self.name)

    def _broadcast_terminator(self) -> None:
        """Put the None terminator on every input queue of every stage.

        Propagating it from the pipeline inputs alone does not terminate cyclic
        graphs. Must only be called after the pipeline has drained.
        """
        for stage in self.stages.values():
            for queue_ in stage._input_queues.values():
                try:
                    queue_.put_nowait(None)
                except Full:
                    # The pipeline did not drain; do not block shutdown.
                    self._logger.warning(
                        "pipeline %s: input queue of stage %r was full at "
                        "shutdown; its terminator was not delivered",
                        self.name, stage.name)

    def join_threads(self, timeout: float | None = None) -> None:
        """
        Wait for all the stage threads to finish, with optional per-stage timeout.

        The caller is expected to force-exit the process afterwards so any
        straggler threads are reclaimed.
        """
        for stage in self.stages.values():
            stage._thread.join(timeout=timeout)

    def _next_result(self) -> int | None:
        """Block until some output queue has a result; return its index.

        Returns None once every output queue has delivered its terminator.
        The condition is held across the scan, so a concurrent put cannot be
        missed; the wait timeout is only a safety net.
        """
        with self._output_cond:
            while True:
                for idx, output_queue in enumerate(self._output_queues):
                    if self._outputs_drained[idx]:
                        continue
                    try:
                        head = output_queue.peek()
                    except Empty:
                        continue
                    if head is None:
                        output_queue.get()
                        self._outputs_drained[idx] = True
                        continue
                    return idx

                # Exit only once every output has terminated, so results still
                # queued on another output are not discarded.
                if all(self._outputs_drained):
                    return None

                self._output_cond.wait(timeout=1.0)

    def retrieve_results(self, event: Event) -> None:
        """Collect finished queries from the pipeline output queues.

        Args:
            event (Event): signalled once per completed query, for the
                schedulers that pace themselves against completions.
        """
        while True:
            idx = self._next_result()
            if idx is None:
                return                      # every output stage terminated

            # Sole consumer and the queue was peeked non-empty, so this cannot
            # raise. Done outside _output_cond so it cannot stall a producer.
            with trace_span(
                name="pipeline retrieve_results",
                attributes={"thread_id": threading.get_ident()},
            ):
                new_query = self._output_queues[idx].get_nowait()

            if not self._disable_logs:
                self._logger.info(
                    "%s, pipeline - %s, run, end, %d, %.6f, %d, %d",
                    self.name,
                    new_query.split,
                    new_query.query_id,
                    new_query.query_submitted_timestamp,
                    new_query.epoch,
                    new_query.batch + 1,
                )

            with trace_span(
                name="pipeline query processed",
                attributes=({
                    "in_flow_id": new_query.out_flow_id,
                    "thread_id": threading.get_ident(),
                    "pipeline": self.name,
                    "epoch": new_query.epoch,
                    "split": new_query.split,
                    "batch": new_query.batch,
                    "query_id": new_query.query_id,
                } if SPANS_ENABLED else None),
            ):
                # Increment under the lock the waiter tests it under.
                with self._drain_cond:
                    self.queries_processed += 1
                    self._drain_cond.notify_all()
            # The schedulers' wakeup; not signalled while holding _drain_cond.
            event.set()

    def _wait_for_drain(self, target_processed: int, timeout: float) -> bool:
        """Block until queries_processed >= target_processed, or timeout.

        Waits on _drain_cond rather than the loadgen Event, which the
        scheduler also waits on and clears.

        Returns:
            bool: False if the timeout expired first.
        """
        deadline = monotonic() + timeout
        with self._drain_cond:
            while self.queries_processed < target_processed:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    self._logger.warning(
                        "%s, pipeline, drain, timeout, sent=%d, processed=%d",
                        self.name, target_processed, self.queries_processed,
                    )
                    return False
                self._drain_cond.wait(remaining)
        return True

    def run(self, query_queue: Queue, event: Event) -> None:
        """
        Executes the pipeline by reading queries from the query queue and processing them until a termination signal is received.

        This method starts a separate thread for result retrieval and processes incoming queries from the sample queue.
        It continues to process queries until a `None` value is retrieved from the queue, indicating termination.
        Upon termination, it sends a termination signal to subsequent stages and joins all threads.

        If `serialize_queries` is set, each query is admitted only after the
        previous one has left the pipeline. Waits are bounded by
        PIPELINE_DRAIN_TIMEOUT_S so a dropped query cannot hang the run.

        Args:
            query_queue (Queue): The queue containing queries from the load generator.
            event (Event): An event used for synchronization between the pipeline and the load generator.
        """

        queries_sent = 0
        self.queries_processed = 0

        result_retrieval_thread = Thread(
            target=self.retrieve_results, args=[event]
        )
        result_retrieval_thread.start()

        dataset_splits = self.get_dataset_splits()
        epoch_dict = {split: 0 for split in dataset_splits}

        serialize = bool(getattr(self._pipeline_config, "serialize_queries", False))

        # Per-query drain budget (s) for serialized mode and for shutdown.
        DRAIN_TIMEOUT = float(os.environ.get("PIPELINE_DRAIN_TIMEOUT_S", "3600"))

        while True:
            query: Query | None = query_queue.get()

            if query is None:
                self._wait_for_drain(queries_sent, DRAIN_TIMEOUT)
                self._broadcast_terminator()
                # Bounded join; main.py force-exits afterwards.
                self.join_threads(timeout=10.0)
                break

            if query.batch == 0:
                epoch_dict[query.split] += 1

            query.epoch = epoch_dict[query.split]

            if not self._disable_logs:
                self._logger.info(
                    "%s, pipeline - %s, run, start, %d, %.6f, %d, %d",
                    self.name,
                    query.split,
                    query.query_id,
                    query.query_submitted_timestamp,
                    query.epoch,
                    query.batch + 1,
                )

            # No flow id or attribute dict when spans are off.
            if SPANS_ENABLED:
                out_flow_id = new_flow_id()
                attributes = {
                    # the scheduler hands over a uuid.UUID; downstream ids are str
                    "in_flow_id": str(query.out_flow_id) if query.out_flow_id else None,
                    "out_flow_id": out_flow_id,
                    "thread_id": threading.get_ident(),
                    "pipeline": self.name,
                    "epoch": query.epoch,
                    "split": query.split,
                    "batch": query.batch,
                    "query_id": query.query_id,
                }
            else:
                out_flow_id, attributes = None, None
            with trace_span(name="pipeline query", attributes=attributes):
                query.out_flow_id = out_flow_id
                for input_queue in self._input_queues:
                    input_queue.put(query)

            queries_sent += 1

            if serialize:
                self._wait_for_drain(queries_sent, DRAIN_TIMEOUT)

        # Bounded in case an output stage never delivered its terminator.
        result_retrieval_thread.join(timeout=10.0)
