from __future__ import annotations  # postponed evaluation of annotations
from queue import Queue
from threading import Thread
from functools import wraps
import logging
import threading
from typing import Any
import json

import mlflow
from utils.trace_span import trace_span, SPANS_ENABLED, new_flow_id

from utils.queues.polling.polling_policy import PollingPolicy
from utils.queues.peekable_queue import PeekableQueue
from utils.component import get_component
from utils.schemas import StageModel, PipelineModel, Query


def log_phase(f):
    """
    Decorator to log the start and end of a function's execution.

    This decorator wraps a function to log its execution phases (start and end) using the `logging` module.
    It logs the parent name, stage name, and function name at the start and end of the function execution.

    Args:
        f (function): The function to be wrapped with logging functionality.

    Returns:
        function: The wrapped function with logging.

    Usage:
        @log_phase
        def some_function(self, *args, **kwargs):
            # Function implementation
    """

    @wraps(f)
    def wrapper(self, *args, **kw):
        if not self.disable_logs:
            logging.getLogger("benchmark").info(
                "%s, %s, %s, start", self.parent_name, self.name, f.__name__
            )
        result = f(self, *args, **kw)
        if not self.disable_logs:
            logging.getLogger("benchmark").info(
                "%s, %s, %s, end", self.parent_name, self.name, f.__name__
            )
        return result

    return wrapper


def log_phase_single(parent_name, name, phase, start):
    """Logs the stage execution status

    Args:
        parent_name (str): Name of the pipeline
        name (str): Stage name
        phase (str): Name of phase (prepare or run)
        start (str): Execution status (start or end)
    """
    logging.getLogger("benchmark").info(
        "%s, %s, %s, %s", parent_name, name, phase, start
    )


def marker_span(stage: "Stage", name: str, attributes: dict | None = None) -> None:
    """Emit a zero-duration span marking an instant in `stage`'s current query.

    Not gated by `disable_logs`, so the timing is available from spans alone.
    Names must be unique per query.
    """
    if not SPANS_ENABLED:
        return
    attrs = {"stage": stage.name,
             "query_id": getattr(stage, "current_query_id", None)}
    if attributes:
        attrs.update(attributes)
    with trace_span(name=f"{stage.name}.{name}", attributes=attrs):
        pass


def log_first_token(stage: "Stage", event: str) -> None:
    """Log one edge of a generator stage's "first_token" sub-phase:

        wall, <pipeline>, <stage>, first_token, {start|end}, perf

    Emit "start" right before generation and "end" at the first produced token,
    once per run() call.
    """
    marker_span(stage, f"first_token_{event}")
    if not stage.disable_logs:
        log_phase_single(stage.parent_name, stage.name, "first_token", event)


def log_generated_tokens(stage: "Stage", n_tokens: int) -> None:
    """Log the per-query generated-token count:

        wall, <pipeline>, <stage>, n_generated_tokens, <int>, perf
    """
    marker_span(stage, "generated_tokens", {"n_generated_tokens": n_tokens})
    if not stage.disable_logs:
        logging.getLogger("benchmark").info(
            "%s, %s, n_generated_tokens, %d", stage.parent_name, stage.name,
            n_tokens,
        )


def log_reasoning_tokens(stage: "Stage", n_reasoning: int, n_answer: int,
                         closed: bool, forced: bool) -> None:
    """Log the split of a generation into reasoning and answer tokens:

        wall, <pipeline>, <stage>, n_reasoning_tokens, <int>, perf
        wall, <pipeline>, <stage>, n_answer_tokens, <int>, perf
        wall, <pipeline>, <stage>, thinking_closed, {0|1|2}, perf

    thinking_closed: 1 = the model closed its reasoning block, 2 = the budget
    ran out and the block was forced closed, 0 = no answer was produced.
    """
    marker_span(stage, "reasoning_tokens",
                {"n_reasoning_tokens": n_reasoning, "n_answer_tokens": n_answer,
                 "thinking_closed": bool(closed), "thinking_forced": bool(forced)})
    if not stage.disable_logs:
        lg = logging.getLogger("benchmark")
        lg.info("%s, %s, n_reasoning_tokens, %d", stage.parent_name, stage.name,
                n_reasoning)
        lg.info("%s, %s, n_answer_tokens, %d", stage.parent_name, stage.name,
                n_answer)
        lg.info("%s, %s, thinking_closed, %d", stage.parent_name, stage.name,
                2 if forced else (1 if closed else 0))


def log_peak_memory(stage: "Stage", peak_bytes: int, source: str) -> None:
    """Log the process-level peak accelerator memory, as reported by the
    backend's allocator, at stage teardown:

        wall, <pipeline>, <stage>, peak_memory_bytes, <int>, perf
    """
    marker_span(stage, "peak_memory", {"peak_memory_bytes": int(peak_bytes),
                                       "source": source})
    if not stage.disable_logs:
        logging.getLogger("benchmark").info(
            "%s, %s, peak_memory_bytes, %d", stage.parent_name, stage.name,
            int(peak_bytes),
        )


def log_prompt_tokens(stage: "Stage", n_tokens: int) -> None:
    """Log the per-query prompt (prefill) token count:

        wall, <pipeline>, <stage>, n_prompt_tokens, <int>, perf
    """
    marker_span(stage, "prompt_tokens", {"n_prompt_tokens": n_tokens})
    if not stage.disable_logs:
        logging.getLogger("benchmark").info(
            "%s, %s, n_prompt_tokens, %d", stage.parent_name, stage.name,
            n_tokens,
        )


class Stage:
    """This is the building block of the pipelines. A stage can perform tasks such as data
    loading, data preprocessing or model execution. The stages are separated in order to
    make the development of specific part of a pipeline and subsequent evaluation as
    easy as possible."""

    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        self.id = stage_config.id
        self.name = stage_config.name
        self.parent_name = pipeline_config.name
        self.disable_logs = stage_config.disable_logs
        self._stage_config = stage_config
        self._polling_policy = stage_config.polling_policy
        self._output_stage_ids = stage_config.outputs
        self.extra_config = stage_config.config
        self._stage_dict: dict[int, Stage] = {}
        self._input_queues: dict[int, PeekableQueue] = {}
        # Shared by the input queues and the polling policy, so fan-in blocks
        # instead of busy-waiting.
        self._input_cond = threading.Condition()
        self._keep_payload_at_exit = getattr(
            self._stage_config, "keep_payload_at_exit", False)
        self.output_queues: dict[int, Queue] = {}
        self._logger = logging.getLogger("benchmark")
        # Per thread, because some stages process queries on several workers.
        self._current = threading.local()

    @property
    def current_query_id(self):
        """Query id this thread is processing, or None outside a query."""
        return getattr(self._current, "query_id", None)

    def __str__(self) -> str:
        """
        Returns a string representation of the object.
        The string includes the object's ID, name, and a JSON dump of its stage configuration.
        Additionally, it lists all output stage IDs in the format "current_id -> output_stage_id".
        Returns:
            str: A formatted string representing the object.
        """
        encoded_config = (
            json.dumps(self._stage_config.__dict__, indent=4)
            .replace('"', "'")
            .replace("    ", "&emsp;")
        )
        s = f'{self.id}["`{self.name}\n{encoded_config}`"]\nstyle {self.id} text-align:left\n'
        for output_stage_id in self._output_stage_ids:
            s += f"{self.id} --> {output_stage_id}\n"
        return s

    def set_stage_dict(self, stage_dict: dict[int, Stage]) -> None:
        """Set the stage dictionary, which is used for dynamic method invocation.

        Args:
            stage_dict (dict[int, Stage]): Dictionary mapping stage IDs (int) to their corresponding Stage objects.
        """
        self._stage_dict = stage_dict

    def set_output_queues(self):
        """Set the output queues of the stage by calling get_input_queue on the outgoing stages.

        Note: This method is automatically called by the pipeline after setting the stage_dict.
        """
        for out_stage_id in self._output_stage_ids:
            self.output_queues[out_stage_id] = self.dispatch_call(
                out_stage_id, "get_input_queue", self.id
            )

    def set_output_queue(self, queue: Queue):
        """Set the output queue of the stage manually. Only used for output stages.

        Args:
            queue (Queue):
        """
        self.output_queues = {-1: queue}

    def get_input_queue(self, idx: int) -> Queue:
        """Get the input queue for the given stage ID.

        If the queue does not exist yet, it is created. Honors
        `max_input_queue_depth` from the stage config (None / 0 == unbounded).

        Args:
            id (int): The ID of the stage to get the input queue for.

        Returns:
            queue.Queue: The input queue for the given stage ID.
        """
        if idx not in self._input_queues:
            maxsize = self._stage_config.max_input_queue_depth or 0
            self._input_queues[idx] = PeekableQueue(
                maxsize=maxsize, notify_condition=self._input_cond
            )
        return self._input_queues[idx]

    def dispatch_call(
        self, stage_id: int, method_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        """Invoke a method on a stage by its ID.

        The method is invoked on the stage with the given ID. The method to invoke
        is specified by the method_name parameter. The arguments to the method are
        passed in as *args and **kwargs.

        Args:
            stage_id (int): The ID of the stage to invoke the method on.
            method_name (str): The name of the method to invoke.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            Any: The result of the invoked method.
        """
        return getattr(self._stage_dict[stage_id], method_name)(*args, **kwargs)

    def join_thread(self) -> None:
        """Wait for the stage thread to join."""
        self._thread.join()

    def get_dataset_splits(self) -> dict[str, int]:
        """
        Get the number of batches in each dataset split.
        Default implementation returns a single batch for 'train' split.
        """
        return {'train': 1}

    def prepare(self) -> None:
        """
        Prepare the stage for execution.
        """
        if (
            len(self._input_queues) > 1
            and self._polling_policy.rsplit(".", 1)[-1] == "SingleQueuePolicy"
        ):
            raise ValueError(
                f"SingleQueuePolicy only works with one input queue "
                f"(stage {self.name!r} has {len(self._input_queues)}); "
                f"use a multi-queue polling policy"
            )
        self._polling_policy_obj: PollingPolicy = get_component(self._polling_policy)(
            self._input_queues, self._input_cond
        )
        self._thread = Thread(target=self.run_wrapper)
        self._thread.start()

    def _get_input_from_queues(self) -> Query | None:
        """Retrieve items from all input queues

        Returns:
            Query | None: The first query from all input queues or None if terminating character is received.
        """
        return self._polling_policy_obj.get_input_from_queues()

    def _push_to_all_outputs(self, output: any) -> None:
        """Push the same data to all output queues

        Args:
            output (any): Element to be pushed to all output queues
        """
        for output_queue in self.output_queues.values():
            output_queue.put(output)

    def _push_to_outputs(self, outputs: dict[int, Query]) -> None:
        """Push each output to its corresponding output queue.

        Args:
            outputs (dict[int, any]): dictionary of outputs, where keys are the stage IDs
                and values are the outputs to be pushed to the corresponding output queue.
        """
        qid = next((getattr(o, "query_id", None) for o in outputs.values()
                    if o is not None), None)
        # The payload of a query leaving the pipeline is not read again; release
        # it on this stage's thread rather than on the result collector's.
        if -1 in self.output_queues and not self._keep_payload_at_exit:
            for output in outputs.values():
                if output is not None:
                    output.data = None

        with trace_span(
            name=f"{self.name}.push_to_outputs",
            attributes=({"thread_id": threading.get_ident(),
                         "stage": self.name, "query_id": qid}
                        if SPANS_ENABLED else None),
        ):
            for idx, output in outputs.items():
                self.output_queues[idx].put(output)

    def run(self, query: Query) -> dict[int, Query]:
        """
        Run function of the Identity stage.

        This function simply returns the first value it receives from any of the input queues.
        It does not perform any operation on the inputs.

        Args:
            query (Query): Inputs retrieved from input queues

        Returns:
            dict[int, Query]: Dictionary of queries, where keys are the stage IDs
                and values are the queries to be pushed to the corresponding output queue.
        """
        return {idx: query for idx in self.output_queues}

    def pre_run(self) -> None:
        """
        This function is run at the very beginning of the run wrapper function.
        This can be used to perform any setup operation that requires to happen in the same thread as run (prepare function is run in the main pipeline thread).
        """
        pass

    def post_run(self) -> None:
        """
        This function is run at the very end of the run wrapper function.
        This can be used to perform any cleanup operations such as closing files or connections.
        """
        pass

    def _process_query(self, query: Query) -> None:
        """Run a single query through self.run, wrapped in its tracing span,
        and push the results to the output queues. Also used by stages that
        process queries on their own worker threads.
        """
        if not self.disable_logs:
            log_phase_single(self.parent_name, self.name, "run", "start")

        self._current.query_id = query.query_id
        # Kept minimal: this sits on the hand-off path between stages.
        if SPANS_ENABLED:
            out_flow_id = new_flow_id()
            attributes = {
                "in_flow_id": query.out_flow_id,
                "out_flow_id": out_flow_id,
                "thread_id": threading.get_ident(),
                "stage": self.name,
                "epoch": query.epoch,
                "split": query.split,
                "batch": query.batch,
                "query_id": query.query_id,
            }
        else:
            out_flow_id, attributes = None, None
        with trace_span(name=f"{self.name}.run", attributes=attributes):
            query.out_flow_id = out_flow_id
            outputs = self.run(query)

        self._push_to_outputs(outputs)
        self._current.query_id = None
        if not self.disable_logs:
            log_phase_single(self.parent_name, self.name, "run", "end")

    def run_wrapper(self) -> None:
        """Continuously poll for the incoming data in the input queues,
        perform actions on them and push the results onto the output queues."""
        self.pre_run()
        while True:
            # The wait is not spanned: it cannot be attributed to a query.
            query = self._get_input_from_queues()
            if not query:
                # received terminating element (None)
                self._push_to_all_outputs(None)
                break

            try:
                self._process_query(query)
            except Exception:
                # Propagate the terminator so the pipeline fails fast instead of
                # hanging until the run timeout.
                import traceback
                print(f"[stage FATAL] {self.name}: exception in run(); "
                      f"terminating pipeline", flush=True)
                traceback.print_exc()
                self._push_to_all_outputs(None)
                raise
        self.post_run()
