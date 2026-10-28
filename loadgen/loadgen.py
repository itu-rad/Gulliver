import logging
import os
import threading
import time
from logging.handlers import QueueHandler
from threading import Thread, Event
from queue import Queue
import multiprocessing

import mlflow
from utils.trace_span import trace_span

from loadgen.schedulers.scheduler import LoadScheduler
from pipeline import Pipeline
from utils.component import get_component
from utils.schemas import PipelineModel
from utils.tracing import flush_traces


def run_loadgen(pipeline_config: PipelineModel, stop_marker: str | None = None) -> None:
    """
    Initialization and invokation of LoadGen. Initialization is necessary after process
    creation, because fork is not the default start method on all platforms (spawn does
    not copy all resources).

    Args:
        pipeline_config (PipelineModel): Pipeline configuration (parsed from YAML file)
        stop_marker (str): stop generating load once this file exists
    """
    loadgen = LoadGen(pipeline_config)
    loadgen.run(stop_marker)


class LoadGen:
    """
    Wrapper around execution of a single pipeline. This includes the pipeline itself,
    the load generation, logging and communication between all of these components.
    """

    def __init__(self, pipeline_config: PipelineModel):
        self._pipeline_config = pipeline_config

        self._pipeline = Pipeline(pipeline_config)
        self._pipeline.prepare()
        dataset_splits = self._pipeline.get_dataset_splits()

        loadgen_config = pipeline_config.loadgen
        load_scheduler_config = loadgen_config.config
        self._load_scheduler: LoadScheduler = get_component(loadgen_config.component)(
            loadgen_config.max_queries,
            loadgen_config.timeout,
            load_scheduler_config,
            dataset_splits,
        )
        self._load_scheduler.prepare()

        sample_queue = Queue(maxsize=loadgen_config.queue_depth)

        # Signals query completions to schedulers that pace on them.
        event = Event()

        self.pipeline_thread = Thread(
            target=self._pipeline.run, args=[sample_queue, event]
        )
        self.scheduler_thread = Thread(
            target=self._load_scheduler.generate, args=[sample_queue, event]
        )

    def __str__(self) -> str:
        s = f"---\ntitle: 'Pipeline: {self._pipeline_config.name}'\n---\nflowchart LR\n"
        s += str(self._load_scheduler)
        s += str(self._pipeline)
        return s

    def _stop_on(self, marker: str) -> None:
        while self.scheduler_thread.is_alive():
            if os.path.exists(marker):
                print(f"{os.path.basename(marker)} finished; stopping load generation", flush=True)
                self._load_scheduler.stop = True
                return
            time.sleep(0.5)

    def run(self, stop_marker: str | None = None) -> None:
        """
        Start the pipeline and load generation threads and join them
        after execution is finished.

        The execution is stopped when the max_queries is reached or timer elapses.
        """
        with trace_span(
            name="LoadGen.run",
            attributes={"thread_id": threading.get_ident()},
        ):
            start_time = time.perf_counter()
            self.pipeline_thread.start()
            self.scheduler_thread.start()
            if stop_marker:
                Thread(target=self._stop_on, args=[stop_marker], daemon=True).start()
            self.scheduler_thread.join()
            self.pipeline_thread.join()
            duration = time.perf_counter() - start_time
            print(f"Pipeline execution completed in {duration:.2f} seconds", flush=True)

        # main.py exits with os._exit, which skips atexit handlers, so flush here.
        flush_traces()
