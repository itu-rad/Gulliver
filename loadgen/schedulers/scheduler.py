from threading import Timer
from typing import Any
from abc import ABC, abstractmethod
import json


class LoadScheduler(ABC):
    """Base load scheduler class, responsible for generating load for a single pipeline."""

    def __init__(
        self,
        max_queries: int,
        timeout: int,
        load_scheduler_config: dict[str, Any],
        dataset_splits: dict[str, int],
    ):
        self.max_queries = max_queries
        self.stop = False
        self.timer = Timer(timeout, self.stop_generator)
        self.dataset_splits = dataset_splits
        self.extra_config = load_scheduler_config

    def __str__(self) -> str:
        d = {
            "max_queries": self.max_queries,
            "timeout": self.timer.interval,
            "config": self.extra_config,
        }
        encoded_d = json.dumps(d, indent=4).replace('"', "'").replace("    ", "&emsp;")
        return f'load_sched["`{self.__class__.__name__}\n{encoded_d}`"]\nstyle load_sched text-align:left\n'

    def prepare(self):
        """
        Prepares the scheduler for execution.

        This method is intended to be overridden by subclasses to perform any
        necessary setup or initialization before the scheduler starts its tasks.
        """
        pass

    def stop_generator(self):
        """
        Set a flag to gracefully end load generation.

        This method sets the `stop` attribute to `True`, which signals the load
        generator to stop its operation. It also prints "TIMEOUT!" to indicate
        that the stop signal has been issued.
        """
        print("TIMEOUT!")
        self.stop = True

    @abstractmethod
    def generate(self, queue, event):
        """Generate load based on the specified distribution.

        Args:
            queue (queue.Queue): Pipeline's input queue.
            event (threading.Event): Conditional variable for synchronizing between the load generation and the pipeline's execution.
        """
        pass
