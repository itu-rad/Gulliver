import threading

from utils.queues.peekable_queue import PeekableQueue
from utils.schemas import Query
from abc import ABC, abstractmethod


class PollingPolicy(ABC):
    """
    Base class for all queue polling policies.
    """

    def __init__(
        self,
        input_queues: dict[int, PeekableQueue],
        notify_condition: threading.Condition | None = None,
    ):
        self.input_queues = input_queues
        # Signalled by PeekableQueue.put. Without one, a private condition makes
        # ``wait(timeout=...)`` degrade to slow polling.
        self._cond = notify_condition if notify_condition is not None else threading.Condition()

    @abstractmethod
    def get_input_from_queues(self) -> Query | None:
        """
        Poll the input queues and return query based on the policy

        Returns:
            Query | None: Thequery or None if no query is available.
        """
        pass
