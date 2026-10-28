import logging
from logging.handlers import QueueListener
import os
import re
import time


# Trace CSV row: wall-clock time (used to align with radT listener samples
# across processes), the message, and a process-local monotonic
# perf_counter_ns for per-stage latency.
PERF_FORMAT = "%(created)f, %(message)s, %(perf)d"

_perf_clock_installed = False


def install_perf_clock():
    """Install a LogRecord factory that adds ``record.perf`` (perf_counter_ns).

    Idempotent; must be called in every process before any record is formatted
    with ``PERF_FORMAT``. The stamp is taken at record creation and survives
    pickling through the multiprocessing log queue.
    """
    global _perf_clock_installed
    if _perf_clock_installed:
        return
    _orig_factory = logging.getLogRecordFactory()

    def _factory(*args, **kwargs):
        record = _orig_factory(*args, **kwargs)
        record.perf = time.perf_counter_ns()
        return record

    logging.setLogRecordFactory(_factory)
    _perf_clock_installed = True


class BenchmarkFilter(logging.Filter):
    """Only allow logs from the 'benchmark' logger"""

    def filter(self, record):
        return record.name == "benchmark"


class Logger:
    """
    Multiprocessing-safe logger, listening on events in a queue.

    Attributes:
        queue (multiprocessing.Queue): The queue to listen on for logging events.
        queue_listener (logging.handlers.QueueListener): The listener that processes log records from the queue.

    Methods:
        __init__(queue, benchmark_name):
            Initializes the Logger with a queue and a benchmark name.

            Args:
                queue (multiprocessing.Queue): The queue to listen on for logging events.
                benchmark_name (str): The name of the benchmark, used to create a filename-safe log file.

        stop_queue_listener():
            Stops the queue listener.
    """

    queue = None
    queue_listener = None

    def __init__(self, queue, benchmark_name):
        self.queue = queue

        install_perf_clock()
        formatter = logging.Formatter(PERF_FORMAT)

        benchmark_filter = BenchmarkFilter()

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        # make benchmark name filename-safe
        benchmark_name = benchmark_name.lower()
        benchmark_name = re.sub(r"\s", "_", benchmark_name)
        benchmark_name = re.sub(r"[^\w_]+", "", benchmark_name)
        # Same output directory variable as main.py.
        log_dir = os.environ.get("SUITE_OUTPUT_DIR") or os.path.join(os.getcwd(), "tmp")
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            filename=os.path.join(log_dir, f"{benchmark_name}.csv")
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(benchmark_filter)

        self.queue_listener = QueueListener(self.queue, stream_handler, file_handler)
        self.queue_listener.start()

        logger = logging.getLogger("benchmark")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    def stop_queue_listener(self):
        """Stop the queue listener"""
        self.queue_listener.stop()
