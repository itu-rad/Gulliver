"""MLflow trace export configuration and flushing.

The entry points exit with ``os._exit``, which skips atexit handlers, so
``flush_traces`` must be called before exiting or in-flight spans are lost.
"""

from __future__ import annotations

import os
import sys


# Read by MLflow's tracing exporter at initialization; must be set before
# the first span is created.
_ASYNC_TRACE_ENV_VAR = "MLFLOW_ENABLE_ASYNC_TRACE_LOGGING"


def configure_sync_export() -> None:
    """Force MLflow tracing to export spans synchronously.

    Idempotent. Call once per process, as early as possible (before any
    ``mlflow.start_span`` call).
    """
    # setdefault: an explicit user override in the shell wins.
    os.environ.setdefault(_ASYNC_TRACE_ENV_VAR, "false")


def configure_async_export() -> None:
    """Enable MLflow async trace export (used by radT's batched uploader).

    Sync export blocks worker threads on every root span. Idempotent; call
    once per process before any ``mlflow.start_span``.
    """
    # setdefault: an explicit user override in the shell wins.
    os.environ.setdefault(_ASYNC_TRACE_ENV_VAR, "true")


def flush_traces() -> None:
    """Drain pending MLflow trace spans and run-level async logs.

    Safe to call multiple times. Never raises — flush errors are printed
    to stderr so the caller (typically ``os._exit`` right after) can keep
    going.
    """
    try:
        import mlflow  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        print(f"[tracing] mlflow unavailable, cannot flush: {exc}",
              file=sys.stderr, flush=True)
        return

    # Run-level async logger (params, metrics, artifacts).
    try:
        mlflow.flush_async_logging()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[tracing] flush_async_logging failed: {exc}",
              file=sys.stderr, flush=True)

    # Trace async logger (spans). Skipped in sync mode, where there is no
    # queue and MLflow would log a spurious error.
    if os.environ.get(_ASYNC_TRACE_ENV_VAR, "true").lower() != "false":
        try:
            mlflow.flush_trace_async_logging(terminate=True)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[tracing] flush_trace_async_logging failed: {exc}",
                  file=sys.stderr, flush=True)
