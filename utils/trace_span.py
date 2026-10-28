"""Span creation via radT (``SUITE_PROC_TRACE``), mlflow (default) or not at all.

Tracing is off when ``SUITE_DISABLE_TRACING`` is set. Each span carries a
process-local ``perf_counter_ns`` start stamp (:data:`PERF_ATTR`).
"""
import contextlib
import itertools
import logging
import os
import time

_TRUTHY = ("1", "true", "yes")

#: Span-attribute key carrying ``time.perf_counter_ns()`` at span start.
PERF_ATTR = "perf_start_ns"

# Namespace for the span-count CSV marker row and run tag. Changing it breaks
# readers of existing traces.
NAMESPACE = "suite"

#: Tag / CSV marker under which the emitted-span count is reported.
COUNT_KEY = f"{NAMESPACE}.spans_emitted"

# The proc backend drops events on queue overflow, so the number of spans handed
# over is counted and published for comparison with the exported event count.
# itertools.count().__next__ is atomic under the GIL; `n += 1` is not.
_span_seq = itertools.count()
_next_seq = _span_seq.__next__

# Flow ids link consecutive spans of one query and are treated as opaque
# strings. A counter is used instead of uuid4 because it is cheaper on the
# stage hand-off path; the per-process random prefix keeps ids unique across
# pipelines that write to one run.
_FLOW_PREFIX = "%08x%04x" % (int.from_bytes(os.urandom(4), "big"), os.getpid() & 0xFFFF)
_flow_next = itertools.count().__next__


def new_flow_id():
    """A process-unique flow id, cheap enough for the hand-off path."""
    return f"{_FLOW_PREFIX}-{_flow_next()}"


_final_count = None


def emitted_count():
    """How many spans this process handed to the backend.

    Reading the counter consumes an index, so the first result is cached.
    Call at shutdown; spans emitted after the first call are not counted.
    """
    global _final_count
    if _final_count is None:
        _final_count = _next_seq()
    return _final_count


_reported = False

# Forked children (radT listeners) inherit the counter; only the owner reports.
_owner_pid = os.getpid()


def report_span_count():
    """Publish the emitted-span count so a reader can verify completeness.

    Written as an mlflow tag and as a `benchmark` CSV row (not gated by
    `disable_logs`). Must be called while the run is active and the logger is
    open; later calls are no-ops. The count is a lower bound, as a few spans may
    start after it is taken.
    """
    global _reported
    if MODE != "proc" or _reported or os.getpid() != _owner_pid:
        return _final_count
    _reported = True
    n = emitted_count()
    logging.getLogger("benchmark").info("%s, spans, emitted, %d", NAMESPACE, n)
    try:
        import mlflow

        if mlflow.active_run() is not None:
            mlflow.set_tag(COUNT_KEY, str(n))
    except Exception:
        pass          # the CSV row is the primary record
    print(f"[{NAMESPACE}] spans emitted: {n}", flush=True)
    return n


def _resolve_mode():
    if os.environ.get("SUITE_PROC_TRACE", "").lower() in _TRUTHY:
        return "proc"
    if os.environ.get("SUITE_DISABLE_TRACING", "").lower() in _TRUTHY:
        return "off"
    return "mlflow"


MODE = _resolve_mode()

#: False when tracing is off, so callers can skip building attribute dicts.
SPANS_ENABLED = MODE != "off"


def _with_perf(attributes):
    """Return a copy of `attributes` with the monotonic clock stamped in."""
    attrs = dict(attributes) if attributes else {}
    attrs[PERF_ATTR] = time.perf_counter_ns()
    return attrs


if MODE == "proc":
    import radt

    def trace_span(name, attributes=None):
        _next_seq()
        return radt.trace.span(name, _with_perf(attributes))

elif MODE == "off":

    def trace_span(name, attributes=None):
        return contextlib.nullcontext()

else:
    import mlflow

    def trace_span(name, attributes=None):
        return mlflow.start_span(name=name, attributes=_with_perf(attributes))
