"""Read the span artifacts radT writes and verify they are complete.

radT spools spans to the run's ``radt-trace/`` artifact directory: a
``manifest.json`` and gzipped JSONL batches of positional records.

Each span carries a wall-clock start and end (comparable across processes) and a
``perf_start_ns`` attribute (monotonic, comparable only within one run), so
per-stage latencies are computed as start-to-start intervals.

Completeness is checked against the span count each run publishes: fewer start
events than emitted spans, or an end without a start, raises
:class:`IncompleteTrace`. Spans still open at process exit are reported as notes.
"""
from __future__ import annotations

import glob
import gzip
import json
import os
from dataclasses import dataclass, field

SCHEMA_VERSION = 1
PERF_ATTR = "perf_start_ns"
# Current namespace first, then earlier spellings found in existing traces.
_NAMESPACES = ("suite", "choreo")
COUNT_TAG = f"{_NAMESPACES[0]}.spans_emitted"
COUNT_TAGS = tuple(f"{n}.spans_emitted" for n in _NAMESPACES)


class IncompleteTrace(RuntimeError):
    """The artifact does not hold every span the workload emitted."""


_GZIP_MAGIC = b"\x1f\x8b"


def _open_batch(path):
    """Open a span batch, gzipped or not, deciding by content: artifacts fetched
    from a tracking server can arrive decompressed under their ``.gz`` name.
    """
    with open(path, "rb") as probe:
        gzipped = probe.read(2) == _GZIP_MAGIC
    if gzipped:
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


@dataclass
class Span:
    span_id: int
    parent_id: int | None
    trace_id: int
    name: str
    attributes: dict
    wall_start_ns: int
    wall_end_ns: int | None = None

    @property
    def perf_start_ns(self) -> int | None:
        """Monotonic start instant, or None for runs traced before it existed."""
        v = self.attributes.get(PERF_ATTR)
        return int(v) if v is not None else None

    @property
    def wall_duration_ns(self) -> int | None:
        if self.wall_end_ns is None:
            return None
        return self.wall_end_ns - self.wall_start_ns

    @property
    def query_id(self):
        return self.attributes.get("query_id")

    @property
    def stage(self):
        return self.attributes.get("stage")


@dataclass
class Completeness:
    """What the verification found. Truthy when nothing was lost.

    `problems` is loss or corruption and makes this falsy; `notes` is the
    benign structure described in the module docstring (parked threads), kept
    visible rather than hidden so an unexpected count still gets looked at.
    """
    event_count_manifest: int
    events_read: int
    starts: int
    ends: int
    emitted_expected: int | None
    unmatched_ends: int
    unclosed_spans: int
    unclosed_names: dict = field(default_factory=dict)
    problems: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def __bool__(self):
        return not self.problems

    def __str__(self):
        head = (f"{self.starts} spans ({self.events_read} events); "
                f"manifest says {self.event_count_manifest}")
        if self.emitted_expected is not None:
            head += f"; workload emitted {self.emitted_expected}"
        if self.problems:
            return head + " — PROBLEMS: " + "; ".join(self.problems)
        tail = " — complete"
        if self.notes:
            tail += " (" + "; ".join(self.notes) + ")"
        return head + tail


@dataclass
class SpanTrace:
    spans: list
    manifest: dict
    completeness: Completeness

    def __post_init__(self):
        self._by_name = {}
        for s in self.spans:
            self._by_name.setdefault(s.name, []).append(s)

    @property
    def names(self):
        """{span name: count} — what this run actually recorded."""
        return {k: len(v) for k, v in sorted(self._by_name.items())}

    def named(self, name):
        return self._by_name.get(name, [])

    def by_query(self, name):
        """{query_id: Span} for one span name.

        Raises on a duplicate query_id rather than silently keeping the last:
        two spans of the same name for one query means the caller's assumption
        about the pipeline's shape is wrong, and quietly dropping one would
        turn that into a plausible-looking number.
        """
        out = {}
        for s in self.named(name):
            q = s.query_id
            if q is None:
                continue
            if q in out:
                raise ValueError(
                    f"two {name!r} spans for query {q} — this span is not "
                    f"once-per-query, so by_query() is the wrong accessor")
            out[q] = s
        return out

    def interval(self, start_name, end_name, clock="perf"):
        """{query_id: ns} between the STARTS of two once-per-query spans.

        Start-to-start, because the perf clock is attached at span start only
        (see the module docstring). Queries missing either end are omitted, and
        the count is left to the caller to check against what it expected --
        this deliberately does not fill gaps.
        """
        if clock not in ("perf", "wall"):
            raise ValueError("clock must be 'perf' or 'wall'")
        a, b = self.by_query(start_name), self.by_query(end_name)
        pick = (lambda s: s.perf_start_ns) if clock == "perf" else (lambda s: s.wall_start_ns)
        out = {}
        for q in a.keys() & b.keys():
            t0, t1 = pick(a[q]), pick(b[q])
            if t0 is not None and t1 is not None:
                out[q] = t1 - t0
        return out

    def has_perf_clock(self):
        return any(s.perf_start_ns is not None for s in self.spans)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def emitted_from_csv(csv_path):
    """The `<namespace>, spans, emitted, N` row a run writes, or None."""
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.split(",")]
                if (len(parts) >= 5 and parts[2:4] == ["spans", "emitted"]
                        and parts[1] in _NAMESPACES):
                    return int(parts[4])
    except (OSError, ValueError):
        return None
    return None


def _verify(manifest, spans, events_read, starts, ends, unmatched_ends, emitted):
    problems, notes = [], []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"schema_version {manifest.get('schema_version')} != {SCHEMA_VERSION} "
            f"(record layout may have moved)")
    if events_read != manifest.get("event_count"):
        problems.append(
            f"read {events_read} events but manifest claims "
            f"{manifest.get('event_count')} (a batch file is missing or truncated)")
    if unmatched_ends:
        problems.append(f"{unmatched_ends} end record(s) with no matching start")

    # Fewer starts than emitted spans means events were dropped; more starts
    # come from threads still parked inside a span at exit.
    if emitted is not None:
        if starts < emitted:
            problems.append(
                f"only {starts} start records for {emitted} spans emitted — the "
                f"backend dropped {emitted - starts} on queue overflow "
                f"(raise RADT_TRACE_PROC_QUEUE_SIZE)")
        elif starts > emitted:
            notes.append(
                f"{starts - emitted} span(s) started after the count was taken "
                f"(threads parked at teardown)")

    unclosed = [s for s in spans if s.wall_end_ns is None]
    by_name = {}
    for s in unclosed:
        by_name[s.name] = by_name.get(s.name, 0) + 1
    if unclosed:
        notes.append(
            f"{len(unclosed)} span(s) still open at exit: "
            + ", ".join(f"{n} x{k}" if k > 1 else n for n, k in sorted(by_name.items())))
    return Completeness(
        event_count_manifest=manifest.get("event_count", -1),
        events_read=events_read, starts=starts, ends=ends,
        emitted_expected=emitted, unmatched_ends=unmatched_ends,
        unclosed_spans=len(unclosed), unclosed_names=by_name,
        problems=problems, notes=notes)


def read_dir(trace_dir, emitted=None, strict=True):
    """Read a ``radt-trace/`` directory into a :class:`SpanTrace`.

    `emitted` is the workload's own span count (mlflow tag, or
    :func:`emitted_from_csv`); pass it whenever it is available, since without
    it the drop check cannot run. `strict` raises :class:`IncompleteTrace` on
    any problem; set it False only to inspect a trace you already know is
    damaged.
    """
    with open(os.path.join(trace_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)

    batches = manifest.get("batches") or sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(trace_dir, "spans-*.jsonl.gz")))
    missing = [b for b in batches if not os.path.exists(os.path.join(trace_dir, b))]
    if missing:
        raise IncompleteTrace(f"manifest names batch files that are absent: {missing}")

    live, spans = {}, []
    events = starts = ends = unmatched = 0
    skipped = 0
    for b in batches:
        with _open_batch(os.path.join(trace_dir, b)) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # A batch cut off at process exit can end in a truncated line.
                    # Strict mode treats that as corrupt; otherwise it is skipped.
                    if strict:
                        raise
                    skipped += 1
                    continue
                events += 1
                if rec[0] == "s":
                    _, sid, parent, trace, name, attrs, ts = rec
                    sp = Span(sid, parent, trace, name, attrs or {}, ts)
                    live[sid] = sp
                    spans.append(sp)
                    starts += 1
                else:
                    _, sid, ts = rec
                    sp = live.pop(sid, None)
                    if sp is None:
                        unmatched += 1
                    else:
                        sp.wall_end_ns = ts
                    ends += 1

    completeness = _verify(manifest, spans, events, starts, ends, unmatched, emitted)
    if strict and not completeness:
        raise IncompleteTrace(str(completeness))
    return SpanTrace(spans, manifest, completeness)


def read_run(run_id, tracking_uri=None, dest_dir=None, strict=True):
    """Download a run's ``radt-trace/`` artifacts and read them.

    The emitted-span count is taken from the run's ``<namespace>.spans_emitted``
    tag, so the drop check runs without the caller supplying anything.
    """
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    local = client.download_artifacts(run_id, "radt-trace", dest_dir)
    tags = client.get_run(run_id).data.tags
    tag = next((tags[t] for t in COUNT_TAGS if t in tags), None)
    return read_dir(local, emitted=int(tag) if tag else None, strict=strict)
