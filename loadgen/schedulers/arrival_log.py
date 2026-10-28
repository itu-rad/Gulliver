"""Intended-vs-actual arrival log for open-loop load schedulers.

Records per query the intended arrival time, the actual enqueue time and how
long the blocking put() stalled, written as ``<label>_arrivals.csv``.
"""

from __future__ import annotations

import os


class ArrivalLog:
    """Accumulates (epoch, intended_ts, actual_ts, block_s) rows; writes on close."""

    def __init__(self):
        self.rows: list[tuple[int, float, float, float]] = []

    def record(self, epoch: int, intended_ts: float, actual_ts: float,
               block_s: float) -> None:
        self.rows.append((epoch, intended_ts, actual_ts, block_s))

    def write(self, results_dir: str | None = None,
              label: str | None = None) -> str | None:
        """Write the sidecar CSV; label defaults to SUITE_OUTPUT_LABEL (the
        same env var main.py uses to name the trace CSV). Returns the path."""
        if not self.rows:
            return None
        label = label or os.environ.get("SUITE_OUTPUT_LABEL")
        if not label:
            return None
        results_dir = results_dir or os.environ.get("SUITE_OUTPUT_DIR", "results")
        try:
            os.makedirs(results_dir, exist_ok=True)
            path = os.path.join(results_dir, f"{label}_arrivals.csv")
            with open(path, "w", encoding="utf-8") as f:
                f.write("epoch,intended_ts,actual_ts,block_s\n")
                for epoch, its, ats, bs in self.rows:
                    f.write(f"{epoch},{its:.6f},{ats:.6f},{bs:.6f}\n")
            return path
        except OSError:
            return None  # never let telemetry kill a run
