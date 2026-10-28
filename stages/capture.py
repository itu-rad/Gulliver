"""Output-capture sidecar stage.

Drop-in replacement for the identity `Stage` used at pipeline tails.
Writes one JSONL record per completed query to
`results/<pipeline_name>_outputs.jsonl`, then forwards the query unchanged.

YAML:
    - name: End stage
      id: 12
      component: stages.TerminalCapture
      polling_policy: utils.queues.polling.FirstSubmittedPolicy
"""

from __future__ import annotations

import json
import os
from threading import Lock

from stages.stage import Stage
from utils.schemas import Query


_DEFAULT_OUTPUT_DIR = os.environ.get("SUITE_OUTPUT_DIR", "results")


def _serializable(value):
    """Coerce a value to something json.dumps can handle."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serializable(v) for k, v in value.items()}
    return repr(value)


class TerminalCapture(Stage):
    """Identity end-stage that also persists each query's final state."""

    def __init__(self, stage_config, pipeline_config):
        super().__init__(stage_config, pipeline_config)

        default_label = pipeline_config.name.replace(" ", "_").lower()
        # Set by main.py from --label, so the JSONL name matches the timing CSV.
        pipeline_name = os.environ.get("SUITE_OUTPUT_LABEL", default_label)
        output_dir = self.extra_config.get("output_dir", _DEFAULT_OUTPUT_DIR)
        self._output_path = os.path.join(
            output_dir, f"{pipeline_name}_outputs.jsonl"
        )
        os.makedirs(output_dir, exist_ok=True)
        # Truncate so a new run does not append to stale data.
        open(self._output_path, "w").close()
        self._write_lock = Lock()

    def run(self, query: Query) -> dict[int, Query]:
        ctx = query.context or {}
        record = {
            "query_id": str(query.query_id),
            "epoch": getattr(query, "epoch", None),
            "batch": getattr(query, "batch", None),
            "split": getattr(query, "split", None),
            "question": _serializable(
                ctx.get("question") or ctx.get("original_query")
            ),
            "golden_answers": _serializable(ctx.get("golden_answers")),
            "retrieved_documents": _serializable(
                ctx.get("retrieved_documents")
            ),
            "generated_answer": _serializable(ctx.get("generated_answer")),
            "final_data": _serializable(query.data),
            "retries": ctx.get("retries", 0),
            "retries_exhausted": bool(ctx.get("retries_exhausted", False)),
            "llm_calls": _serializable(ctx.get("llm_calls")),
            "reasoning_tokens": _serializable(ctx.get("reasoning_tokens")),
        }
        with self._write_lock:
            with open(self._output_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        return {idx: query for idx in self.output_queues}
