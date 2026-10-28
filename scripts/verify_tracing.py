"""Verify MLflow tracing on a noop pipeline (span counts, flow chains, ordering).

``--mode executor`` runs the pipeline in-process; ``--mode radt`` runs it via
main.py. Exits 0 if all checks pass; JSON report on stdout, summary on stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Any, Optional

# Make the framework importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["executor", "radt"],
        default="executor",
        help="executor: run pipeline in-process (fast). radt: full subprocess via main.py.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=3,
        help="max_queries override for the noop pipeline (default 3).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="number of identity stages in the noop chain (default 1).",
    )
    parser.add_argument(
        "--experiment",
        type=int,
        required=True,
        help="radT/MLflow experiment id.",
    )
    parser.add_argument(
        "--keep-config",
        action="store_true",
        help="don't delete the temp config after the run (debugging).",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="write JSON report to this file instead of stdout "
             "(stdout is polluted by the loadgen mermaid diagram).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def _build_noop_config(depth: int, max_queries: int, run_label: str) -> str:
    """Build a tiny noop-chain config as a YAML string.

    Single pipeline, ``depth`` identity stages chained linearly, offline
    scheduler with ``max_queries``. Listeners default to ``[macmon]`` from
    BenchmarkModel.
    """
    stages_yaml = []
    for i in range(depth):
        outputs = [i + 1] if i < depth - 1 else []
        stages_yaml.append(
            f"    - name: Identity Stage {i}\n"
            f"      id: {i}\n"
            f"      component: stages.stage.Stage\n"
            f"      outputs: {outputs}\n"
            f"      polling_policy: utils.queues.polling.SingleQueuePolicy\n"
            f"      disable_logs: false\n"
        )

    return (
        f"name: verify_tracing_{run_label}\n"
        "pipelines:\n"
        f"  - name: verify_tracing_pipeline_{run_label}\n"
        "    inputs:\n"
        "      - 0\n"
        "    outputs:\n"
        f"      - {depth - 1}\n"
        "    dataset_stage_id: 0\n"
        "    loadgen:\n"
        "      component: loadgen.OfflineLoadScheduler\n"
        "      queue_depth: 100\n"
        f"      max_queries: {max_queries}\n"
        "      timeout: 60000\n"
        "      config:\n"
        "        rate: 0\n"
        "    stages:\n"
        + "".join(stages_yaml)
    )


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def _run_executor_mode(config_path: str) -> str:
    """Run the pipeline in-process and return the MLflow run_id."""
    import mlflow  # pylint: disable=import-outside-toplevel
    from pydantic_yaml import parse_yaml_raw_as  # pylint: disable=import-outside-toplevel

    from loadgen import run_loadgen  # pylint: disable=import-outside-toplevel
    from utils.schemas import BenchmarkModel  # pylint: disable=import-outside-toplevel
    from utils.tracing import configure_sync_export  # pylint: disable=import-outside-toplevel

    configure_sync_export()

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = parse_yaml_raw_as(BenchmarkModel, fh.read())

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"[verify_tracing] mlflow run_id = {run_id}", file=sys.stderr, flush=True)
        run_loadgen(cfg.pipelines[0])

    return run_id


def _finalize_verifier_parent_run(
    experiment_id: int, started_at_ms: int
) -> int:
    """Mark the RadT parent run for this verifier invocation FINISHED.

    Needed when main.py is killed on timeout and the parent run stays RUNNING.
    Only runs started after ``started_at_ms`` whose name starts with
    ``verify_tracing_`` are touched, never the ``(0 0) `` pipeline child run.

    Returns the number of runs finalized.
    """
    import mlflow  # pylint: disable=import-outside-toplevel
    client = mlflow.MlflowClient()
    try:
        running = client.search_runs(
            experiment_ids=[str(experiment_id)],
            filter_string="attributes.status = 'RUNNING'",
            max_results=100,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[verify_tracing] sweep search failed: {exc}",
              file=sys.stderr, flush=True)
        return 0

    finalized = 0
    for r in running:
        if r.info.start_time < started_at_ms:
            continue
        name = r.info.run_name or ""
        if not name.startswith("verify_tracing_"):
            continue
        if name.startswith("(0 0) "):
            continue
        try:
            client.set_terminated(r.info.run_id, status="FINISHED")
            print(f"[verify_tracing] finalized parent run {r.info.run_id[:12]} ({name})",
                  file=sys.stderr, flush=True)
            finalized += 1
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[verify_tracing] failed to finalize {r.info.run_id[:12]}: {exc}",
                  file=sys.stderr, flush=True)
    return finalized


def _run_radt_mode(config_path: str, experiment_id: int) -> list[str]:
    """Run via main.py + RadT, return list of MLflow run_ids spawned.

    radT spawns one MLflow run per pipeline; the noop config has one pipeline.
    """
    env = os.environ.copy()
    marker = f"verify_tracing_{int(time.time())}"
    env["SUITE_VERIFY_MARKER"] = marker

    cmd = [
        sys.executable,
        os.path.join(_REPO_ROOT, "main.py"),
        config_path,
        "-e",
        str(experiment_id),
    ]
    print(f"[verify_tracing] running: {' '.join(cmd)}", file=sys.stderr, flush=True)
    started_at = time.time()
    # If main.py hangs, kill it and verify whatever child runs landed in MLflow.
    try:
        proc = subprocess.run(cmd, env=env, cwd=_REPO_ROOT, check=False, timeout=180)
        if proc.returncode != 0:
            print(
                f"[verify_tracing] WARNING: main.py exited with code {proc.returncode}",
                file=sys.stderr,
                flush=True,
            )
    except subprocess.TimeoutExpired:
        print(
            "[verify_tracing] WARNING: main.py timed out after 180s — killing and verifying any child runs",
            file=sys.stderr,
            flush=True,
        )

    _finalize_verifier_parent_run(experiment_id, int(started_at * 1000))

    # Find runs created since started_at.
    import mlflow  # pylint: disable=import-outside-toplevel

    client = mlflow.MlflowClient()
    started_ms = int(started_at * 1000)
    runs = client.search_runs(
        experiment_ids=[str(experiment_id)],
        filter_string=f"attributes.start_time >= {started_ms}",
        order_by=["attributes.start_time DESC"],
        max_results=10,
    )
    run_ids = [r.info.run_id for r in runs]
    print(f"[verify_tracing] discovered {len(run_ids)} new run_id(s): {run_ids}",
          file=sys.stderr, flush=True)
    return run_ids


# ---------------------------------------------------------------------------
# Span retrieval
# ---------------------------------------------------------------------------

def _retrieve_spans_for_run(run_id: str, max_attempts: int = 5, delay: float = 1.0):
    """Pull spans from Postgres with a short retry budget.

    Tolerates spans arriving shortly after the run ends. Returns an empty
    DataFrame if none arrive, so the checks report the failure.
    """
    from utils.export_trace import explode_spans_content, retrieve_spans  # pylint: disable=import-outside-toplevel

    last_df = None
    for attempt in range(1, max_attempts + 1):
        df = retrieve_spans(run_id)
        last_df = df
        if not df.empty:
            return explode_spans_content(df)
        if attempt < max_attempts:
            print(
                f"[verify_tracing] no spans yet for run {run_id}, retry {attempt}/{max_attempts}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)

    return explode_spans_content(last_df) if last_df is not None else None


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""

@dataclass
class Report:
    run_id: str
    mode: str
    queries: int
    depth: int
    span_count: int
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


def _attr(row, key: str):
    """Read an attribute from the per-span normalized columns OR raw content."""
    val = row.get(f"attributes.{key}")
    if val is not None and not _is_nan(val):
        return val
    # Some attributes (epoch, batch, query_id, stage, pipeline, split) won't be
    # in the pre-extracted columns; fall back to raw content.
    content = row.get("content")
    if isinstance(content, str):
        try:
            doc = json.loads(content)
            return doc.get("attributes", {}).get(key)
        except Exception:  # pylint: disable=broad-except
            return None
    return None


def _is_nan(x: Any) -> bool:
    try:
        import math
        return isinstance(x, float) and math.isnan(x)
    except Exception:  # pylint: disable=broad-except
        return False


def _build_report(spans_df, run_id: str, mode: str, queries: int, depth: int) -> Report:
    rep = Report(run_id=run_id, mode=mode, queries=queries, depth=depth, span_count=len(spans_df) if spans_df is not None else 0)

    if spans_df is None or spans_df.empty:
        rep.checks.append(Check("spans_found", False, f"no spans retrieved for run {run_id}"))
        return rep
    rep.checks.append(Check("spans_found", True, f"{len(spans_df)} spans"))

    # Bucket spans by name for count assertions.
    by_name: dict[str, list] = defaultdict(list)
    for _, row in spans_df.iterrows():
        by_name[row["name"]].append(row)

    Q = queries
    D = depth

    # -------- count assertions --------
    def _check_count(name: str, expected: int, *, op: str = "eq") -> Check:
        got = len(by_name.get(name, []))
        if op == "eq":
            ok = got == expected
        elif op == "ge":
            ok = got >= expected
        else:
            raise ValueError(op)
        return Check(
            f"count[{name}]",
            ok,
            f"expected {op} {expected}, got {got}",
        )

    rep.checks.append(_check_count("generate query", Q))
    rep.checks.append(_check_count("pipeline query", Q))
    rep.checks.append(_check_count("pipeline query processed", Q))
    rep.checks.append(_check_count("LoadGen.run", 1))

    # Per-stage span counts.
    for stage_idx in range(D):
        stage_name = f"Identity Stage {stage_idx}"
        rep.checks.append(_check_count(f"{stage_name}.run", Q))
        rep.checks.append(_check_count(f"{stage_name}.push_to_outputs", Q))
        rep.checks.append(_check_count(f"{stage_name}.get_input", Q, op="ge"))

    # -------- thread_id presence --------
    missing_tid = []
    for _, row in spans_df.iterrows():
        tid = _attr(row, "thread_id")
        if tid is None or (isinstance(tid, str) and not tid):
            missing_tid.append(row["name"])
    rep.checks.append(
        Check(
            "thread_id_present",
            not missing_tid,
            f"missing on {len(missing_tid)} span(s)" + (f": {missing_tid[:3]}" if missing_tid else ""),
        )
    )

    # -------- per-stage thread distinctness --------
    stage_tids: dict[str, set] = defaultdict(set)
    for _, row in spans_df.iterrows():
        name = row["name"]
        if name.endswith(".run") and "Identity Stage" in name:
            tid = _attr(row, "thread_id")
            if tid is not None:
                stage_tids[name].add(int(tid))
    tids_distinct_per_stage = all(len(v) == 1 for v in stage_tids.values())
    rep.checks.append(
        Check(
            "stage_thread_consistency",
            tids_distinct_per_stage,
            f"each stage runs on exactly one thread: {dict((k, list(v)) for k, v in stage_tids.items())}",
        )
    )

    # -------- flow chain integrity --------
    # For each "generate query" span, follow out_flow_id → ... → in_flow_id
    # through pipeline query → stage[0].run → ... → stage[D-1].run → pipeline query processed.
    def _build_index_by_in_flow(name: str) -> dict[str, Any]:
        return {
            str(_attr(row, "in_flow_id")): row
            for row in by_name.get(name, [])
            if _attr(row, "in_flow_id")
        }

    pipeline_query_by_in = _build_index_by_in_flow("pipeline query")
    pipeline_processed_by_in = _build_index_by_in_flow("pipeline query processed")
    stage_run_by_in: list[dict[str, Any]] = [
        _build_index_by_in_flow(f"Identity Stage {i}.run") for i in range(D)
    ]

    chain_failures = []
    chain_ok = 0
    for gen_row in by_name.get("generate query", []):
        gen_flow = _attr(gen_row, "out_flow_id")
        if not gen_flow:
            chain_failures.append({"step": "generate query missing out_flow_id"})
            continue
        # Step 1: pipeline query keyed by in_flow == gen_flow
        pq = pipeline_query_by_in.get(str(gen_flow))
        if pq is None:
            chain_failures.append({"step": "no pipeline query for gen flow", "gen_flow": str(gen_flow)})
            continue
        prev_flow = _attr(pq, "out_flow_id")
        # Step 2..N: each stage in order
        broken = False
        for stage_idx in range(D):
            sr = stage_run_by_in[stage_idx].get(str(prev_flow))
            if sr is None:
                chain_failures.append({
                    "step": f"no stage[{stage_idx}].run for flow",
                    "flow": str(prev_flow),
                })
                broken = True
                break
            prev_flow = _attr(sr, "out_flow_id")
        if broken:
            continue
        # Step N+1: pipeline query processed
        pp = pipeline_processed_by_in.get(str(prev_flow))
        if pp is None:
            chain_failures.append({"step": "no pipeline query processed for final flow", "flow": str(prev_flow)})
            continue
        chain_ok += 1

    rep.checks.append(
        Check(
            "flow_chain_complete",
            chain_ok == Q and not chain_failures,
            f"{chain_ok}/{Q} queries traced end-to-end"
            + (f"; first failures: {chain_failures[:2]}" if chain_failures else ""),
        )
    )

    # -------- required attributes on .run spans --------
    required_run_attrs = ("stage", "epoch", "split", "batch", "query_id", "thread_id")
    missing_run_attrs = []
    for stage_idx in range(D):
        for row in by_name.get(f"Identity Stage {stage_idx}.run", []):
            for attr in required_run_attrs:
                if _attr(row, attr) is None:
                    missing_run_attrs.append((row["name"], attr))
    rep.checks.append(
        Check(
            "stage_run_attrs",
            not missing_run_attrs,
            f"missing {len(missing_run_attrs)} attr(s)"
            + (f": {missing_run_attrs[:3]}" if missing_run_attrs else ""),
        )
    )

    # -------- temporal monotonicity along chain --------
    # Order: generate query < pipeline query < stage[0].run < ... < stage[D-1].run < pipeline query processed
    out_of_order = []
    for gen_row in by_name.get("generate query", []):
        gen_flow = _attr(gen_row, "out_flow_id")
        if not gen_flow:
            continue
        pq = pipeline_query_by_in.get(str(gen_flow))
        if pq is None:
            continue
        ts_chain = [(gen_row["name"], int(gen_row["start_time_unix_nano"])),
                    (pq["name"], int(pq["start_time_unix_nano"]))]
        prev_flow = _attr(pq, "out_flow_id")
        broken = False
        for stage_idx in range(D):
            sr = stage_run_by_in[stage_idx].get(str(prev_flow))
            if sr is None:
                broken = True
                break
            ts_chain.append((sr["name"], int(sr["start_time_unix_nano"])))
            prev_flow = _attr(sr, "out_flow_id")
        if broken:
            continue
        pp = pipeline_processed_by_in.get(str(prev_flow))
        if pp is not None:
            ts_chain.append((pp["name"], int(pp["start_time_unix_nano"])))
        # check monotonic
        for (a_name, a_ts), (b_name, b_ts) in zip(ts_chain, ts_chain[1:]):
            if a_ts > b_ts:
                out_of_order.append((a_name, a_ts, b_name, b_ts))

    rep.checks.append(
        Check(
            "temporal_monotonicity",
            not out_of_order,
            f"{len(out_of_order)} inversions"
            + (f": e.g. {out_of_order[0]}" if out_of_order else ""),
        )
    )

    return rep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    run_label = uuid.uuid4().hex[:8]
    config_yaml = _build_noop_config(args.depth, args.queries, run_label)

    tmpfd, tmp_path = tempfile.mkstemp(suffix=f"_verify_{run_label}.yml", prefix="noop_")
    with os.fdopen(tmpfd, "w") as fh:
        fh.write(config_yaml)
    print(f"[verify_tracing] config at {tmp_path}", file=sys.stderr, flush=True)

    try:
        if args.mode == "executor":
            run_id = _run_executor_mode(tmp_path)
            run_ids = [run_id]
        else:
            run_ids = _run_radt_mode(tmp_path, args.experiment)

        if not run_ids:
            print(json.dumps({"error": "no run_ids produced"}), flush=True)
            return 1

        all_reports = []
        skipped = []
        any_failed = False
        for run_id in run_ids:
            spans_df = _retrieve_spans_for_run(run_id)
            # Under radT the parent run has no spans; skip it rather than fail.
            if args.mode == "radt" and (spans_df is None or spans_df.empty):
                skipped.append(run_id)
                continue
            rep = _build_report(spans_df, run_id, args.mode, args.queries, args.depth)
            all_reports.append(rep)
            if not rep.passed:
                any_failed = True

        # In radt mode at least one pipeline run must have spans.
        if args.mode == "radt" and not all_reports:
            print(
                f"[verify_tracing] FAIL: no pipeline runs with spans found "
                f"(skipped {len(skipped)} zero-span run(s): {skipped})",
                file=sys.stderr,
                flush=True,
            )
            any_failed = True

        # Pretty stderr summary.
        for rep in all_reports:
            print("", file=sys.stderr)
            print(f"=== {rep.mode} mode | run {rep.run_id} | {rep.span_count} spans ===",
                  file=sys.stderr)
            for c in rep.checks:
                marker = "PASS" if c.passed else "FAIL"
                print(f"  [{marker}] {c.name}: {c.detail}", file=sys.stderr)
            print(f"  --> overall: {'PASS' if rep.passed else 'FAIL'}", file=sys.stderr)

        # JSON report for programmatic use.
        out = {
            "mode": args.mode,
            "queries": args.queries,
            "depth": args.depth,
            "any_failed": any_failed,
            "skipped_zero_span_runs": skipped if args.mode == "radt" else [],
            "reports": [
                {
                    "run_id": r.run_id,
                    "passed": r.passed,
                    "span_count": r.span_count,
                    "checks": [asdict(c) for c in r.checks],
                }
                for r in all_reports
            ],
        }
        json_payload = json.dumps(out, indent=2, default=str)
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as fh:
                fh.write(json_payload)
            print(f"[verify_tracing] JSON report written to {args.json_out}",
                  file=sys.stderr, flush=True)
        else:
            print(json_payload, flush=True)
        return 1 if any_failed else 0

    finally:
        if not args.keep_config:
            try:
                os.unlink(tmp_path)
            except Exception:  # pylint: disable=broad-except
                pass


if __name__ == "__main__":
    sys.exit(main())
