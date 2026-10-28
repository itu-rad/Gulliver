import argparse
import mlflow
import signal
import sys
import os
import logging
import shutil
import tempfile
import time

import radt
from radt.run.listeners import listeners

import numpy as np
import pandas as pd
from pydantic import BaseModel
from multiprocessing import Process, Queue
from pydantic_yaml import parse_yaml_raw_as
from typing import Literal
from utils.logger import Logger, PERF_FORMAT, install_perf_clock
from loadgen import run_loadgen
from utils.orchestrator_watchdog import OrchestratorWatchdog
from utils.schemas import BenchmarkModel
from utils.server_manager import kill_all as kill_all_servers
from utils.tracing import configure_async_export, flush_traces
from utils.trace_span import NAMESPACE as TRACE_NAMESPACE, report_span_count


def parse_args():
    parser = argparse.ArgumentParser(description="Gulliver benchmark runner.")
    parser.add_argument(
        "config_file_path",
        type=str,
        help="path to the pipeline configuration file.",
    )
    parser.add_argument(
        "-p",
        type=int,
        dest="pipeline_id",
        default=-1,
        help="Run only this pipeline of the config in the current process.",
    )
    parser.add_argument(
        "-e",
        type=int,
        dest="experiment_id",
        default=0,
        help="radT experiment id",
    )
    parser.add_argument(
        "--serialize",
        dest="serialize_override",
        choices=["true", "false"],
        default=None,
        help="Override pipeline's serialize_queries flag (true/false).",
    )
    parser.add_argument(
        "--pipeline-logs",
        dest="pipeline_logs_override",
        choices=["on", "off"],
        default=None,
        help="Override the pipeline-level per-query CSV rows "
             "(PipelineModel.disable_logs): 'on' writes them, 'off' suppresses them.",
    )
    parser.add_argument(
        "--max-queries",
        dest="max_queries_override",
        type=int,
        default=None,
        help="Override the load generator's max_queries.",
    )
    parser.add_argument(
        "--label",
        dest="label",
        type=str,
        default=None,
        help="Override the per-run output filename suffix "
             "(default: pipeline name lowercased).",
    )
    parser.add_argument("--sync-dir", dest="sync_dir", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _slug(name: str) -> str:
    """Pipeline name -> filename-safe token, matching the default label style."""
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


def convert_listeners(listeners: list[Literal[listeners.keys()]]) -> str:
    return "+".join(listeners)


def radt_entrypoint(args):
    # Must run before the first span is created; the flag is read at exporter init.
    configure_async_export()

    if os.environ.get("SUITE_DISABLE_TRACING", "").lower() in ("1", "true", "yes"):
        mlflow.tracing.disable()

    # Out-of-process tracing: spans are sent to a radT-owned exporter process.
    # Started on the main thread, before pipeline threads and CUDA init.
    if os.environ.get("SUITE_PROC_TRACE", "").lower() in ("1", "true", "yes"):
        radt.trace.start(experiment_id=args.experiment_id)

    # Forked children (radT listeners) inherit this handler; only the owner flushes.
    _owner_pid = os.getpid()

    def _on_sigterm(signum, frame):  # pylint: disable=unused-argument
        if os.getpid() != _owner_pid:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return
        kill_all_servers()
        flush_traces()
        report_span_count()
        radt.trace.shutdown()
        # Close the run and mark it KILLED so it is not mistaken for a clean run.
        active = mlflow.active_run()
        radt.shutdown()
        if active is not None:
            try:
                mlflow.MlflowClient().set_terminated(active.info.run_id, "KILLED")
            except Exception:  # pylint: disable=broad-except
                pass
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    signal.signal(signal.SIGTERM, _on_sigterm)

    with open(args.config_file_path, "r", encoding="utf-8") as file:
        yaml_config = file.read()
        benchmark_config = parse_yaml_raw_as(BenchmarkModel, yaml_config)

        pipeline_cfg = benchmark_config.pipelines[args.pipeline_id]
        # Per-pipeline environment, e.g. an MPS client's SM cap; must precede CUDA init.
        for _k, _v in pipeline_cfg.env.items():
            os.environ[_k] = str(_v)
        if args.serialize_override is not None:
            pipeline_cfg.serialize_queries = (args.serialize_override == "true")
        if args.pipeline_logs_override is not None:
            pipeline_cfg.disable_logs = (args.pipeline_logs_override == "off")
        if args.max_queries_override is not None and pipeline_cfg.stop_with is None:
            pipeline_cfg.loadgen.max_queries = args.max_queries_override
        stop_marker = None
        if pipeline_cfg.stop_with is not None and args.sync_dir:
            stop_marker = os.path.join(args.sync_dir, _slug(pipeline_cfg.stop_with))

        default_label = pipeline_cfg.name.replace(" ", "_").lower()
        pipeline_name = args.label if args.label else default_label
        log_dir = os.environ.get("SUITE_OUTPUT_DIR", "results")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{pipeline_name}.csv")
        # TerminalCapture names its JSONL after the same label.
        os.environ["SUITE_OUTPUT_LABEL"] = pipeline_name

        install_perf_clock()
        formatter = logging.Formatter(PERF_FORMAT)
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)

        logger = logging.getLogger("benchmark")
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        import atexit as _atexit

        def _cleanup_log_handler():
            try:
                file_handler.flush()
                file_handler.close()
                logger.removeHandler(file_handler)
            except Exception:  # pylint: disable=broad-except
                pass

        _atexit.register(_cleanup_log_handler)

        # Flatten the config into mlflow params.
        def build_mlflow_config(
            config: dict, data: BaseModel | list | dict, directory: str
        ) -> None:
            if isinstance(data, BaseModel):
                for k in data.model_fields_set:
                    v = getattr(data, k)
                    build_mlflow_config(config, v, f"{directory}.{k}")
            elif isinstance(data, list):
                for i, v in enumerate(data):
                    build_mlflow_config(config, v, f"{directory}:{i}")
            elif isinstance(data, dict):
                for k, v in data.items():
                    build_mlflow_config(config, v, f"{directory}.{k}")
            else:
                config[f"{directory}"] = data

        # Select the experiment before the first mlflow call starts a run.
        # MLFLOW_EXPERIMENT_ID, set by radT for its children, takes precedence.
        if "MLFLOW_EXPERIMENT_ID" not in os.environ and args.experiment_id:
            mlflow.set_experiment(experiment_id=str(args.experiment_id))

        mlflow.log_artifact(args.config_file_path, "pipeline")

        mlflow_config = {}
        build_mlflow_config(
            mlflow_config, benchmark_config.pipelines[args.pipeline_id], "pipeline"
        )
        mlflow.log_params(mlflow_config)

    # Starts the radT listeners when RADT_PRESENT is set; a no-op otherwise.
    with radt.run.RADTBenchmark():
        label = os.environ.get("SUITE_OUTPUT_LABEL", "")
        mlflow.set_tag(f"{TRACE_NAMESPACE}.label", label)
        if os.environ.get("SUITE_PROC_TRACE", "").lower() in ("1", "true", "yes"):
            _ar = mlflow.active_run()
            if _ar is not None:
                radt.trace.set_run(_ar.info.run_id)
        pipe_name = benchmark_config.pipelines[args.pipeline_id].name
        mlflow.set_tag("mlflow.runName",
                       f"{label} | {pipe_name}" if label else pipe_name)
        try:
            run_loadgen(benchmark_config.pipelines[args.pipeline_id], stop_marker)
        finally:
            if args.sync_dir:
                open(os.path.join(args.sync_dir, _slug(pipe_name)), "w").close()

        # Report while the run is still active.
        report_span_count()

    try:
        file_handler.flush()
        file_handler.close()
        logger.removeHandler(file_handler)
    except Exception:  # pylint: disable=broad-except
        pass

    # Interpreter shutdown can hang on telemetry sockets and backend teardown,
    # so flush everything explicitly and force-exit.
    kill_all_servers()
    flush_traces()
    report_span_count()
    radt.trace.shutdown()
    radt.shutdown()
    os._exit(0)


def main(args):
    process_start_time = time.time()

    with open(args.config_file_path, "r", encoding="utf-8") as file:
        yaml_config = file.read()
        benchmark_config = parse_yaml_raw_as(BenchmarkModel, yaml_config)

    print("listeners", benchmark_config.listeners)

    # Bounds cleanup if radt.schedule_external hangs after the runs finish.
    watchdog = OrchestratorWatchdog(
        experiment_id=args.experiment_id,
        parent_name=benchmark_config.name,
        pipeline_names=[p.name for p in benchmark_config.pipelines],
        process_start_time=process_start_time,
    )
    watchdog.start()

    logger_queue = Queue()
    logger = Logger(logger_queue, benchmark_config.name)

    names = {p.name for p in benchmark_config.pipelines}
    for p in benchmark_config.pipelines:
        if p.stop_with is not None and p.stop_with not in names:
            sys.exit(f"stop_with: no pipeline named {p.stop_with!r}")
    sync_dir = tempfile.mkdtemp(prefix="gulliver-")

    pipeline_ids = [i for i in range(len(benchmark_config.pipelines))]
    df_schedule = pd.DataFrame(np.empty(0, dtype=radt.constants.CSV_FORMAT))
    for pipeline_id in pipeline_ids:
        df_schedule.loc[pipeline_id] = pd.Series(
            {
                "Experiment": args.experiment_id,
                "Workload": 0,
                "Name": benchmark_config.pipelines[pipeline_id].name,
                "Status": "",
                "Run": "",
                "Devices": 0,
                "Collocation": benchmark_config.collocation,
                "Listeners": convert_listeners(benchmark_config.listeners).lower(),
                "File": "main.py",
                # radT re-executes main.py with these params, one process per
                # pipeline, so CLI overrides and a per-pipeline label go here.
                "Params": f"{args.config_file_path} -p {pipeline_id} --sync-dir {sync_dir}"
                          + (f" --label {args.label}_{_slug(benchmark_config.pipelines[pipeline_id].name)}"
                             if args.label else "")
                          + (f" --serialize {args.serialize_override}"
                             if args.serialize_override else "")
                          + (f" --pipeline-logs {args.pipeline_logs_override}"
                             if args.pipeline_logs_override else "")
                          + (f" --max-queries {args.max_queries_override}"
                             if args.max_queries_override is not None else ""),
            }
        )

    radt.schedule_external(
        [],
        df_schedule,
        group_name=benchmark_config.name,
    )
    watchdog.notify_schedule_returned()
    shutil.rmtree(sync_dir, ignore_errors=True)

    logger.stop_queue_listener()


if __name__ == "__main__":
    args = parse_args()
    if args.pipeline_id != -1:
        radt_entrypoint(args)
    else:
        main(args)
