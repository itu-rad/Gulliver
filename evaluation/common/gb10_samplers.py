#!/usr/bin/env python3
"""Machine-wide hardware samplers for NVIDIA GB10 collocation runs.

Writes sidecar CSVs with wall-clock timestamps: nvidia-smi (nvsmi), per-process
pmon, Nsight Systems GPU metrics (nsys), Arm PMU perf counters on one process
(perf) and process snapshots (pids). Stop with SIGTERM.

    python evaluation/common/gb10_samplers.py all --label <label> --dir <results> [--perf-pattern '-p 1']
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NSYS_CONFIG = HERE / "nsys_gb10_probe.config"
NVSMI_FIELDS = ("timestamp,power.draw,clocks.sm,pstate,utilization.gpu,utilization.memory,"
                "temperature.gpu,temperature.gpu.tlimit,clocks_event_reasons.active")
PERF_EVENTS = ("armv8_pmuv3_0/l2d_cache_refill/,armv8_pmuv3_1/l2d_cache_refill/,"
               "armv8_pmuv3_0/l2d_cache_wb/,armv8_pmuv3_1/l2d_cache_wb/,"
               "armv8_pmuv3_0/cpu_cycles/,armv8_pmuv3_1/cpu_cycles/")
PID_PATTERNS = ("main.py", "vllm", "VLLM", "nvidia-cuda-mps")


def _log(msg: str) -> None:
    print(f"[gb10_samplers {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
class _LineSampler:
    """Run a streaming command; hand each stdout line to `on_line` with a wall ts."""

    def __init__(self, cmd: list[str], out: Path, header: list[str], stderr_too=False):
        self.cmd, self.out, self.header = cmd, Path(out), header
        self.stderr_too = stderr_too
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._fh = None
        self._w = None
        self.rows = 0

    def parse(self, line: str, ts: float):
        raise NotImplementedError

    def start(self):
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.out, "w", newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow(self.header)
        self._fh.flush()
        self._proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT if self.stderr_too else subprocess.DEVNULL,
                                      text=True, bufsize=1)
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        return self

    def _pump(self):
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            ts = time.time()
            try:
                for row in self.parse(line.rstrip("\n"), ts) or ():
                    self._w.writerow(row)
                    self.rows += 1
            except Exception as e:  # never let a parse error kill the sampler
                _log(f"{self.__class__.__name__}: parse error {e!r} on {line[:80]!r}")
            if self.rows % 20 == 0:
                self._fh.flush()

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=5)
        if self._fh:
            self._fh.flush()
            self._fh.close()
        _log(f"{self.__class__.__name__}: {self.rows} rows -> {self.out}")


class NvsmiSampler(_LineSampler):
    def __init__(self, out, interval_s=1.0):
        cmd = ["nvidia-smi", f"--query-gpu={NVSMI_FIELDS}", "--format=csv,noheader,nounits",
               f"-lms", str(int(interval_s * 1000))]
        super().__init__(cmd, out, ["wall_ts", "smi_timestamp", "power_w", "sm_mhz", "pstate",
                                    "gpu_util_pct", "mem_util_pct_DEAD", "temp_c", "tlimit_c",
                                    "throttle_reasons"])

    def parse(self, line, ts):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9 or not parts[0]:
            return []
        return [[f"{ts:.3f}"] + parts[:9]]


class PmonSampler(_LineSampler):
    """nvidia-smi pmon -s um: one line per process per sample."""

    def __init__(self, out, interval_s=1.0):
        cmd = ["nvidia-smi", "pmon", "-s", "um", "-d", str(max(1, int(interval_s))), "-o", "T"]
        super().__init__(cmd, out, ["wall_ts", "smi_time", "gpu", "pid", "type", "sm_pct",
                                    "mem_pct_DEAD", "fb_mib", "command"])

    def parse(self, line, ts):
        if not line or line.lstrip().startswith("#"):
            return []
        p = line.split()
        # with -o T: HH:MM:SS gpu pid type sm mem enc dec jpg ofa fb ccpm command
        if len(p) < 13:
            return []
        t, gpu, pid, typ, sm, mem = p[0], p[1], p[2], p[3], p[4], p[5]
        fb, command = p[10], " ".join(p[12:])
        if pid == "-":
            return []
        return [[f"{ts:.3f}", t, gpu, pid, typ, sm, mem, fb, command]]


class PerfSampler(_LineSampler):
    """perf stat -I on one pid; perf writes its CSV to stderr."""

    def __init__(self, out, pid: int, interval_s=1.0):
        cmd = ["perf", "stat", "-p", str(pid), "-I", str(int(interval_s * 1000)), "-x", ",",
               "-e", PERF_EVENTS]
        super().__init__(cmd, out, ["wall_ts", "perf_time_s", "count", "unit", "event",
                                    "run_pct"], stderr_too=True)
        self.pid = pid

    def parse(self, line, ts):
        if not line or line.startswith("#"):
            return []
        p = line.split(",")
        if len(p) < 6:
            return []
        return [[f"{ts:.3f}", p[0].strip(), p[1].strip(), p[2].strip(), p[3].strip(),
                 p[5].strip() if len(p) > 5 else ""]]


# ---------------------------------------------------------------------------
class NsysSampler:
    """Nsight GPU-metrics session for the whole cell; CSV on stop."""

    def __init__(self, out: Path, config: Path = NSYS_CONFIG, hz: int = 100, cap_s: int = 7200,
                 keep: bool = False):
        self.out, self.config, self.hz, self.cap_s, self.keep = Path(out), Path(config), hz, cap_s, keep
        # Build paths by string, not with_suffix(): labels may contain dots.
        self.rep = Path(str(self.out)[:-4] if str(self.out).endswith(".csv") else str(self.out))
        self._proc: subprocess.Popen | None = None
        self.t_launch = None

    def start(self):
        if not self.config.exists():
            raise FileNotFoundError(self.config)
        for f in (Path(str(self.rep) + ".nsys-rep"), Path(str(self.rep) + ".sqlite")):
            if f.exists():
                f.unlink()
        self.out.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["nsys", "profile", "--gpu-metrics-devices=0", f"--gpu-metrics-set=file:{self.config}",
               f"--gpu-metrics-frequency={self.hz}", "--trace=none", "--sample=none",
               "--cpuctxsw=none", "-d", str(self.cap_s), "-o", str(self.rep), "sleep", str(self.cap_s)]
        # The previous cell's profiling session can take a while to release the GPU.
        for attempt in range(25):
            self.t_launch = time.time()
            self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            time.sleep(1.0)
            if self._proc.poll() is None:
                break
            err = self._proc.stderr.read() if self._proc.stderr else ""
            if "Already under profiling" not in err or attempt == 24:
                raise RuntimeError(f"nsys exited immediately rc={self._proc.returncode}: {err[-400:]}")
            time.sleep(5.0)
        _log(f"nsys sampling at {self.hz} Hz -> {self.rep}.nsys-rep")
        return self

    def stop(self):
        if self._proc is None:
            return
        if self._proc.poll() is None:
            # SIGINT makes nsys finalise the report; killing the launched `sleep` does not.
            self._proc.send_signal(signal.SIGINT)
            try:
                self._proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                _log("nsys did not finalise in 300 s after SIGINT; terminating")
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
        rep = Path(str(self.rep) + ".nsys-rep")
        if not rep.exists():
            _log(f"nsys: no report at {rep}; nothing exported")
            return
        self.export(rep, self.out, keep=self.keep)

    @staticmethod
    def export(rep: Path, out_csv: Path, keep: bool = False) -> int:
        sql = Path(str(rep)[:-len(".nsys-rep")] + ".sqlite") if str(rep).endswith(".nsys-rep") else rep.with_suffix(".sqlite")
        if sql.exists():
            sql.unlink()
        r = subprocess.run(["nsys", "export", "--type", "sqlite", "-o", str(sql), str(rep)],
                           capture_output=True, text=True)
        if r.returncode != 0 or not sql.exists():
            _log(f"nsys export failed rc={r.returncode}: {r.stderr[-300:]}")
            return 0
        n = NsysSampler.sqlite_to_csv(sql, out_csv)
        if not keep:
            rep.unlink(missing_ok=True)
            sql.unlink(missing_ok=True)
        _log(f"nsys: {n} samples -> {out_csv}")
        return n

    @staticmethod
    def sqlite_to_csv(sql: Path, out_csv: Path) -> int:
        con = sqlite3.connect(str(sql))
        c = con.cursor()
        epoch_ns = c.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
        names = dict(c.execute("select metricId, metricName from TARGET_INFO_GPU_METRICS").fetchall())
        cols = [names[k] for k in sorted(names)]
        ids = sorted(names)
        rows: dict[int, dict[int, float]] = {}
        for ts, mid, val in c.execute("select timestamp, metricId, value from GPU_METRICS"):
            rows.setdefault(ts, {})[mid] = val
        con.close()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["wall_ts", "session_ns"] + [_slug(cn) for cn in cols])
            for ts in sorted(rows):
                r = rows[ts]
                w.writerow([f"{(epoch_ns + ts) / 1e9:.4f}", ts] + [r.get(m, "") for m in ids])
        meta = Path(str(out_csv)[:-4] + ".meta.json") if str(out_csv).endswith(".csv") else out_csv.with_suffix(".meta.json")
        meta.write_text(json.dumps({"utcEpochNs": epoch_ns, "columns": cols, "samples": len(rows)}, indent=1))
        return len(rows)


def _slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return s.lower()


# ---------------------------------------------------------------------------
def snapshot_pids(patterns=PID_PATTERNS) -> list[dict]:
    r = subprocess.run(["ps", "-eo", "pid,ppid,etimes,comm,args"], capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines()[1:]:
        p = line.split(None, 4)
        if len(p) < 5:
            continue
        pid, ppid, et, comm, args = p
        if any(pat in args for pat in patterns) and "gb10_samplers" not in args:
            out.append({"pid": int(pid), "ppid": int(ppid), "etimes": int(et), "comm": comm,
                        "args": args[:300]})
    return out


class PidsSnapshotter:
    def __init__(self, out: Path, interval_s=30.0, perf_pattern: str | None = None,
                 perf_out: Path | None = None):
        self.out, self.interval_s = Path(out), interval_s
        self.perf_pattern, self.perf_out = perf_pattern, perf_out
        self.perf: PerfSampler | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.snaps: list[dict] = []

    def start(self):
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            procs = snapshot_pids()
            self.snaps.append({"wall_ts": time.time(), "procs": procs})
            self.out.write_text(json.dumps(self.snaps, indent=0))
            if self.perf_pattern and self.perf is None:
                # radt runs main.py in-process and its listeners are forks with the same
                # command line, so the pipeline process is the lowest matching pid.
                cand = sorted([p for p in procs if self.perf_pattern in p["args"] and "main.py" in p["args"]],
                              key=lambda p: p["pid"])
                if cand:
                    pid = cand[0]["pid"]
                    try:
                        self.perf = PerfSampler(self.perf_out, pid).start()
                        _log(f"perf attached to pid {pid} ({cand[0]['args'][:60]})")
                    except Exception as e:
                        _log(f"perf attach failed: {e!r}")
                        self.perf_pattern = None
            self._stop.wait(self.interval_s)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        if self.perf:
            self.perf.stop()
        self.snaps.append({"wall_ts": time.time(), "procs": snapshot_pids()})
        self.out.write_text(json.dumps(self.snaps, indent=0))


# ---------------------------------------------------------------------------
def run_until_signalled(samplers, duration: float = 0.0):
    stop = threading.Event()

    def _on_term(signum, frame):
        stop.set()

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    try:
        if duration > 0:
            stop.wait(duration)
        else:
            while not stop.is_set():
                stop.wait(3600)
    finally:
        for s in reversed(samplers):
            try:
                s.stop()
            except Exception as e:
                _log(f"stop failed for {s.__class__.__name__}: {e!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for m in ("nvsmi", "pmon", "nsys", "perf", "pids", "all"):
        s = sub.add_parser(m)
        s.add_argument("--out", help="output path (single sampler modes)")
        s.add_argument("--label", help="label for 'all': <dir>/<label>_<sampler>.csv")
        s.add_argument("--dir", default=".", help="output directory for 'all'")
        s.add_argument("--interval", type=float, default=1.0, help="seconds (nvsmi/pmon/perf/pids)")
        s.add_argument("--duration", type=float, default=0.0, help="stop after N s (0 = until SIGTERM)")
        s.add_argument("--hz", type=int, default=100, help="nsys sampling frequency")
        s.add_argument("--cap", type=int, default=7200, help="nsys session cap (s)")
        s.add_argument("--config", default=str(NSYS_CONFIG))
        s.add_argument("--keep-rep", action="store_true", help="keep the .nsys-rep and .sqlite")
        s.add_argument("--pid", type=int, help="process for perf mode")
        s.add_argument("--perf-pattern", default=None,
                       help="'all' mode: attach perf to the first main.py process whose args contain this")
        s.add_argument("--export", help="nsys mode: only export an existing .nsys-rep to --out")
    a = ap.parse_args()

    if a.mode == "nsys" and a.export:
        n = NsysSampler.export(Path(a.export), Path(a.out), keep=True)
        return 0 if n else 1

    samplers = []
    if a.mode == "all":
        if not a.label:
            ap.error("--label required for all")
        d = Path(a.dir)
        samplers.append(NvsmiSampler(d / f"{a.label}_nvsmi.csv", a.interval).start())
        samplers.append(PmonSampler(d / f"{a.label}_pmon.csv", a.interval).start())
        try:
            samplers.append(NsysSampler(d / f"{a.label}_nsys.csv", Path(a.config), a.hz, a.cap,
                                        a.keep_rep).start())
        except Exception as e:
            _log(f"nsys unavailable for this cell: {e!r}")
        samplers.append(PidsSnapshotter(d / f"{a.label}_pids.json", 30.0, a.perf_pattern,
                                        d / f"{a.label}_perf.csv").start())
    else:
        if not a.out:
            ap.error("--out required")
        if a.mode == "nvsmi":
            samplers.append(NvsmiSampler(a.out, a.interval).start())
        elif a.mode == "pmon":
            samplers.append(PmonSampler(a.out, a.interval).start())
        elif a.mode == "nsys":
            samplers.append(NsysSampler(a.out, Path(a.config), a.hz, a.cap, a.keep_rep).start())
        elif a.mode == "perf":
            if not a.pid:
                ap.error("--pid required")
            samplers.append(PerfSampler(a.out, a.pid, a.interval).start())
        elif a.mode == "pids":
            samplers.append(PidsSnapshotter(a.out, a.interval).start())
    _log(f"{a.mode}: {len(samplers)} sampler(s) running (SIGTERM to stop)")
    run_until_signalled(samplers, a.duration)
    return 0


if __name__ == "__main__":
    sys.exit(main())
