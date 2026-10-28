#!/usr/bin/env python3
"""Hold a CPU latency QoS request (/dev/cpu_dma_latency) while a command runs (Linux).

Keeps deep CPU idle states off for the command's duration; the request lapses when
the file is closed. The device is root-only by default (e.g. `sudo chmod 666` it).

    python evaluation/common/cpu_latency_qos.py [--max-us N] [--require] -- <command...>
"""
import argparse
import os
import struct
import subprocess
import sys

DEV = "/dev/cpu_dma_latency"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-us", type=int, default=0, help="permitted wake-up latency (default 0)")
    ap.add_argument("--require", action="store_true", help="fail instead of warn when the device is not writable")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command to run under the QoS")
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command given (use -- <command...>)")
    fd = None
    try:
        fd = os.open(DEV, os.O_RDWR)
        os.write(fd, struct.pack("i", a.max_us))
        print(f"[cpu_latency_qos] holding {DEV} at {a.max_us} us for: {' '.join(cmd)}", flush=True)
    except OSError as e:
        msg = f"[cpu_latency_qos] cannot hold {DEV} ({e}); deep idle states stay ENABLED"
        if a.require:
            sys.exit(msg)
        print("!! " + msg, file=sys.stderr, flush=True)
    try:
        rc = subprocess.call(cmd)
    finally:
        if fd is not None:
            os.close(fd)
    sys.exit(rc)


if __name__ == "__main__":
    main()
