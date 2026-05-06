from __future__ import annotations

import argparse
import time
from pathlib import Path

from dummy_stop import main as stop_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit a stop action, then keep the process alive.")
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--hang-sec", type=float, default=3600.0)
    parser.add_argument("--wait-timeout-sec", type=float, default=120.0)
    parser.add_argument("--poll-interval-sec", type=float, default=0.01)
    args = parser.parse_args()

    stop_code = stop_main_args(
        [
            "--session-dir",
            str(args.session_dir),
            "--wait-timeout-sec",
            str(args.wait_timeout_sec),
            "--poll-interval-sec",
            str(args.poll_interval_sec),
        ]
    )
    if stop_code != 0:
        return stop_code
    time.sleep(max(0.0, float(args.hang_sec)))
    return 0


def stop_main_args(argv: list[str]) -> int:
    import sys

    old_argv = sys.argv
    try:
        sys.argv = [str(Path(__file__).with_name("dummy_stop.py")), *argv]
        return int(stop_main())
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
