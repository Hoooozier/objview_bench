from __future__ import annotations

import argparse
from pathlib import Path

from dummy_common import interaction_paths, read_done_if_available, wait_for_benchmark_step, wait_for_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for the first observation, then never submit an action.")
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--sleep-sec", type=float, default=3600.0)
    parser.add_argument("--wait-timeout-sec", type=float, default=120.0)
    parser.add_argument("--poll-interval-sec", type=float, default=0.01)
    args = parser.parse_args()

    config = wait_for_config(args.session_dir, args.wait_timeout_sec, args.poll_interval_sec)
    paths = interaction_paths(args.session_dir, config)
    wait_for_benchmark_step(
        session_dir=args.session_dir,
        current_step_path=paths["current_step"],
        episode_done_path=paths["episode_done"],
        episode_id=str(config["episode_id"]),
        timeout_sec=args.wait_timeout_sec,
        poll_interval_sec=args.poll_interval_sec,
    )
    import time

    start = time.perf_counter()
    while time.perf_counter() - start <= max(0.0, float(args.sleep_sec)):
        if read_done_if_available(paths["episode_done"], str(config["episode_id"])) is not None:
            return 0
        time.sleep(args.poll_interval_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

