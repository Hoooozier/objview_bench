from __future__ import annotations

import argparse
from pathlib import Path

from dummy_common import interaction_paths, wait_for_benchmark_step, wait_for_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Write malformed action.json and signal ready.")
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--wait-timeout-sec", type=float, default=120.0)
    parser.add_argument("--poll-interval-sec", type=float, default=0.01)
    args = parser.parse_args()

    config = wait_for_config(args.session_dir, args.wait_timeout_sec, args.poll_interval_sec)
    paths = interaction_paths(args.session_dir, config)
    step = wait_for_benchmark_step(
        session_dir=args.session_dir,
        current_step_path=paths["current_step"],
        episode_done_path=paths["episode_done"],
        episode_id=str(config["episode_id"]),
        timeout_sec=args.wait_timeout_sec,
        poll_interval_sec=args.poll_interval_sec,
    )
    if step is None:
        return 0
    paths["action"].parent.mkdir(parents=True, exist_ok=True)
    paths["action"].write_text("{ bad json\n", encoding="utf-8")
    paths["ready_algorithm"].parent.mkdir(parents=True, exist_ok=True)
    paths["ready_algorithm"].touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
