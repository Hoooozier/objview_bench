from __future__ import annotations

import argparse
import time
from pathlib import Path

from dummy_common import interaction_paths, submit_action, wait_for_benchmark_step, wait_for_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit move with malformed pose.")
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--algorithm-runtime-sec", type=float, default=0.01)
    parser.add_argument("--wait-timeout-sec", type=float, default=120.0)
    parser.add_argument("--poll-interval-sec", type=float, default=0.01)
    args = parser.parse_args()

    config = wait_for_config(args.session_dir, args.wait_timeout_sec, args.poll_interval_sec)
    episode_id = str(config["episode_id"])
    paths = interaction_paths(args.session_dir, config)
    step = wait_for_benchmark_step(
        session_dir=args.session_dir,
        current_step_path=paths["current_step"],
        episode_done_path=paths["episode_done"],
        episode_id=episode_id,
        timeout_sec=args.wait_timeout_sec,
        poll_interval_sec=args.poll_interval_sec,
    )
    if step is None:
        return 0
    submit_action(
        action_path=paths["action"],
        ready_algorithm_path=paths["ready_algorithm"],
        action={
            "action": "move",
            "episode_id": episode_id,
            "step_index": int(step["step_index"]),
            "pose": [1.0, 2.0, 3.0],
            "algorithm_runtime_sec": float(args.algorithm_runtime_sec),
            "submitted_at": time.time(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
