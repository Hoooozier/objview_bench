from __future__ import annotations

import argparse
import time
from pathlib import Path

from dummy_common import interaction_paths, load_pose_sequence, same_pose, submit_action, wait_for_benchmark_step, wait_for_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit several legal moves, then candidate_exhausted.")
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--cache-index-json", type=Path, required=True)
    parser.add_argument("--view-set", type=str, default="tammes_128")
    parser.add_argument("--num-moves", type=int, default=3)
    parser.add_argument("--algorithm-runtime-sec", type=float, default=0.01)
    parser.add_argument("--wait-timeout-sec", type=float, default=120.0)
    parser.add_argument("--poll-interval-sec", type=float, default=0.01)
    args = parser.parse_args()

    config = wait_for_config(args.session_dir, args.wait_timeout_sec, args.poll_interval_sec)
    episode_id = str(config["episode_id"])
    uid = str(config["uid"])
    paths = interaction_paths(args.session_dir, config)
    poses = load_pose_sequence(args.cache_index_json, uid, args.view_set)
    cursor = 0
    last_seen_step = None

    for move_idx in range(max(0, int(args.num_moves))):
        step = wait_for_benchmark_step(
            session_dir=args.session_dir,
            current_step_path=paths["current_step"],
            episode_done_path=paths["episode_done"],
            episode_id=episode_id,
            last_seen_step=last_seen_step,
            timeout_sec=args.wait_timeout_sec,
            poll_interval_sec=args.poll_interval_sec,
        )
        if step is None:
            return 0
        last_seen_step = int(step["step_index"])
        current_pose = [float(v) for v in step["current_pose"]]
        while cursor < len(poses) and same_pose(poses[cursor], current_pose):
            cursor += 1
        pose = poses[min(cursor, len(poses) - 1)]
        cursor += 1
        submit_action(
            action_path=paths["action"],
            ready_algorithm_path=paths["ready_algorithm"],
            action={
                "action": "move",
                "episode_id": episode_id,
                "step_index": last_seen_step,
                "pose": pose,
                "algorithm_runtime_sec": float(args.algorithm_runtime_sec),
                "submitted_at": time.time(),
            },
        )

    step = wait_for_benchmark_step(
        session_dir=args.session_dir,
        current_step_path=paths["current_step"],
        episode_done_path=paths["episode_done"],
        episode_id=episode_id,
        last_seen_step=last_seen_step,
        timeout_sec=args.wait_timeout_sec,
        poll_interval_sec=args.poll_interval_sec,
    )
    if step is None:
        return 0
    submit_action(
        action_path=paths["action"],
        ready_algorithm_path=paths["ready_algorithm"],
        action={
            "action": "stop",
            "episode_id": episode_id,
            "step_index": int(step["step_index"]),
            "stop_reason": "candidate_exhausted",
            "algorithm_runtime_sec": float(args.algorithm_runtime_sec),
            "submitted_at": time.time(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
