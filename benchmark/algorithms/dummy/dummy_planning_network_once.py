from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dummy_common import (
    interaction_paths,
    submit_action,
    wait_for_benchmark_step,
    wait_for_config,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planning_network_client import request_planning_network_infer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue one planning-network RPC request during an episode, then stop.")
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--service-name", required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--wait-timeout-sec", type=float, default=120.0)
    parser.add_argument("--poll-interval-sec", type=float, default=0.01)
    args = parser.parse_args()

    episode_config = wait_for_config(args.session_dir, args.wait_timeout_sec, args.poll_interval_sec)
    episode_id = str(episode_config["episode_id"])
    paths = interaction_paths(args.session_dir, episode_config)

    current_step = wait_for_benchmark_step(
        session_dir=args.session_dir,
        current_step_path=paths["current_step"],
        episode_done_path=paths["episode_done"],
        episode_id=episode_id,
        timeout_sec=args.wait_timeout_sec,
        poll_interval_sec=args.poll_interval_sec,
    )
    if current_step is None:
        return 0

    request_name = f"{args.service_name}_infer_step{int(current_step['step_index']):03d}"
    rpc_start = time.perf_counter()
    result = request_planning_network_infer(
        session_dir=args.session_dir,
        service_name=args.service_name,
        input_npz=args.input_npz,
        request_id=request_name,
        topk=args.topk,
        wait_timeout_sec=args.wait_timeout_sec,
        poll_interval_sec=args.poll_interval_sec,
    )
    rpc_elapsed = time.perf_counter() - rpc_start
    runtime_total_sec = float(result.get("runtime", {}).get("total_sec", 0.0))

    action = {
        "action": "stop",
        "episode_id": episode_id,
        "step_index": int(current_step["step_index"]),
        "stop_reason": "plan_end",
        "algorithm_runtime_sec": runtime_total_sec,
        "submitted_at": time.time(),
        "debug_planning_network": {
            "service_name": args.service_name,
            "request_id": request_name,
            "rpc_elapsed_sec": rpc_elapsed,
            "service_runtime_sec": runtime_total_sec,
            "backend": result.get("backend"),
            "batch_size": result.get("batch_size"),
            "num_candidates": result.get("num_candidates"),
            "topk_indices": result.get("topk_indices"),
            "best_index": result.get("best_index"),
            "decodes": result.get("decodes"),
        },
    }
    submit_action(
        action_path=paths["action"],
        ready_algorithm_path=paths["ready_algorithm"],
        action=action,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
