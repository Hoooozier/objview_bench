from __future__ import annotations

import argparse
import time
from pathlib import Path

from dummy_common import (
    interaction_paths,
    read_json,
    shape_completion_paths,
    submit_action,
    wait_for_benchmark_step,
    wait_for_config,
    wait_for_file,
    write_json_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue one shape-completion RPC request during an episode, then stop.")
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument(
        "--partial-pointcloud-path",
        type=Path,
        default=None,
        help="Explicit partial point cloud file to send to the shape completion service for this smoke test.",
    )
    parser.add_argument("--wait-timeout-sec", type=float, default=120.0)
    parser.add_argument("--poll-interval-sec", type=float, default=0.01)
    args = parser.parse_args()

    episode_config = wait_for_config(args.session_dir, args.wait_timeout_sec, args.poll_interval_sec)
    episode_id = str(episode_config["episode_id"])
    paths = interaction_paths(args.session_dir, episode_config)
    completion = shape_completion_paths(args.session_dir, episode_config)

    wait_for_file(completion["ready_path"], args.wait_timeout_sec, args.poll_interval_sec)

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

    if args.partial_pointcloud_path is None:
        raise RuntimeError(
            "dummy_shape_completion_once requires --partial-pointcloud-path for this smoke test. "
            "The benchmark observation manifest is not itself a point cloud file."
        )
    partial_path = args.partial_pointcloud_path.resolve()
    request_name = f"shape_complete_step{int(current_step['step_index']):03d}"
    request_path = completion["requests_dir"] / f"{request_name}.json"
    request_ready_path = request_path.with_suffix(request_path.suffix + ".ready")
    response_path = completion["responses_dir"] / request_path.name
    response_ready_path = response_path.with_suffix(response_path.suffix + ".ready")
    output_path = completion["outputs_dir"] / f"{request_name}_completed.pcd"

    request = {
        "jsonrpc": "2.0",
        "id": request_name,
        "method": "complete_shape",
        "params": {
            "partial_pointcloud_path": str(partial_path),
            "output_pointcloud_path": str(output_path),
        },
    }
    rpc_start = time.perf_counter()
    write_json_atomic(request_path, request)
    request_ready_path.touch()
    wait_for_file(response_ready_path, args.wait_timeout_sec, args.poll_interval_sec)
    response = read_json(response_path)
    response_ready_path.unlink(missing_ok=True)
    if "error" in response:
        raise RuntimeError(f"Shape completion RPC failed: {response['error']}")
    result = response.get("result", {})
    runtime_total_sec = float(result.get("runtime", {}).get("total_sec", 0.0))
    rpc_elapsed = time.perf_counter() - rpc_start

    action = {
        "action": "stop",
        "episode_id": episode_id,
        "step_index": int(current_step["step_index"]),
        "stop_reason": "plan_end",
        "algorithm_runtime_sec": runtime_total_sec,
        "submitted_at": time.time(),
        "debug_shape_completion": {
            "request_id": request_name,
            "rpc_elapsed_sec": rpc_elapsed,
            "service_runtime_sec": runtime_total_sec,
            "completed_pointcloud_path": result.get("completed_pointcloud_path"),
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
