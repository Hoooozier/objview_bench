from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_DUMMY_DIR = Path(__file__).resolve().parents[1] / "dummy"
if str(_DUMMY_DIR) not in sys.path:
    sys.path.insert(0, str(_DUMMY_DIR))

from dummy_common import (  # noqa: E402
    interaction_paths,
    read_json,
    submit_action,
    wait_for_benchmark_step,
    wait_for_config,
    wait_for_file,
    write_json_atomic,
)


def _count_views(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            if raw.strip():
                count += 1
    if count <= 0:
        raise ValueError(f"No candidate views found in {path}")
    return count


def _rpc_call(
    *,
    session_dir: Path,
    request_name: str,
    request: dict[str, Any],
    timeout_sec: float,
    poll_interval_sec: float,
) -> dict[str, Any]:
    request_path = session_dir / "requests" / f"{request_name}.json"
    response_path = session_dir / "responses" / request_path.name
    request_ready_path = request_path.with_suffix(request_path.suffix + ".ready")
    response_ready_path = response_path.with_suffix(response_path.suffix + ".ready")

    for stale_path in (request_path, request_ready_path, response_path, response_ready_path):
        try:
            stale_path.unlink()
        except FileNotFoundError:
            pass

    write_json_atomic(request_path, request)
    request_ready_path.touch()
    wait_for_file(response_ready_path, timeout_sec, poll_interval_sec)
    response = read_json(response_path)
    try:
        response_ready_path.unlink()
    except FileNotFoundError:
        pass
    if "error" in response:
        raise RuntimeError(f"RPC {request_name} failed: {response['error']}")
    result = response.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"RPC {request_name} returned invalid response: {response}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Oracle greedy planner that repeatedly selects the tammes_128 view with max NSC@0.01 gain."
    )
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--views", type=Path, default=Path("Tammes_sphere/128_xyz.txt"))
    parser.add_argument("--view-set", type=str, default="tammes_128")
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--max-actions", type=int, default=360)
    parser.add_argument("--algorithm-runtime-sec", type=float, default=0.0)
    parser.add_argument("--wait-timeout-sec", type=float, default=3600.0)
    parser.add_argument("--poll-interval-sec", type=float, default=0.01)
    parser.add_argument("--include-scores", action="store_true")
    args = parser.parse_args()

    episode_config = wait_for_config(args.session_dir, args.wait_timeout_sec, args.poll_interval_sec)
    episode_id = str(episode_config["episode_id"])
    paths = interaction_paths(args.session_dir, episode_config)

    num_views = _count_views(args.views)
    remaining = set(range(num_views))
    submitted_actions = 0
    last_seen_step: int | None = None

    while True:
        current_step = wait_for_benchmark_step(
            session_dir=args.session_dir,
            current_step_path=paths["current_step"],
            episode_done_path=paths["episode_done"],
            episode_id=episode_id,
            timeout_sec=args.wait_timeout_sec,
            poll_interval_sec=args.poll_interval_sec,
            last_seen_step=last_seen_step,
        )
        if current_step is None:
            return 0

        step_index = int(current_step["step_index"])
        last_seen_step = step_index

        if submitted_actions >= int(args.max_actions) or not remaining:
            submit_action(
                action_path=paths["action"],
                ready_algorithm_path=paths["ready_algorithm"],
                action={
                    "action": "stop",
                    "episode_id": episode_id,
                    "step_index": step_index,
                    "stop_reason": "candidate_exhausted",
                    "algorithm_runtime_sec": float(args.algorithm_runtime_sec),
                    "submitted_at": time.time(),
                },
            )
            return 0

        request_id = f"oracle_best_nsc_gain_step{step_index:03d}"
        result = _rpc_call(
            session_dir=args.session_dir,
            request_name=request_id,
            request={
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "oracle_best_nsc_gain",
                "params": {
                    "view_set": str(args.view_set),
                    "threshold": float(args.threshold),
                    "candidate_view_ids": sorted(remaining),
                    "include_scores": bool(args.include_scores),
                },
            },
            timeout_sec=args.wait_timeout_sec,
            poll_interval_sec=args.poll_interval_sec,
        )
        best = result.get("best")
        if not isinstance(best, dict) or best.get("pose") is None:
            submit_action(
                action_path=paths["action"],
                ready_algorithm_path=paths["ready_algorithm"],
                action={
                    "action": "stop",
                    "episode_id": episode_id,
                    "step_index": step_index,
                    "stop_reason": "candidate_exhausted",
                    "algorithm_runtime_sec": float(args.algorithm_runtime_sec),
                    "submitted_at": time.time(),
                },
            )
            return 0

        view_id = int(best["view_id"])
        remaining.discard(view_id)
        submit_action(
            action_path=paths["action"],
            ready_algorithm_path=paths["ready_algorithm"],
            action={
                "action": "move",
                "episode_id": episode_id,
                "step_index": step_index,
                "pose": [float(v) for v in best["pose"]],
                "oracle_selected_view_id": view_id,
                "oracle_threshold": float(args.threshold),
                "oracle_gain": float(best.get("gain", 0.0)),
                "oracle_current_nsc": float(result.get("current_nsc", 0.0)),
                "oracle_candidate_nsc": float(best.get("candidate_nsc", 0.0)),
                "algorithm_runtime_sec": float(args.algorithm_runtime_sec),
                "submitted_at": time.time(),
            },
        )
        submitted_actions += 1


if __name__ == "__main__":
    raise SystemExit(main())
