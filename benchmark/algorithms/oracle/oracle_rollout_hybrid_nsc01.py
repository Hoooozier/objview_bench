from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

_DUMMY_DIR = Path(__file__).resolve().parents[1] / "dummy"
if str(_DUMMY_DIR) not in sys.path:
    sys.path.insert(0, str(_DUMMY_DIR))

from dummy_common import (  # noqa: E402
    interaction_paths,
    load_pose_sequence,
    read_json,
    submit_action,
    wait_for_benchmark_step,
    wait_for_config,
    wait_for_file,
    write_json_atomic,
)


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


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(float(x) * float(x) for x in v))
    if n <= 1e-12:
        return [0.0, 0.0, 0.0]
    return [float(x) / n for x in v]


def _nearest_view_id(current_pose: list[float], poses: list[list[float]]) -> tuple[int, float]:
    current_dir = _unit([float(current_pose[0]), float(current_pose[1]), float(current_pose[2])])
    best_id = 0
    best_dot = -float("inf")
    for view_id, pose in enumerate(poses):
        view_dir = _unit([float(pose[0]), float(pose[1]), float(pose[2])])
        dot = sum(a * b for a, b in zip(current_dir, view_dir))
        if dot > best_dot:
            best_id = view_id
            best_dot = dot
    return best_id, best_dot


def _load_rollout_sequence(path: Path, start_view_id: int) -> list[int]:
    data = read_json(path)
    if int(data.get("expected_num_views", 0)) <= 0:
        raise ValueError(f"Invalid rollout expected_num_views in {path}")
    rollouts = data.get("rollouts")
    if not isinstance(rollouts, list):
        raise ValueError(f"Rollout JSON must contain a list field 'rollouts': {path}")

    chosen = None
    for rollout in rollouts:
        if isinstance(rollout, dict) and int(rollout.get("start_view_id", -1)) == int(start_view_id):
            chosen = rollout
            break
    if chosen is None:
        raise KeyError(f"start_view_id={start_view_id} not found in {path}")

    sequence: list[int] = []
    seen = {int(start_view_id)}
    steps = chosen.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"Rollout start_view_id={start_view_id} has no steps list in {path}")
    for step in steps:
        if not isinstance(step, dict):
            continue
        gain = int(step.get("oracle_nbv_gain", 0))
        if gain <= 0:
            break
        view_id = int(step["oracle_nbv_view_id"])
        if view_id in seen:
            continue
        sequence.append(view_id)
        seen.add(view_id)
    return sequence


def _deterministic_shuffle(ids: list[int], *, seed_text: str) -> list[int]:
    seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    return shuffled


def _stop_action(*, episode_id: str, step_index: int, reason: str, runtime_sec: float) -> dict[str, Any]:
    return {
        "action": "stop",
        "episode_id": episode_id,
        "step_index": int(step_index),
        "stop_reason": reason,
        "algorithm_runtime_sec": float(runtime_sec),
        "submitted_at": time.time(),
    }


def _move_action(
    *,
    episode_id: str,
    step_index: int,
    pose: list[float],
    view_id: int,
    source: str,
    runtime_sec: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    action = {
        "action": "move",
        "episode_id": episode_id,
        "step_index": int(step_index),
        "pose": [float(v) for v in pose],
        "oracle_selected_view_id": int(view_id),
        "oracle_source": str(source),
        "algorithm_runtime_sec": float(runtime_sec),
        "submitted_at": time.time(),
    }
    if extra:
        action.update(extra)
    return action


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a precomputed voxel-0.02 oracle rollout, then fill the remaining "
            "candidate views using a configurable fallback. The default fallback is "
            "ascending view-id order, which is cheap and transparent for full-space "
            "ceiling runs."
        )
    )
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--cache-index-json", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--view-set", type=str, default="tammes_128")
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--max-actions", type=int, default=128)
    parser.add_argument(
        "--fallback-mode",
        choices=("deterministic_shuffle", "sorted", "online_nsc01"),
        default="sorted",
    )
    parser.add_argument("--algorithm-runtime-sec", type=float, default=0.0)
    parser.add_argument("--wait-timeout-sec", type=float, default=3600.0)
    parser.add_argument("--poll-interval-sec", type=float, default=0.01)
    parser.add_argument("--include-online-scores", action="store_true")
    args = parser.parse_args()

    episode_config = wait_for_config(args.session_dir, args.wait_timeout_sec, args.poll_interval_sec)
    uid = str(episode_config["uid"])
    episode_id = str(episode_config["episode_id"])
    paths = interaction_paths(args.session_dir, episode_config)

    poses = load_pose_sequence(args.cache_index_json, uid, args.view_set)
    if not poses:
        raise ValueError(f"No poses found for uid={uid}, view_set={args.view_set}")

    rollout_path = args.rollout_root / f"{uid}.json"
    if not rollout_path.exists():
        raise FileNotFoundError(f"Rollout JSON not found: {rollout_path}")

    remaining = set(range(len(poses)))
    replay_sequence: list[int] | None = None
    replay_cursor = 0
    fallback_sequence: list[int] | None = None
    fallback_cursor = 0
    mapped_start_view_id: int | None = None
    mapped_start_dot: float | None = None
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

        if replay_sequence is None:
            current_pose = [float(v) for v in current_step["current_pose"]]
            mapped_start_view_id, mapped_start_dot = _nearest_view_id(current_pose, poses)
            # The nearest start only selects which offline rollout to replay; it is not
            # treated as a consumed tammes_128 candidate because the benchmark initial
            # pose may not be exactly that candidate.
            replay_sequence = _load_rollout_sequence(rollout_path, mapped_start_view_id)

        if submitted_actions >= int(args.max_actions) or not remaining:
            submit_action(
                action_path=paths["action"],
                ready_algorithm_path=paths["ready_algorithm"],
                action=_stop_action(
                    episode_id=episode_id,
                    step_index=step_index,
                    reason="candidate_exhausted",
                    runtime_sec=float(args.algorithm_runtime_sec),
                ),
            )
            return 0

        while replay_cursor < len(replay_sequence) and replay_sequence[replay_cursor] not in remaining:
            replay_cursor += 1

        if replay_cursor < len(replay_sequence):
            view_id = int(replay_sequence[replay_cursor])
            replay_cursor += 1
            remaining.discard(view_id)
            submit_action(
                action_path=paths["action"],
                ready_algorithm_path=paths["ready_algorithm"],
                action=_move_action(
                    episode_id=episode_id,
                    step_index=step_index,
                    pose=poses[view_id],
                    view_id=view_id,
                    source="voxel_rollout_0p02",
                    runtime_sec=float(args.algorithm_runtime_sec),
                    extra={
                        "rollout_path": str(rollout_path),
                        "rollout_mapped_start_view_id": int(mapped_start_view_id),
                        "rollout_mapped_start_dot": float(mapped_start_dot),
                        "rollout_replay_cursor": int(replay_cursor),
                    },
                ),
            )
            submitted_actions += 1
            continue

        if args.fallback_mode in ("deterministic_shuffle", "sorted"):
            if fallback_sequence is None:
                fallback_candidates = sorted(remaining)
                if args.fallback_mode == "deterministic_shuffle":
                    fallback_sequence = _deterministic_shuffle(
                        fallback_candidates,
                        seed_text=f"{uid}:{args.view_set}:{mapped_start_view_id}:remaining",
                    )
                else:
                    fallback_sequence = fallback_candidates

            while fallback_cursor < len(fallback_sequence) and fallback_sequence[fallback_cursor] not in remaining:
                fallback_cursor += 1

            if fallback_cursor >= len(fallback_sequence):
                submit_action(
                    action_path=paths["action"],
                    ready_algorithm_path=paths["ready_algorithm"],
                    action=_stop_action(
                        episode_id=episode_id,
                        step_index=step_index,
                        reason="candidate_exhausted",
                        runtime_sec=float(args.algorithm_runtime_sec),
                    ),
                )
                return 0

            view_id = int(fallback_sequence[fallback_cursor])
            fallback_cursor += 1
            remaining.discard(view_id)
            submit_action(
                action_path=paths["action"],
                ready_algorithm_path=paths["ready_algorithm"],
                action=_move_action(
                    episode_id=episode_id,
                    step_index=step_index,
                    pose=poses[view_id],
                    view_id=view_id,
                    source=f"{args.fallback_mode}_fallback",
                    runtime_sec=float(args.algorithm_runtime_sec),
                    extra={
                        "rollout_path": str(rollout_path),
                        "rollout_mapped_start_view_id": int(mapped_start_view_id),
                        "rollout_mapped_start_dot": float(mapped_start_dot),
                        "rollout_replay_cursor": int(replay_cursor),
                        "fallback_mode": str(args.fallback_mode),
                        "fallback_cursor": int(fallback_cursor),
                    },
                ),
            )
            submitted_actions += 1
            continue

        request_id = f"hybrid_online_nsc_gain_step{step_index:03d}"
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
                    "include_scores": bool(args.include_online_scores),
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
                action=_stop_action(
                    episode_id=episode_id,
                    step_index=step_index,
                    reason="plan_end",
                    runtime_sec=float(args.algorithm_runtime_sec),
                ),
            )
            return 0

        view_id = int(best["view_id"])
        remaining.discard(view_id)
        current_nsc = float(result.get("current_nsc", 0.0))
        submit_action(
            action_path=paths["action"],
            ready_algorithm_path=paths["ready_algorithm"],
            action=_move_action(
                episode_id=episode_id,
                step_index=step_index,
                pose=[float(v) for v in best["pose"]],
                view_id=view_id,
                source="online_nsc01_fallback",
                runtime_sec=float(args.algorithm_runtime_sec),
                extra={
                    "oracle_threshold": float(args.threshold),
                    "oracle_gain": float(best.get("gain", 0.0)),
                    "oracle_current_nsc": current_nsc,
                    "oracle_candidate_nsc": float(best.get("candidate_nsc", 0.0)),
                },
            ),
        )
        submitted_actions += 1


if __name__ == "__main__":
    raise SystemExit(main())
