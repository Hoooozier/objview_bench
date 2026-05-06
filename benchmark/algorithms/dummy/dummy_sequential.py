from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)
        f.write("\n")
    tmp_path.replace(path)


def _wait_for_file(path: Path, timeout_sec: float, poll_interval_sec: float) -> None:
    start = time.perf_counter()
    while not path.exists():
        if time.perf_counter() - start > timeout_sec:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(poll_interval_sec)


def _consume_ready(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _publish_startup_ready(session_dir: Path) -> None:
    ready_path = session_dir / "actions" / "algorithm_started"
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    ready_path.touch()


def _read_done_if_available(path: Path, episode_id: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        done = _read_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if str(done.get("episode_id")) != str(episode_id):
        return None
    return done


def _wait_for_next_benchmark_step(
    *,
    session_dir: Path,
    current_step_path: Path,
    episode_done_path: Path,
    episode_id: str,
    last_seen_step: int | None,
    timeout_sec: float,
    poll_interval_sec: float,
) -> dict[str, Any] | None:
    ready_path = session_dir / "state" / "ready_benchmark"
    start = time.perf_counter()
    while True:
        if _read_done_if_available(episode_done_path, episode_id) is not None:
            return None
        if ready_path.exists():
            try:
                current_step = _read_json(current_step_path)
            except (FileNotFoundError, json.JSONDecodeError):
                current_step = None
            if current_step is not None:
                step_index = int(current_step.get("step_index", -1))
                if str(current_step.get("episode_id")) == str(episode_id) and step_index != last_seen_step:
                    _consume_ready(ready_path)
                    return current_step
        if time.perf_counter() - start > timeout_sec:
            raise TimeoutError(f"Timed out waiting for benchmark ready signal at {ready_path}")
        time.sleep(poll_interval_sec)


def _pose_to_7d(pose_dict: dict[str, Any]) -> list[float]:
    return [
        float(pose_dict["camera_xyz"][0]),
        float(pose_dict["camera_xyz"][1]),
        float(pose_dict["camera_xyz"][2]),
        float(pose_dict["lookat_xyz"][0]),
        float(pose_dict["lookat_xyz"][1]),
        float(pose_dict["lookat_xyz"][2]),
        float(pose_dict["roll_rad"]),
    ]


def _load_pose_sequence(cache_index_path: Path, uid: str, view_set: str) -> list[list[float]]:
    data = _read_json(cache_index_path)
    objects = data.get("objects")
    if not isinstance(objects, dict) or uid not in objects:
        raise KeyError(f"uid={uid} not found in cache index")
    views = objects[uid].get("views")
    if not isinstance(views, list):
        raise ValueError(f"cache index views missing for uid={uid}")

    selected = [entry for entry in views if entry.get("view_set") == view_set]
    selected.sort(key=lambda entry: int(entry.get("view_idx", 0)))
    return [_pose_to_7d(entry["pose"]) for entry in selected]


def _same_pose(a: list[float], b: list[float], atol: float = 1e-6) -> bool:
    return len(a) == len(b) == 7 and all(abs(float(x) - float(y)) <= atol for x, y in zip(a, b))


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit tammes_128 poses sequentially, then stop.")
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--cache-index-json", type=Path, required=True)
    parser.add_argument("--view-set", type=str, default="tammes_128")
    parser.add_argument("--algorithm-runtime-sec", type=float, default=0.01)
    parser.add_argument("--wait-timeout-sec", type=float, default=120.0)
    parser.add_argument("--poll-interval-sec", type=float, default=0.01)
    args = parser.parse_args()

    episode_config_path = args.session_dir / "config" / "episode_config.json"
    current_step_path = args.session_dir / "state" / "current_step.json"

    _wait_for_file(episode_config_path, args.wait_timeout_sec, args.poll_interval_sec)
    episode_config = _read_json(episode_config_path)
    _publish_startup_ready(args.session_dir)
    uid = str(episode_config["uid"])
    episode_id = str(episode_config["episode_id"])
    interaction_paths = episode_config.get("interaction_paths", {})
    action_rel = interaction_paths.get("action_path", "actions/action.json")
    ready_rel = interaction_paths.get("ready_algorithm_path", "actions/ready_algorithm")
    done_rel = interaction_paths.get("episode_done_path", "state/episode_done.json")
    action_path = args.session_dir / Path(str(action_rel))
    ready_algorithm_path = args.session_dir / Path(str(ready_rel))
    episode_done_path = args.session_dir / Path(str(done_rel))

    poses = _load_pose_sequence(args.cache_index_json, uid, args.view_set)
    if not poses:
        raise ValueError(f"No poses found for uid={uid}, view_set={args.view_set}")

    cursor = 0
    matched_current_pose = False
    last_seen_step: int | None = None
    while True:
        current_step = _wait_for_next_benchmark_step(
            session_dir=args.session_dir,
            current_step_path=current_step_path,
            episode_done_path=episode_done_path,
            episode_id=episode_id,
            last_seen_step=last_seen_step,
            timeout_sec=args.wait_timeout_sec,
            poll_interval_sec=args.poll_interval_sec,
        )
        if current_step is None:
            return 0
        step_index = int(current_step["step_index"])
        current_pose = [float(v) for v in current_step["current_pose"]]

        last_seen_step = step_index

        while cursor < len(poses) and _same_pose(poses[cursor], current_pose):
            matched_current_pose = True
            cursor += 1

        if not matched_current_pose and step_index == 0:
            cursor = 0

        if cursor >= len(poses):
            action = {
                "action": "stop",
                "episode_id": episode_id,
                "step_index": step_index,
                "stop_reason": "candidate_exhausted",
                "algorithm_runtime_sec": float(args.algorithm_runtime_sec),
                "submitted_at": time.time(),
            }
            _write_json_atomic(action_path, action)
            ready_algorithm_path.parent.mkdir(parents=True, exist_ok=True)
            ready_algorithm_path.touch()
            return 0

        action = {
            "action": "move",
            "episode_id": episode_id,
            "step_index": step_index,
            "pose": list(poses[cursor]),
            "algorithm_runtime_sec": float(args.algorithm_runtime_sec),
            "submitted_at": time.time(),
        }
        _write_json_atomic(action_path, action)
        ready_algorithm_path.parent.mkdir(parents=True, exist_ok=True)
        ready_algorithm_path.touch()
        cursor += 1


if __name__ == "__main__":
    raise SystemExit(main())

