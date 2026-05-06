from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)
        f.write("\n")
    tmp_path.replace(path)


def wait_for_file(path: Path, timeout_sec: float, poll_interval_sec: float) -> None:
    start = time.perf_counter()
    while not path.exists():
        if time.perf_counter() - start > timeout_sec:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(poll_interval_sec)


def wait_for_config(session_dir: Path, timeout_sec: float, poll_interval_sec: float) -> dict[str, Any]:
    config_path = session_dir / "config" / "episode_config.json"
    wait_for_file(config_path, timeout_sec, poll_interval_sec)
    config = read_json(config_path)
    publish_startup_ready(session_dir)
    return config


def read_done_if_available(path: Path, episode_id: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        done = read_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if str(done.get("episode_id")) != str(episode_id):
        return None
    return done


def consume_ready(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def publish_startup_ready(session_dir: Path) -> None:
    ready_path = session_dir / "actions" / "algorithm_started"
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    ready_path.touch()


def interaction_paths(session_dir: Path, episode_config: dict[str, Any]) -> dict[str, Path]:
    paths = episode_config.get("interaction_paths", {})
    return {
        "action": session_dir / Path(str(paths.get("action_path", "actions/action.json"))),
        "ready_algorithm": session_dir / Path(str(paths.get("ready_algorithm_path", "actions/ready_algorithm"))),
        "current_step": session_dir / Path(str(paths.get("current_step_path", "state/current_step.json"))),
        "episode_done": session_dir / Path(str(paths.get("episode_done_path", "state/episode_done.json"))),
    }


def shape_completion_paths(session_dir: Path, episode_config: dict[str, Any]) -> dict[str, Path]:
    capability = episode_config.get("optional_capabilities", {}).get("shape_completion", {})
    workspace = capability.get("session_workspace", {})
    return {
        "service_root": session_dir / Path(str(workspace.get("service_root", "shape_completion"))),
        "requests_dir": session_dir / Path(str(workspace.get("requests_dir", "shape_completion/requests"))),
        "responses_dir": session_dir / Path(str(workspace.get("responses_dir", "shape_completion/responses"))),
        "outputs_dir": session_dir / Path(str(workspace.get("outputs_dir", "shape_completion/outputs"))),
        "ready_path": session_dir / Path(str(workspace.get("ready_path", "shape_completion/service_ready"))),
    }


def wait_for_benchmark_step(
    *,
    session_dir: Path,
    current_step_path: Path,
    episode_done_path: Path,
    episode_id: str,
    timeout_sec: float,
    poll_interval_sec: float,
    last_seen_step: int | None = None,
) -> dict[str, Any] | None:
    ready_path = session_dir / "state" / "ready_benchmark"
    start = time.perf_counter()
    while True:
        if read_done_if_available(episode_done_path, episode_id) is not None:
            return None
        if ready_path.exists():
            try:
                current_step = read_json(current_step_path)
            except (FileNotFoundError, json.JSONDecodeError):
                current_step = None
            if current_step is not None:
                step_index = int(current_step.get("step_index", -1))
                if str(current_step.get("episode_id")) == str(episode_id) and step_index != last_seen_step:
                    consume_ready(ready_path)
                    return current_step
        if time.perf_counter() - start > timeout_sec:
            raise TimeoutError(f"Timed out waiting for benchmark ready signal at {ready_path}")
        time.sleep(poll_interval_sec)


def submit_action(
    *,
    action_path: Path,
    ready_algorithm_path: Path,
    action: dict[str, Any],
) -> None:
    write_json_atomic(action_path, action)
    ready_algorithm_path.parent.mkdir(parents=True, exist_ok=True)
    ready_algorithm_path.touch()


def pose_to_7d(pose_dict: dict[str, Any]) -> list[float]:
    return [
        float(pose_dict["camera_xyz"][0]),
        float(pose_dict["camera_xyz"][1]),
        float(pose_dict["camera_xyz"][2]),
        float(pose_dict["lookat_xyz"][0]),
        float(pose_dict["lookat_xyz"][1]),
        float(pose_dict["lookat_xyz"][2]),
        float(pose_dict["roll_rad"]),
    ]


def load_pose_sequence(cache_index_path: Path, uid: str, view_set: str) -> list[list[float]]:
    data = read_json(cache_index_path)
    objects = data.get("objects")
    if not isinstance(objects, dict) or uid not in objects:
        raise KeyError(f"uid={uid} not found in cache index")
    views = objects[uid].get("views")
    if not isinstance(views, list):
        raise ValueError(f"cache index views missing for uid={uid}")

    selected = [entry for entry in views if entry.get("view_set") == view_set]
    selected.sort(key=lambda entry: int(entry.get("view_idx", 0)))
    return [pose_to_7d(entry["pose"]) for entry in selected]


def same_pose(a: list[float], b: list[float], atol: float = 1e-6) -> bool:
    return len(a) == len(b) == 7 and all(abs(float(x) - float(y)) <= atol for x, y in zip(a, b))
