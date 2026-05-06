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


def _wait_for_benchmark_step(
    *,
    session_dir: Path,
    current_step_path: Path,
    episode_done_path: Path,
    episode_id: str,
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
            if current_step is not None and str(current_step.get("episode_id")) == str(episode_id):
                _consume_ready(ready_path)
                return current_step
        if time.perf_counter() - start > timeout_sec:
            raise TimeoutError(f"Timed out waiting for benchmark ready signal at {ready_path}")
        time.sleep(poll_interval_sec)


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal algorithm that immediately stops.")
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--stop-reason", type=str, default="plan_end", choices=["plan_end", "candidate_exhausted"])
    parser.add_argument("--algorithm-runtime-sec", type=float, default=0.01)
    parser.add_argument("--wait-timeout-sec", type=float, default=120.0)
    parser.add_argument("--poll-interval-sec", type=float, default=0.01)
    args = parser.parse_args()

    episode_config_path = args.session_dir / "config" / "episode_config.json"
    current_step_path = args.session_dir / "state" / "current_step.json"

    _wait_for_file(episode_config_path, args.wait_timeout_sec, args.poll_interval_sec)
    episode_config = _read_json(episode_config_path)
    _publish_startup_ready(args.session_dir)
    episode_id = str(episode_config["episode_id"])

    interaction_paths = episode_config.get("interaction_paths", {})
    action_rel = interaction_paths.get("action_path", "actions/action.json")
    ready_rel = interaction_paths.get("ready_algorithm_path", "actions/ready_algorithm")
    done_rel = interaction_paths.get("episode_done_path", "state/episode_done.json")
    action_path = args.session_dir / Path(str(action_rel))
    ready_algorithm_path = args.session_dir / Path(str(ready_rel))
    episode_done_path = args.session_dir / Path(str(done_rel))

    current_step = _wait_for_benchmark_step(
        session_dir=args.session_dir,
        current_step_path=current_step_path,
        episode_done_path=episode_done_path,
        episode_id=episode_id,
        timeout_sec=args.wait_timeout_sec,
        poll_interval_sec=args.poll_interval_sec,
    )
    if current_step is None:
        return 0
    step_index = int(current_step["step_index"])

    action = {
        "action": "stop",
        "episode_id": episode_id,
        "step_index": step_index,
        "stop_reason": args.stop_reason,
        "algorithm_runtime_sec": float(args.algorithm_runtime_sec),
        "submitted_at": time.time(),
    }
    _write_json_atomic(action_path, action)
    ready_algorithm_path.parent.mkdir(parents=True, exist_ok=True)
    ready_algorithm_path.touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

