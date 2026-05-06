from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from api_feasibility import CameraPose, FeasibilityAPI


JSONRPC_VERSION = "2.0"
PROTOCOL_NAME = "objview_interaction"
PROTOCOL_VERSION = "v1"
POSE_FORMAT = "camera_lookat_roll"
SUPPORTED_METHODS = {
    "describe_interface",
    "is_feasible",
    "pose_to_matrix",
    "submit_action",
    "oracle_best_nsc_gain",
}


class InteractionSession:
    """
    Shared-folder benchmark/algorithm interaction boundary manager.

    Responsibilities:
    - Publish released benchmark-side files for the algorithm to read.
    - Expose a small, safe JSON-RPC surface for algorithm queries/actions.
    - Record interaction logs.

    Non-responsibilities:
    - Rendering new observations.
    - Evaluating metrics.
    - Advancing the episode or deciding stop policy.
    - Exposing GT geometry or other hidden benchmark internals.
    """

    def __init__(
        self,
        session_dir: str | Path,
        feasibility_api: FeasibilityAPI,
        *,
        action_filename: str = "action.json",
        oracle_request_handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.feasibility_api = feasibility_api
        self.action_filename = action_filename
        self.oracle_request_handler = oracle_request_handler

        self.config_dir = self.session_dir / "config"
        self.state_dir = self.session_dir / "state"
        self.observations_dir = self.session_dir / "observations"
        self.requests_dir = self.session_dir / "requests"
        self.responses_dir = self.session_dir / "responses"
        self.actions_dir = self.session_dir / "actions"
        self.logs_dir = self.session_dir / "logs"

        for directory in (
            self.config_dir,
            self.state_dir,
            self.observations_dir,
            self.requests_dir,
            self.responses_dir,
            self.actions_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.rpc_log_path = self.logs_dir / "rpc_log.jsonl"
        self.submitted_action_path = self.actions_dir / self.action_filename
        self.interface_json_path = self.config_dir / "interface.json"
        self.episode_config_path = self.config_dir / "episode_config.json"
        self.current_step_path = self.state_dir / "current_step.json"
        self.ready_benchmark_path = self.state_dir / "ready_benchmark"
        self.episode_done_path = self.state_dir / "episode_done.json"
        self.legacy_current_observation_path = self.state_dir / "current_observation.json"
        self.history_jsonl_path = self.state_dir / "history.jsonl"
        self.ready_algorithm_path = self.actions_dir / "ready_algorithm"

        if self.legacy_current_observation_path.exists():
            self.legacy_current_observation_path.unlink()

        self.publish_interface_description()

    def handle_request_file(self, request_path: str | Path) -> dict[str, Any]:
        request_path = Path(request_path)
        with request_path.open("r", encoding="utf-8") as f:
            request = json.load(f)

        response = self.handle_request(request)
        response_path = self.responses_dir / request_path.name
        self._write_json(response_path, response)
        self._touch(self._ready_path(response_path))
        return response

    def process_pending_requests(self) -> list[dict[str, Any]]:
        processed = []
        for request_ready_path in sorted(self.requests_dir.glob("*.ready")):
            request_path = self._json_path_from_ready(request_ready_path)
            response_path = self.responses_dir / request_path.name
            response_ready_path = self._ready_path(response_path)
            if response_ready_path.exists():
                try:
                    request_ready_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            try:
                response = self.handle_request_file(request_path)
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            try:
                request_ready_path.unlink()
            except FileNotFoundError:
                pass
            processed.append(
                {
                    "request_path": self._session_relpath(request_path),
                    "response_path": self._session_relpath(response_path),
                    "request_ready_path": self._session_relpath(request_ready_path),
                    "response_ready_path": self._session_relpath(response_ready_path),
                    "request_id": response.get("id"),
                    "has_error": "error" in response,
                }
            )
        return processed

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request.get("id")
        try:
            self._validate_jsonrpc_request(request)
            method = request["method"]
            params = request.get("params", {})

            if method == "describe_interface":
                result = self.describe_interface()
            elif method == "is_feasible":
                result = self._handle_is_feasible(params)
            elif method == "pose_to_matrix":
                result = self._handle_pose_to_matrix(params)
            elif method == "submit_action":
                result = self._handle_submit_action(params)
            elif method == "oracle_best_nsc_gain":
                if self.oracle_request_handler is None:
                    raise InteractionError(
                        "ORACLE_UNAVAILABLE",
                        "oracle_best_nsc_gain is only available for oracle-family methods.",
                    )
                result = self.oracle_request_handler(params)
            else:
                raise InteractionError("METHOD_NOT_FOUND", f"Unsupported method: {method}")

            response = {
                "jsonrpc": JSONRPC_VERSION,
                "id": request_id,
                "result": result,
            }
        except InteractionError as exc:
            response = self._error_response(request_id, exc.code, exc.message)
        except Exception as exc:
            response = self._error_response(request_id, "INTERNAL_ERROR", str(exc))

        self._append_log({"request": request, "response": response, "time": time.time()})
        return response

    def has_submitted_action(self) -> bool:
        return self.submitted_action_path.exists()

    def read_submitted_action(
        self,
        *,
        expected_step_index: int | None = None,
        expected_episode_id: str | None = None,
    ) -> dict[str, Any] | None:
        if not self.ready_algorithm_path.exists():
            return None
        if not self.has_submitted_action():
            return None
        try:
            with self.submitted_action_path.open("r", encoding="utf-8") as f:
                action = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        if not isinstance(action, dict):
            return None
        if expected_step_index is not None and "step_index" in action:
            if int(action["step_index"]) != int(expected_step_index):
                return None
        if expected_episode_id is not None and "episode_id" in action:
            if str(action["episode_id"]) != str(expected_episode_id):
                return None
        return action

    def clear_submitted_action(self) -> None:
        if self.submitted_action_path.exists():
            self.submitted_action_path.unlink()
        if self.ready_algorithm_path.exists():
            self.ready_algorithm_path.unlink()

    def clear_algorithm_ready(self) -> None:
        if self.ready_algorithm_path.exists():
            self.ready_algorithm_path.unlink()

    def clear_benchmark_ready(self) -> None:
        if self.ready_benchmark_path.exists():
            self.ready_benchmark_path.unlink()

    def describe_interface(self) -> dict[str, Any]:
        return {
            "protocol_name": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "pose_format": POSE_FORMAT,
            "methods": {
                "describe_interface": {
                    "description": "Return supported RPC methods, file layout, and protocol semantics.",
                },
                "is_feasible": {
                    "description": "Check whether one or more poses satisfy the current feasibility constraint.",
                    "supports_batch": True,
                    "pose_format": POSE_FORMAT,
                },
                "pose_to_matrix": {
                    "description": "Convert one or more benchmark poses to camera/world extrinsic matrices.",
                    "supports_batch": True,
                    "pose_format": POSE_FORMAT,
                },
                "submit_action": {
                    "description": "Submit exactly one final action for the current step.",
                    "allowed_actions": ["move", "stop"],
                    "pose_format": POSE_FORMAT,
                    "max_actions_per_step": 1,
                },
                "oracle_best_nsc_gain": {
                    "description": (
                        "Oracle-only method. Given candidate tammes view ids, return the candidate "
                        "with the largest one-step gain in Normalized Surface Coverage."
                    ),
                    "status": "oracle_family_only",
                    "ground_truth_dependent": True,
                    "params": {
                        "view_set": "tammes_360",
                        "candidate_view_ids": "list[int]",
                        "threshold": 0.01,
                        "include_scores": False,
                    },
                },
            },
            "files": {
                "interface": self._session_relpath(self.interface_json_path),
                "episode_config": self._session_relpath(self.episode_config_path),
                "current_step": self._session_relpath(self.current_step_path),
                "ready_benchmark": self._session_relpath(self.ready_benchmark_path),
                "episode_done": self._session_relpath(self.episode_done_path),
                "history": self._session_relpath(self.history_jsonl_path),
                "observations_root": self._session_relpath(self.observations_dir),
                "observation_manifest_pattern": "observations/step_{step_index:03d}.json",
                "requests_root": self._session_relpath(self.requests_dir),
                "responses_root": self._session_relpath(self.responses_dir),
                "request_ready_pattern": "requests/{request_name}.ready",
                "response_ready_pattern": "responses/{request_name}.ready",
                "action_output": self._session_relpath(self.submitted_action_path),
                "ready_algorithm": self._session_relpath(self.ready_algorithm_path),
                "rpc_log": self._session_relpath(self.rpc_log_path),
            },
            "filesystem_contract": {
                "benchmark_writes": [
                    "config/",
                    "state/",
                    "observations/",
                    "responses/",
                    "logs/",
                ],
                "algorithm_writes": [
                    "requests/",
                    "actions/",
                ],
                "algorithm_final_action_path": self._session_relpath(self.submitted_action_path),
                "algorithm_ready_path": self._session_relpath(self.ready_algorithm_path),
                "benchmark_managed_paths_are_read_only_for_algorithm": True,
            },
            "semantics": {
                "released_observations_only": True,
                "ground_truth_access": False,
                "benchmark_controls_episode_progression": True,
                "algorithm_controls_internal_state": True,
                "handshake": (
                    "Benchmark writes state/current_step.json and observations/step_XXX.json, "
                    "then creates state/ready_benchmark as a signal file. Algorithm writes "
                    "actions/action.json, then creates actions/ready_algorithm as a signal file. "
                    "For JSON-RPC calls, clients write requests/<name>.json, then create "
                    "requests/<name>.ready. The benchmark consumes the request ready file, writes "
                    "responses/<name>.json, then creates responses/<name>.ready. Clients consume "
                    "the response ready file after reading the response."
                ),
            },
            "optional_capabilities": {
                "shape_completion": {
                    "status": "metadata_only",
                    "description": (
                        "Static completion-backend metadata that algorithms may query from "
                        "interface.json / episode_config.json. This does not imply that a "
                        "shape-completion RPC service is currently running for the episode."
                    ),
                    "session_workspace_schema": {
                        "service_root": "shape_completion/",
                        "requests_dir": "shape_completion/requests/",
                        "responses_dir": "shape_completion/responses/",
                        "outputs_dir": "shape_completion/outputs/",
                        "ready_path": "shape_completion/service_ready",
                    },
                    "backends": {
                        "PoinTr-C": {
                            "num_input_points_model": 2048,
                            "num_output_points": 8192,
                        }
                    },
                }
            },
        }

    def publish_interface_description(self) -> dict[str, Any]:
        interface = self.describe_interface()
        self._write_json(self.interface_json_path, interface)
        return interface

    def publish_episode_config(self, episode_config: dict[str, Any]) -> dict[str, Any]:
        config = dict(episode_config)
        config.setdefault("protocol_name", PROTOCOL_NAME)
        config.setdefault("protocol_version", PROTOCOL_VERSION)
        config.setdefault("pose_format", POSE_FORMAT)
        config.setdefault(
            "interaction_paths",
            {
                "requests_dir": self._session_relpath(self.requests_dir),
                "responses_dir": self._session_relpath(self.responses_dir),
                "request_ready_suffix": ".ready",
                "response_ready_suffix": ".ready",
                "action_path": self._session_relpath(self.submitted_action_path),
                "ready_algorithm_path": self._session_relpath(self.ready_algorithm_path),
                "current_step_path": self._session_relpath(self.current_step_path),
                "ready_benchmark_path": self._session_relpath(self.ready_benchmark_path),
                "episode_done_path": self._session_relpath(self.episode_done_path),
                "observations_root": self._session_relpath(self.observations_dir),
            },
        )
        self._write_json(self.episode_config_path, config)
        return config

    def publish_observation_manifest(
        self,
        *,
        step_index: int,
        observation_manifest: dict[str, Any],
    ) -> dict[str, Any]:
        record = dict(observation_manifest)
        record["step_index"] = int(step_index)
        manifest_path = self.observations_dir / f"step_{int(step_index):03d}.json"
        self._write_json(manifest_path, record)
        return {
            "manifest_path": self._session_relpath(manifest_path),
            "record": record,
        }

    def publish_current_step(
        self,
        *,
        step_index: int,
        current_step_record: dict[str, Any],
    ) -> dict[str, Any]:
        record = dict(current_step_record)
        record["step_index"] = int(step_index)
        self._write_json(self.current_step_path, record)
        return record

    def publish_benchmark_ready(
        self,
        *,
        episode_id: str,
        step_index: int,
        visited_view_num: int,
    ) -> dict[str, Any]:
        record = {
            "episode_id": str(episode_id),
            "step_index": int(step_index),
            "visited_view_num": int(visited_view_num),
            "current_step_path": self._session_relpath(self.current_step_path),
            "ready_at": time.time(),
        }
        self._touch(self.ready_benchmark_path)
        return record

    def publish_episode_done(
        self,
        *,
        episode_id: str,
        terminal_source: str | None,
        terminal_reason: str | None,
        algorithm_stop_reason: str | None,
        benchmark_stop_reason: str | None,
        step_index: int | None,
        visited_view_num: int,
    ) -> dict[str, Any]:
        record = {
            "episode_id": str(episode_id),
            "terminal_source": terminal_source,
            "terminal_reason": terminal_reason,
            "algorithm_stop_reason": algorithm_stop_reason,
            "benchmark_stop_reason": benchmark_stop_reason,
            "step_index": None if step_index is None else int(step_index),
            "visited_view_num": int(visited_view_num),
            "done_at": time.time(),
        }
        self._write_json(self.episode_done_path, record)
        return record

    def append_history(self, entry: dict[str, Any]) -> dict[str, Any]:
        record = dict(entry)
        with self.history_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
        return record

    def _validate_jsonrpc_request(self, request: dict[str, Any]) -> None:
        if not isinstance(request, dict):
            raise InteractionError("INVALID_REQUEST", "Request must be a JSON object.")
        if request.get("jsonrpc") != JSONRPC_VERSION:
            raise InteractionError("INVALID_REQUEST", "jsonrpc must be '2.0'.")
        if "id" not in request:
            raise InteractionError("INVALID_REQUEST", "Missing request id.")
        if "method" not in request:
            raise InteractionError("INVALID_REQUEST", "Missing method.")
        if request["method"] not in SUPPORTED_METHODS:
            raise InteractionError("METHOD_NOT_FOUND", f"Unsupported method: {request['method']}")
        if "params" in request and not isinstance(request["params"], dict):
            raise InteractionError("INVALID_PARAMS", "params must be a JSON object.")

    def _handle_is_feasible(self, params: dict[str, Any]) -> dict[str, Any]:
        if "pose" in params and "poses" in params:
            raise InteractionError("INVALID_PARAMS", "Use either pose or poses, not both.")
        if "pose" in params:
            return self._single_feasibility_result(params["pose"])
        if "poses" in params:
            poses = params["poses"]
            if not isinstance(poses, list):
                raise InteractionError("INVALID_PARAMS", "poses must be a list.")
            return {"results": [self._single_feasibility_result(pose) for pose in poses]}
        raise InteractionError("INVALID_PARAMS", "Missing pose or poses.")

    def _single_feasibility_result(self, pose: Sequence[float]) -> dict[str, Any]:
        result = self.feasibility_api.check(pose_from_7d(pose))
        return {
            "feasible": bool(result.feasible),
            "reason": result.reason,
        }

    def _handle_pose_to_matrix(self, params: dict[str, Any]) -> dict[str, Any]:
        if "pose" in params and "poses" in params:
            raise InteractionError("INVALID_PARAMS", "Use either pose or poses, not both.")
        if "pose" in params:
            return self._single_pose_matrix_result(params["pose"])
        if "poses" in params:
            poses = params["poses"]
            if not isinstance(poses, list):
                raise InteractionError("INVALID_PARAMS", "poses must be a list.")
            return {"results": [self._single_pose_matrix_result(pose) for pose in poses]}
        raise InteractionError("INVALID_PARAMS", "Missing pose or poses.")

    def _single_pose_matrix_result(self, pose: Sequence[float]) -> dict[str, Any]:
        camera_pose = pose_from_7d(pose)
        camera_to_world, world_to_camera = pose_to_extrinsics(camera_pose)
        return {
            "pose_format": POSE_FORMAT,
            "camera_to_world": camera_to_world.tolist(),
            "world_to_camera": world_to_camera.tolist(),
        }

    def _handle_submit_action(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.has_submitted_action():
            raise InteractionError("ACTION_ALREADY_SUBMITTED", "Only one action may be submitted per step.")
        action = params.get("action")
        algorithm_runtime_sec = self._parse_algorithm_runtime(params)
        step_index = self._current_or_param_step_index(params)
        episode_id = self._current_episode_id()
        if action == "move":
            if "pose" not in params:
                raise InteractionError("INVALID_PARAMS", "move action requires pose.")
            pose = pose_to_7d(pose_from_7d(params["pose"]))
            action_record = {
                "action": "move",
                "episode_id": episode_id,
                "step_index": step_index,
                "pose": list(pose),
                "pose_format": POSE_FORMAT,
                "algorithm_runtime_sec": algorithm_runtime_sec,
                "submitted_at": time.time(),
            }
        elif action == "stop":
            action_record = {
                "action": "stop",
                "episode_id": episode_id,
                "step_index": step_index,
                "stop_reason": params.get("stop_reason", "plan_end"),
                "algorithm_runtime_sec": algorithm_runtime_sec,
                "submitted_at": time.time(),
            }
        else:
            raise InteractionError("INVALID_PARAMS", "action must be 'move' or 'stop'.")

        self._write_json(self.submitted_action_path, action_record)
        self._touch(self.ready_algorithm_path)
        return {
            "accepted": True,
            "action": action_record["action"],
            "action_path": self._session_relpath(self.submitted_action_path),
            "ready_algorithm_path": self._session_relpath(self.ready_algorithm_path),
        }

    def _session_relpath(self, path: Path) -> str:
        return str(path.relative_to(self.session_dir)).replace("\\", "/")

    @staticmethod
    def _parse_algorithm_runtime(params: dict[str, Any]) -> float | None:
        if "algorithm_runtime_sec" not in params:
            return None
        try:
            runtime = float(params["algorithm_runtime_sec"])
        except (TypeError, ValueError) as exc:
            raise InteractionError("INVALID_PARAMS", "algorithm_runtime_sec must be a number.") from exc
        if runtime < 0:
            raise InteractionError("INVALID_PARAMS", "algorithm_runtime_sec must be non-negative.")
        return runtime

    def _append_log(self, record: dict[str, Any]) -> None:
        with self.rpc_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _current_or_param_step_index(self, params: dict[str, Any]) -> int:
        if "step_index" in params:
            return int(params["step_index"])
        if not self.current_step_path.exists():
            raise InteractionError("INVALID_STATE", "current_step.json is not available.")
        current_step = self._read_json(self.current_step_path)
        return int(current_step["step_index"])

    def _current_episode_id(self) -> str:
        if not self.episode_config_path.exists():
            return ""
        config = self._read_json(self.episode_config_path)
        return str(config.get("episode_id", ""))

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise InteractionError("INVALID_JSON", f"Expected JSON object in {path}")
        return data

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=True)
            f.write("\n")
        tmp_path.replace(path)

    @staticmethod
    def _touch(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    @staticmethod
    def _ready_path(json_path: Path) -> Path:
        return json_path.with_suffix(json_path.suffix + ".ready")

    @staticmethod
    def _json_path_from_ready(ready_path: Path) -> Path:
        if ready_path.suffix != ".ready":
            raise ValueError(f"Expected .ready path, got {ready_path}")
        return ready_path.with_suffix("")

    @staticmethod
    def _error_response(request_id: Any, code: str, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "error": {
                "code": code,
                "message": message,
            },
        }


class InteractionError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def pose_from_7d(values: Sequence[float]) -> CameraPose:
    if len(values) != 7:
        raise InteractionError("INVALID_POSE", f"Expected 7 pose values, got {len(values)}.")
    vals = [float(v) for v in values]
    return CameraPose(
        camera_xyz=(vals[0], vals[1], vals[2]),
        lookat_xyz=(vals[3], vals[4], vals[5]),
        roll_rad=vals[6],
    )


def pose_to_7d(pose: CameraPose) -> tuple[float, float, float, float, float, float, float]:
    return (
        float(pose.camera_xyz[0]),
        float(pose.camera_xyz[1]),
        float(pose.camera_xyz[2]),
        float(pose.lookat_xyz[0]),
        float(pose.lookat_xyz[1]),
        float(pose.lookat_xyz[2]),
        float(pose.roll_rad),
    )


def pose_to_extrinsics(pose: CameraPose) -> tuple[np.ndarray, np.ndarray]:
    camera_pos = np.asarray(pose.camera_xyz, dtype=np.float32)
    look_at = np.asarray(pose.lookat_xyz, dtype=np.float32)

    cam_z = look_at - camera_pos
    cam_z = cam_z / max(np.linalg.norm(cam_z), 1e-12)
    if np.linalg.norm(cam_z - np.array([0.0, 0.0, -1.0], dtype=np.float32)) < 1e-6:
        cam_z = np.array([1e-8, 1e-8, -1.0], dtype=np.float32)
        cam_z = cam_z / max(np.linalg.norm(cam_z), 1e-12)
    if np.linalg.norm(cam_z - np.array([0.0, 0.0, 1.0], dtype=np.float32)) < 1e-6:
        cam_z = np.array([1e-8, 1e-8, 1.0], dtype=np.float32)
        cam_z = cam_z / max(np.linalg.norm(cam_z), 1e-12)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    cam_x_left = np.cross(-cam_z, world_up)
    cam_x_left = cam_x_left / max(np.linalg.norm(cam_x_left), 1e-12)
    cam_y_up = np.cross(cam_x_left, -cam_z)
    cam_y_up = cam_y_up / max(np.linalg.norm(cam_y_up), 1e-12)

    c = np.cos(float(pose.roll_rad)).astype(np.float32)
    s = np.sin(float(pose.roll_rad)).astype(np.float32)
    cam_x_left_0 = cam_x_left
    cam_y_up_0 = cam_y_up
    cam_x_left = c * cam_x_left_0 + s * cam_y_up_0
    cam_y_up = -s * cam_x_left_0 + c * cam_y_up_0

    cam_x_right = -cam_x_left
    cam_y_down = -cam_y_up

    camera_to_world = np.eye(4, dtype=np.float32)
    camera_to_world[:3, :3] = np.stack([cam_x_right, cam_y_down, cam_z], axis=1)
    camera_to_world[:3, 3] = camera_pos

    world_to_camera = np.eye(4, dtype=np.float32)
    world_to_camera[:3, :3] = camera_to_world[:3, :3].T
    world_to_camera[:3, 3] = -world_to_camera[:3, :3] @ camera_pos
    return camera_to_world, world_to_camera


def main() -> None:
    parser = argparse.ArgumentParser(description="Process or test JSON-RPC interaction requests.")
    parser.add_argument("--session-dir", type=Path, default=Path("interaction/session_debug"))
    parser.add_argument("--feasibility-json", type=Path, default=Path("configs/feasibility/whole.json"))
    parser.add_argument("--request-json", type=Path, default=None)
    parser.add_argument(
        "--run-default-tests",
        action="store_true",
        help="Run built-in checks for is_feasible, pose_to_matrix, and submit_action.",
    )
    args = parser.parse_args()

    session = InteractionSession(
        session_dir=args.session_dir,
        feasibility_api=FeasibilityAPI(args.feasibility_json),
    )

    if args.run_default_tests:
        _run_default_smoke_tests(session)
        return

    if args.request_json is None:
        parser.error("--request-json is required unless --run-default-tests is used")

    response = session.handle_request_file(args.request_json)
    print(json.dumps(response, indent=2, ensure_ascii=True))


def _run_default_smoke_tests(session: InteractionSession) -> None:
    session.clear_submitted_action()
    session.publish_episode_config(
        {
            "episode_id": "smoke_episode",
            "uid": "debug_uid",
            "task": {
                "type": "object_centric_active_3d_reconstruction",
                "evaluation_target": "geometry_only",
                "single_object": True,
            },
            "coordinate_frame": "object_normalized_frame",
            "depth_semantics": "standard_rgbd_z",
            "object_normalization": {
                "object_center": [0.0, 0.0, 0.0],
                "unit_sphere_radius": 1.0,
                "farthest_point_radius_le": 1.0,
            },
            "camera_intrinsics": {
                "image_width": 512,
                "image_height": 512,
                "fov_x_rad": 0.6981317,
                "fov_y_rad": 0.6981317,
                "principal_x": 256.0,
                "principal_y": 256.0,
            },
            "viewspace_constraint": {
                "name": "quarter",
            },
            "interaction": {
                "allowed_actions": ["move", "stop"],
                "max_actions_per_step": 1,
            },
            "algorithm_stop_reasons": [
                "plan_end",
                "candidate_exhausted",
                "next_view_visited",
                "next_view_infeasible",
            ],
            "benchmark_terminal_reasons": [
                "invalid_action",
                "max_cap_reached",
            ],
            "start_state": {
                "start_view_id": 0,
                "start_pose": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            },
        }
    )
    published_manifest = session.publish_observation_manifest(
        step_index=0,
        observation_manifest={
            "rgb_path": "render_cache/frames/debug_uid/tammes_128/view_000/rgb.png",
            "depth_path": "render_cache/frames/debug_uid/tammes_128/view_000/depth.npz",
            "mask_path": "render_cache/frames/debug_uid/tammes_128/view_000/mask.png",
            "frame_meta_path": "render_cache/frames/debug_uid/tammes_128/view_000/frame_meta.json",
        },
    )
    session.publish_current_step(
        step_index=0,
        current_step_record={
            "visited_view_num": 1,
            "current_pose": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "observation_manifest_path": published_manifest["manifest_path"],
        },
    )
    session.append_history(
        {
            "event": "observation_published",
            "step_index": 0,
            "visited_view_num": 1,
        }
    )

    requests = [
        {
            "name": "describe interface",
            "request": {
                "jsonrpc": JSONRPC_VERSION,
                "id": "test_describe_interface",
                "method": "describe_interface",
                "params": {},
            },
        },
        {
            "name": "single is_feasible",
            "request": {
                "jsonrpc": JSONRPC_VERSION,
                "id": "test_feasible_single",
                "method": "is_feasible",
                "params": {
                    "pose": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                },
            },
        },
        {
            "name": "batch is_feasible",
            "request": {
                "jsonrpc": JSONRPC_VERSION,
                "id": "test_feasible_batch",
                "method": "is_feasible",
                "params": {
                    "poses": [
                        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                },
            },
        },
        {
            "name": "single pose_to_matrix",
            "request": {
                "jsonrpc": JSONRPC_VERSION,
                "id": "test_matrix_single",
                "method": "pose_to_matrix",
                "params": {
                    "pose": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                },
            },
        },
        {
            "name": "batch pose_to_matrix",
            "request": {
                "jsonrpc": JSONRPC_VERSION,
                "id": "test_matrix_batch",
                "method": "pose_to_matrix",
                "params": {
                    "poses": [
                        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                },
            },
        },
        {
            "name": "submit move",
            "request": {
                "jsonrpc": JSONRPC_VERSION,
                "id": "test_submit_move",
                "method": "submit_action",
                "params": {
                    "action": "move",
                    "pose": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    "algorithm_runtime_sec": 0.01,
                },
            },
        },
        {
            "name": "submit stop",
            "request": {
                "jsonrpc": JSONRPC_VERSION,
                "id": "test_submit_stop",
                "method": "submit_action",
                "params": {
                    "action": "stop",
                    "algorithm_runtime_sec": 0.02,
                },
            },
        },
    ]

    for item in requests:
        response = session.handle_request(item["request"])
        status = "ok" if "result" in response else "error"
        print(f"[{status}] {item['name']}")
        print(json.dumps(response, indent=2, ensure_ascii=True))

    action = session.read_submitted_action()
    print("[submitted_action]")
    print(json.dumps(action, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

"""
python api_interaction.py --session-dir interaction/session_debug --feasibility-json configs/feasibility/quarter.json --run-default-tests
python api_interaction.py --session-dir interaction/session_debug --feasibility-json configs/feasibility/quarter.json --request-json path/to/request.json
"""
