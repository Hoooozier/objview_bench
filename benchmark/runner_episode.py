from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from api_evaluation import EvaluateAPI, points_from_rgbd_frame
from api_feasibility import FeasibilityAPI
from api_interaction import InteractionError, InteractionSession

if TYPE_CHECKING:
    from api_render import CameraIntrinsics, CameraPose, RGBDFrame, Render


ALGORITHM_STOP_REASONS = {
    "plan_end",
    "candidate_exhausted",
    "next_view_visited",
    "next_view_infeasible",
}
BENCHMARK_TERMINAL_REASONS = {"invalid_action", "max_cap_reached", "enough_info"}
BENCHMARK_STOP_REASONS = {"checkpoint_coverage_complete"}


@dataclass
class EpisodeConfig:
    episode_id: str
    uid: str
    obj_path: str | Path
    gt_pointcloud_path: str | Path
    start_pose: Any
    start_view_set: str = "benchmark_start_3"
    start_view_id: int = 0
    viewspace_constraint_name: str = "whole"
    method_name: str = "unknown_method"
    method_display_name: str | None = None
    family: str = "unknown_family"
    execution_mode_compatibility: str = "unknown"
    execution_mode: str = "unknown"
    budget_checkpoints: tuple[int, ...] = (5, 10, 30, 50)
    enable_iterative_enough_info_stop: bool = True
    max_visited_view_num: int = 129
    action_poll_interval_sec: float = 0.01
    action_wait_timeout_sec: float = 100.0
    camera_intrinsics: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class EpisodeRunnerError(RuntimeError):
    def __init__(self, message: str, *, exception_info: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.exception_info = exception_info or {"error": message}


class EpisodeRunner:
    def __init__(
        self,
        *,
        config: EpisodeConfig,
        render_api: Any,
        feasibility_api: FeasibilityAPI,
        evaluator: EvaluateAPI,
        interaction: InteractionSession,
        session_observations_root: str | Path,
    ) -> None:
        self.config = config
        self.render_api = render_api
        self.feasibility_api = feasibility_api
        self.evaluator = evaluator
        self.interaction = interaction
        self.session_observations_root = Path(session_observations_root)
        self.session_observations_root.mkdir(parents=True, exist_ok=True)

        self.has_started = False
        self.has_terminated = False
        self.current_step_index: int | None = None
        self.current_pose: CameraPose | None = None
        self.visited_view_num = 0
        self.last_action: dict[str, Any] | None = None
        self.algorithm_action_history: list[dict[str, Any]] = []
        self.terminal_source: str | None = None
        self.terminal_reason: str | None = None
        self.algorithm_stop_reason: str | None = None
        self.benchmark_stop_reason: str | None = None
        self.exception_info: dict[str, Any] | None = None
        self._session_metadata_published = False
        self.interaction.oracle_request_handler = self._handle_oracle_best_nsc_gain

    def run(self) -> dict[str, Any]:
        try:
            self._publish_initial_step()
            while True:
                action = self._wait_for_action()
                self.last_action = action
                self.interaction.clear_algorithm_ready()
                action_record = self._action_history_record(action)
                self.algorithm_action_history.append(action_record)
                self.interaction.append_history(action_record)

                if action.get("action") == "stop":
                    self._handle_stop(action)
                    return self._finalize()

                if action.get("action") != "move":
                    self.evaluator.record_algorithm_runtime(action.get("algorithm_runtime_sec"))
                    self._set_benchmark_terminal("invalid_action")
                    return self._finalize()

                should_finalize = self._handle_move(action)
                if should_finalize:
                    return self._finalize()
        except EpisodeRunnerError as exc:
            if self.exception_info is None:
                self.exception_info = exc.exception_info
            self._publish_exception_done()
            raise
        except Exception as exc:  # pragma: no cover - defensive
            self.exception_info = {
                "type": type(exc).__name__,
                "message": str(exc),
                "step_index": self.current_step_index,
            }
            raise EpisodeRunnerError("EpisodeRunner failed.", exception_info=self.exception_info) from exc

    def publish_session_metadata(self) -> None:
        if self._session_metadata_published:
            return
        self.interaction.publish_interface_description()
        self.interaction.publish_episode_config(self._episode_config_dict())
        self._session_metadata_published = True

    def _publish_initial_step(self) -> None:
        self.publish_session_metadata()

        render_start = time.perf_counter()
        frame = self.render_api.render_loaded(self.config.start_pose)
        render_sec = time.perf_counter() - render_start

        eval_start = time.perf_counter()
        step_metrics = self.evaluator.record_step(
            step_index=0,
            pose=self.config.start_pose,
            rgbd_frame=frame,
            benchmark_wall_time_sec=render_sec,
        )
        eval_sec = time.perf_counter() - eval_start

        self.current_step_index = 0
        self.current_pose = self.config.start_pose
        self.visited_view_num = int(step_metrics["visited_view_num"])

        manifest_path = self._publish_step_artifacts(step_index=0, pose=self.config.start_pose, frame=frame)
        self.interaction.clear_submitted_action()
        self.interaction.clear_benchmark_ready()
        self.interaction.publish_current_step(
            step_index=0,
            current_step_record={
                "episode_id": self.config.episode_id,
                "visited_view_num": self.visited_view_num,
                "current_pose": self._pose_to_7d(self.config.start_pose),
                "observation_manifest_path": manifest_path,
            },
        )
        self.interaction.publish_benchmark_ready(
            episode_id=self.config.episode_id,
            step_index=0,
            visited_view_num=self.visited_view_num,
        )
        self.interaction.append_history(
            {
                "event": "observation_published",
                "step_index": 0,
                "visited_view_num": self.visited_view_num,
                "render_sec": render_sec,
                "eval_sec": eval_sec,
            }
        )
        self.has_started = True

    def _wait_for_action(self) -> dict[str, Any]:
        start = time.perf_counter()
        while True:
            self._process_pending_rpc_requests()
            action = self.interaction.read_submitted_action(
                expected_step_index=int(self.current_step_index or 0),
                expected_episode_id=self.config.episode_id,
            )
            if action is not None:
                return action
            if time.perf_counter() - start > self.config.action_wait_timeout_sec:
                self.exception_info = {
                    "type": "action_wait_timeout",
                    "message": f"Timed out after {self.config.action_wait_timeout_sec:.3f}s waiting for action.",
                    "step_index": self.current_step_index,
                }
                raise EpisodeRunnerError("Timed out waiting for algorithm action.", exception_info=self.exception_info)
            time.sleep(self.config.action_poll_interval_sec)

    def _process_pending_rpc_requests(self) -> None:
        processed = self.interaction.process_pending_requests()
        for record in processed:
            self.interaction.append_history(
                {
                    "event": "rpc_request_processed",
                    "step_index": self.current_step_index,
                    **record,
                }
            )

    def _handle_oracle_best_nsc_gain(self, params: dict[str, Any]) -> dict[str, Any]:
        if str(self.config.family) != "oracle":
            raise InteractionError(
                "ORACLE_FORBIDDEN",
                "oracle_best_nsc_gain is only available when episode family is 'oracle'.",
            )
        if self.current_step_index is None:
            raise InteractionError("INVALID_STATE", "Episode has not published an initial observation yet.")

        view_set = str(params.get("view_set", "tammes_360"))
        threshold = float(params.get("threshold", 0.01))
        include_scores = bool(params.get("include_scores", False))
        candidate_view_ids = params.get("candidate_view_ids")
        if not isinstance(candidate_view_ids, list):
            raise InteractionError("INVALID_PARAMS", "candidate_view_ids must be a list of integer view ids.")

        current_nsc = self.evaluator.normalized_surface_coverage(threshold=threshold)
        best: dict[str, Any] | None = None
        scores: list[dict[str, Any]] = []

        for raw_view_id in candidate_view_ids:
            try:
                view_id = int(raw_view_id)
                pose = self._pose_from_render_cache_view(view_set=view_set, view_id=view_id)
            except Exception as exc:
                record = {
                    "view_id": int(raw_view_id) if isinstance(raw_view_id, int) else raw_view_id,
                    "status": "invalid_candidate",
                    "reason": str(exc),
                }
                if include_scores:
                    scores.append(record)
                continue

            feasibility = self.feasibility_api.check(pose)
            if not feasibility.feasible:
                record = {
                    "view_id": view_id,
                    "status": "infeasible",
                    "reason": feasibility.reason,
                }
                if include_scores:
                    scores.append(record)
                continue

            frame = self.render_api.render_loaded(pose)
            gain_info = self.evaluator.candidate_normalized_surface_coverage_gain(
                points_from_rgbd_frame(frame),
                threshold=threshold,
                current_nsc=current_nsc,
            )
            record = {
                "view_id": view_id,
                "status": "ok",
                "pose": self._pose_to_7d(pose),
                **gain_info,
            }
            if include_scores:
                scores.append(record)
            if best is None or (
                float(record["gain"]) > float(best["gain"]) + 1e-12
                or (
                    abs(float(record["gain"]) - float(best["gain"])) <= 1e-12
                    and int(record["view_id"]) < int(best["view_id"])
                )
            ):
                best = record

        result = {
            "step_index": int(self.current_step_index),
            "visited_view_num": int(self.visited_view_num),
            "view_set": view_set,
            "threshold": float(threshold),
            "current_nsc": float(current_nsc),
            "num_requested_candidates": int(len(candidate_view_ids)),
            "best": best,
        }
        if include_scores:
            result["scores"] = scores
        return result

    def _handle_stop(self, action: dict[str, Any]) -> None:
        self.evaluator.record_algorithm_runtime(action.get("algorithm_runtime_sec"))
        stop_reason = action.get("stop_reason", "plan_end")
        if stop_reason not in ALGORITHM_STOP_REASONS:
            self._set_benchmark_terminal("invalid_action")
            return
        self.terminal_source = "algorithm"
        self.terminal_reason = "algorithm_stop"
        self.algorithm_stop_reason = str(stop_reason)
        self.interaction.append_history(
            {
                "event": "algorithm_stop",
                "step_index": self.current_step_index,
                "algorithm_stop_reason": self.algorithm_stop_reason,
            }
        )

    def _handle_move(self, action: dict[str, Any]) -> bool:
        pose_values = action.get("pose")
        if not isinstance(pose_values, list) or len(pose_values) != 7:
            self.evaluator.record_algorithm_runtime(action.get("algorithm_runtime_sec"))
            self._set_benchmark_terminal("invalid_action")
            return True

        try:
            next_pose = self._pose_from_7d(pose_values)
        except Exception:
            self.evaluator.record_algorithm_runtime(action.get("algorithm_runtime_sec"))
            self._set_benchmark_terminal("invalid_action")
            return True

        feasibility = self.feasibility_api.check(next_pose)
        if not feasibility.feasible:
            self.evaluator.record_algorithm_runtime(action.get("algorithm_runtime_sec"))
            self.interaction.append_history(
                {
                    "event": "invalid_action",
                    "step_index": self.current_step_index,
                    "reason": feasibility.reason,
                }
            )
            self._set_benchmark_terminal("invalid_action")
            return True

        render_start = time.perf_counter()
        frame = self.render_api.render_loaded(next_pose)
        render_sec = time.perf_counter() - render_start

        next_step_index = int(self.current_step_index or 0) + 1
        step_metrics = self.evaluator.record_step(
            step_index=next_step_index,
            pose=next_pose,
            rgbd_frame=frame,
            algorithm_runtime_sec=action.get("algorithm_runtime_sec"),
            benchmark_wall_time_sec=render_sec,
        )

        self.current_step_index = next_step_index
        self.current_pose = next_pose
        self.visited_view_num = int(step_metrics["visited_view_num"])

        manifest_path = self._publish_step_artifacts(step_index=next_step_index, pose=next_pose, frame=frame)
        self.interaction.clear_submitted_action()
        self.interaction.clear_benchmark_ready()
        self.interaction.publish_current_step(
            step_index=next_step_index,
            current_step_record={
                "episode_id": self.config.episode_id,
                "visited_view_num": self.visited_view_num,
                "current_pose": self._pose_to_7d(next_pose),
                "observation_manifest_path": manifest_path,
            },
        )
        self.interaction.append_history(
            {
                "event": "move_accepted",
                "step_index": next_step_index,
                "visited_view_num": self.visited_view_num,
                "render_sec": render_sec,
            }
        )
        self._record_checkpoints()

        if self._should_stop_after_checkpoint_coverage():
            self._set_benchmark_terminal("enough_info", benchmark_stop_reason="checkpoint_coverage_complete")
            self.interaction.append_history(
                {
                    "event": "benchmark_stop",
                    "step_index": next_step_index,
                    "benchmark_stop_reason": self.benchmark_stop_reason,
                    "visited_view_num": self.visited_view_num,
                    "checkpoint_names": sorted(self.evaluator.checkpoints.keys()),
                }
            )
            return True

        if self.visited_view_num >= self.config.max_visited_view_num:
            self._set_benchmark_terminal("max_cap_reached")
            return True

        self.interaction.publish_benchmark_ready(
            episode_id=self.config.episode_id,
            step_index=next_step_index,
            visited_view_num=self.visited_view_num,
        )
        return False

    def _record_checkpoints(self) -> None:
        accepted_nbv_action_num = max(0, int(self.visited_view_num) - 1)
        if accepted_nbv_action_num in self.config.budget_checkpoints:
            self.evaluator.evaluate_checkpoint(
                f"K={accepted_nbv_action_num}",
                reason="fixed_budget",
            )
            self.interaction.append_history(
                {
                    "event": "checkpoint",
                    "name": f"K={accepted_nbv_action_num}",
                    "accepted_nbv_action_num": accepted_nbv_action_num,
                    "visited_view_num": self.visited_view_num,
                }
            )

        if self.evaluator.ms_history:
            stats = self.evaluator.ms_history[-1]["stats"]
            for ms_key in ("0.01", "0.02", "0.03"):
                key = ms_key
                if key in stats and stats[key].get("triggered_now"):
                    name = f"MS@{key}"
                    self.evaluator.evaluate_checkpoint(name, reason=name)
                    self.interaction.append_history(
                        {
                            "event": "checkpoint",
                            "name": name,
                            "visited_view_num": self.visited_view_num,
                        }
                    )

    def _publish_step_artifacts(self, *, step_index: int, pose: Any, frame: Any) -> str:
        refs = self._resolve_frame_file_refs(step_index=step_index, pose=pose, frame=frame)
        published = self.interaction.publish_observation_manifest(
            step_index=step_index,
            observation_manifest=refs,
        )
        return str(published["manifest_path"])

    def _resolve_frame_file_refs(self, *, step_index: int, pose: Any, frame: Any) -> dict[str, Any]:
        if frame.source == "cache":
            cache_refs = self._cache_entry_file_refs(pose)
            if cache_refs is not None:
                cache_refs["source"] = "render_cache"
                return cache_refs

        step_root = self.session_observations_root / f"step_{int(step_index):03d}"
        step_root.mkdir(parents=True, exist_ok=True)

        rgb_path = step_root / "rgb.png"
        depth_path = step_root / "depth.npz"
        mask_path = step_root / "mask.png"
        frame_meta_path = step_root / "frame_meta.json"

        Image.fromarray(np.asarray(frame.rgb, dtype=np.uint8)).save(rgb_path)
        np.savez_compressed(depth_path, depth=np.asarray(frame.depth, dtype=np.float32))
        Image.fromarray((np.asarray(frame.mask, dtype=bool).astype(np.uint8) * 255)).save(mask_path)
        from api_render import frame_meta_dict

        frame_meta_path.write_text(
            json.dumps(frame_meta_dict(frame), indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

        return {
            "source": "session_observations",
            "rgb_path": self._relpath(rgb_path),
            "depth_path": self._relpath(depth_path),
            "mask_path": self._relpath(mask_path),
            "frame_meta_path": self._relpath(frame_meta_path),
        }

    def _cache_entry_file_refs(self, pose: Any) -> dict[str, Any] | None:
        cache_index = getattr(self.render_api, "_cache_index", None)
        if cache_index is None:
            return None
        objects = cache_index.get("objects")
        if not isinstance(objects, dict):
            return None
        uid_entry = objects.get(self.config.uid)
        if not isinstance(uid_entry, dict):
            return None
        views = uid_entry.get("views")
        if not isinstance(views, list):
            return None
        parse_pose = getattr(self.render_api, "_parse_pose_entry", None)
        pose_equal = getattr(self.render_api, "_pose_equal", None)
        resolve_cache_path = getattr(self.render_api, "_resolve_cache_path", None)
        if parse_pose is None or pose_equal is None or resolve_cache_path is None:
            return None
        for entry in views:
            try:
                entry_pose = parse_pose(entry)
            except Exception:
                continue
            if not pose_equal(pose, entry_pose, atol=getattr(self.render_api, "pose_match_atol", 1e-4)):
                continue
            refs = {}
            for key in ("rgb", "depth", "mask", "frame_meta"):
                if key not in entry:
                    return None
                refs[f"{key}_path"] = self._relpath(resolve_cache_path(entry[key]))
            return refs
        return None

    def _finalize(self) -> dict[str, Any]:
        if self.terminal_reason is None:
            self._set_benchmark_terminal("invalid_action")

        terminal_checkpoint = self.evaluator.evaluate_checkpoint("terminal", reason=self.terminal_reason)
        self.interaction.clear_benchmark_ready()
        self.interaction.publish_episode_done(
            episode_id=self.config.episode_id,
            terminal_source=self.terminal_source,
            terminal_reason=self.terminal_reason,
            algorithm_stop_reason=self.algorithm_stop_reason,
            benchmark_stop_reason=self.benchmark_stop_reason,
            step_index=self.current_step_index,
            visited_view_num=self.visited_view_num,
        )
        self.interaction.append_history(
            {
                "event": "episode_terminated",
                "terminal_source": self.terminal_source,
                "terminal_reason": self.terminal_reason,
                "algorithm_stop_reason": self.algorithm_stop_reason,
                "benchmark_stop_reason": self.benchmark_stop_reason,
            }
        )
        self.has_terminated = True
        return {
            "episode_id": self.config.episode_id,
            "uid": self.config.uid,
            "method_name": self.config.method_name,
            "method_display_name": self.config.method_display_name or self.config.method_name,
            "family": self.config.family,
            "execution_mode_compatibility": self.config.execution_mode_compatibility,
            "execution_mode": self.config.execution_mode,
            "viewspace_constraint_name": self.config.viewspace_constraint_name,
            "start_view_id": self.config.start_view_id,
            "start_view_set": self.config.start_view_set,
            "start_pose": self._pose_to_7d(self.config.start_pose),
            "has_started": self.has_started,
            "has_terminated": self.has_terminated,
            "terminal_source": self.terminal_source,
            "terminal_reason": self.terminal_reason,
            "algorithm_stop_reason": self.algorithm_stop_reason,
            "benchmark_stop_reason": self.benchmark_stop_reason,
            "effective_stop_step": self.current_step_index,
            "effective_stop_view_num": self.visited_view_num,
            "accepted_nbv_action_num": max(0, int(self.visited_view_num) - 1),
            "current_step_index": self.current_step_index,
            "visited_view_num": self.visited_view_num,
            "max_visited_view_num": self.config.max_visited_view_num,
            "checkpoints": dict(self.evaluator.checkpoints),
            "terminal_checkpoint": terminal_checkpoint,
            "evaluator_summary": self.evaluator.summary(),
            "algorithm_action_history": list(self.algorithm_action_history),
            "exception_info": self.exception_info,
        }

    def _publish_exception_done(self) -> None:
        reason = "episode_runner_error"
        if isinstance(self.exception_info, dict) and self.exception_info.get("type"):
            reason = str(self.exception_info["type"])
        try:
            self.interaction.clear_benchmark_ready()
            self.interaction.clear_algorithm_ready()
            self.interaction.publish_episode_done(
                episode_id=self.config.episode_id,
                terminal_source="exception",
                terminal_reason=reason,
                algorithm_stop_reason=None,
                benchmark_stop_reason=None,
                step_index=self.current_step_index,
                visited_view_num=self.visited_view_num,
            )
            self.interaction.append_history(
                {
                    "event": "episode_exception",
                    "terminal_source": "exception",
                    "terminal_reason": reason,
                    "exception_info": self.exception_info,
                }
            )
        except Exception:
            pass

    def _set_benchmark_terminal(self, reason: str, *, benchmark_stop_reason: str | None = None) -> None:
        if reason not in BENCHMARK_TERMINAL_REASONS:
            raise ValueError(f"Unsupported benchmark terminal reason: {reason}")
        if benchmark_stop_reason is not None and benchmark_stop_reason not in BENCHMARK_STOP_REASONS:
            raise ValueError(f"Unsupported benchmark stop reason: {benchmark_stop_reason}")
        self.terminal_source = "benchmark"
        self.terminal_reason = str(reason)
        self.algorithm_stop_reason = None
        self.benchmark_stop_reason = benchmark_stop_reason

    def _action_history_record(self, action: dict[str, Any]) -> dict[str, Any]:
        record = {
            "event": "action_received",
            "step_index": self.current_step_index,
            "action": action.get("action"),
        }
        for key in (
            "oracle_source",
            "oracle_selected_view_id",
            "oracle_gain",
            "oracle_current_nsc",
            "oracle_candidate_nsc",
            "rollout_mapped_start_view_id",
            "rollout_mapped_start_dot",
            "rollout_replay_cursor",
        ):
            if key in action:
                record[key] = action[key]
        return record

    def _is_iterative_method(self) -> bool:
        compatibility = str(self.config.execution_mode_compatibility or "").strip().lower()
        return compatibility in {"iterative", "both", "iterative_or_both"}

    def _required_checkpoint_names(self) -> set[str]:
        return {f"K={int(k)}" for k in self.config.budget_checkpoints if int(k) in {5, 10, 30}} | {
            "MS@0.01",
            "MS@0.02",
            "MS@0.03",
        }

    def _should_stop_after_checkpoint_coverage(self) -> bool:
        if not self.config.enable_iterative_enough_info_stop:
            return False
        if not self._is_iterative_method():
            return False
        checkpoint_names = set(self.evaluator.checkpoints.keys())
        return self._required_checkpoint_names().issubset(checkpoint_names)

    def _episode_config_dict(self) -> dict[str, Any]:
        camera_intrinsics = self.config.camera_intrinsics
        if camera_intrinsics is None:
            camera_intrinsics = {
                "image_width": int(self.render_api.intrinsics.image_width),
                "image_height": int(self.render_api.intrinsics.image_height),
                "fov_x_rad": float(self.render_api.intrinsics.fov_x_rad),
                "fov_y_rad": float(self.render_api.intrinsics.fov_y_rad),
                "principal_x": float(self.render_api.intrinsics.principal_x),
                "principal_y": float(self.render_api.intrinsics.principal_y),
            }
        return {
            "episode_id": self.config.episode_id,
            "uid": self.config.uid,
            "task": {
                "type": "object_centric_active_3d_reconstruction",
                "evaluation_target": "geometry_only",
                "single_object": True,
            },
            "coordinate_frame": "object_normalized_frame",
            "camera_model": {
                "type": "pinhole",
                "distortion": "none",
                "depth_value": "camera_z",
                "depth_unit": "object_normalized_unit",
            },
            "depth_semantics": {
                "type": "camera_z",
                "unit": "object_normalized_unit",
                "invalid_value": 0,
                "background_value": 0,
            },
            "object_normalization": {
                "object_center": [0.0, 0.0, 0.0],
                "unit_sphere_radius": 1.0,
                "farthest_point_radius_le": 1.0,
            },
            "camera_intrinsics": camera_intrinsics,
            "viewspace_constraint": {
                "name": self.config.viewspace_constraint_name,
            },
            "interaction": {
                "allowed_actions": ["move", "stop"],
                "max_actions_per_step": 1,
                "budget_semantics": {
                    "fixed_budget_K": "number of algorithm-selected NBV actions after the standardized initial observation",
                    "views": "total acquired observations including the initial observation",
                    "max_visited_view_num": self.config.max_visited_view_num,
                    "budget_checkpoints": [int(k) for k in self.config.budget_checkpoints],
                },
            },
            "optional_capabilities": {
                "shape_completion": {
                    "status": "metadata_only",
                    "session_workspace": {
                        "service_root": self._relpath(self.interaction.session_dir / "shape_completion"),
                        "requests_dir": self._relpath(self.interaction.session_dir / "shape_completion" / "requests"),
                        "responses_dir": self._relpath(self.interaction.session_dir / "shape_completion" / "responses"),
                        "outputs_dir": self._relpath(self.interaction.session_dir / "shape_completion" / "outputs"),
                        "ready_path": self._relpath(self.interaction.session_dir / "shape_completion" / "service_ready"),
                    },
                    "backends": {
                        "PoinTr-C": {
                            "num_input_points_model": 2048,
                            "num_output_points": 8192,
                        }
                    },
                },
                "planning_network": {
                    "status": "metadata_only",
                    "session_workspace": {
                        "service_root": self._relpath(self.interaction.session_dir / "planning_network"),
                        "service_name_template": "planning_network/{service_name}",
                        "requests_dir_template": "planning_network/{service_name}/requests",
                        "responses_dir_template": "planning_network/{service_name}/responses",
                        "outputs_dir_template": "planning_network/{service_name}/outputs",
                        "ready_path_template": "planning_network/{service_name}/service_ready",
                    },
                    "protocol": "objview_planning_network",
                    "methods": ["infer"],
                    "backends": {
                        "MASCVP": {
                            "output_semantics": "scores over view candidates plus optional gamma decode",
                        },
                        "BENBV": {
                            "output_semantics": "scores over boundary candidates; highest score is the next candidate",
                        },
                    },
                },
            },
            "algorithm_stop_reasons": sorted(ALGORITHM_STOP_REASONS),
            "benchmark_terminal_reasons": sorted(BENCHMARK_TERMINAL_REASONS),
            "start_state": {
                "start_view_set": self.config.start_view_set,
                "start_view_id": self.config.start_view_id,
                "start_pose": self._pose_to_7d(self.config.start_pose),
            },
            "method_name": self.config.method_name,
            "family": self.config.family,
            "enable_iterative_enough_info_stop": self.config.enable_iterative_enough_info_stop,
            **self.config.extra,
        }

    @staticmethod
    def _pose_to_7d(pose: Any) -> list[float]:
        return [
            float(pose.camera_xyz[0]),
            float(pose.camera_xyz[1]),
            float(pose.camera_xyz[2]),
            float(pose.lookat_xyz[0]),
            float(pose.lookat_xyz[1]),
            float(pose.lookat_xyz[2]),
            float(pose.roll_rad),
        ]

    @staticmethod
    def _pose_from_7d(values: list[float]) -> Any:
        from api_render import CameraPose

        vals = [float(v) for v in values]
        if len(vals) != 7:
            raise ValueError(f"Expected 7 pose values, got {len(vals)}")
        return CameraPose(
            camera_xyz=(vals[0], vals[1], vals[2]),
            lookat_xyz=(vals[3], vals[4], vals[5]),
            roll_rad=vals[6],
        )

    def _pose_from_render_cache_view(self, *, view_set: str, view_id: int) -> Any:
        cache_index = getattr(self.render_api, "_cache_index", None)
        if cache_index is None:
            raise RuntimeError("render_api has no loaded cache index")
        objects = cache_index.get("objects")
        if not isinstance(objects, dict):
            raise ValueError("cache index must contain an objects dict")
        uid_entry = objects.get(self.config.uid)
        if not isinstance(uid_entry, dict):
            raise KeyError(f"uid={self.config.uid} not found in cache index")
        views = uid_entry.get("views")
        if not isinstance(views, list):
            raise ValueError(f"cache index views missing for uid={self.config.uid}")
        parse_pose = getattr(self.render_api, "_parse_pose_entry", None)
        if parse_pose is None:
            raise RuntimeError("render_api does not expose _parse_pose_entry")
        for entry in views:
            if str(entry.get("view_set")) != str(view_set):
                continue
            if int(entry.get("view_idx", -1)) != int(view_id):
                continue
            return parse_pose(entry)
        raise KeyError(f"view_set={view_set!r}, view_id={view_id} not found in render cache")

    def _relpath(self, path: str | Path) -> str:
        path = Path(path).resolve()
        try:
            return str(path.relative_to(self.interaction.session_dir)).replace("\\", "/")
        except ValueError:
            return str(path).replace("\\", "/")


def _load_cache_index(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Cache index must be a JSON object: {path}")
    return data


def _resolve_start_pose(
    *,
    cache_index: dict[str, Any],
    uid: str,
    start_view_set: str,
    start_view_id: int,
    explicit_start_pose: list[float] | None,
) -> Any:
    if explicit_start_pose is not None:
        return EpisodeRunner._pose_from_7d([float(v) for v in explicit_start_pose])

    objects = cache_index.get("objects")
    if not isinstance(objects, dict):
        raise ValueError("cache index must contain an objects dict")
    uid_entry = objects.get(uid)
    if not isinstance(uid_entry, dict):
        raise KeyError(f"uid={uid} not found in cache index")
    views = uid_entry.get("views")
    if not isinstance(views, list):
        raise ValueError(f"cache index views missing for uid={uid}")

    for entry in views:
        if entry.get("view_set") != start_view_set:
            continue
        if int(entry.get("view_idx", -1)) != int(start_view_id):
            continue
        pose_dict = entry.get("pose")
        if not isinstance(pose_dict, dict):
            raise ValueError("cache view entry missing pose dict")
        return EpisodeRunner._pose_from_7d(
            [
                float(pose_dict["camera_xyz"][0]),
                float(pose_dict["camera_xyz"][1]),
                float(pose_dict["camera_xyz"][2]),
                float(pose_dict["lookat_xyz"][0]),
                float(pose_dict["lookat_xyz"][1]),
                float(pose_dict["lookat_xyz"][2]),
                float(pose_dict["roll_rad"]),
            ]
        )

    raise KeyError(
        f"Start view not found in cache index for uid={uid}, "
        f"view_set={start_view_set}, view_idx={start_view_id}"
    )


def _fallback_start_pose_from_view_set(
    *,
    start_view_set: str,
    start_view_id: int,
) -> Any:
    benchmark_root = Path(__file__).resolve().parent
    view_set_files = {
        "benchmark_start_3": benchmark_root / "Tammes_sphere" / "benchmark_start_3_xyz.txt",
        "tammes_128": benchmark_root / "Tammes_sphere" / "128_xyz.txt",
        "tammes_360": benchmark_root / "Tammes_sphere" / "360_xyz.txt",
    }
    view_set_path = view_set_files.get(str(start_view_set))
    if view_set_path is None:
        raise KeyError(
            f"Auto start-pose fallback does not know view_set={start_view_set!r}. "
            f"Known view sets: {sorted(view_set_files)}"
        )
    if not view_set_path.exists():
        raise FileNotFoundError(f"View-set file not found for auto start-pose fallback: {view_set_path}")

    xyzs: list[tuple[float, float, float]] = []
    with view_set_path.open("r", encoding="utf-8") as f:
        for line_idx, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"{view_set_path}:{line_idx} must contain exactly 3 floats, got {line!r}")
            xyzs.append(tuple(float(x) * 3.0 for x in parts))

    if not (0 <= int(start_view_id) < len(xyzs)):
        raise IndexError(
            f"start_view_id={start_view_id} out of range for view_set={start_view_set} "
            f"(num_views={len(xyzs)})"
        )
    camera_xyz = xyzs[int(start_view_id)]
    return EpisodeRunner._pose_from_7d(
        [
            float(camera_xyz[0]),
            float(camera_xyz[1]),
            float(camera_xyz[2]),
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )


def _resolve_start_pose_with_auto_fallback(
    *,
    cache_index: dict[str, Any],
    uid: str,
    start_view_set: str,
    start_view_id: int,
    explicit_start_pose: list[float] | None,
) -> Any:
    try:
        return _resolve_start_pose(
            cache_index=cache_index,
            uid=uid,
            start_view_set=start_view_set,
            start_view_id=start_view_id,
            explicit_start_pose=explicit_start_pose,
        )
    except KeyError as exc:
        fallback_pose = _fallback_start_pose_from_view_set(
            start_view_set=start_view_set,
            start_view_id=start_view_id,
        )
        print(
            "[WARN] Start pose missing from render cache index for "
            f"uid={uid}, view_set={start_view_set}, view_idx={start_view_id}. "
            "Falling back to geometric view-set pose and relying on Render(mode=auto) "
            "for the initial observation.",
            file=sys.stderr,
        )
        print(f"[WARN] Original cache lookup error: {exc}", file=sys.stderr)
        return fallback_pose


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one benchmark episode.")
    parser.add_argument("--episode-id", type=str, required=True)
    parser.add_argument("--uid", type=str, required=True)
    parser.add_argument("--obj-path", type=Path, required=True)
    parser.add_argument("--gt-pointcloud", type=Path, required=True)
    parser.add_argument("--cache-index-json", type=Path, default=Path("render_cache/cache_index.json"))
    parser.add_argument("--feasibility-json", type=Path, required=True)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--session-observations-root", type=Path, required=True)
    parser.add_argument("--start-pose", type=float, nargs=7, default=None)
    parser.add_argument("--start-view-id", type=int, default=0)
    parser.add_argument("--start-view-set", type=str, default="benchmark_start_3")
    parser.add_argument("--method-name", type=str, default="manual_method")
    parser.add_argument("--method-display-name", type=str, default=None)
    parser.add_argument("--family", type=str, default="manual_family")
    parser.add_argument("--execution-mode-compatibility", type=str, default="unknown")
    parser.add_argument("--execution-compatibility", type=str, default=None, help="Deprecated alias for --execution-mode-compatibility.")
    parser.add_argument("--execution-mode", type=str, default="unknown")
    parser.add_argument(
        "--enable-iterative-enough-info-stop",
        dest="enable_iterative_enough_info_stop",
        action="store_true",
        help="Enable benchmark-side enough-info stopping for iterative methods.",
    )
    parser.add_argument(
        "--disable-iterative-enough-info-stop",
        dest="enable_iterative_enough_info_stop",
        action="store_false",
        help="Disable benchmark-side enough-info stopping for iterative methods.",
    )
    parser.set_defaults(enable_iterative_enough_info_stop=True)
    parser.add_argument(
        "--max-visited-view-num",
        type=int,
        default=129,
        help="Safety cap on total acquired views, including the initial observation.",
    )
    parser.add_argument(
        "--max-cap",
        type=int,
        default=None,
        help="Deprecated alias for --max-visited-view-num.",
    )
    parser.add_argument("--action-wait-timeout-sec", type=float, default=100.0)
    parser.add_argument("--action-poll-interval-sec", type=float, default=0.01)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def main() -> int:
    args = _build_argparser().parse_args()
    from api_render import Render, intrinsics_from_dict

    cache_index = _load_cache_index(args.cache_index_json)
    intrinsics_data = cache_index.get("intrinsics")
    if not isinstance(intrinsics_data, dict):
        raise ValueError("cache index must contain intrinsics")
    intrinsics = intrinsics_from_dict(intrinsics_data)

    session_dir = args.session_dir.resolve()
    session_dir.mkdir(parents=True, exist_ok=True)

    feasibility_api = FeasibilityAPI(args.feasibility_json)
    render_api = Render(
        uid=args.uid,
        obj_path=args.obj_path,
        intrinsics=intrinsics,
        device=args.device,
        mode="auto",
        cache_index_json=args.cache_index_json,
    )
    interaction = InteractionSession(session_dir=session_dir, feasibility_api=feasibility_api)
    evaluator = EvaluateAPI(
        uid=args.uid,
        gt_pointcloud_path=args.gt_pointcloud,
        render_api=render_api,
        feasibility_api=feasibility_api,
    )

    config = EpisodeConfig(
        episode_id=args.episode_id,
        uid=args.uid,
        obj_path=args.obj_path,
        gt_pointcloud_path=args.gt_pointcloud,
        start_pose=_resolve_start_pose_with_auto_fallback(
            cache_index=cache_index,
            uid=args.uid,
            start_view_set=args.start_view_set,
            start_view_id=args.start_view_id,
            explicit_start_pose=None if args.start_pose is None else [float(v) for v in args.start_pose],
        ),
        start_view_set=args.start_view_set,
        start_view_id=args.start_view_id,
        viewspace_constraint_name=Path(args.feasibility_json).stem,
        method_name=args.method_name,
        method_display_name=args.method_display_name,
        family=args.family,
        execution_mode_compatibility=args.execution_compatibility or args.execution_mode_compatibility,
        execution_mode=args.execution_mode,
        enable_iterative_enough_info_stop=bool(args.enable_iterative_enough_info_stop),
        max_visited_view_num=args.max_visited_view_num if args.max_cap is None else args.max_cap,
        action_wait_timeout_sec=args.action_wait_timeout_sec,
        action_poll_interval_sec=args.action_poll_interval_sec,
    )

    runner = EpisodeRunner(
        config=config,
        render_api=render_api,
        feasibility_api=feasibility_api,
        evaluator=evaluator,
        interaction=interaction,
        session_observations_root=args.session_observations_root,
    )
    result = runner.run()

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    main()
