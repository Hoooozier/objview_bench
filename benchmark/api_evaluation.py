from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np

try:
    from api_render import CameraPose
except ModuleNotFoundError:
    @dataclass(frozen=True)
    class CameraPose:
        camera_xyz: tuple[float, float, float]
        lookat_xyz: tuple[float, float, float]
        roll_rad: float = 0.0


DEFAULT_COVERAGE_THRESHOLDS = (0.01, 0.02, 0.03)


def _view_set_xyzs_from_geometry(view_set: str, *, radius: float = 3.0) -> list[tuple[float, float, float]]:
    benchmark_root = Path(__file__).resolve().parent
    view_set_files = {
        "benchmark_start_3": benchmark_root / "Tammes_sphere" / "benchmark_start_3_xyz.txt",
        "tammes_128": benchmark_root / "Tammes_sphere" / "128_xyz.txt",
        "tammes_360": benchmark_root / "Tammes_sphere" / "360_xyz.txt",
    }
    path = view_set_files.get(str(view_set))
    if path is None:
        raise KeyError(f"Unknown view_set for geometric pose fallback: {view_set!r}")
    xyzs: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_idx} must contain exactly 3 floats, got {line!r}")
            xyzs.append(tuple(float(x) * float(radius) for x in parts))
    return xyzs


class EvaluateAPI:
    """
    Episode-level metric accumulator for accepted benchmark observations.

    Responsibilities:
    - Fuse released observation point clouds.
    - Update Chamfer Distance, Surface Coverage, Normalized Surface Coverage,
      path cost, and runtime metrics.
    - Return per-step metrics and final summary dictionaries.

    Non-responsibilities:
    - Protocol control, feasibility checking, interaction logging, rendering new
      algorithm queries, stop reasons, invalid actions, or output path policy.
    """

    def __init__(
        self,
        uid: str,
        gt_pointcloud_path: str | Path,
        render_api: Any,
        *,
        observable_view_set: str = "tammes_360",
        coverage_thresholds: Sequence[float] = DEFAULT_COVERAGE_THRESHOLDS,
        unit_sphere_radius: float = 1.0,
        fusion_voxel_size: float = 0.005,
        feasibility_api: Any | None = None,
        observable_cache_root: str | Path | None = Path("eval_cache"),
        observable_cache_constraint: str | None = None,
        map_stabilization_voxel_sizes: Sequence[float] = DEFAULT_COVERAGE_THRESHOLDS,
        map_stabilization_tau: float = 0.01,
        map_stabilization_patience: int = 3,
        map_stabilization_skip_after_trigger: bool = True,
    ) -> None:
        self.uid = str(uid)
        self.gt_pointcloud_path = Path(gt_pointcloud_path)
        self.render_api = render_api
        self.observable_view_set = str(observable_view_set)
        self.coverage_thresholds = tuple(float(t) for t in coverage_thresholds)
        self.unit_sphere_radius = float(unit_sphere_radius)
        self.fusion_voxel_size = float(fusion_voxel_size)
        self.feasibility_api = feasibility_api
        self.observable_cache_root = None if observable_cache_root is None else Path(observable_cache_root)
        self.observable_cache_constraint = (
            str(observable_cache_constraint)
            if observable_cache_constraint is not None
            else infer_observable_cache_constraint(feasibility_api)
        )
        self.map_stabilization_voxel_sizes = tuple(float(v) for v in map_stabilization_voxel_sizes)
        self.map_stabilization_tau = float(map_stabilization_tau)
        self.map_stabilization_patience = int(map_stabilization_patience)
        self.map_stabilization_skip_after_trigger = bool(map_stabilization_skip_after_trigger)

        if self.unit_sphere_radius <= 0:
            raise ValueError("unit_sphere_radius must be positive")
        if self.fusion_voxel_size <= 0:
            raise ValueError("fusion_voxel_size must be positive")
        if any(t <= 0 for t in self.coverage_thresholds):
            raise ValueError("coverage_thresholds must be positive")
        if any(v <= 0 for v in self.map_stabilization_voxel_sizes):
            raise ValueError("map_stabilization_voxel_sizes must be positive")
        if self.map_stabilization_tau < 0:
            raise ValueError("map_stabilization_tau must be non-negative")
        if self.map_stabilization_patience <= 0:
            raise ValueError("map_stabilization_patience must be positive")

        self.num_observable_views = 0
        self.gt_pointcloud = load_pointcloud(self.gt_pointcloud_path)
        self.observable_reference_cache_metadata: dict[str, Any] = {}
        self.observable_reference_source = "built"
        cached_observable = self._load_observable_pointcloud_from_cache()
        if cached_observable is None:
            self.observable_pointcloud = self._build_observable_pointcloud()
            self.observable_surface_ratio = {
                _threshold_key(t): point_cloud_coverage(self.gt_pointcloud, self.observable_pointcloud, threshold=t)
                for t in self.coverage_thresholds
            }
        else:
            self.observable_pointcloud, self.observable_reference_cache_metadata = cached_observable
            self.observable_reference_source = "eval_cache"
            self.num_observable_views = int(
                self.observable_reference_cache_metadata.get("num_observable_views", 0)
            )
            cached_ratio = self.observable_reference_cache_metadata.get("observable_surface_ratio")
            if isinstance(cached_ratio, dict) and all(_threshold_key(t) in cached_ratio for t in self.coverage_thresholds):
                self.observable_surface_ratio = {
                    _threshold_key(t): float(cached_ratio[_threshold_key(t)])
                    for t in self.coverage_thresholds
                }
            else:
                self.observable_surface_ratio = {
                    _threshold_key(t): point_cloud_coverage(self.gt_pointcloud, self.observable_pointcloud, threshold=t)
                    for t in self.coverage_thresholds
                }

        self.visited_poses: list[list[float]] = []
        self.fused_observed_pointcloud: np.ndarray | None = None
        self.per_step_metrics: list[dict[str, Any]] = []
        self.total_path_cost = 0.0
        self.total_algorithm_runtime_sec = 0.0
        self.total_benchmark_wall_time_sec = 0.0
        self.checkpoints: dict[str, dict[str, Any]] = {}
        self._heavy_metric_cache_by_visited_view_num: dict[int, dict[str, Any]] = {}
        self.ms_previous_counts = {_threshold_key(v): None for v in self.map_stabilization_voxel_sizes}
        self.ms_stable_counts = {_threshold_key(v): 0 for v in self.map_stabilization_voxel_sizes}
        self.ms_trigger_step_indices = {_threshold_key(v): None for v in self.map_stabilization_voxel_sizes}
        self.ms_trigger_visited_view_nums = {_threshold_key(v): None for v in self.map_stabilization_voxel_sizes}
        self.ms_history: list[dict[str, Any]] = []

    def record_step(
        self,
        step_index: int,
        pose: CameraPose | Sequence[float],
        rgbd_frame: Any,
        *,
        algorithm_runtime_sec: float | None = None,
        benchmark_wall_time_sec: float | None = None,
        update_map_stabilization: bool = True,
    ) -> dict[str, Any]:
        camera_pose = coerce_pose(pose)
        pose_7d = pose_to_7d(camera_pose)
        observed_points = points_from_rgbd_frame(rgbd_frame)

        self.visited_poses.append(list(pose_7d))
        self.fused_observed_pointcloud = fuse_pointclouds(
            [self.fused_observed_pointcloud, observed_points],
            voxel_size=self.fusion_voxel_size,
        )

        path_cost_increment = 0.0
        if len(self.visited_poses) >= 2:
            prev_pose = pose_from_7d(self.visited_poses[-2])
            path_cost_increment = collision_avoid_unit_sphere_distance(
                np.asarray(prev_pose.camera_xyz, dtype=np.float64),
                np.asarray(camera_pose.camera_xyz, dtype=np.float64),
                radius=self.unit_sphere_radius,
            )
        self.total_path_cost += float(path_cost_increment)

        if algorithm_runtime_sec is not None:
            algorithm_runtime_sec = _nonnegative_float(algorithm_runtime_sec, "algorithm_runtime_sec")
            self.total_algorithm_runtime_sec += algorithm_runtime_sec
        if benchmark_wall_time_sec is not None:
            benchmark_wall_time_sec = _nonnegative_float(benchmark_wall_time_sec, "benchmark_wall_time_sec")
            self.total_benchmark_wall_time_sec += benchmark_wall_time_sec

        map_stabilization = None
        if update_map_stabilization:
            map_stabilization = self._record_map_stabilization(step_index)

        visited_view_num = len(self.visited_poses)
        step_metrics = {
            "uid": self.uid,
            "step_index": int(step_index),
            "visited_view_num": int(visited_view_num),
            "pose": list(pose_7d),
            "path_cost_increment": float(path_cost_increment),
            "total_path_cost": float(self.total_path_cost),
            "algorithm_runtime_sec": algorithm_runtime_sec,
            "total_algorithm_runtime_sec": float(self.total_algorithm_runtime_sec),
            "benchmark_wall_time_sec": benchmark_wall_time_sec,
            "total_benchmark_wall_time_sec": float(self.total_benchmark_wall_time_sec),
            "map_stabilization": map_stabilization,
            "num_observed_points": int(len(observed_points)),
            "num_fused_points": int(len(self.fused_observed_pointcloud)),
        }
        self.per_step_metrics.append(step_metrics)
        return step_metrics

    def record_algorithm_runtime(self, algorithm_runtime_sec: float | None) -> None:
        """Accumulate algorithm decision time for actions that do not add a new observation."""
        if algorithm_runtime_sec is None:
            return
        self.total_algorithm_runtime_sec += _nonnegative_float(
            algorithm_runtime_sec,
            "algorithm_runtime_sec",
        )

    def normalized_surface_coverage(self, *, threshold: float = 0.01) -> float:
        if self.fused_observed_pointcloud is None:
            raise RuntimeError("Cannot compute NSC before any observed point cloud is recorded")
        return point_cloud_coverage(
            self.observable_pointcloud,
            self.fused_observed_pointcloud,
            threshold=float(threshold),
        )

    def candidate_normalized_surface_coverage_gain(
        self,
        candidate_points: np.ndarray,
        *,
        threshold: float = 0.01,
        current_nsc: float | None = None,
    ) -> dict[str, Any]:
        if self.fused_observed_pointcloud is None:
            raise RuntimeError("Cannot compute candidate NSC gain before any observed point cloud is recorded")
        if current_nsc is None:
            current_nsc = self.normalized_surface_coverage(threshold=threshold)
        candidate_fused = fuse_pointclouds(
            [self.fused_observed_pointcloud, candidate_points],
            voxel_size=self.fusion_voxel_size,
        )
        candidate_nsc = point_cloud_coverage(
            self.observable_pointcloud,
            candidate_fused,
            threshold=float(threshold),
        )
        return {
            "current_nsc": float(current_nsc),
            "candidate_nsc": float(candidate_nsc),
            "gain": float(candidate_nsc - float(current_nsc)),
            "num_candidate_points": int(len(candidate_points)),
            "num_candidate_fused_points": int(len(candidate_fused)),
        }

    def evaluate_checkpoint(
        self,
        name: str,
        *,
        reason: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        if self.fused_observed_pointcloud is None:
            raise RuntimeError("Cannot evaluate checkpoint before any observed point cloud is recorded")

        name = str(name)
        if name in self.checkpoints and not overwrite:
            return self.checkpoints[name]

        visited_view_num = int(len(self.visited_poses))
        heavy_metrics = self._heavy_metric_cache_by_visited_view_num.get(visited_view_num)
        heavy_metrics_source = "cache"
        if heavy_metrics is None:
            heavy_metrics_source = "computed"
            heavy_metrics = {
                "chamfer_distance": chamfer_distance_np(self.gt_pointcloud, self.fused_observed_pointcloud),
                "surface_coverage": {
                    _threshold_key(t): point_cloud_coverage(self.gt_pointcloud, self.fused_observed_pointcloud, threshold=t)
                    for t in self.coverage_thresholds
                },
                "normalized_surface_coverage": {
                    _threshold_key(t): point_cloud_coverage(self.observable_pointcloud, self.fused_observed_pointcloud, threshold=t)
                    for t in self.coverage_thresholds
                },
            }
            self._heavy_metric_cache_by_visited_view_num[visited_view_num] = heavy_metrics

        checkpoint = {
            "uid": self.uid,
            "name": name,
            "reason": reason,
            "visited_view_num": visited_view_num,
            "heavy_metrics_source": heavy_metrics_source,
            "chamfer_distance": heavy_metrics["chamfer_distance"],
            "surface_coverage": dict(heavy_metrics["surface_coverage"]),
            "normalized_surface_coverage": dict(heavy_metrics["normalized_surface_coverage"]),
            "total_path_cost": float(self.total_path_cost),
            "total_algorithm_runtime_sec": float(self.total_algorithm_runtime_sec),
            "total_benchmark_wall_time_sec": float(self.total_benchmark_wall_time_sec),
            "num_fused_points": int(len(self.fused_observed_pointcloud)),
            "map_stabilization_trigger_step_indices": dict(self.ms_trigger_step_indices),
            "map_stabilization_trigger_visited_view_nums": dict(self.ms_trigger_visited_view_nums),
        }
        self.checkpoints[name] = checkpoint
        return checkpoint

    def summary(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "num_steps": int(len(self.per_step_metrics)),
            "visited_view_num": int(len(self.visited_poses)),
            "observable_view_set": self.observable_view_set,
            "coverage_thresholds": list(self.coverage_thresholds),
            "observable_surface_ratio": dict(self.observable_surface_ratio),
            "observable_num_points": int(len(self.observable_pointcloud)),
            "num_observable_views": int(self.num_observable_views),
            "observable_reference_source": self.observable_reference_source,
            "observable_cache_constraint": self.observable_cache_constraint,
            "observable_reference_cache_metadata": dict(self.observable_reference_cache_metadata),
            "final_path_cost": float(self.total_path_cost),
            "total_algorithm_runtime_sec": float(self.total_algorithm_runtime_sec),
            "total_benchmark_wall_time_sec": float(self.total_benchmark_wall_time_sec),
            "checkpoints": dict(self.checkpoints),
            "map_stabilization": {
                "voxel_sizes": list(self.map_stabilization_voxel_sizes),
                "tau": float(self.map_stabilization_tau),
                "patience": int(self.map_stabilization_patience),
                "skip_after_trigger": bool(self.map_stabilization_skip_after_trigger),
                "trigger_step_indices": dict(self.ms_trigger_step_indices),
                "trigger_visited_view_nums": dict(self.ms_trigger_visited_view_nums),
                "history": list(self.ms_history),
            },
            "num_fused_points": 0 if self.fused_observed_pointcloud is None else int(len(self.fused_observed_pointcloud)),
            "visited_poses": list(self.visited_poses),
            "per_step_metrics": list(self.per_step_metrics),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.summary()

    def _build_observable_pointcloud(self) -> np.ndarray:
        poses = self._observable_poses_from_render_cache()
        pointclouds = []
        num_observable_views = 0
        for pose in poses:
            if self.feasibility_api is not None and not self.feasibility_api.is_feasible(pose):
                continue
            frame = self.render_api.render_loaded(pose)
            pointclouds.append(points_from_rgbd_frame(frame))
            num_observable_views += 1
        if not pointclouds:
            raise ValueError(
                f"No feasible observable views found for uid={self.uid}, "
                f"view_set={self.observable_view_set!r}"
            )
        self.num_observable_views = num_observable_views
        return fuse_pointclouds(pointclouds, voxel_size=self.fusion_voxel_size)

    def _load_observable_pointcloud_from_cache(self) -> tuple[np.ndarray, dict[str, Any]] | None:
        if self.observable_cache_root is None:
            return None
        if not self.observable_cache_constraint:
            return None

        npz_path, json_path = observable_reference_cache_paths(
            self.observable_cache_root,
            self.observable_cache_constraint,
            self.uid,
            self.observable_view_set,
            self.fusion_voxel_size,
        )
        if not npz_path.exists() or not json_path.exists():
            return None

        with np.load(npz_path) as data:
            if "observable_pointcloud" not in data:
                raise KeyError(f"'observable_pointcloud' not found in {npz_path}")
            observable_pointcloud = _to_points_array(data["observable_pointcloud"])
        with json_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        if not isinstance(metadata, dict):
            raise ValueError(f"Observable cache metadata must be a JSON object: {json_path}")

        if str(metadata.get("uid")) != self.uid:
            raise ValueError(f"Observable cache uid mismatch in {json_path}")
        if str(metadata.get("constraint")) != self.observable_cache_constraint:
            raise ValueError(f"Observable cache constraint mismatch in {json_path}")
        if str(metadata.get("observable_view_set")) != self.observable_view_set:
            raise ValueError(f"Observable cache view_set mismatch in {json_path}")
        cached_fusion = float(metadata.get("fusion_voxel_size"))
        if abs(cached_fusion - self.fusion_voxel_size) > 1e-12:
            raise ValueError(f"Observable cache fusion_voxel_size mismatch in {json_path}")

        metadata = dict(metadata)
        metadata["npz_path"] = str(npz_path)
        metadata["json_path"] = str(json_path)
        return observable_pointcloud, metadata

    def _observable_poses_from_render_cache(self) -> list[CameraPose]:
        cache_index = getattr(self.render_api, "_cache_index", None)
        if cache_index is None:
            raise RuntimeError("render_api must have a loaded cache index to build observable point cloud")

        objects = cache_index.get("objects")
        if not isinstance(objects, dict):
            raise ValueError("render cache index must contain an objects dictionary")

        uid = getattr(self.render_api, "uid", self.uid)
        uid_entry = objects.get(uid)
        if uid_entry is None:
            try:
                xyzs = _view_set_xyzs_from_geometry(self.observable_view_set, radius=3.0)
                print(
                    "[WARN] Observable reference uid missing from render cache index for "
                    f"uid={uid}, view_set={self.observable_view_set}. "
                    "Falling back to geometric view-set poses and Render(mode=auto).",
                    file=sys.stderr,
                )
                return [
                    CameraPose(
                        camera_xyz=tuple(float(v) for v in camera_xyz),
                        lookat_xyz=(0.0, 0.0, 0.0),
                        roll_rad=0.0,
                    )
                    for camera_xyz in xyzs
                ]
            except Exception as exc:
                raise KeyError(f"uid={uid} not found in render cache index") from exc

        views = uid_entry.get("views")
        if not isinstance(views, list):
            raise ValueError(f"cache entry for uid={uid} must contain a views list")

        parse_pose = getattr(self.render_api, "_parse_pose_entry", None)
        poses = []
        for entry in views:
            if entry.get("view_set") != self.observable_view_set:
                continue
            if parse_pose is not None:
                pose = parse_pose(entry)
            else:
                pose_dict = entry["pose"]
                pose = CameraPose(
                    camera_xyz=tuple(float(x) for x in pose_dict["camera_xyz"]),
                    lookat_xyz=tuple(float(x) for x in pose_dict["lookat_xyz"]),
                    roll_rad=float(pose_dict["roll_rad"]),
                )
            poses.append(pose)

        if not poses:
            raise ValueError(f"No views found for observable_view_set={self.observable_view_set!r}")
        return poses

    def _record_map_stabilization(self, step_index: int) -> dict[str, Any]:
        if self.fused_observed_pointcloud is None:
            raise RuntimeError("fused_observed_pointcloud is unavailable")

        stats: dict[str, Any] = {}
        for voxel_size in self.map_stabilization_voxel_sizes:
            key = _threshold_key(voxel_size)
            if self.map_stabilization_skip_after_trigger and self.ms_trigger_step_indices[key] is not None:
                stats[key] = {
                    "num_voxels": None,
                    "growth_ratio": None,
                    "stable_count": int(self.ms_stable_counts[key]),
                    "triggered": True,
                    "triggered_now": False,
                    "trigger_step_index": self.ms_trigger_step_indices[key],
                    "trigger_visited_view_num": self.ms_trigger_visited_view_nums[key],
                    "skipped_after_trigger": True,
                }
                continue

            num_voxels = count_occupied_voxels(self.fused_observed_pointcloud, voxel_size=voxel_size)
            previous_count = self.ms_previous_counts[key]

            if previous_count is None:
                growth_ratio = None
                self.ms_stable_counts[key] = 0
            else:
                growth_ratio = (num_voxels - int(previous_count)) / max(int(previous_count), 1)
                if growth_ratio < self.map_stabilization_tau:
                    self.ms_stable_counts[key] += 1
                else:
                    self.ms_stable_counts[key] = 0

            triggered_now = False
            if (
                previous_count is not None
                and self.ms_trigger_step_indices[key] is None
                and self.ms_stable_counts[key] >= self.map_stabilization_patience
            ):
                self.ms_trigger_step_indices[key] = int(step_index)
                self.ms_trigger_visited_view_nums[key] = int(len(self.visited_poses))
                triggered_now = True

            self.ms_previous_counts[key] = int(num_voxels)

            stats[key] = {
                "num_voxels": int(num_voxels),
                "growth_ratio": None if growth_ratio is None else float(growth_ratio),
                "stable_count": int(self.ms_stable_counts[key]),
                "triggered": self.ms_trigger_step_indices[key] is not None,
                "triggered_now": bool(triggered_now),
                "trigger_step_index": self.ms_trigger_step_indices[key],
                "trigger_visited_view_num": self.ms_trigger_visited_view_nums[key],
                "skipped_after_trigger": False,
            }

        record = {
            "step_index": int(step_index),
            "stats": stats,
        }
        self.ms_history.append(record)
        return record


def load_pointcloud(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        for key in ("points", "pointcloud", "vertices", "xyz"):
            if key in data:
                arr = data[key]
                break
        else:
            first_key = data.files[0]
            arr = data[first_key]
    elif suffix in {".ply", ".pcd"}:
        arr = _load_pointcloud_open3d(path)
    else:
        raise ValueError(f"Unsupported point cloud suffix: {suffix}")

    return _to_points_array(arr)


def points_from_rgbd_frame(frame: Any) -> np.ndarray:
    mask = np.asarray(frame.mask).astype(bool)
    if getattr(frame, "points_world_from_depth", None) is not None:
        points = np.asarray(frame.points_world_from_depth)
    elif getattr(frame, "points_world", None) is not None:
        points = np.asarray(frame.points_world)
    else:
        raise ValueError("rgbd_frame must contain points_world_from_depth or points_world")

    if points.ndim == 3:
        if mask.shape != points.shape[:2]:
            raise ValueError(f"mask shape {mask.shape} does not match points shape {points.shape}")
        points = points[mask]
    return _to_points_array(points)


def chamfer_distance_np(pc_gt: np.ndarray, pc_pred: np.ndarray, p_norm: int = 2) -> float:
    gt = _to_points_array(pc_gt)
    pred = _to_points_array(pc_pred)
    if len(gt) == 0 or len(pred) == 0:
        return float("inf")

    import torch
    from pytorch3d.loss import chamfer_distance

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss, _ = chamfer_distance(
        _to_torch(gt, device),
        _to_torch(pred, device),
        norm=p_norm,
        batch_reduction="mean",
        point_reduction="mean",
    )
    return float(loss.item())


def point_cloud_coverage(pc_gt: np.ndarray, pc_pred: np.ndarray, threshold: float = 0.01) -> float:
    gt = _to_points_array(pc_gt)
    pred = _to_points_array(pc_pred)
    if len(gt) == 0:
        return 0.0
    if len(pred) == 0:
        return 0.0

    import torch
    from pytorch3d.ops import knn_points

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresh_sq = float(threshold) ** 2
    gt_t = _to_torch(gt, device)
    pred_t = _to_torch(pred, device)
    d_gt2pred = knn_points(gt_t, pred_t, K=1).dists[..., 0]
    return float((d_gt2pred < thresh_sq).float().mean().item())


def fuse_pointclouds(pointclouds: Sequence[np.ndarray | None], voxel_size: float = 0.005) -> np.ndarray:
    valid = [np.asarray(pc, dtype=np.float32) for pc in pointclouds if pc is not None and len(pc) > 0]
    if not valid:
        return np.zeros((0, 3), dtype=np.float32)

    pts = _to_points_array(np.concatenate(valid, axis=0))
    if len(pts) == 0:
        return pts

    voxel_size = float(voxel_size)
    if voxel_size <= 0:
        return pts

    voxel_idx = np.floor(pts / voxel_size).astype(np.int64)
    _, inverse = np.unique(voxel_idx, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float32)
    fused = np.zeros((len(counts), 3), dtype=np.float32)
    np.add.at(fused, inverse, pts)
    fused /= counts[:, None]
    return fused.astype(np.float32)


def count_occupied_voxels(points: np.ndarray, voxel_size: float) -> int:
    pts = _to_points_array(points)
    if len(pts) == 0:
        return 0
    voxel_size = float(voxel_size)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    voxel_idx = np.floor(pts / voxel_size).astype(np.int64)
    unique_voxels = np.unique(voxel_idx, axis=0)
    return int(len(unique_voxels))


def collision_avoid_unit_sphere_distance(
    p: np.ndarray,
    q: np.ndarray,
    radius: float,
) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    radius = float(radius)
    if radius <= 0:
        return float(np.linalg.norm(p - q))

    r1 = float(np.linalg.norm(p))
    r2 = float(np.linalg.norm(q))
    if r1 <= radius + 1e-9 or r2 <= radius + 1e-9:
        return float(np.linalg.norm(p - q))

    if _segment_point_min_distance(p, q) >= radius - 1e-9:
        return float(np.linalg.norm(p - q))

    tan1 = math.sqrt(max(r1 * r1 - radius * radius, 0.0))
    tan2 = math.sqrt(max(r2 * r2 - radius * radius, 0.0))

    cos_theta = float(np.dot(p, q) / (r1 * r2))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = float(math.acos(cos_theta))

    gamma1 = math.acos(max(-1.0, min(1.0, radius / r1)))
    gamma2 = math.acos(max(-1.0, min(1.0, radius / r2)))
    two_pi = 2.0 * math.pi

    def _norm_ang(a: float) -> float:
        a = a % two_pi
        if a < 0:
            a += two_pi
        return a

    t_a_candidates = [_norm_ang(+gamma1), _norm_ang(-gamma1)]
    t_b_candidates = [_norm_ang(theta + gamma2), _norm_ang(theta - gamma2)]

    best = float("inf")
    for t_a in t_a_candidates:
        for t_b in t_b_candidates:
            delta = abs(t_b - t_a)
            delta = min(delta, two_pi - delta)
            best = min(best, float(tan1 + tan2 + radius * delta))
    return float(best)


def coerce_pose(pose: CameraPose | Sequence[float]) -> CameraPose:
    if isinstance(pose, CameraPose):
        return pose
    return pose_from_7d(pose)


def pose_from_7d(values: Sequence[float]) -> CameraPose:
    if len(values) != 7:
        raise ValueError(f"Expected 7 pose values, got {len(values)}")
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


def _to_points_array(pc: Any) -> np.ndarray:
    arr = np.asarray(pc)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"point cloud should have shape (N, 3), got {arr.shape}")
    return arr.astype(np.float32)


def _to_torch(arr: np.ndarray, device: Any) -> Any:
    import torch

    return torch.from_numpy(arr.astype(np.float32)).to(device).unsqueeze(0)


def _load_pointcloud_open3d(path: Path) -> np.ndarray:
    try:
        import open3d as o3d  # type: ignore
    except ImportError as exc:
        raise ImportError(f"open3d is required to read {path.suffix} point clouds") from exc

    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points, dtype=np.float32)


def _segment_point_min_distance(a: np.ndarray, b: np.ndarray) -> float:
    v = b - a
    vv = float(np.dot(v, v))
    if vv < 1e-12:
        return float(np.linalg.norm(a))
    t = -float(np.dot(a, v)) / vv
    t = max(0.0, min(1.0, t))
    x = a + t * v
    return float(np.linalg.norm(x))


def _threshold_key(threshold: float) -> str:
    return f"{float(threshold):.3f}".rstrip("0").rstrip(".")


def _float_key(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def infer_observable_cache_constraint(feasibility_api: Any | None) -> str:
    if feasibility_api is None:
        return "whole"
    constraint_json = getattr(feasibility_api, "constraint_json", None)
    if constraint_json is not None:
        return Path(constraint_json).stem
    return "custom"


def observable_reference_cache_paths(
    cache_root: str | Path,
    constraint: str,
    uid: str,
    observable_view_set: str,
    fusion_voxel_size: float,
) -> tuple[Path, Path]:
    stem = f"{uid}__{observable_view_set}__fusion_voxel_{_float_key(fusion_voxel_size)}"
    out_dir = Path(cache_root) / "observable_reference" / str(constraint)
    return out_dir / f"{stem}.npz", out_dir / f"{stem}.json"


def _nonnegative_float(value: float, name: str) -> float:
    value = float(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual tester for EvaluateAPI.")
    parser.add_argument("--uid", type=str, required=True)
    parser.add_argument("--gt-pointcloud", type=Path, required=True)
    parser.add_argument("--obj-path", type=Path, required=True)
    parser.add_argument("--cache-index-json", type=Path, default=Path("render_cache/cache_index.json"))
    parser.add_argument("--test-view-set", type=str, default="tammes_128")
    parser.add_argument("--observable-view-set", type=str, default="tammes_360")
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--feasibility-json", type=Path, default=None)
    parser.add_argument("--eval-cache-root", type=Path, default=Path("eval_cache"))
    parser.add_argument("--observable-cache-constraint", type=str, default=None)
    parser.add_argument("--disable-eval-cache", action="store_true")
    parser.add_argument("--fusion-voxel-size", type=float, default=0.005)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser


def _load_cache_index(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"cache index must be a JSON object: {path}")
    return data


def _poses_from_cache_index(
    cache_index: dict[str, Any],
    uid: str,
    view_set: str,
) -> list[CameraPose]:
    objects = cache_index.get("objects")
    if not isinstance(objects, dict):
        raise ValueError("cache index must contain an objects dict")
    uid_entry = objects.get(uid)
    if uid_entry is None:
        raise KeyError(f"uid={uid} not found in cache index")
    views = uid_entry.get("views")
    if not isinstance(views, list):
        raise ValueError(f"cache entry for uid={uid} must contain a views list")

    selected = [entry for entry in views if entry.get("view_set") == view_set]
    selected.sort(key=lambda entry: int(entry.get("view_idx", 0)))

    poses = []
    for entry in selected:
        pose_dict = entry.get("pose")
        if not isinstance(pose_dict, dict):
            raise ValueError("Each cache view entry must contain a pose dict")
        poses.append(
            CameraPose(
                camera_xyz=tuple(float(x) for x in pose_dict["camera_xyz"]),
                lookat_xyz=tuple(float(x) for x in pose_dict["lookat_xyz"]),
                roll_rad=float(pose_dict["roll_rad"]),
            )
        )
    return poses


def main() -> int:
    args = _build_argparser().parse_args()

    from api_render import CameraIntrinsics, Render, intrinsics_from_dict

    cache_index = _load_cache_index(args.cache_index_json)
    intrinsics_data = cache_index.get("intrinsics")
    if not isinstance(intrinsics_data, dict):
        raise ValueError("cache index must contain intrinsics")
    intrinsics: CameraIntrinsics = intrinsics_from_dict(intrinsics_data)

    feasibility_api = None
    if args.feasibility_json is not None:
        from api_feasibility import FeasibilityAPI

        feasibility_api = FeasibilityAPI(args.feasibility_json)

    renderer = Render(
        uid=args.uid,
        obj_path=args.obj_path,
        intrinsics=intrinsics,
        device=args.device,
        mode="cache",
        cache_index_json=args.cache_index_json,
    )

    init_start = time.perf_counter()
    evaluator = EvaluateAPI(
        uid=args.uid,
        gt_pointcloud_path=args.gt_pointcloud,
        render_api=renderer,
        observable_view_set=args.observable_view_set,
        fusion_voxel_size=args.fusion_voxel_size,
        feasibility_api=feasibility_api,
        observable_cache_root=None if args.disable_eval_cache else args.eval_cache_root,
        observable_cache_constraint=args.observable_cache_constraint,
    )
    init_sec = time.perf_counter() - init_start
    print(
        f"[INIT] evaluator_init_sec={init_sec:.6f} "
        f"observable_reference_source={evaluator.observable_reference_source} "
        f"observable_cache_constraint={evaluator.observable_cache_constraint}"
    )

    poses_all = _poses_from_cache_index(cache_index, args.uid, args.test_view_set)
    if feasibility_api is not None:
        poses = [pose for pose in poses_all if feasibility_api.is_feasible(pose)]
    else:
        poses = poses_all
    feasible_pose_count = len(poses)
    if args.max_steps >= 0:
        poses = poses[: min(args.max_steps, feasible_pose_count)]
    if not poses:
        raise ValueError(f"No poses found for uid={args.uid}, view_set={args.test_view_set}")
    print(
        f"[POSES] view_set={args.test_view_set} "
        f"total={len(poses_all)} feasible={feasible_pose_count} "
        f"scheduled={len(poses)} max_steps={args.max_steps}"
    )

    for step_index, pose in enumerate(poses):
        step_start = time.perf_counter()
        render_start = time.perf_counter()
        frame = renderer.render_loaded(pose)
        render_sec = time.perf_counter() - render_start

        eval_start = time.perf_counter()
        metrics = evaluator.record_step(
            step_index=step_index,
            pose=pose,
            rgbd_frame=frame,
            algorithm_runtime_sec=None,
            benchmark_wall_time_sec=None,
            update_map_stabilization=True,
        )
        eval_sec = time.perf_counter() - eval_start
        step_sec = time.perf_counter() - step_start
        ms_stats = metrics["map_stabilization"]["stats"] if metrics["map_stabilization"] else {}
        ms_growth = {
            key: value.get("growth_ratio")
            for key, value in ms_stats.items()
        }
        ms_trigger_visited = {
            key: value.get("trigger_visited_view_num")
            for key, value in ms_stats.items()
        }
        visited_view_num = metrics["visited_view_num"]
        print(
            f"step={step_index:03d} "
            f"visited_view_num={visited_view_num:03d} "
            f"path={metrics['total_path_cost']:.6f} "
            f"fused_points={metrics['num_fused_points']} "
            f"MS_growth={_format_metric_dict(ms_growth)} "
            f"MS_trigger_visited_view_nums={ms_trigger_visited} "
            f"time={{render:{render_sec:.6f}, eval:{eval_sec:.6f}, step:{step_sec:.6f}}}"
        )
        if visited_view_num in {5, 10, 30, 50}:
            checkpoint_start = time.perf_counter()
            checkpoint = evaluator.evaluate_checkpoint(f"K={visited_view_num}", reason="fixed_budget")
            checkpoint_sec = time.perf_counter() - checkpoint_start
            print(f"[CHECKPOINT] {_format_checkpoint(checkpoint)} time={checkpoint_sec:.6f}")
        for ms_key, ms_value in ms_stats.items():
            if ms_value.get("triggered_now"):
                checkpoint_start = time.perf_counter()
                checkpoint = evaluator.evaluate_checkpoint(f"MS@{ms_key}", reason=f"MS@{ms_key}")
                checkpoint_sec = time.perf_counter() - checkpoint_start
                print(f"[CHECKPOINT] {_format_checkpoint(checkpoint)} time={checkpoint_sec:.6f}")

    if args.max_steps >= 0 and args.max_steps < feasible_pose_count:
        terminal_reason = "max_steps_reached"
    elif feasibility_api is not None:
        terminal_reason = "candidate_exhausted"
    else:
        terminal_reason = "test_sequence_end"

    terminal_start = time.perf_counter()
    terminal_checkpoint = evaluator.evaluate_checkpoint("terminal", reason=terminal_reason)
    terminal_sec = time.perf_counter() - terminal_start
    summary = evaluator.summary()
    print(f"[CHECKPOINT] {_format_checkpoint(terminal_checkpoint)} time={terminal_sec:.6f}")
    print("[SUMMARY]")
    print(json.dumps(
        {
            "uid": summary["uid"],
            "num_steps": summary["num_steps"],
            "final_path_cost": summary["final_path_cost"],
            "checkpoints": {
                key: {
                    "visited_view_num": value["visited_view_num"],
                    "reason": value["reason"],
                    "heavy_metrics_source": value["heavy_metrics_source"],
                    "chamfer_distance": value["chamfer_distance"],
                    "surface_coverage": value["surface_coverage"],
                    "normalized_surface_coverage": value["normalized_surface_coverage"],
                    "total_path_cost": value["total_path_cost"],
                }
                for key, value in summary["checkpoints"].items()
            },
            "map_stabilization": {
                "trigger_step_indices": summary["map_stabilization"]["trigger_step_indices"],
                "trigger_visited_view_nums": summary["map_stabilization"]["trigger_visited_view_nums"],
                "tau": summary["map_stabilization"]["tau"],
                "patience": summary["map_stabilization"]["patience"],
            },
        },
        indent=2,
    ))
    print(f"[TIME] evaluator_init_sec={init_sec:.6f}")

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        print(f"summary_json: {args.summary_json}")

    return 0


def _format_metric_dict(values: dict[str, Any]) -> str:
    parts = []
    for key in sorted(values, key=lambda x: float(x)):
        value = values[key]
        if value is None:
            parts.append(f"{key}:None")
        elif isinstance(value, (float, int)):
            parts.append(f"{key}:{float(value):.6f}")
        else:
            parts.append(f"{key}:{value}")
    return "{" + ", ".join(parts) + "}"


def _format_checkpoint(checkpoint: dict[str, Any]) -> str:
    return (
        f"name={checkpoint['name']} "
        f"reason={checkpoint['reason']} "
        f"visited_view_num={checkpoint['visited_view_num']} "
        f"SC={_format_metric_dict(checkpoint['surface_coverage'])} "
        f"NSC={_format_metric_dict(checkpoint['normalized_surface_coverage'])} "
        f"CD={checkpoint['chamfer_distance']:.6f} "
        f"path={checkpoint['total_path_cost']:.6f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
