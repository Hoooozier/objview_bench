from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

BENCHMARK_ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class MethodSpec:
    method_name: str
    family: str
    command: tuple[str, ...]
    planning_network_command: Optional[tuple[str, ...]] = None
    display_name: str | None = None
    execution_mode_compatibility: str = "unknown"
    execution_mode: str = "unknown"
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


METHODS: dict[str, MethodSpec] = {
    "oracle_rollout_hybrid_nsc01": MethodSpec(
        method_name="oracle_rollout_hybrid_nsc01",
        family="oracle",
        display_name="VoxelRollout+SortedRemaining",
        execution_mode_compatibility="oracle_only",
        execution_mode="online_decision",
        command=(
            "{python}",
            "algorithms/oracle/oracle_rollout_hybrid_nsc01.py",
            "--session-dir",
            "{session_dir}",
            "--cache-index-json",
            "{cache_index_json}",
            "--rollout-root",
            "/mnt/d/ObjView-Bench/object_dataset/clean_pool/oracle_rollout",
            "--view-set",
            "tammes_128",
            "--threshold",
            "0.01",
            "--max-actions",
            "128",
            "--fallback-mode",
            "sorted",
        ),
        description=(
            "Replay a precomputed voxel-0.02 tammes_128 oracle rollout, then fill the "
            "remaining tammes_128 candidates in ascending view-id order. Runs until "
            "max actions or candidate exhaustion."
        ),
        metadata={
            "oracle": True,
            "ground_truth_dependent": True,
            "view_set": "tammes_128",
            "rollout_resolution": 0.02,
            "fallback_mode": "sorted",
            "online_threshold": 0.01,
            "max_actions": 128,
        },
    ),
    "oracle_greedy_nsc01": MethodSpec(
        method_name="oracle_greedy_nsc01",
        family="oracle",
        display_name="Oracle Greedy NSC@0.01",
        execution_mode_compatibility="oracle_only",
        execution_mode="online_decision",
        command=(
            "{python}",
            "algorithms/oracle/oracle_greedy_nsc01.py",
            "--session-dir",
            "{session_dir}",
            "--views",
            "{benchmark_root}/Tammes_sphere/128_xyz.txt",
            "--view-set",
            "tammes_128",
            "--threshold",
            "0.01",
            "--max-actions",
            "128",
        ),
        description=(
            "Analysis-only oracle that greedily selects the feasible tammes_128 view "
            "with maximum one-step NSC@0.01 gain."
        ),
        metadata={
            "oracle": True,
            "ground_truth_dependent": True,
            "view_set": "tammes_128",
            "threshold": 0.01,
            "max_actions": 128,
        },
    ),
    "classical_voxel_ig_rse": MethodSpec(
        method_name="classical_voxel_ig_rse",
        family="classical_NBV",
        display_name="VoxelIG-RSE",
        execution_mode_compatibility="iterative",
        execution_mode="online_decision",
        command=(
            "bash",
            "algorithms/cpp_enroot/run_cpp_enroot_algorithm.sh",
            "--session-dir",
            "{session_dir}",
            "--cache-index-json",
            "{cache_index_json}",
            "--algorithm",
            "voxel_ig_iterative",
            "--views",
            "{benchmark_root}/Tammes_sphere/360_xyz.txt",
            "--view-radius",
            "3.0",
            "--voxel-ig-method",
            "rse",
            "--voxel-ig-grid-dim",
            "64",
            "--voxel-ig-map-bbox-min",
            "-1.0",
            "--voxel-ig-map-bbox-max",
            "1.0",
            "--voxel-ig-ray-stride",
            "0",
            "--voxel-ig-filter-rays-to-bbox",
            "1",
            "--voxel-ig-backend",
            "cuda",
        ),
        description="Classical voxel information gain baseline using Rear Side Entropy VI.",
        metadata={
            "representation": "voxel_ig",
            "planning": "iterative",
            "method": "rse",
            "grid_dim": 64,
            "bbox": [-1.0, 1.0],
            "default_view_set": "tammes_360",
        },
    ),
    "classical_voxel_ig_rse_mov": MethodSpec(
        method_name="classical_voxel_ig_rse_mov",
        family="classical_NBV",
        display_name="VoxelIG-RSE+Mov",
        execution_mode_compatibility="iterative",
        execution_mode="online_decision",
        command=(
            "bash",
            "algorithms/cpp_enroot/run_cpp_enroot_algorithm.sh",
            "--session-dir",
            "{session_dir}",
            "--cache-index-json",
            "{cache_index_json}",
            "--algorithm",
            "voxel_ig_iterative",
            "--views",
            "{benchmark_root}/Tammes_sphere/360_xyz.txt",
            "--view-radius",
            "3.0",
            "--voxel-ig-method",
            "rse",
            "--voxel-ig-grid-dim",
            "64",
            "--voxel-ig-map-bbox-min",
            "-1.0",
            "--voxel-ig-map-bbox-max",
            "1.0",
            "--voxel-ig-ray-stride",
            "0",
            "--voxel-ig-filter-rays-to-bbox",
            "1",
            "--voxel-ig-backend",
            "cuda",
            "--voxel-ig-use-movement-cost",
            "1",
            "--voxel-ig-movement-cost-weight",
            "0.7",
        ),
        description="Classical voxel information gain baseline using Rear Side Entropy VI with movement-aware utility.",
        metadata={
            "representation": "voxel_ig",
            "planning": "iterative",
            "method": "rse_mov",
            "backend": "cuda",
            "grid_dim": 64,
            "bbox": [-1.0, 1.0],
            "default_view_set": "tammes_360",
            "movement_cost_weight": 0.5,
        },
    ),
    "benbv_planning_network": MethodSpec(
        method_name="benbv_planning_network",
        family="learned_nbv",
        display_name="BENBV Planning Network",
        execution_mode_compatibility="iterative",
        execution_mode="online_decision",
        command=(
            "bash",
            "algorithms/cpp_enroot/run_cpp_enroot_algorithm.sh",
            "--session-dir",
            "{session_dir}",
            "--cache-index-json",
            "{cache_index_json}",
            "--algorithm",
            "benbv_planning_network",
            "--benbv-service-name",
            "benbv",
            "--benbv-resolution",
            "0.02",
            "--benbv-camera-distance",
            "2.0",
            "--benbv-point-sample-count",
            "4096",
            "--benbv-candidate-count",
            "20",
            "--benbv-knn",
            "30",
            "--benbv-boundary-angle-deg",
            "120",
            "--benbv-partial-voxel-leaf",
            "0.015625",
            "--benbv-pointcloud-bbox-min",
            "-1.0",
            "--benbv-pointcloud-bbox-max",
            "1.0",
            "--benbv-topk",
            "5",
            "--benbv-debug-save",
            "0",
            "--benbv-debug-dir",
            "benbv_planning_network_debug",
            "--silent",
            "1",
        ),
        planning_network_command=(
            "python",
            "api_planning_network.py",
            "--session-dir",
            "{session_dir}",
            "--service-name",
            "benbv",
            "--backend",
            "benbv",
            "--ckpt",
            "planning_network/BENBV/best_val_model.pth",
            "--device",
            "cuda:0",
        ),
        description=(
            "BENBV planning-network baseline. The C++ algorithm aggregates the observed "
            "partial point cloud, estimates normals, extracts boundary points with PCL, "
            "clusters them into 20 boundary candidates, writes BENBV P/S/C network input, "
            "calls the benchmark planning-network service, ranks all candidate scores, "
            "filters candidates through dynamic feasibility RPC, and executes the highest "
            "scoring feasible continuous camera-lookat pose."
        ),
        metadata={
            "representation": "partial_point_cloud",
            "planning": "learned_boundary_nbv",
            "network_backend": "benbv",
            "checkpoint": "planning_network/BENBV/best_val_model.pth",
            "requires_planning_network": True,
            "planning_network_service_name": "benbv",
            "candidate_type": "dynamic_boundary_candidate",
            "candidate_count": 20,
            "network_input": ["P", "S", "C"],
            "point_sample_count": 4096,
            "normal_estimation": "pcl_normal_estimation_omp",
            "boundary_detector": "pcl_boundary_estimation",
            "boundary_angle_deg": 120.0,
            "knn": 30,
            "camera_distance": 2.0,
            "partial_voxel_leaf": 0.015625,
            "bbox": [-1.0, 1.0],
            "pose_format": "camera_lookat_roll",
            "candidate_pose": "[camera_xyz, boundary_target_xyz, roll]",
            "selection": "score_sorted_all_candidates_then_dynamic_feasibility",
        },
    ),
    "nbvnet_planning_network": MethodSpec(
        method_name="nbvnet_planning_network",
        family="learned_nbv",
        display_name="NBVNet Planning Network",
        execution_mode_compatibility="iterative",
        execution_mode="online_decision",
        command=(
            "bash",
            "algorithms/cpp_enroot/run_cpp_enroot_algorithm.sh",
            "--session-dir",
            "{session_dir}",
            "--cache-index-json",
            "{cache_index_json}",
            "--algorithm",
            "nbvnet_planning_network",
            "--views",
            "{benchmark_root}/Tammes_sphere/128_xyz.txt",
            "--view-radius",
            "3.0",
            "--obstacle-radius",
            "1.0",
            "--nbvnet-grid-dim",
            "64",
            "--nbvnet-map-bbox-min",
            "-1.0",
            "--nbvnet-map-bbox-max",
            "1.0",
            "--nbvnet-unknown-occ",
            "0.5",
            "--nbvnet-service-name",
            "nbvnet",
            "--nbvnet-topk",
            "5",
            "--nbvnet-debug-save",
            "0",
            "--nbvnet-debug-save-ot",
            "0",
            "--nbvnet-debug-dir",
            "nbvnet_planning_network_debug",
            "--silent",
            "1",
        ),
        planning_network_command=(
            "python",
            "api_planning_network.py",
            "--session-dir",
            "{session_dir}",
            "--service-name",
            "nbvnet",
            "--backend",
            "nbvnet",
            "--ckpt",
            "planning_network/NBVNET/best_val_loss.pt",
            "--views",
            "Tammes_sphere/128_xyz.txt",
            "--device",
            "cuda:0",
            "--view-position-radius",
            "1.0",
        ),
        description=(
            "NBVNet planning-network baseline. The C++ algorithm maintains an "
            "RSE-style occupancy map, exports a dense 64^3 grid each step, calls "
            "the benchmark planning-network service, and executes the single "
            "predicted NBV class if it is feasible and unvisited. If the predicted "
            "view has already been executed, it stops with next_view_visited; if "
            "the predicted view is missing, out of range, or infeasible, it stops "
            "with next_view_infeasible."
        ),
        metadata={
            "representation": "dense_occupancy_grid",
            "planning": "direct_nbv_classification",
            "network_backend": "nbvnet",
            "checkpoint": "planning_network/NBVNET/best_val_loss.pt",
            "requires_planning_network": True,
            "planning_network_service_name": "nbvnet",
            "default_view_set": "tammes_128",
            "view_count": 128,
            "view_radius": 3.0,
            "network_view_position_radius": 1.0,
            "grid_size": 64,
            "bbox": [-1.0, 1.0],
            "unknown_occupancy": 0.5,
            "selection": "best_index_only",
            "fallback": "none",
            "stop_reasons": ["next_view_visited", "next_view_infeasible"],
            "debug_save": False,
        },
    ),
    "pointr_c_nbv": MethodSpec(
        method_name="pointr_c_nbv",
        family="completion_planning",
        display_name="PoinTr-C+NBV(Pred-NBV)",
        execution_mode_compatibility="iterative",
        execution_mode="online_decision",
        command=(
            "bash",
            "algorithms/cpp_enroot/run_cpp_enroot_algorithm.sh",
            "--session-dir",
            "{session_dir}",
            "--cache-index-json",
            "{cache_index_json}",
            "--algorithm",
            "pointr_c_nbv",
            "--views",
            "{benchmark_root}/Tammes_sphere/360_xyz.txt",
            "--view-radius",
            "3.0",
            "--pointr-partial-voxel-leaf-size",
            "0.015625",
            "--pointr-planning-voxel-size",
            "0.03125",
            "--pointr-pointcloud-bbox-min",
            "-1.0",
            "--pointr-pointcloud-bbox-max",
            "1.0",
            "--pointr-hpr-radius-scale",
            "100.0",
            "--pointr-tau",
            "0.95",
            "--pointr-completion-backend",
            "PoinTr-C",
        ),
        description="Prediction-guided online NBV method using PoinTr-C completion, HPR visibility scoring, and local path-cost tie-breaking.",
        metadata={
            "representation": "pointcloud_partial_map",
            "planning": "iterative_nbv",
            "completion_backend": "PoinTr-C",
            "requires_shape_completion": True,
            "default_view_set": "tammes_360",
            "partial_voxel_leaf_size": 0.015625,
            "planning_voxel_size": 0.03125,
            "hpr_radius_scale": 100.0,
            "tau": 0.95,
            "bbox": [-1.0, 1.0],
        },
    ),
    "pointr_c_scp": MethodSpec(
        method_name="pointr_c_scp",
        family="completion_planning",
        display_name="PoinTr-C+SCP",
        execution_mode_compatibility="automatic_only",
        execution_mode="online_decision",
        command=(
            "bash",
            "algorithms/cpp_enroot/run_cpp_enroot_algorithm.sh",
            "--session-dir",
            "{session_dir}",
            "--cache-index-json",
            "{cache_index_json}",
            "--algorithm",
            "pointr_c_scp",
            "--views",
            "{benchmark_root}/Tammes_sphere/360_xyz.txt",
            "--view-radius",
            "3.0",
            "--obstacle-radius",
            "1.0",
            "--pointr-partial-voxel-leaf-size",
            "0.015625",
            "--pointr-planning-voxel-size",
            "0.03125",
            "--pointr-pointcloud-bbox-min",
            "-1.0",
            "--pointr-pointcloud-bbox-max",
            "1.0",
            "--pointr-sor-mean-k",
            "16",
            "--pointr-sor-stddev-mul",
            "1.5",
            "--pointr-visibility-max-range",
            "6.0",
            "--pointr-visibility-mode",
            "inverse_cuda",
            "--pointr-min-visible-views",
            "1",
            "--pointr-scp-time-limit",
            "10.0",
            "--pointr-completion-backend",
            "PoinTr-C",
            "--silent",
            "1",
        ),
        description=(
            "PoinTr-C shape-completion-guided set cover planning with deterministic "
            "bootstrap, light completion denoising, voxelized current-plus-predicted "
            "reference geometry, predicted-minus-current target selection, inverse-visibility "
            "coverage modeling, and minimum-view coverage planning."
        ),
        metadata={
            "representation": "pointcloud_partial_map",
            "planning": "set_cover",
            "completion_backend": "PoinTr-C",
            "requires_shape_completion": True,
            "default_view_set": "tammes_360",
            "partial_voxel_leaf_size": 0.015625,
            "planning_voxel_size": 0.03125,
            "bbox": [-1.0, 1.0],
            "bootstrap": "farthest_path_cost",
            "reference_geometry": "current_union_predicted",
            "target_geometry": "predicted_minus_current",
            "sor_mean_k": 16,
            "sor_stddev_mul": 1.5,
            "visibility_mode": "inverse_cuda",
            "min_visible_views": 1,
            "scp_time_limit_sec": 10.0,
        },
    ),
    "pointr_c_mcp_5": MethodSpec(
        method_name="pointr_c_mcp_5",
        family="completion_planning",
        display_name="PoinTr-C+MCP (K=5)",
        execution_mode_compatibility="fixed_budget_only",
        execution_mode="online_decision",
        command=(
            "bash",
            "algorithms/cpp_enroot/run_cpp_enroot_algorithm.sh",
            "--session-dir",
            "{session_dir}",
            "--cache-index-json",
            "{cache_index_json}",
            "--algorithm",
            "pointr_c_mcp",
            "--budget",
            "5",
            "--views",
            "{benchmark_root}/Tammes_sphere/360_xyz.txt",
            "--view-radius",
            "3.0",
            "--obstacle-radius",
            "1.0",
            "--pointr-partial-voxel-leaf-size",
            "0.015625",
            "--pointr-planning-voxel-size",
            "0.03125",
            "--pointr-pointcloud-bbox-min",
            "-1.0",
            "--pointr-pointcloud-bbox-max",
            "1.0",
            "--pointr-sor-mean-k",
            "16",
            "--pointr-sor-stddev-mul",
            "1.5",
            "--pointr-visibility-max-range",
            "6.0",
            "--pointr-visibility-mode",
            "inverse_cuda",
            "--pointr-min-visible-views",
            "1",
            "--pointr-mcp-time-limit",
            "10.0",
            "--pointr-completion-backend",
            "PoinTr-C",
            "--silent",
            "1",
            "--pointr-mcp-debug-save",
            "0",
            "--pointr-mcp-debug-dir",
            "pointr_c_mcp_debug",
        ),
        description=(
            "PoinTr-C shape-completion-guided maximum coverage planning with deterministic "
            "bootstrap, light completion denoising, voxelized current-plus-predicted "
            "reference geometry, predicted-minus-current target selection, inverse-visibility "
            "coverage modeling, budgeted maximum coverage optimization, and deterministic "
            "farthest-point budget fill."
        ),
        metadata={
            "representation": "pointcloud_partial_map",
            "planning": "maximum_coverage",
            "completion_backend": "PoinTr-C",
            "requires_shape_completion": True,
            "default_view_set": "tammes_360",
            "budget": 5,
            "partial_voxel_leaf_size": 0.015625,
            "planning_voxel_size": 0.03125,
            "bbox": [-1.0, 1.0],
            "bootstrap": "farthest_path_cost",
            "reference_geometry": "current_union_predicted",
            "target_geometry": "predicted_minus_current",
            "sor_mean_k": 16,
            "sor_stddev_mul": 1.5,
            "visibility_mode": "inverse_cuda",
            "min_visible_views": 1,
            "mcp_time_limit_sec": 10.0,
            "budget_fill": "deterministic_farthest_point",
        },
    ),
    "pointr_c_mcp_10": MethodSpec(
        method_name="pointr_c_mcp_10",
        family="completion_planning",
        display_name="PoinTr-C+MCP (K=10)",
        execution_mode_compatibility="fixed_budget_only",
        execution_mode="online_decision",
        command=(
            "bash",
            "algorithms/cpp_enroot/run_cpp_enroot_algorithm.sh",
            "--session-dir",
            "{session_dir}",
            "--cache-index-json",
            "{cache_index_json}",
            "--algorithm",
            "pointr_c_mcp",
            "--budget",
            "10",
            "--views",
            "{benchmark_root}/Tammes_sphere/360_xyz.txt",
            "--view-radius",
            "3.0",
            "--obstacle-radius",
            "1.0",
            "--pointr-partial-voxel-leaf-size",
            "0.015625",
            "--pointr-planning-voxel-size",
            "0.03125",
            "--pointr-pointcloud-bbox-min",
            "-1.0",
            "--pointr-pointcloud-bbox-max",
            "1.0",
            "--pointr-sor-mean-k",
            "16",
            "--pointr-sor-stddev-mul",
            "1.5",
            "--pointr-visibility-max-range",
            "6.0",
            "--pointr-visibility-mode",
            "inverse_cuda",
            "--pointr-min-visible-views",
            "1",
            "--pointr-mcp-time-limit",
            "10.0",
            "--pointr-completion-backend",
            "PoinTr-C",
            "--silent",
            "1",
            "--pointr-mcp-debug-save",
            "0",
            "--pointr-mcp-debug-dir",
            "pointr_c_mcp_debug",
        ),
        description=(
            "PoinTr-C shape-completion-guided maximum coverage planning with deterministic "
            "bootstrap, light completion denoising, voxelized current-plus-predicted "
            "reference geometry, predicted-minus-current target selection, inverse-visibility "
            "coverage modeling, budgeted maximum coverage optimization, and deterministic "
            "farthest-point budget fill."
        ),
        metadata={
            "representation": "pointcloud_partial_map",
            "planning": "maximum_coverage",
            "completion_backend": "PoinTr-C",
            "requires_shape_completion": True,
            "default_view_set": "tammes_360",
            "budget": 10,
            "partial_voxel_leaf_size": 0.015625,
            "planning_voxel_size": 0.03125,
            "bbox": [-1.0, 1.0],
            "bootstrap": "farthest_path_cost",
            "reference_geometry": "current_union_predicted",
            "target_geometry": "predicted_minus_current",
            "sor_mean_k": 16,
            "sor_stddev_mul": 1.5,
            "visibility_mode": "inverse_cuda",
            "min_visible_views": 1,
            "mcp_time_limit_sec": 10.0,
            "budget_fill": "deterministic_farthest_point",
        },
    ),
    "pointr_c_mcp_30": MethodSpec(
        method_name="pointr_c_mcp_30",
        family="completion_planning",
        display_name="PoinTr-C+MCP (K=30)",
        execution_mode_compatibility="fixed_budget_only",
        execution_mode="online_decision",
        command=(
            "bash",
            "algorithms/cpp_enroot/run_cpp_enroot_algorithm.sh",
            "--session-dir",
            "{session_dir}",
            "--cache-index-json",
            "{cache_index_json}",
            "--algorithm",
            "pointr_c_mcp",
            "--budget",
            "30",
            "--views",
            "{benchmark_root}/Tammes_sphere/360_xyz.txt",
            "--view-radius",
            "3.0",
            "--obstacle-radius",
            "1.0",
            "--pointr-partial-voxel-leaf-size",
            "0.015625",
            "--pointr-planning-voxel-size",
            "0.03125",
            "--pointr-pointcloud-bbox-min",
            "-1.0",
            "--pointr-pointcloud-bbox-max",
            "1.0",
            "--pointr-sor-mean-k",
            "16",
            "--pointr-sor-stddev-mul",
            "1.5",
            "--pointr-visibility-max-range",
            "6.0",
            "--pointr-visibility-mode",
            "inverse_cuda",
            "--pointr-min-visible-views",
            "1",
            "--pointr-mcp-time-limit",
            "10.0",
            "--pointr-completion-backend",
            "PoinTr-C",
            "--silent",
            "1",
            "--pointr-mcp-debug-save",
            "0",
            "--pointr-mcp-debug-dir",
            "pointr_c_mcp_debug",
        ),
        description=(
            "PoinTr-C shape-completion-guided maximum coverage planning with deterministic "
            "bootstrap, light completion denoising, voxelized current-plus-predicted "
            "reference geometry, predicted-minus-current target selection, inverse-visibility "
            "coverage modeling, budgeted maximum coverage optimization, and deterministic "
            "farthest-point budget fill."
        ),
        metadata={
            "representation": "pointcloud_partial_map",
            "planning": "maximum_coverage",
            "completion_backend": "PoinTr-C",
            "requires_shape_completion": True,
            "default_view_set": "tammes_360",
            "budget": 30,
            "partial_voxel_leaf_size": 0.015625,
            "planning_voxel_size": 0.03125,
            "bbox": [-1.0, 1.0],
            "bootstrap": "farthest_path_cost",
            "reference_geometry": "current_union_predicted",
            "target_geometry": "predicted_minus_current",
            "sor_mean_k": 16,
            "sor_stddev_mul": 1.5,
            "visibility_mode": "inverse_cuda",
            "min_visible_views": 1,
            "mcp_time_limit_sec": 10.0,
            "budget_fill": "deterministic_farthest_point",
        },
    ),
    "pointr_c_mcp_50": MethodSpec(
        method_name="pointr_c_mcp_50",
        family="completion_planning",
        display_name="PoinTr-C+MCP (K=50)",
        execution_mode_compatibility="fixed_budget_only",
        execution_mode="online_decision",
        command=(
            "bash",
            "algorithms/cpp_enroot/run_cpp_enroot_algorithm.sh",
            "--session-dir",
            "{session_dir}",
            "--cache-index-json",
            "{cache_index_json}",
            "--algorithm",
            "pointr_c_mcp",
            "--budget",
            "50",
            "--views",
            "{benchmark_root}/Tammes_sphere/360_xyz.txt",
            "--view-radius",
            "3.0",
            "--obstacle-radius",
            "1.0",
            "--pointr-partial-voxel-leaf-size",
            "0.015625",
            "--pointr-planning-voxel-size",
            "0.03125",
            "--pointr-pointcloud-bbox-min",
            "-1.0",
            "--pointr-pointcloud-bbox-max",
            "1.0",
            "--pointr-sor-mean-k",
            "16",
            "--pointr-sor-stddev-mul",
            "1.5",
            "--pointr-visibility-max-range",
            "6.0",
            "--pointr-visibility-mode",
            "inverse_cuda",
            "--pointr-min-visible-views",
            "1",
            "--pointr-mcp-time-limit",
            "10.0",
            "--pointr-completion-backend",
            "PoinTr-C",
            "--silent",
            "1",
            "--pointr-mcp-debug-save",
            "0",
            "--pointr-mcp-debug-dir",
            "pointr_c_mcp_debug",
        ),
        description=(
            "PoinTr-C shape-completion-guided maximum coverage planning with deterministic "
            "bootstrap, light completion denoising, voxelized current-plus-predicted "
            "reference geometry, predicted-minus-current target selection, inverse-visibility "
            "coverage modeling, budgeted maximum coverage optimization, and deterministic "
            "farthest-point budget fill."
        ),
        metadata={
            "representation": "pointcloud_partial_map",
            "planning": "maximum_coverage",
            "completion_backend": "PoinTr-C",
            "requires_shape_completion": True,
            "default_view_set": "tammes_360",
            "budget": 50,
            "partial_voxel_leaf_size": 0.015625,
            "planning_voxel_size": 0.03125,
            "bbox": [-1.0, 1.0],
            "bootstrap": "farthest_path_cost",
            "reference_geometry": "current_union_predicted",
            "target_geometry": "predicted_minus_current",
            "sor_mean_k": 16,
            "sor_stddev_mul": 1.5,
            "visibility_mode": "inverse_cuda",
            "min_visible_views": 1,
            "mcp_time_limit_sec": 10.0,
            "budget_fill": "deterministic_farthest_point",
        },
    ),
    "mascvp_planning_network": MethodSpec(
        method_name="mascvp_planning_network",
        family="learned_set_cover",
        display_name="MA-SCVP Planning Network",
        execution_mode_compatibility="automatic_only",
        execution_mode="online_decision",
        command=(
            "bash",
            "algorithms/cpp_enroot/run_cpp_enroot_algorithm.sh",
            "--session-dir",
            "{session_dir}",
            "--cache-index-json",
            "{cache_index_json}",
            "--algorithm",
            "mascvp_planning_network",
            "--views",
            "{benchmark_root}/Tammes_sphere/128_xyz.txt",
            "--view-radius",
            "3.0",
            "--obstacle-radius",
            "1.0",
            "--tsp-time-limit",
            "-1",
            "--mascvp-grid-dim",
            "64",
            "--mascvp-map-bbox-min",
            "-1.0",
            "--mascvp-map-bbox-max",
            "1.0",
            "--mascvp-unknown-occ",
            "0.5",
            "--mascvp-service-name",
            "mascvp",
            "--mascvp-decode-key",
            "gamma_0.3",
            "--mascvp-topk",
            "5",
            "--mascvp-debug-save",
            "0",
            "--mascvp-debug-save-ot",
            "0",
            "--mascvp-debug-dir",
            "mascvp_planning_network_debug",
            "--silent",
            "1",
        ),
        planning_network_command=(
            "python",
            "api_planning_network.py",
            "--session-dir",
            "{session_dir}",
            "--service-name",
            "mascvp",
            "--backend",
            "mascvp",
            "--ckpt",
            "planning_network/MASCVP/main_train_public/best_f1.pt",
            "--views",
            "Tammes_sphere/128_xyz.txt",
            "--device",
            "cuda:0",
            "--view-position-radius",
            "1.0",
        ),
        description=(
            "MA-SCVP planning-network baseline. The C++ algorithm builds an RSE-style "
            "dense occupancy grid and 128-view state after an init-plus-farthest-point "
            "bootstrap, calls the benchmark planning-network service, decodes the "
            "gamma=0.3 predicted residual view set, filters visited and infeasible "
            "Tammes-128 views, orders the remaining set with TSP, and executes the fixed plan."
        ),
        metadata={
            "representation": "dense_occupancy_grid",
            "planning": "learned_set_cover",
            "network_backend": "mascvp",
            "checkpoint": "planning_network/MASCVP/main_train_public/best_f1.pt",
            "requires_planning_network": True,
            "planning_network_service_name": "mascvp",
            "default_view_set": "tammes_128",
            "view_count": 128,
            "view_radius": 3.0,
            "network_view_position_radius": 1.0,
            "bootstrap": "init_plus_farthest_point",
            "init_mapping": "nearest_tammes_128_for_view_state_only",
            "grid_size": 64,
            "bbox": [-1.0, 1.0],
            "unknown_occupancy": 0.5,
            "decode": "gamma_0.3",
            "ordering": "tsp",
        },
    ),
    "simple_random_tsporder_5": MethodSpec(
        method_name="simple_random_tsporder_5",
        family="simple",
        display_name="Random-TSPOrder-5",
        execution_mode_compatibility="fixed_budget_only",
        execution_mode="finite_plan",
        command=(
            "bash",
            "algorithms/cpp_enroot/run_cpp_enroot_algorithm.sh",
            "--session-dir",
            "{session_dir}",
            "--cache-index-json",
            "{cache_index_json}",
            "--view-set",
            "tammes_360",
            "--algorithm",
            "random_tsp_order",
            "--budget",
            "5",
            "--seed",
            "42",
        ),
        description="Randomly sample 5 tammes_360 views, then submit them in TSP order.",
        metadata={"selection": "random", "ordering": "tsp_order", "default_view_set": "tammes_360", "budget": 5},
    ),
    "simple_random_tsporder_10": MethodSpec(
        method_name="simple_random_tsporder_10",
        family="simple",
        display_name="Random-TSPOrder-10",
        execution_mode_compatibility="fixed_budget_only",
        execution_mode="finite_plan",
        command=(
            "bash",
            "algorithms/cpp_enroot/run_cpp_enroot_algorithm.sh",
            "--session-dir",
            "{session_dir}",
            "--cache-index-json",
            "{cache_index_json}",
            "--view-set",
            "tammes_360",
            "--algorithm",
            "random_tsp_order",
            "--budget",
            "10",
            "--seed",
            "42",
        ),
        description="Randomly sample 10 tammes_360 views, then submit them in TSP order.",
        metadata={"selection": "random", "ordering": "tsp_order", "default_view_set": "tammes_360", "budget": 10},
    ),
    "simple_random_tsporder_30": MethodSpec(
        method_name="simple_random_tsporder_30",
        family="simple",
        display_name="Random-TSPOrder-30",
        execution_mode_compatibility="fixed_budget_only",
        execution_mode="finite_plan",
        command=(
            "bash",
            "algorithms/cpp_enroot/run_cpp_enroot_algorithm.sh",
            "--session-dir",
            "{session_dir}",
            "--cache-index-json",
            "{cache_index_json}",
            "--view-set",
            "tammes_360",
            "--algorithm",
            "random_tsp_order",
            "--budget",
            "30",
            "--seed",
            "42",
        ),
        description="Randomly sample 30 tammes_360 views, then submit them in TSP order.",
        metadata={"selection": "random", "ordering": "tsp_order", "default_view_set": "tammes_360", "budget": 30},
    ),
    "simple_random_tsporder_50": MethodSpec(
        method_name="simple_random_tsporder_50",
        family="simple",
        display_name="Random-TSPOrder-50",
        execution_mode_compatibility="fixed_budget_only",
        execution_mode="finite_plan",
        command=(
            "bash",
            "algorithms/cpp_enroot/run_cpp_enroot_algorithm.sh",
            "--session-dir",
            "{session_dir}",
            "--cache-index-json",
            "{cache_index_json}",
            "--view-set",
            "tammes_360",
            "--algorithm",
            "random_tsp_order",
            "--budget",
            "50",
            "--seed",
            "42",
        ),
        description="Randomly sample 50 tammes_360 views, then submit them in TSP order.",
        metadata={"selection": "random", "ordering": "tsp_order", "default_view_set": "tammes_360", "budget": 50},
    ),
    "dummy_stop": MethodSpec(
        method_name="dummy_stop",
        family="dummy",
        command=(
            "{python}",
            "algorithms/dummy/dummy_stop.py",
            "--session-dir",
            "{session_dir}",
        ),
        description="Immediately stop with plan_end.",
    ),
    "dummy_candidate_exhausted": MethodSpec(
        method_name="dummy_candidate_exhausted",
        family="dummy",
        command=(
            "{python}",
            "algorithms/dummy/dummy_candidate_exhausted.py",
            "--session-dir",
            "{session_dir}",
        ),
        description="Immediately stop with candidate_exhausted.",
    ),
    "dummy_sequential": MethodSpec(
        method_name="dummy_sequential",
        family="dummy",
        command=(
            "{python}",
            "algorithms/dummy/dummy_sequential.py",
            "--session-dir",
            "{session_dir}",
            "--cache-index-json",
            "{cache_index_json}",
            "--view-set",
            "{view_set}",
        ),
        description="Submit feasible Tammes candidates in index order.",
        metadata={
            "default_view_set": "tammes_128",
        },
    ),
    "dummy_one_move_then_stop": MethodSpec(
        method_name="dummy_one_move_then_stop",
        family="dummy",
        command=(
            "{python}",
            "algorithms/dummy/dummy_one_move_then_stop.py",
            "--session-dir",
            "{session_dir}",
            "--cache-index-json",
            "{cache_index_json}",
        ),
        description="Submit one cached move, then stop.",
    ),
    "dummy_rpc_during_episode": MethodSpec(
        method_name="dummy_rpc_during_episode",
        family="dummy",
        command=(
            "{python}",
            "algorithms/dummy/dummy_rpc_during_episode.py",
            "--session-dir",
            "{session_dir}",
        ),
        description="Query feasibility through file RPC during an episode, then stop.",
    ),
    "dummy_stop_then_hang": MethodSpec(
        method_name="dummy_stop_then_hang",
        family="dummy",
        command=(
            "{python}",
            "algorithms/dummy/dummy_stop_then_hang.py",
            "--session-dir",
            "{session_dir}",
            "--hang-sec",
            "{hang_sec}",
        ),
        description="Submit stop, then keep the process alive.",
    ),
    "dummy_crash_early": MethodSpec(
        method_name="dummy_crash_early",
        family="dummy",
        command=(
            "{python}",
            "algorithms/dummy/dummy_crash_early.py",
            "--session-dir",
            "{session_dir}",
        ),
        description="Exit immediately with a non-zero code.",
    ),
}


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a registered algorithm adapter.")
    parser.add_argument("--method", type=str, default=None, help="Registered method name.")
    parser.add_argument("--session-dir", type=Path, default=None, help="Interaction session directory.")
    parser.add_argument("--cache-index-json", type=Path, default=Path("render_cache/cache_index.json"))
    parser.add_argument("--view-set", type=str, default="tammes_128")
    parser.add_argument("--hang-sec", type=float, default=3600.0)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--list-methods", action="store_true")
    parser.add_argument("--describe-method", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print method lists/descriptions as JSON.")
    return parser


def _format_token(token: str, values: dict[str, str]) -> str:
    out = str(token)
    for key, value in values.items():
        out = out.replace("{" + key + "}", value)
    return out


def build_command(spec: MethodSpec, *, args: argparse.Namespace) -> list[str]:
    if args.session_dir is None:
        raise ValueError("--session-dir is required when launching a method.")
    values = {
        "benchmark_root": str(BENCHMARK_ROOT),
        "python": str(args.python),
        "session_dir": str(args.session_dir),
        "cache_index_json": str(args.cache_index_json),
        "view_set": str(args.view_set),
        "hang_sec": str(args.hang_sec),
    }
    return [_format_token(token, values) for token in spec.command]


def _method_description(spec: MethodSpec) -> dict[str, Any]:
    return {
        "method_name": spec.method_name,
        "display_name": spec.display_name or spec.method_name,
        "family": spec.family,
        "execution_mode_compatibility": spec.execution_mode_compatibility,
        "execution_mode": spec.execution_mode,
        "command": list(spec.command),
        "planning_network_command": None if spec.planning_network_command is None else list(spec.planning_network_command),
        "description": spec.description,
        "metadata": dict(spec.metadata),
    }


def _print_methods(*, as_json: bool) -> None:
    rows = [_method_description(METHODS[name]) for name in sorted(METHODS)]
    if as_json:
        print(json.dumps(rows, indent=2, ensure_ascii=True))
        return
    for row in rows:
        print(
            f"{row['method_name']}\t{row['display_name']}\t{row['family']}\t"
            f"{row['execution_mode_compatibility']}\t{row['execution_mode']}\t{row['description']}"
        )


def _print_method(name: str, *, as_json: bool) -> int:
    spec = METHODS.get(name)
    if spec is None:
        print(f"Unknown method: {name}", file=sys.stderr)
        return 2
    data = _method_description(spec)
    if as_json:
        print(json.dumps(data, indent=2, ensure_ascii=True))
    else:
        print(f"method_name: {data['method_name']}")
        print(f"display_name: {data['display_name']}")
        print(f"family: {data['family']}")
        print(f"execution_mode_compatibility: {data['execution_mode_compatibility']}")
        print(f"execution_mode: {data['execution_mode']}")
        print(f"description: {data['description']}")
        print("command:")
        print("  " + " ".join(data["command"]))
        if data["metadata"]:
            print("metadata:")
            print(json.dumps(data["metadata"], indent=2, ensure_ascii=True))
    return 0


def main() -> int:
    args = _build_argparser().parse_args()

    if args.list_methods:
        _print_methods(as_json=args.json)
        return 0

    if args.describe_method is not None:
        return _print_method(args.describe_method, as_json=args.json)

    if args.method is None:
        print("--method is required unless --list-methods or --describe-method is used.", file=sys.stderr)
        return 2

    spec = METHODS.get(args.method)
    if spec is None:
        print(f"Unknown method: {args.method}", file=sys.stderr)
        return 2

    command = build_command(spec, args=args)
    if args.dry_run:
        if args.json:
            print(
                json.dumps(
                    {
                        "method_name": spec.method_name,
                        "display_name": spec.display_name or spec.method_name,
                        "family": spec.family,
                        "execution_mode_compatibility": spec.execution_mode_compatibility,
                        "execution_mode": spec.execution_mode,
                        "command": command,
                    },
                    indent=2,
                    ensure_ascii=True,
                )
            )
        else:
            print(" ".join(command))
        return 0

    completed = subprocess.run(command, cwd=str(Path.cwd()))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

"""
python launcher_algorithm.py \
  --method simple_random_tsporder_5 \
  --session-dir interaction/session_simple_probe \
  --dry-run
"""
