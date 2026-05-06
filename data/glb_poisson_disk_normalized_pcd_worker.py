# -*- coding: utf-8 -*-
import os
import json
import argparse
import gc
from pathlib import Path

import numpy as np
import open3d as o3d


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json_atomic(data: dict, json_path: str):
    tmp_path = json_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, json_path)


def validate_pcd_file(
    pcd_path: str,
    radius_tol: float = 1e-3,
    center_tol: float = 1e-3,
):
    info = {
        "exists": False,
        "readable": False,
        "num_points": 0,
        "max_radius": None,
        "mean_center_norm": None,
        "all_finite": False,
        "has_colors": False,
        "error": None,
    }

    if not os.path.exists(pcd_path):
        info["error"] = "file_not_found"
        return False, info

    info["exists"] = True

    if os.path.getsize(pcd_path) <= 0:
        info["error"] = "file_empty"
        return False, info

    try:
        pcd = o3d.io.read_point_cloud(pcd_path, print_progress=False)
        if pcd is None:
            info["error"] = "open3d_returned_none"
            return False, info

        pts = np.asarray(pcd.points)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
            info["error"] = "invalid_point_shape"
            return False, info

        info["readable"] = True
        info["num_points"] = int(pts.shape[0])

        all_finite = np.isfinite(pts).all()
        info["all_finite"] = bool(all_finite)
        if not all_finite:
            info["error"] = "non_finite_points"
            return False, info

        cols = np.asarray(pcd.colors)
        info["has_colors"] = bool(
            cols.ndim == 2 and cols.shape[0] == pts.shape[0] and cols.shape[1] == 3
        )

        mean_center = pts.mean(axis=0)
        mean_center_norm = float(np.linalg.norm(mean_center))
        info["mean_center_norm"] = mean_center_norm

        radii = np.linalg.norm(pts, axis=1)
        max_radius = float(radii.max())
        info["max_radius"] = max_radius

        radius_ok = abs(max_radius - 1.0) <= radius_tol
        center_ok = mean_center_norm <= center_tol

        if not radius_ok:
            info["error"] = f"max_radius_not_close_to_1 (got {max_radius})"
            return False, info

        if not center_ok:
            info["error"] = f"mean_center_not_close_to_0 (got {mean_center_norm})"
            return False, info

        return True, info

    except Exception as e:
        info["error"] = str(e)
        return False, info


def load_glb_as_triangle_mesh(glb_path: str):
    mesh = o3d.io.read_triangle_mesh(
        glb_path,
        enable_post_processing=True,
        print_progress=False,
    )

    if mesh is None:
        raise RuntimeError("Open3D returned None for mesh.")
    if len(mesh.vertices) == 0:
        raise RuntimeError("Mesh has no vertices.")
    if len(mesh.triangles) == 0:
        raise RuntimeError("Mesh has no triangles.")

    return mesh


def sample_points_poisson(mesh, num_points: int):
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)

    if pcd is None:
        raise RuntimeError("Poisson sampling returned None.")

    pts = np.asarray(pcd.points)
    if pts.shape[0] == 0:
        raise RuntimeError("Poisson sampling produced zero points.")

    return pcd


def compute_center_and_scale(points: np.ndarray):
    center = points.mean(axis=0)
    dists = np.linalg.norm(points - center[None, :], axis=1)
    scale = float(dists.max())

    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError("Invalid scale computed.")

    return center, scale


def normalize_points(points: np.ndarray, center: np.ndarray, scale: float):
    return (points - center[None, :]) / scale


def save_pcd(points: np.ndarray, out_path: str):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    ok = o3d.io.write_point_cloud(
        out_path,
        pcd,
        write_ascii=False,
        compressed=False,
        print_progress=False,
    )
    if not ok:
        raise RuntimeError(f"Failed to write PCD: {out_path}")


def process_one_file(glb_path: str, output_dir: str, num_points: int, radius_tol: float, center_tol: float):
    uid = Path(glb_path).stem
    out_pcd_path = os.path.join(output_dir, f"{uid}.pcd")

    mesh = None
    sampled_pcd = None
    points = None
    center = None
    normalized_points = None

    try:
        mesh = load_glb_as_triangle_mesh(glb_path)
        sampled_pcd = sample_points_poisson(mesh, num_points=num_points)
        points = np.asarray(sampled_pcd.points)

        center, scale = compute_center_and_scale(points)
        normalized_points = normalize_points(points, center, scale)

        save_pcd(normalized_points, out_pcd_path)

        ok, info = validate_pcd_file(
            out_pcd_path,
            radius_tol=radius_tol,
            center_tol=center_tol,
        )
        if not ok:
            raise RuntimeError(f"Written PCD failed validation: {info}")

        return {
            "uid": uid,
            "translation_center": [float(x) for x in center.tolist()],
            "scale": float(scale),
            "num_points": int(points.shape[0]),
        }

    finally:
        del mesh
        del sampled_pcd
        del points
        del center
        del normalized_points
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Process one GLB into normalized PCD.")
    parser.add_argument("--glb_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_points", type=int, required=True)
    parser.add_argument("--result_json", type=str, required=True)
    parser.add_argument("--radius_tol", type=float, default=1e-3)
    parser.add_argument("--center_tol", type=float, default=1e-3)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(os.path.dirname(args.result_json) or ".")

    record = process_one_file(
        glb_path=args.glb_path,
        output_dir=args.output_dir,
        num_points=args.num_points,
        radius_tol=args.radius_tol,
        center_tol=args.center_tol,
    )
    save_json_atomic(record, args.result_json)


if __name__ == "__main__":
    main()
