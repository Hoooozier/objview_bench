from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from api_evaluation import (
    DEFAULT_COVERAGE_THRESHOLDS,
    fuse_pointclouds,
    load_pointcloud,
    point_cloud_coverage,
    points_from_rgbd_frame,
)
from api_feasibility import FeasibilityAPI
from api_render import CameraPose, Render, intrinsics_from_dict


DEFAULT_CONSTRAINTS = ("whole", "hemi", "quarter")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _threshold_key(threshold: float) -> str:
    return f"{float(threshold):.3f}".rstrip("0").rstrip(".")


def _float_key(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_existing_path(path_text: str, roots: Sequence[Path]) -> Path:
    raw = Path(path_text)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        for root in roots:
            candidates.append((root / raw).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve path {path_text!r}. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def _resolve_gt_pointcloud(
    uid: str,
    *,
    gt_pointcloud_root: Path | None,
    gt_pointcloud_template: str | None,
    gt_pointcloud_map: dict[str, Any] | None,
    cwd: Path,
) -> Path:
    if gt_pointcloud_map is not None and uid in gt_pointcloud_map:
        return _resolve_existing_path(str(gt_pointcloud_map[uid]), [cwd])

    if gt_pointcloud_root is None or gt_pointcloud_template is None:
        raise ValueError(
            "GT point cloud is required. Provide --gt-pointcloud-root and "
            "--gt-pointcloud-template, or --gt-pointcloud-map-json."
        )

    rel = gt_pointcloud_template.format(uid=uid)
    return _resolve_existing_path(rel, [gt_pointcloud_root])


def _pose_from_entry(entry: dict[str, Any]) -> CameraPose:
    pose = entry.get("pose")
    if not isinstance(pose, dict):
        raise ValueError("Each cache view entry must contain pose dict")
    return CameraPose(
        camera_xyz=tuple(float(v) for v in pose["camera_xyz"]),
        lookat_xyz=tuple(float(v) for v in pose["lookat_xyz"]),
        roll_rad=float(pose["roll_rad"]),
    )


def _view_entries_for_uid(
    cache_index: dict[str, Any],
    uid: str,
    view_set: str,
) -> list[dict[str, Any]]:
    objects = cache_index.get("objects")
    if not isinstance(objects, dict):
        raise ValueError("cache index must contain top-level objects dict")
    object_record = objects.get(uid)
    if not isinstance(object_record, dict):
        raise KeyError(f"uid={uid} not found in cache index")
    views = object_record.get("views")
    if not isinstance(views, list):
        raise ValueError(f"cache entry for uid={uid} must contain views list")
    return [entry for entry in views if entry.get("view_set") == view_set]


def _constraint_json_path(name: str, config_root: Path) -> Path:
    path = config_root / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Constraint JSON not found for {name!r}: {path}")
    return path


def _cache_paths(
    output_root: Path,
    constraint_name: str,
    uid: str,
    observable_view_set: str,
    fusion_voxel_size: float,
) -> tuple[Path, Path]:
    stem = f"{uid}__{observable_view_set}__fusion_voxel_{_float_key(fusion_voxel_size)}"
    out_dir = output_root / "observable_reference" / constraint_name
    return out_dir / f"{stem}.npz", out_dir / f"{stem}.json"


def _write_reference_cache(
    npz_path: Path,
    json_path: Path,
    *,
    observable_pointcloud: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_npz = npz_path.with_suffix(npz_path.suffix + ".tmp")
    tmp_json = json_path.with_suffix(json_path.suffix + ".tmp")

    with tmp_npz.open("wb") as f:
        np.savez_compressed(
            f,
            observable_pointcloud=observable_pointcloud.astype(np.float32),
        )
    with tmp_json.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    tmp_npz.replace(npz_path)
    tmp_json.replace(json_path)


def build_observable_reference(
    *,
    uid: str,
    cache_index: dict[str, Any],
    cache_index_json: Path,
    obj_root: Path,
    gt_pointcloud_path: Path,
    constraint_name: str,
    constraint_json: Path,
    observable_view_set: str,
    fusion_voxel_size: float,
    coverage_thresholds: Sequence[float],
    output_root: Path,
    device: str,
    overwrite: bool,
) -> dict[str, Any]:
    npz_path, json_path = _cache_paths(
        output_root,
        constraint_name,
        uid,
        observable_view_set,
        fusion_voxel_size,
    )
    if npz_path.exists() and json_path.exists() and not overwrite:
        return {
            "uid": uid,
            "constraint": constraint_name,
            "status": "skipped_exists",
            "npz": str(npz_path),
            "json": str(json_path),
        }

    objects = cache_index["objects"]
    object_record = objects[uid]
    cache_index_parent = cache_index_json.parent
    benchmark_root = cache_index_parent.parent
    obj_path = _resolve_existing_path(
        str(object_record["obj_path"]),
        [
            obj_root,
            Path.cwd(),
            cache_index_parent,
            benchmark_root,
            benchmark_root.parent / "object_dataset",
        ],
    )
    intrinsics = intrinsics_from_dict(cache_index["intrinsics"])
    renderer = Render(
        uid=uid,
        obj_path=obj_path,
        intrinsics=intrinsics,
        device=device,
        mode="cache",
        cache_index_json=cache_index_json,
    )
    feasibility = FeasibilityAPI(constraint_json)

    entries = _view_entries_for_uid(cache_index, uid, observable_view_set)
    if not entries:
        raise ValueError(f"No views found for uid={uid}, view_set={observable_view_set!r}")

    start = time.perf_counter()
    pointclouds: list[np.ndarray] = []
    feasible_view_indices: list[int] = []
    for entry in entries:
        pose = _pose_from_entry(entry)
        if not feasibility.is_feasible(pose):
            continue
        frame = renderer.render_loaded(pose)
        pointclouds.append(points_from_rgbd_frame(frame))
        feasible_view_indices.append(int(entry.get("view_idx", len(feasible_view_indices))))

    if not pointclouds:
        raise ValueError(
            f"No feasible observable views for uid={uid}, constraint={constraint_name}, "
            f"view_set={observable_view_set}"
        )

    observable_pointcloud = fuse_pointclouds(pointclouds, voxel_size=fusion_voxel_size)
    build_sec = time.perf_counter() - start

    gt_pointcloud = load_pointcloud(gt_pointcloud_path)
    ratio_start = time.perf_counter()
    observable_surface_ratio = {
        _threshold_key(t): point_cloud_coverage(gt_pointcloud, observable_pointcloud, threshold=float(t))
        for t in coverage_thresholds
    }
    ratio_sec = time.perf_counter() - ratio_start

    metadata = {
        "uid": uid,
        "constraint": constraint_name,
        "constraint_json": str(constraint_json),
        "constraint_json_sha256": _sha256_file(constraint_json),
        "observable_view_set": observable_view_set,
        "fusion_voxel_size": float(fusion_voxel_size),
        "coverage_thresholds": [float(t) for t in coverage_thresholds],
        "observable_surface_ratio": observable_surface_ratio,
        "num_observable_views": int(len(feasible_view_indices)),
        "num_candidate_views": int(len(entries)),
        "num_observable_points": int(len(observable_pointcloud)),
        "feasible_view_indices": feasible_view_indices,
        "source_cache_index_json": str(cache_index_json),
        "gt_pointcloud_path": str(gt_pointcloud_path),
        "obj_path": str(obj_path),
        "build_sec": float(build_sec),
        "ratio_sec": float(ratio_sec),
        "created_by": "eval_cache_worker.py",
    }
    _write_reference_cache(
        npz_path,
        json_path,
        observable_pointcloud=observable_pointcloud,
        metadata=metadata,
    )
    return {
        "uid": uid,
        "constraint": constraint_name,
        "status": "built",
        "num_observable_views": int(len(feasible_view_indices)),
        "num_observable_points": int(len(observable_pointcloud)),
        "build_sec": float(build_sec),
        "ratio_sec": float(ratio_sec),
        "npz": str(npz_path),
        "json": str(json_path),
    }


def _load_uid_list(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(x["uid"] if isinstance(x, dict) and "uid" in x else x) for x in data]
    if isinstance(data, dict):
        if "uids" in data and isinstance(data["uids"], list):
            return [str(x) for x in data["uids"]]
        if "objects" in data and isinstance(data["objects"], dict):
            return [str(x) for x in data["objects"].keys()]
    raise ValueError(f"Cannot parse uid list from {path}")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build evaluator observable-reference cache from render_cache frames."
    )
    parser.add_argument("--cache-index-json", type=Path, default=Path("render_cache/cache_index.json"))
    parser.add_argument("--output-root", type=Path, default=Path("eval_cache"))
    parser.add_argument("--observable-view-set", type=str, default="tammes_360")
    parser.add_argument("--constraints", nargs="+", default=list(DEFAULT_CONSTRAINTS))
    parser.add_argument("--constraint-config-root", type=Path, default=Path("configs/feasibility"))
    parser.add_argument("--uid", action="append", default=None, help="Build one uid. Can be repeated.")
    parser.add_argument("--uid-list-json", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0, help="Start index in selected uid list, inclusive.")
    parser.add_argument("--end", type=int, default=-1, help="End index in selected uid list, exclusive. Use -1 for the end.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--obj-root", type=Path, default=Path("."))
    parser.add_argument("--gt-pointcloud-root", type=Path, default=None)
    parser.add_argument(
        "--gt-pointcloud-template",
        type=str,
        default=None,
        help="Template relative to --gt-pointcloud-root, e.g. '{uid}/{uid}.pcd' or '{uid}.npy'.",
    )
    parser.add_argument("--gt-pointcloud-map-json", type=Path, default=None)
    parser.add_argument("--fusion-voxel-size", type=float, default=0.005)
    parser.add_argument("--coverage-thresholds", type=float, nargs="+", default=list(DEFAULT_COVERAGE_THRESHOLDS))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    cache_index_json = args.cache_index_json.resolve()
    cache_index = _load_json(cache_index_json)
    objects = cache_index.get("objects")
    if not isinstance(objects, dict):
        raise ValueError("cache index must contain objects dict")

    if args.uid:
        uids = [str(uid) for uid in args.uid]
    elif args.uid_list_json is not None:
        uids = _load_uid_list(args.uid_list_json)
    else:
        uids = list(objects.keys())
    start = int(args.start)
    if start < 0:
        raise ValueError("--start must be non-negative")
    end_arg = int(args.end)
    end = None if end_arg < 0 else end_arg
    if end is not None and end < start:
        raise ValueError("--end must be greater than or equal to --start")
    uids = uids[start:end]
    if args.limit is not None:
        uids = uids[: int(args.limit)]

    gt_pointcloud_map = None
    if args.gt_pointcloud_map_json is not None:
        gt_pointcloud_map = _load_json(args.gt_pointcloud_map_json)

    total_start = time.perf_counter()
    results = []
    for uid in uids:
        if uid not in objects:
            raise KeyError(f"uid={uid} not found in cache index")
        gt_pointcloud_path = _resolve_gt_pointcloud(
            uid,
            gt_pointcloud_root=args.gt_pointcloud_root,
            gt_pointcloud_template=args.gt_pointcloud_template,
            gt_pointcloud_map=gt_pointcloud_map,
            cwd=Path.cwd(),
        )
        for constraint_name in args.constraints:
            constraint_json = _constraint_json_path(str(constraint_name), args.constraint_config_root)
            item_start = time.perf_counter()
            result = build_observable_reference(
                uid=uid,
                cache_index=cache_index,
                cache_index_json=cache_index_json,
                obj_root=args.obj_root,
                gt_pointcloud_path=gt_pointcloud_path,
                constraint_name=str(constraint_name),
                constraint_json=constraint_json,
                observable_view_set=args.observable_view_set,
                fusion_voxel_size=args.fusion_voxel_size,
                coverage_thresholds=args.coverage_thresholds,
                output_root=args.output_root,
                device=args.device,
                overwrite=args.overwrite,
            )
            result["elapsed_sec"] = float(time.perf_counter() - item_start)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False))

    print(json.dumps(
        {
            "status": "done",
            "num_items": len(results),
            "uid_start": start,
            "uid_end": end,
            "num_uids": len(uids),
            "total_sec": float(time.perf_counter() - total_start),
            "output_root": str(args.output_root),
        },
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
