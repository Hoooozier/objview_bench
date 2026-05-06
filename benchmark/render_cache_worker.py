from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from api_render import RGBDFrame


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def _load_json_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _load_uid_entries(long_tail_json: Path, test_json: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for path in [long_tail_json, test_json]:
        for entry in _load_json_list(path):
            uid = str(entry["uid"])
            entries[uid] = entry
    return entries


def _load_xyz_file(path: Path) -> list[tuple[float, float, float]]:
    xyzs: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_idx} must contain exactly 3 floats, got {line!r}")
            xyz = tuple(float(x) * 3.0 for x in parts)
            xyzs.append(xyz)
    return xyzs


def _build_view_sets(args: argparse.Namespace) -> dict[str, list[tuple[float, float, float]]]:
    available = {
        "benchmark_start_3": _load_xyz_file(Path(args.benchmark_start_3)),
        "tammes_128": _load_xyz_file(Path(args.tammes_128)),
        "tammes_360": _load_xyz_file(Path(args.tammes_360)),
    }
    requested = [str(name) for name in args.view_sets]
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise ValueError(
            f"Unknown --view-sets entries: {unknown}. "
            f"Available view sets: {sorted(available)}"
        )
    return {name: available[name] for name in requested}


def _relative_to(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _resolve_obj_path(obj_root: Path, entry: dict[str, Any]) -> Path:
    uid = str(entry["uid"])
    obj_rel = Path(str(entry["obj_path"]))

    candidates = [
        (obj_root / obj_rel).resolve(),
        (obj_root / uid / f"{uid}.obj").resolve(),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "OBJ not found for uid="
        f"{uid}. Tried: {', '.join(str(p) for p in candidates)}"
    )


def _merge_cache_indices(
    index_paths: list[Path],
    output_path: Path,
    *,
    allow_overwrite: bool = False,
) -> dict[str, int]:
    if len(index_paths) < 2:
        raise ValueError("At least two index files are required for merge-index")

    merged: dict[str, Any] | None = None
    uid_count = 0
    view_count = 0

    def _cache_spec_without_view_sets(spec: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in spec.items() if key != "view_sets"}

    def _view_key(view: dict[str, Any]) -> tuple[str, int]:
        return (str(view.get("view_set")), int(view.get("view_idx", -1)))

    for index_path in index_paths:
        data = _load_json_dict(index_path)
        for key in ("intrinsics", "cache_spec", "objects"):
            if key not in data:
                raise ValueError(f"{index_path} is missing required key {key!r}")

        cache_spec = data["cache_spec"]
        if not isinstance(cache_spec, dict):
            raise ValueError(f"{index_path} field 'cache_spec' must be a dict")
        objects = data["objects"]
        if not isinstance(objects, dict):
            raise ValueError(f"{index_path} field 'objects' must be a dict")

        if merged is None:
            merged = {
                "intrinsics": data["intrinsics"],
                "cache_spec": dict(cache_spec),
                "objects": {},
            }
            merged["cache_spec"]["view_sets"] = list(cache_spec.get("view_sets", []))
        else:
            if data["intrinsics"] != merged["intrinsics"]:
                raise ValueError(f"intrinsics mismatch in {index_path}")
            if _cache_spec_without_view_sets(cache_spec) != _cache_spec_without_view_sets(merged["cache_spec"]):
                raise ValueError(f"cache_spec mismatch in {index_path}")
            merged_view_sets = list(merged["cache_spec"].get("view_sets", []))
            for view_set in cache_spec.get("view_sets", []):
                if view_set not in merged_view_sets:
                    merged_view_sets.append(view_set)
            merged["cache_spec"]["view_sets"] = merged_view_sets

        for uid, object_record in objects.items():
            uid = str(uid)
            if not isinstance(object_record, dict):
                raise ValueError(f"{index_path} object record for uid={uid} must be a dict")
            incoming_views = object_record.get("views", [])
            if not isinstance(incoming_views, list):
                raise ValueError(f"{index_path} object record for uid={uid} has invalid views")

            if uid not in merged["objects"]:
                merged["objects"][uid] = dict(object_record)
                merged["objects"][uid]["views"] = list(incoming_views)
                continue

            merged_record = merged["objects"][uid]
            merged_views = merged_record.get("views", [])
            if not isinstance(merged_views, list):
                raise ValueError(f"merged object record for uid={uid} has invalid views")

            if object_record.get("obj_path") != merged_record.get("obj_path"):
                raise ValueError(f"obj_path mismatch for duplicate uid={uid} while merging {index_path}")

            view_by_key = {_view_key(view): idx for idx, view in enumerate(merged_views)}
            for view in incoming_views:
                key = _view_key(view)
                if key in view_by_key:
                    if not allow_overwrite:
                        raise ValueError(
                            f"Duplicate uid/view entry {uid}/{key[0]}/{key[1]} while merging {index_path}"
                        )
                    merged_views[view_by_key[key]] = view
                else:
                    view_by_key[key] = len(merged_views)
                    merged_views.append(view)

            merged_views.sort(key=lambda view: (str(view.get("view_set")), int(view.get("view_idx", -1))))
            merged_record["views"] = merged_views

    assert merged is not None
    sorted_objects = {uid: merged["objects"][uid] for uid in sorted(merged["objects"])}
    merged["objects"] = sorted_objects

    uid_count = len(sorted_objects)
    view_count = sum(
        len(record.get("views", []))
        for record in sorted_objects.values()
        if isinstance(record, dict)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
        f.write("\n")
    tmp_path.replace(output_path)

    return {"uid_count": uid_count, "view_count": int(view_count)}


def _save_frame_outputs(
    frame: "RGBDFrame",
    out_dir: Path,
    *,
    uid: str,
    view_set: str,
    view_idx: int,
    save_points_world: bool,
) -> dict[str, str]:
    from api_render import frame_meta_dict

    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = out_dir / "rgb.png"
    depth_path = out_dir / "depth.npz"
    mask_path = out_dir / "mask.png"
    meta_path = out_dir / "frame_meta.json"

    Image.fromarray(frame.rgb).save(rgb_path)
    np.savez_compressed(depth_path, depth=frame.depth.astype(np.float32))
    Image.fromarray(frame.mask.astype(np.uint8) * 255).save(mask_path)

    meta = frame_meta_dict(
        frame,
        extra={
            "uid": uid,
            "view_set": view_set,
            "view_idx": int(view_idx),
            "depth_semantics": "standard_rgbd_z",
        },
    )
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    result = {
        "rgb": str(rgb_path),
        "depth": str(depth_path),
        "mask": str(mask_path),
        "frame_meta": str(meta_path),
    }

    if save_points_world and frame.points_world is not None:
        points_world_path = out_dir / "points_world.npy"
        np.save(points_world_path, frame.points_world.astype(np.float32))
        result["points_world"] = str(points_world_path)

    return result


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pre-render benchmark RGB-D cache")

    parser.add_argument(
        "--mode",
        type=str,
        default="render",
        choices=["render", "merge-index"],
        help="render: build cache frames/index; merge-index: merge existing cache index JSON files.",
    )
    parser.add_argument(
        "--merge-indices",
        type=str,
        nargs="+",
        default=None,
        help="Input cache index JSON files for --mode merge-index.",
    )
    parser.add_argument(
        "--merge-output",
        type=str,
        default=None,
        help="Output cache index JSON path for --mode merge-index.",
    )
    parser.add_argument(
        "--merge-allow-overwrite",
        action="store_true",
        help="Allow later merge indices to overwrite duplicate uid entries.",
    )
    parser.add_argument(
        "--cache-root",
        type=str,
        default=None,
        help="Output cache root. cache_index.json and frames/ will be created here.",
    )
    parser.add_argument(
        "--obj-root",
        type=str,
        default=None,
        help="Root directory that contains object_dataset assets referenced by obj_path fields.",
    )
    parser.add_argument(
        "--long-tail-json",
        type=str,
        default="final_clean_pool/final_clean_long_tail_pool.json",
    )
    parser.add_argument(
        "--test-json",
        type=str,
        default="released_benchmark_splits/main_hidden_test.json",
    )
    parser.add_argument(
        "--benchmark-start-3",
        type=str,
        default="Tammes_sphere/benchmark_start_3_xyz.txt",
    )
    parser.add_argument(
        "--tammes-128",
        type=str,
        default="Tammes_sphere/128_xyz.txt",
    )
    parser.add_argument(
        "--tammes-360",
        type=str,
        default="Tammes_sphere/360_xyz.txt",
    )
    parser.add_argument(
        "--view-sets",
        type=str,
        nargs="+",
        default=["benchmark_start_3", "tammes_128", "tammes_360"],
        help=(
            "Subset of view sets to render. "
            "Choices: benchmark_start_3 tammes_128 tammes_360"
        ),
    )
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--fov-x-deg", type=float, default=45.0)
    parser.add_argument("--fov-y-deg", type=float, default=45.0)
    parser.add_argument("--principal-x", type=float, default=256.0)
    parser.add_argument("--principal-y", type=float, default=256.0)
    parser.add_argument("--lookat-x", type=float, default=0.0)
    parser.add_argument("--lookat-y", type=float, default=0.0)
    parser.add_argument("--lookat-z", type=float, default=0.0)
    parser.add_argument("--roll-deg", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save-points-world", action="store_true")
    parser.add_argument("--limit-uids", type=int, default=None)
    parser.add_argument("--start", type=int, default=0, help="Start index in sorted uid list.")
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="End index (exclusive) in sorted uid list. Use -1 for the end of the list.",
    )
    return parser


def main() -> int:
    args = _build_argparser().parse_args()

    if args.mode == "merge-index":
        if not args.merge_indices:
            raise ValueError("--merge-indices is required when --mode merge-index")
        output_path = (
            Path(args.merge_output).resolve()
            if args.merge_output is not None
            else Path("render_cache/cache_index.json").resolve()
        )
        stats = _merge_cache_indices(
            [Path(p).resolve() for p in args.merge_indices],
            output_path,
            allow_overwrite=bool(args.merge_allow_overwrite),
        )
        print("[DONE]")
        print(f"index_json : {output_path}")
        print(f"uid_count  : {stats['uid_count']}")
        print(f"view_count : {stats['view_count']}")
        return 0

    if args.cache_root is None:
        raise ValueError("--cache-root is required when --mode render")
    if args.obj_root is None:
        raise ValueError("--obj-root is required when --mode render")

    from api_render import (
        CameraIntrinsics,
        CameraPose,
        Render,
        intrinsics_to_dict,
    )

    cache_root = Path(args.cache_root).resolve()
    frames_root = cache_root / "frames"
    cache_root.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)

    intr = CameraIntrinsics(
        image_width=args.image_width,
        image_height=args.image_height,
        fov_x_rad=np.deg2rad(args.fov_x_deg),
        fov_y_rad=np.deg2rad(args.fov_y_deg),
        principal_x=args.principal_x,
        principal_y=args.principal_y,
    )
    lookat_xyz = (float(args.lookat_x), float(args.lookat_y), float(args.lookat_z))
    roll_rad = float(np.deg2rad(args.roll_deg))

    uid_entries = _load_uid_entries(
        Path(args.long_tail_json),
        Path(args.test_json),
    )
    sorted_uids = sorted(uid_entries)
    start = int(args.start)
    if start < 0:
        raise ValueError(f"--start must be >= 0, got {start}")
    end = len(sorted_uids) if int(args.end) < 0 else int(args.end)
    if end < start:
        raise ValueError(f"--end must be >= --start, got start={start}, end={end}")
    sorted_uids = sorted_uids[start:end]
    if args.limit_uids is not None:
        sorted_uids = sorted_uids[: int(args.limit_uids)]

    view_sets = _build_view_sets(args)

    cache_index: dict[str, Any] = {
        "intrinsics": intrinsics_to_dict(intr),
        "cache_spec": {
            "lookat_xyz": [float(x) for x in lookat_xyz],
            "roll_rad": float(roll_rad),
            "depth_semantics": "standard_rgbd_z",
            "version": "v1",
            "view_sets": list(view_sets.keys()),
        },
        "objects": {},
    }

    total_views = 0
    for uid_idx, uid in enumerate(sorted_uids, start=1):
        entry = uid_entries[uid]
        obj_rel = Path(str(entry["obj_path"]))
        obj_path = _resolve_obj_path(Path(args.obj_root), entry)

        print(f"[{uid_idx}/{len(sorted_uids)}] uid={uid}")

        renderer = Render(
            uid=uid,
            obj_path=obj_path,
            intrinsics=intr,
            device=args.device,
            mode="online",
        )

        object_record = {
            "obj_path": str(obj_rel).replace("\\", "/"),
            "views": [],
        }

        for view_set_name, xyzs in view_sets.items():
            for view_idx, camera_xyz in enumerate(xyzs):
                pose = CameraPose(
                    camera_xyz=tuple(float(x) for x in camera_xyz),
                    lookat_xyz=lookat_xyz,
                    roll_rad=roll_rad,
                )
                frame = renderer.render_loaded(pose)

                view_dir = frames_root / uid / view_set_name / f"view_{view_idx:03d}"
                saved = _save_frame_outputs(
                    frame,
                    view_dir,
                    uid=uid,
                    view_set=view_set_name,
                    view_idx=view_idx,
                    save_points_world=args.save_points_world,
                )

                view_record = {
                    "pose": {
                        "camera_xyz": [float(x) for x in pose.camera_xyz],
                        "lookat_xyz": [float(x) for x in pose.lookat_xyz],
                        "roll_rad": float(pose.roll_rad),
                    },
                    "rgb": _relative_to(Path(saved["rgb"]), cache_root),
                    "depth": _relative_to(Path(saved["depth"]), cache_root),
                    "mask": _relative_to(Path(saved["mask"]), cache_root),
                    "frame_meta": _relative_to(Path(saved["frame_meta"]), cache_root),
                    "view_set": view_set_name,
                    "view_idx": int(view_idx),
                }
                if "points_world" in saved:
                    view_record["points_world"] = _relative_to(Path(saved["points_world"]), cache_root)

                object_record["views"].append(view_record)
                total_views += 1

        cache_index["objects"][uid] = object_record

        cache_index_path = cache_root / "cache_index.json"
        with cache_index_path.open("w", encoding="utf-8") as f:
            json.dump(cache_index, f, indent=2)

    print("[DONE]")
    print(f"cache_root : {cache_root}")
    print(f"uid_count  : {len(sorted_uids)}")
    print(f"view_count : {total_views}")
    print(f"index_json : {cache_root / 'cache_index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
