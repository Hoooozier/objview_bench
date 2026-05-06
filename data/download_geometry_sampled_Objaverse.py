# -*- coding: utf-8 -*-
import os
import json
import shutil
import argparse
import multiprocessing
from pathlib import Path

import objaverse

# Optional imports for validation mode
try:
    import trimesh
except ImportError:
    trimesh = None

try:
    import open3d as o3d
except ImportError:
    o3d = None


# =========================
# Default settings
# =========================

JSON_PATH = "cleaned_attribute.json"
OUTPUT_ROOT = os.getcwd()
TARGET_GLB_DIR = os.path.join(OUTPUT_ROOT, "geometry_sampled", "glb")
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".objaverse")


# =========================
# Helper functions
# =========================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_uids_from_json(json_path):
    """
    Load all UIDs from a filtered JSON file.

    Expected JSON format:
    [
        {
            "UID": "...",
            ...
        },
        ...
    ]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    uids = []
    for item in data:
        uid = item.get("uid") or item.get("UID") 
        if uid is not None:
            uids.append(str(uid))

    # Deduplicate while preserving order
    seen = set()
    unique_uids = []
    for uid in uids:
        if uid not in seen:
            seen.add(uid)
            unique_uids.append(uid)

    return unique_uids


def clear_objaverse_cache(cache_dir=CACHE_DIR):
    """
    Remove the default objaverse cache directory (~/.objaverse).
    """
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"[INFO] Removed objaverse cache: {cache_dir}")
    else:
        print(f"[INFO] Objaverse cache not found, nothing to clear: {cache_dir}")


def get_target_glb_path(uid, target_dir=TARGET_GLB_DIR):
    """
    Final stored file path: geometry_sampled/glb/<uid>.glb
    """
    return os.path.join(target_dir, f"{uid}.glb")


def copy_downloaded_objects_to_target(objects, target_dir=TARGET_GLB_DIR):
    """
    Copy downloaded objaverse files to geometry_sampled/glb/<uid>.glb

    objects: dict[uid] -> downloaded local path
    """
    ensure_dir(target_dir)

    copied = 0
    failed = []

    for uid, src_path in objects.items():
        if src_path is None:
            failed.append(uid)
            print(f"[WARN] Download path is None for UID={uid}")
            continue

        if not os.path.exists(src_path):
            failed.append(uid)
            print(f"[WARN] Downloaded file does not exist for UID={uid}: {src_path}")
            continue

        dst_path = get_target_glb_path(uid, target_dir)

        try:
            shutil.copy2(src_path, dst_path)
            copied += 1
        except Exception as e:
            failed.append(uid)
            print(f"[ERROR] Failed to copy UID={uid} from {src_path} to {dst_path}: {e}")

    print(f"[INFO] Copied {copied} files to {target_dir}")
    if failed:
        print(f"[WARN] Failed to copy {len(failed)} files.")

    return copied, failed


def list_missing_uids(uids, target_dir=TARGET_GLB_DIR):
    """
    Check which UIDs are missing in geometry_sampled/glb.
    """
    missing = []
    existing = 0

    for uid in uids:
        dst_path = get_target_glb_path(uid, target_dir)
        if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
            existing += 1
        else:
            missing.append(uid)

    print(f"[INFO] Existing files: {existing}")
    print(f"[INFO] Missing files: {len(missing)}")
    return missing


def download_uids(uids, download_processes=None):
    """
    Download Objaverse objects for the given UIDs.
    Returns dict: uid -> local downloaded path
    """
    if not uids:
        print("[INFO] No UIDs to download.")
        return {}

    if download_processes is None or download_processes <= 0:
        download_processes = multiprocessing.cpu_count()

    print(f"[INFO] objaverse version: {objaverse.__version__}")
    print(f"[INFO] Downloading {len(uids)} objects with download_processes={download_processes}")

    objects = objaverse.load_objects(
        uids=uids,
        download_processes=download_processes,
    )

    print(f"[INFO] objaverse returned {len(objects)} downloaded entries")
    return objects


def validate_with_trimesh(file_path):
    """
    Validate whether a mesh file can be loaded by trimesh.
    """
    if trimesh is None:
        return {
            "ok": False,
            "error": "trimesh is not installed"
        }

    try:
        loaded = trimesh.load(file_path, force="scene")
        if loaded is None:
            return {"ok": False, "error": "trimesh.load returned None"}

        # Basic validity check
        if isinstance(loaded, trimesh.Scene):
            geom_count = len(loaded.geometry)
            return {
                "ok": geom_count > 0,
                "error": None if geom_count > 0 else "empty scene",
                "type": "Scene",
                "geometry_count": geom_count
            }
        else:
            v = len(loaded.vertices) if hasattr(loaded, "vertices") else 0
            f = len(loaded.faces) if hasattr(loaded, "faces") else 0
            return {
                "ok": v > 0,
                "error": None if v > 0 else "mesh has no vertices",
                "type": type(loaded).__name__,
                "vertices": v,
                "faces": f
            }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def validate_with_open3d(file_path):
    """
    Validate whether a mesh file can be loaded by open3d.
    """
    if o3d is None:
        return {
            "ok": False,
            "error": "open3d is not installed"
        }

    try:
        mesh = o3d.io.read_triangle_mesh(file_path)
        if mesh is None:
            return {"ok": False, "error": "open3d returned None"}

        v = len(mesh.vertices)
        t = len(mesh.triangles)

        return {
            "ok": v > 0,
            "error": None if v > 0 else "mesh has no vertices",
            "vertices": v,
            "triangles": t
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def validate_all_meshes(target_dir=TARGET_GLB_DIR, report_path=None):
    """
    Validate all .glb files in target_dir with trimesh and open3d.
    """
    ensure_dir(target_dir)

    files = sorted([
        p for p in Path(target_dir).glob("*.glb")
        if p.is_file()
    ])

    print(f"[INFO] Found {len(files)} .glb files in {target_dir}")

    report = []
    trimesh_ok_count = 0
    open3d_ok_count = 0
    both_ok_count = 0

    for i, file_path in enumerate(files, 1):
        uid = file_path.stem

        tri_result = validate_with_trimesh(str(file_path))
        o3d_result = validate_with_open3d(str(file_path))

        tri_ok = tri_result.get("ok", False)
        o3d_ok = o3d_result.get("ok", False)

        if tri_ok:
            trimesh_ok_count += 1
        if o3d_ok:
            open3d_ok_count += 1
        if tri_ok and o3d_ok:
            both_ok_count += 1

        record = {
            "uid": uid,
            "file": str(file_path),
            "trimesh": tri_result,
            "open3d": o3d_result
        }
        report.append(record)

        if i % 100 == 0 or i == len(files):
            print(f"[INFO] Validated {i}/{len(files)} files")

    summary = {
        "total_files": len(files),
        "trimesh_ok": trimesh_ok_count,
        "open3d_ok": open3d_ok_count,
        "both_ok": both_ok_count,
        "trimesh_available": trimesh is not None,
        "open3d_available": o3d is not None,
    }

    result = {
        "summary": summary,
        "details": report
    }

    if report_path is None:
        report_path = os.path.join(target_dir, "mesh_validation_report.json")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("[INFO] Validation summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[INFO] Report saved to: {report_path}")


# =========================
# Modes
# =========================

def mode_download_all(json_path, target_dir, download_processes):
    """
    Download all UIDs from JSON, copy to target dir, then clear cache.
    """
    uids = load_uids_from_json(json_path)
    print(f"[INFO] Loaded {len(uids)} UIDs from {json_path}")

    if len(uids) == 0:
        print("[WARN] No valid UIDs found in the JSON file.")
        return

    objects = download_uids(uids, download_processes=download_processes)
    copy_downloaded_objects_to_target(objects, target_dir)
    clear_objaverse_cache()


def mode_redownload_missing(json_path, target_dir, download_processes):
    """
    Check missing files in target_dir, redownload only missing ones,
    copy them, then clear cache.
    """
    uids = load_uids_from_json(json_path)
    print(f"[INFO] Loaded {len(uids)} UIDs from {json_path}")

    if len(uids) == 0:
        print("[WARN] No valid UIDs found in the JSON file.")
        return

    ensure_dir(target_dir)

    missing_uids = list_missing_uids(uids, target_dir=target_dir)

    if len(missing_uids) == 0:
        print("[INFO] All files already exist. Nothing to redownload.")
        clear_objaverse_cache()
        return

    objects = download_uids(missing_uids, download_processes=download_processes)
    copy_downloaded_objects_to_target(objects, target_dir)
    clear_objaverse_cache()


def mode_validate_mesh_loaders(target_dir, report_path=None):
    """
    Validate all mesh files in target_dir with trimesh and open3d.
    """
    validate_all_meshes(target_dir=target_dir, report_path=report_path)


# =========================
# Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Objaverse download / redownload / mesh validation utility"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["download_all", "redownload_missing", "validate_mesh_loaders"],
        help="Run mode"
    )

    parser.add_argument(
        "--json_path",
        type=str,
        default=JSON_PATH,
        help="Path to cleaned_attribute.json"
    )

    parser.add_argument(
        "--target_dir",
        type=str,
        default=TARGET_GLB_DIR,
        help="Directory to store copied .glb files"
    )

    parser.add_argument(
        "--download_processes",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel download processes"
    )

    parser.add_argument(
        "--report_path",
        type=str,
        default=None,
        help="Output path for mesh validation JSON report"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] Mode: {args.mode}")
    print(f"[INFO] JSON path: {args.json_path}")
    print(f"[INFO] Target dir: {args.target_dir}")

    if args.mode == "download_all":
        mode_download_all(
            json_path=args.json_path,
            target_dir=args.target_dir,
            download_processes=args.download_processes,
        )

    elif args.mode == "redownload_missing":
        mode_redownload_missing(
            json_path=args.json_path,
            target_dir=args.target_dir,
            download_processes=args.download_processes,
        )

    elif args.mode == "validate_mesh_loaders":
        mode_validate_mesh_loaders(
            target_dir=args.target_dir,
            report_path=args.report_path,
        )

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()

"""
python download_geometry_sampled_Objaverse.py \
    --mode download_all \
    --json_path geometry_sampled_12000.json \
    --target_dir geometry_sampled/glb

python download_geometry_sampled_Objaverse.py \
    --mode redownload_missing \
    --json_path geometry_sampled_12000.json \
    --target_dir geometry_sampled/glb

python download_geometry_sampled_Objaverse.py \
    --mode validate_mesh_loaders \
    --target_dir geometry_sampled/glb \
    --report_path geometry_sampled/glb_mesh_validation_report.json
"""
