# -*- coding: utf-8 -*-

import os
import gc
import json
import math
import argparse
import multiprocessing as mp
from typing import Any, Dict, List, Optional

import numpy as np
import trimesh
import objaverse

# ============================================================
# JSON helpers
# ============================================================

def load_items_from_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Load a JSON list from disk.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")

    return data


def extract_uids(items: List[Dict[str, Any]]) -> List[str]:
    """
    Extract UID list from input items.
    """
    uids = []
    seen = set()

    for item in items:
        uid = item.get("UID")
        if uid is None:
            continue
        if uid in seen:
            continue
        seen.add(uid)
        uids.append(uid)

    return uids


# ============================================================
# Mesh loading / lightweight checks
# ============================================================

def load_mesh_safely(mesh_path: str) -> trimesh.Trimesh:
    """
    Load mesh from file safely.

    Key point:
    - If the file is loaded as a Scene, apply scene transforms before merging.
    - Use Scene.to_geometry() instead of deprecated Scene.dump(concatenate=True).
    """
    loaded = trimesh.load(mesh_path, force="scene")

    if isinstance(loaded, trimesh.Scene):
        geom = loaded.to_geometry()

        if isinstance(geom, trimesh.Trimesh):
            mesh = geom

        elif isinstance(geom, list):
            meshes = [
                m for m in geom
                if isinstance(m, trimesh.Trimesh) and len(m.faces) > 0
            ]
            if len(meshes) == 0:
                raise ValueError("No valid mesh geometry found in the scene.")
            mesh = trimesh.util.concatenate(meshes)

        else:
            raise TypeError(f"Unsupported geometry type from Scene.to_geometry(): {type(geom)}")

    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded

    else:
        raise TypeError(f"Unsupported object type: {type(loaded)}")

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Loaded mesh is empty.")

    return mesh


def downcast_mesh_arrays(mesh: trimesh.Trimesh) -> None:
    """
    Downcast arrays to reduce memory use.
    """
    if mesh.vertices.dtype != np.float32:
        mesh.vertices = mesh.vertices.astype(np.float32)

    if mesh.faces.dtype != np.int32:
        mesh.faces = mesh.faces.astype(np.int32)

def check_non_empty(mesh: trimesh.Trimesh) -> bool:
    """
    Check whether the mesh has non-zero vertices and faces.
    """
    return len(mesh.vertices) > 0 and len(mesh.faces) > 0


def check_finite_vertices(mesh: trimesh.Trimesh) -> bool:
    """
    Check whether all vertex coordinates are finite.
    """
    return bool(np.isfinite(mesh.vertices).all())


def check_valid_face_indices(mesh: trimesh.Trimesh) -> bool:
    """
    Check whether face indices are valid.
    """
    if len(mesh.faces) == 0:
        return False

    faces = np.asarray(mesh.faces)
    if not np.issubdtype(faces.dtype, np.integer):
        return False

    if faces.min() < 0:
        return False

    if faces.max() >= len(mesh.vertices):
        return False

    return True

def inspect_single_mesh_ultralight(mesh_path: str) -> Dict[str, Any]:
    """
    Ultra-light single-mesh geometry annotation.
    No hard filtering is done here.
    """
    record: Dict[str, Any] = {
        "load_ok": False,
        "error": None,
        "vertex_count": None,
        "face_count": None,
        "non_empty": None,
        "finite_vertices": None,
        "valid_face_indices": None,
        "is_winding_consistent": None,
        "is_watertight": None,
        "bbox_extent": None,
        "euler_number": None,
        "degenerate_face_ratio": None,
        "area": None,
        "volume": None,
        "convex_hull_area": None,
        "convex_hull_volume": None,
    }

    mesh = None

    try:
        mesh = load_mesh_safely(mesh_path)
        downcast_mesh_arrays(mesh)
        mesh.remove_unreferenced_vertices()

        record["load_ok"] = True
        record["vertex_count"] = int(len(mesh.vertices))
        record["face_count"] = int(len(mesh.faces))

        record["non_empty"] = bool(check_non_empty(mesh))
        record["finite_vertices"] = bool(check_finite_vertices(mesh))
        record["valid_face_indices"] = bool(check_valid_face_indices(mesh))

        record["is_winding_consistent"] = bool(mesh.is_winding_consistent)
        record["is_watertight"] = bool(mesh.is_watertight)

        record["bbox_extent"] = np.asarray(mesh.bounding_box.extents, dtype=np.float64).tolist()
        record["euler_number"] = int(mesh.euler_number)
        record["degenerate_face_ratio"] = float(np.mean(np.asarray(mesh.area_faces, dtype=np.float64) <= 1e-16))

        record["area"] = float(mesh.area)
        record["volume"] = float(mesh.volume)

        hull = mesh.convex_hull
        record["convex_hull_area"] = float(hull.area)
        record["convex_hull_volume"] = float(hull.volume)
        

    except MemoryError:
        record["error"] = "MemoryError"
    except Exception as e:
        record["error"] = f"{type(e).__name__}: {str(e)}"
    finally:
        del mesh
        gc.collect()

    return record


# ============================================================
# Worker
# ============================================================

def annotate_one_uid(uid: str) -> Dict[str, Any]:
    """
    Download one object by UID, inspect the mesh, and return one JSON record.
    """
    base_record: Dict[str, Any] = {
        "UID": uid,
        "load_ok": False,
        "error": None,
        "vertex_count": None,
        "face_count": None,
        "non_empty": None,
        "finite_vertices": None,
        "valid_face_indices": None,
        "is_winding_consistent": None,
        "is_watertight": None,
        "bbox_extent": None,
        "euler_number": None,
        "degenerate_face_ratio": None,
        "area": None,
        "volume": None,
        "convex_hull_area": None,
        "convex_hull_volume": None,
    }

    try:
        objects = objaverse.load_objects(
            uids=[uid],
            download_processes=1,
        )

        mesh_path = objects.get(uid)
        if mesh_path is None:
            base_record["error"] = "download_failed_or_missing"
            return base_record
        
        file_size_bytes = os.path.getsize(mesh_path)
        if file_size_bytes >= 100 * 1024 * 1024:  # max_file_size_mb MB size limit
            base_record["error"] = "file_too_large"
            return base_record

        mesh_record = inspect_single_mesh_ultralight(mesh_path)
        base_record.update(mesh_record)

    except MemoryError:
        base_record["error"] = "MemoryError"
    except Exception as e:
        base_record["error"] = f"{type(e).__name__}: {str(e)}"
    finally:
        gc.collect()

    return base_record


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate Objaverse meshes from a filtered JSON file."
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to cleaned_attribute.json",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index in UID list (inclusive)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index in UID list (exclusive)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, mp.cpu_count() // 2),
        help="Number of worker processes",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1,
        help="Chunksize for multiprocessing.imap_unordered",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Print progress every N finished objects",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] objaverse version: {objaverse.__version__}")
    print(f"[INFO] Reading input JSON: {args.input_json}")

    items = load_items_from_json(args.input_json)
    all_uids = extract_uids(items)

    start = max(0, args.start)
    end = len(all_uids) if args.end is None else min(args.end, len(all_uids))
    uids = all_uids[start:end]

    print(f"[INFO] Total unique UIDs in input: {len(all_uids)}")
    print(f"[INFO] Processing UID range: [{start}, {end})")
    print(f"[INFO] Number of UIDs in this run: {len(uids)}")
    print(f"[INFO] Output JSONL: {args.output_jsonl}")
    print(f"[INFO] num_workers: {args.num_workers}")
    print(f"[INFO] chunksize: {args.chunksize}")
    print(f"[INFO] log_every: {args.log_every}")

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)

    processed = 0
    load_ok_count = 0
    error_count = 0

    with open(args.output_jsonl, "w", encoding="utf-8") as f_out:
        with mp.Pool(processes=args.num_workers, maxtasksperchild=20) as pool:
            for record in pool.imap_unordered(annotate_one_uid, uids, chunksize=args.chunksize):
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed += 1

                if record.get("load_ok", False):
                    load_ok_count += 1
                if record.get("error") is not None:
                    error_count += 1

                if processed % args.log_every == 0 or processed == len(uids):
                    print(
                        f"[INFO] processed={processed}/{len(uids)} | "
                        f"load_ok={load_ok_count} | "
                        f"errors={error_count}"
                    )

    print("[INFO] Done.")
    print(f"[INFO] Saved JSONL to: {args.output_jsonl}")
    print(f"[INFO] Final processed: {processed}")
    print(f"[INFO] Final load_ok: {load_ok_count}")
    print(f"[INFO] Final errors: {error_count}")


if __name__ == "__main__":
    main()

# Example usage:
# python geometry_annotate_from_json.py \
#   --input_json cleaned_attribute.json \
#   --output_jsonl geometry_annotations_part_000000_000100.jsonl \
#   --start 0 \
#   --end 100 \
#   --num_workers 4