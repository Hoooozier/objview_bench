# -*- coding: utf-8 -*-
import os
import json
import shutil
import argparse
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np


INPUT_DIR = "geometry_sampled/obj"
OUTPUT_DIR = "geometry_sampled/obj_normalized"
DEFAULT_JSON_PATH = "geometry_sampled/normalization.json"
DEFAULT_STATS_JSON_PATH = "geometry_sampled/obj_normalized_stats.json"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json_dict(json_path: str):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Normalization JSON not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"JSON root must be a dict: {json_path}")
    return data


def save_json_atomic(data: dict, json_path: str):
    tmp_path = json_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, json_path)


def is_valid_record(record):
    if not isinstance(record, dict):
        return False
    if "uid" not in record or "translation_center" not in record or "scale" not in record:
        return False

    center = record["translation_center"]
    scale = record["scale"]

    if not isinstance(center, list) or len(center) != 3:
        return False

    try:
        _ = [float(x) for x in center]
        scale = float(scale)
    except Exception:
        return False

    return np.isfinite(scale) and scale > 0


def copy_if_exists(src: str, dst: str):
    if os.path.exists(src) and os.path.isfile(src):
        shutil.copy2(src, dst)
        return True
    return False


def cleanup_output_dir(output_obj_dir: str):
    if os.path.isdir(output_obj_dir):
        shutil.rmtree(output_obj_dir)


def find_all_object_dirs(input_dir: str):
    root = Path(input_dir)
    if not root.exists():
        return []

    obj_dirs = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        uid = p.name
        obj_path = p / f"{uid}.obj"
        if obj_path.exists() and obj_path.is_file():
            obj_dirs.append(str(p))

    return sorted(obj_dirs)


def normalize_obj_text(src_obj_path: str, dst_obj_path: str, center, scale):
    center = np.asarray(center, dtype=np.float64)
    scale = float(scale)

    num_vertices = 0
    max_radius = 0.0
    sum_vertices = np.zeros(3, dtype=np.float64)

    with open(src_obj_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(dst_obj_path, "w", encoding="utf-8", newline="\n") as fout:

        for line in fin:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) < 4:
                    fout.write(line)
                    continue

                try:
                    x, y, z = map(float, parts[1:4])
                except Exception:
                    fout.write(line)
                    continue

                v = np.array([x, y, z], dtype=np.float64)
                v = (v - center) / scale

                if not np.isfinite(v).all():
                    raise RuntimeError("non_finite_vertices_after_normalization")

                r = float(np.linalg.norm(v))
                if r > max_radius:
                    max_radius = r
                sum_vertices += v
                num_vertices += 1

                # Rewrite only first three coordinates; keep any extra fields in original v line.
                if len(parts) > 4:
                    tail = " " + " ".join(parts[4:])
                else:
                    tail = ""
                fout.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}{tail}\n")
            else:
                fout.write(line)

    if num_vertices == 0:
        raise RuntimeError("no_vertex_lines_found_in_obj")

    mean_center = sum_vertices / num_vertices
    mean_center_norm = float(np.linalg.norm(mean_center))

    return {
        "num_vertices": int(num_vertices),
        "max_radius": float(max_radius),
        "mean_center_norm": mean_center_norm,
    }


def count_faces_in_obj(obj_path: str):
    num_faces = 0
    num_vt = 0
    num_vn = 0
    has_mtllib = False
    has_usemtl = False

    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("f "):
                num_faces += 1
            elif line.startswith("vt "):
                num_vt += 1
            elif line.startswith("vn "):
                num_vn += 1
            elif line.startswith("mtllib "):
                has_mtllib = True
            elif line.startswith("usemtl "):
                has_usemtl = True

    return {
        "num_faces": int(num_faces),
        "num_vt": int(num_vt),
        "num_vn": int(num_vn),
        "has_mtllib": bool(has_mtllib),
        "has_usemtl": bool(has_usemtl),
    }


def is_output_complete(output_root: str, uid: str):
    out_dir = Path(output_root) / uid
    out_obj = out_dir / f"{uid}.obj"
    blender_json = out_dir / "blender_export_info.json"

    if blender_json.exists():
        return False

    if not out_obj.exists() or not out_obj.is_file() or out_obj.stat().st_size <= 0:
        return False

    allowed_names = {f"{uid}.obj", f"{uid}.mtl", "texture.png"}
    for p in out_dir.iterdir():
        if p.name not in allowed_names:
            return False

    return True


def process_one_object(obj_dir: str, output_root: str, json_data: dict):
    obj_dir = Path(obj_dir)
    uid = obj_dir.name

    if uid not in json_data:
        raise RuntimeError("uid_missing_in_normalization_json")

    record = json_data[uid]
    if not is_valid_record(record):
        raise RuntimeError("invalid_normalization_record")

    src_obj_path = obj_dir / f"{uid}.obj"
    src_mtl_path = obj_dir / f"{uid}.mtl"
    src_tex_path = obj_dir / "texture.png"

    if not src_obj_path.exists():
        raise RuntimeError(f"obj_not_found: {src_obj_path}")

    center = np.asarray(record["translation_center"], dtype=np.float64)
    scale = float(record["scale"])

    out_obj_dir = Path(output_root) / uid
    cleanup_output_dir(str(out_obj_dir))
    ensure_dir(str(out_obj_dir))

    out_obj_path = out_obj_dir / f"{uid}.obj"
    out_mtl_path = out_obj_dir / f"{uid}.mtl"
    out_tex_path = out_obj_dir / "texture.png"

    norm_stats = normalize_obj_text(
        str(src_obj_path),
        str(out_obj_path),
        center,
        scale,
    )

    obj_stats = count_faces_in_obj(str(out_obj_path))

    has_mtl = copy_if_exists(str(src_mtl_path), str(out_mtl_path))
    has_texture = copy_if_exists(str(src_tex_path), str(out_tex_path))

    # Force-remove blender json and avoid copying extra files.
    keep_names = {f"{uid}.obj", f"{uid}.mtl", "texture.png"}
    for p in out_obj_dir.iterdir():
        if p.name not in keep_names:
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)

    return {
        "uid": uid,
        "num_vertices": norm_stats["num_vertices"],
        "num_faces": obj_stats["num_faces"],
        "num_vt": obj_stats["num_vt"],
        "num_vn": obj_stats["num_vn"],
        "max_radius": norm_stats["max_radius"],
        "mean_center_norm": norm_stats["mean_center_norm"],
        "has_mtllib": obj_stats["has_mtllib"],
        "has_usemtl": obj_stats["has_usemtl"],
        "has_mtl_file": has_mtl,
        "has_texture_file": has_texture,
    }


def build_initial_stats(args, normalization_json, all_obj_dirs):
    return {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "json_path": args.json_path,
        "stats_json_path": args.stats_json_path,
        "total_object_dirs_found": len(all_obj_dirs),
        "normalization_json_records": len(normalization_json),
        "skipped_already_complete_count": 0,
        "skipped_already_complete_uids": [],
        "missing_or_invalid_normalization_count": 0,
        "missing_or_invalid_normalization_uids": [],
        "todo_count": 0,
        "success_count": 0,
        "success_uids": [],
        "obj_processing_failed_count": 0,
        "obj_processing_failed_uids": [],
        "obj_processing_failed_details": {},
        "success_examples": [],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Normalize OBJ by directly editing vertex lines, preserve original OBJ/MTL/textures, skip failures, and write stats JSON."
    )
    parser.add_argument("--input_dir", type=str, default=INPUT_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--json_path", type=str, default=DEFAULT_JSON_PATH)
    parser.add_argument("--stats_json_path", type=str, default=DEFAULT_STATS_JSON_PATH)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(os.path.dirname(args.stats_json_path) or ".")

    normalization_json = load_json_dict(args.json_path)
    all_obj_dirs = find_all_object_dirs(args.input_dir)

    print(f"[INFO] Found {len(all_obj_dirs)} object folders in {args.input_dir}", flush=True)
    print(f"[INFO] normalization.json contains {len(normalization_json)} records", flush=True)

    if len(all_obj_dirs) == 0:
        print("[WARN] No valid object folders found.", flush=True)
        empty_stats = build_initial_stats(args, normalization_json, all_obj_dirs)
        save_json_atomic(empty_stats, args.stats_json_path)
        return

    stats = build_initial_stats(args, normalization_json, all_obj_dirs)

    todo_dirs = []
    for obj_dir in all_obj_dirs:
        uid = Path(obj_dir).name

        if uid not in normalization_json or not is_valid_record(normalization_json[uid]):
            stats["missing_or_invalid_normalization_uids"].append(uid)
            continue

        if not args.overwrite and is_output_complete(args.output_dir, uid):
            stats["skipped_already_complete_uids"].append(uid)
            continue

        todo_dirs.append(obj_dir)

    stats["missing_or_invalid_normalization_count"] = len(stats["missing_or_invalid_normalization_uids"])
    stats["skipped_already_complete_count"] = len(stats["skipped_already_complete_uids"])
    stats["todo_count"] = len(todo_dirs)

    print(f"[INFO] Missing/invalid normalization: {stats['missing_or_invalid_normalization_count']}", flush=True)
    print(f"[INFO] Skipped already complete: {stats['skipped_already_complete_count']}", flush=True)
    print(f"[INFO] To process: {stats['todo_count']}", flush=True)

    if len(todo_dirs) == 0:
        print("[INFO] Nothing to do.", flush=True)
        save_json_atomic(stats, args.stats_json_path)
        print(f"[INFO] Stats JSON saved at: {args.stats_json_path}", flush=True)
        return

    ctx = mp.get_context("forkserver")
    max_pending = max(args.workers, args.workers * args.prefetch_factor)

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
        future_to_dir = {}
        todo_iter = iter(todo_dirs)

        for _ in range(min(max_pending, len(todo_dirs))):
            obj_dir = next(todo_iter, None)
            if obj_dir is None:
                break
            future = executor.submit(process_one_object, obj_dir, args.output_dir, normalization_json)
            future_to_dir[future] = obj_dir

        completed = 0
        while future_to_dir:
            done, _ = wait(list(future_to_dir.keys()), return_when=FIRST_COMPLETED)

            for future in done:
                obj_dir = future_to_dir.pop(future)
                uid = Path(obj_dir).name
                completed += 1

                try:
                    record = future.result()
                    stats["success_uids"].append(uid)
                    if len(stats["success_examples"]) < 20:
                        stats["success_examples"].append(record)

                    print(
                        f"[INFO] ({completed}/{len(todo_dirs)}) Done: {uid} | "
                        f"v={record['num_vertices']} | f={record['num_faces']} | "
                        f"max_radius={record['max_radius']:.6f} | mean_center_norm={record['mean_center_norm']:.6f} | "
                        f"mtllib={record['has_mtllib']} | usemtl={record['has_usemtl']} | "
                        f"mtl_file={record['has_mtl_file']} | texture_file={record['has_texture_file']}",
                        flush=True,
                    )

                except Exception as e:
                    err_msg = str(e)
                    stats["obj_processing_failed_uids"].append(uid)
                    stats["obj_processing_failed_details"][uid] = err_msg

                    out_dir = Path(args.output_dir) / uid
                    if out_dir.exists():
                        shutil.rmtree(out_dir, ignore_errors=True)

                    print(f"[WARN] ({completed}/{len(todo_dirs)}) Skipped: {uid} | {err_msg}", flush=True)

                obj_dir_next = next(todo_iter, None)
                if obj_dir_next is not None:
                    new_future = executor.submit(
                        process_one_object,
                        obj_dir_next,
                        args.output_dir,
                        normalization_json,
                    )
                    future_to_dir[new_future] = obj_dir_next

    stats["success_count"] = len(stats["success_uids"])
    stats["obj_processing_failed_count"] = len(stats["obj_processing_failed_uids"])

    save_json_atomic(stats, args.stats_json_path)

    print("[INFO] Finished.", flush=True)
    print(f"[INFO] Total object dirs found: {stats['total_object_dirs_found']}", flush=True)
    print(f"[INFO] normalization.json records: {stats['normalization_json_records']}", flush=True)
    print(f"[INFO] Missing/invalid normalization count: {stats['missing_or_invalid_normalization_count']}", flush=True)
    print(f"[INFO] Skipped already complete count: {stats['skipped_already_complete_count']}", flush=True)
    print(f"[INFO] Success count: {stats['success_count']}", flush=True)
    print(f"[INFO] OBJ processing failed count: {stats['obj_processing_failed_count']}", flush=True)
    print(f"[INFO] Output saved at: {args.output_dir}", flush=True)
    print(f"[INFO] Stats JSON saved at: {args.stats_json_path}", flush=True)


if __name__ == "__main__":
    main()

"""
python normalize_obj.py \
  --input_dir geometry_sampled/obj \
  --output_dir geometry_sampled/obj_normalized \
  --json_path geometry_sampled/normalization.json \
  --stats_json_path geometry_sampled/obj_normalized_stats.json \
  --workers 20
"""
