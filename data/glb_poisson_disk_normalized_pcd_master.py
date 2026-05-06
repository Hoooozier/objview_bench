# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import open3d as o3d


INPUT_DIR = "geometry_sampled/glb"
OUTPUT_DIR = "geometry_sampled/pcd_normalized"
DEFAULT_NUM_POINTS = 200000
DEFAULT_JSON_PATH = "geometry_sampled/normalization.json"
DEFAULT_WORKER_SCRIPT = str(Path(__file__).with_name("glb_poisson_disk_normalized_pcd_worker.py"))


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json_dict(json_path: str):
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"JSON root must be a dict: {json_path}")
    return data


def save_json_dict_atomic(data: dict, json_path: str):
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


def should_skip_uid(
    uid: str,
    output_dir: str,
    json_data: dict,
    radius_tol: float = 1e-3,
    center_tol: float = 1e-3,
):
    pcd_path = os.path.join(output_dir, f"{uid}.pcd")
    pcd_ok, pcd_info = validate_pcd_file(
        pcd_path,
        radius_tol=radius_tol,
        center_tol=center_tol,
    )

    json_ok = uid in json_data and is_valid_record(json_data[uid])

    return pcd_ok and json_ok, pcd_info, json_ok


def find_all_glb_files(input_dir: str) -> List[str]:
    root = Path(input_dir)
    glb_files = []
    glb_files.extend([str(p) for p in root.glob("*.glb") if p.is_file()])
    glb_files.extend([str(p) for p in root.glob("*/*.glb") if p.is_file()])
    return sorted(set(glb_files))


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def launch_worker(glb_path: str, args, result_dir: str, log_dir: str):
    uid = Path(glb_path).stem
    result_json = os.path.join(result_dir, f"{uid}.json")
    stdout_path = os.path.join(log_dir, f"{uid}.stdout.log")
    stderr_path = os.path.join(log_dir, f"{uid}.stderr.log")

    for p in (result_json, stdout_path, stderr_path):
        if os.path.exists(p):
            os.remove(p)

    stdout_f = open(stdout_path, "w", encoding="utf-8")
    stderr_f = open(stderr_path, "w", encoding="utf-8")
    cmd = [
        sys.executable,
        args.worker_script,
        "--glb_path", glb_path,
        "--output_dir", args.output_dir,
        "--num_points", str(args.num_points),
        "--result_json", result_json,
        "--radius_tol", str(args.radius_tol),
        "--center_tol", str(args.center_tol),
    ]
    proc = subprocess.Popen(cmd, stdout=stdout_f, stderr=stderr_f)
    return {
        "uid": uid,
        "glb_path": glb_path,
        "proc": proc,
        "stdout_f": stdout_f,
        "stderr_f": stderr_f,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "result_json": result_json,
    }


def close_launch_files(item: Dict):
    for key in ("stdout_f", "stderr_f"):
        f = item.get(key)
        if f is not None and not f.closed:
            f.close()


def tail_text(path: str, max_chars: int = 2000) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    if len(text) <= max_chars:
        return text.strip()
    return text[-max_chars:].strip()


def main():
    parser = argparse.ArgumentParser(
        description="Parallel GLB -> normalized XYZ-only PCD with isolated worker subprocesses."
    )
    parser.add_argument("--input_dir", type=str, default=INPUT_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--num_points", type=int, default=DEFAULT_NUM_POINTS)
    parser.add_argument("--json_path", type=str, default=DEFAULT_JSON_PATH)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument("--radius_tol", type=float, default=1e-3)
    parser.add_argument("--center_tol", type=float, default=1e-3)
    parser.add_argument("--worker_script", type=str, default=DEFAULT_WORKER_SCRIPT)
    parser.add_argument("--state_dir", type=str, default="")
    parser.add_argument("--poll_interval", type=float, default=0.2)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(os.path.dirname(args.json_path) or ".")

    if not os.path.exists(args.worker_script):
        raise FileNotFoundError(f"Worker script not found: {args.worker_script}")

    state_dir = args.state_dir or os.path.join(args.output_dir, "_worker_state")
    result_dir = os.path.join(state_dir, "results")
    log_dir = os.path.join(state_dir, "logs")
    ensure_dir(state_dir)
    ensure_dir(result_dir)
    ensure_dir(log_dir)

    fail_list_path = os.path.join(state_dir, "failed_uids.txt")

    all_glb_files = find_all_glb_files(args.input_dir)
    print(f"[INFO] Found {len(all_glb_files)} GLB files in {args.input_dir}", flush=True)

    if len(all_glb_files) == 0:
        print("[WARN] No GLB files found.", flush=True)
        return

    json_data = load_json_dict(args.json_path)

    todo_files = []
    skipped = 0
    invalid_pcd = 0
    missing_json = 0

    for glb_path in all_glb_files:
        uid = Path(glb_path).stem
        skip, pcd_info, json_ok = should_skip_uid(
            uid,
            args.output_dir,
            json_data,
            radius_tol=args.radius_tol,
            center_tol=args.center_tol,
        )

        if skip:
            skipped += 1
        else:
            todo_files.append(glb_path)
            if pcd_info["error"] is not None:
                invalid_pcd += 1
            if not json_ok:
                missing_json += 1

    print(f"[INFO] Skipped already valid: {skipped}", flush=True)
    print(f"[INFO] To process: {len(todo_files)}", flush=True)
    print(f"[INFO] Invalid/missing PCD among todo: {invalid_pcd}", flush=True)
    print(f"[INFO] Missing/invalid JSON record among todo: {missing_json}", flush=True)

    if len(todo_files) == 0:
        print("[INFO] Nothing to do. All files already processed and validated.", flush=True)
        return

    counters = {"completed": 0, "success": 0, "fail": 0}
    running: Dict[str, Dict] = {}
    todo_iter = iter(todo_files)

    def refill():
        while len(running) < args.workers:
            glb_path = next(todo_iter, None)
            if glb_path is None:
                break
            item = launch_worker(glb_path, args, result_dir, log_dir)
            running[item["uid"]] = item

    refill()

    while running:
        finished_uids = []
        for uid, item in list(running.items()):
            ret = item["proc"].poll()
            if ret is None:
                continue

            finished_uids.append(uid)
            close_launch_files(item)

            counters["completed"] += 1
            idx = counters["completed"]

            if ret == 0 and os.path.exists(item["result_json"]):
                try:
                    record = read_json(item["result_json"])
                    sampled_n = int(record.pop("num_points", -1))
                    json_data[uid] = record
                    save_json_dict_atomic(json_data, args.json_path)
                    counters["success"] += 1
                    print(f"[INFO] ({idx}/{len(todo_files)}) Done: {uid} | points={sampled_n}", flush=True)
                except Exception as e:
                    counters["fail"] += 1
                    with open(fail_list_path, "a", encoding="utf-8") as f:
                        f.write(uid + "\n")
                    print(f"[ERROR] ({idx}/{len(todo_files)}) Failed postprocess: {uid} | {e}", flush=True)
            else:
                counters["fail"] += 1
                with open(fail_list_path, "a", encoding="utf-8") as f:
                    f.write(uid + "\n")
                stderr_tail = tail_text(item["stderr_path"])
                stdout_tail = tail_text(item["stdout_path"])
                detail = stderr_tail or stdout_tail or f"worker exited with code {ret}"
                detail = detail.replace("\n", " | ")
                print(f"[ERROR] ({idx}/{len(todo_files)}) Failed: {uid} | {detail}", flush=True)

        for uid in finished_uids:
            running.pop(uid, None)

        refill()

        if running:
            time.sleep(args.poll_interval)

    print("[INFO] Finished.", flush=True)
    print(f"[INFO] Success this run: {counters['success']}", flush=True)
    print(f"[INFO] Failed this run: {counters['fail']}", flush=True)
    print(f"[INFO] JSON saved at: {args.json_path}", flush=True)
    print(f"[INFO] Failure list saved at: {fail_list_path}", flush=True)


if __name__ == "__main__":
    main()
