# -*- coding: utf-8 -*-

import os
import json
import math
import shutil
import argparse
import subprocess
import time
from typing import List, Dict, Any
import resource
import gc
import signal

def set_memory_limit(max_memory_gb: float) -> None:
    """
    Set address-space memory limit for the child process.
    Linux/Unix only.
    """
    limit_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))


def load_items_from_json(json_path: str) -> List[Dict[str, Any]]:
    """
    Load a JSON list from disk.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")

    return data


def extract_unique_uids(items: List[Dict[str, Any]]) -> List[str]:
    """
    Extract unique UIDs from input items while preserving order.
    """
    seen = set()
    uids = []

    for item in items:
        uid = item.get("UID")
        if uid is None:
            continue
        if uid in seen:
            continue
        seen.add(uid)
        uids.append(uid)

    return uids


def clear_objaverse_cache(cache_dir: str) -> None:
    """
    Remove the objaverse cache directory if it exists.
    """
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"[INFO] Removed cache directory: {cache_dir}")
    else:
        print(f"[INFO] Cache directory does not exist, skip removing: {cache_dir}")


def build_output_path(output_dir: str, start: int, end: int) -> str:
    """
    Build output JSONL filename for one batch.
    """
    filename = f"geometry_annotations_part_{start:06d}_{end:06d}.jsonl"
    return os.path.join(output_dir, filename)


def format_seconds(seconds: float) -> str:
    """
    Format seconds into HH:MM:SS.
    """
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def remove_output_if_exists(output_jsonl: str) -> None:
    """
    Remove output JSONL if it exists.
    """
    if os.path.exists(output_jsonl):
        try:
            os.remove(output_jsonl)
            print(f"[INFO] Removed output file: {output_jsonl}")
        except OSError as e:
            print(f"[WARN] Failed to remove output file: {output_jsonl} | {e}")


def run_one_batch(
    worker_script: str,
    input_json: str,
    output_jsonl: str,
    start: int,
    end: int,
    num_workers: int,
    chunksize: int,
    log_every: int,
    python_exec: str,
    batch_timeout_sec: int,
    max_memory_gb: float,
) -> float:
    """
    Run one batch by calling the worker script as a subprocess.
    Return the elapsed time in seconds.
    """
    cmd = [
        python_exec,
        worker_script,
        "--input_json", input_json,
        "--output_jsonl", output_jsonl,
        "--start", str(start),
        "--end", str(end),
        "--num_workers", str(num_workers),
        "--chunksize", str(chunksize),
        "--log_every", str(log_every),
    ]

    print("[INFO] Running command:")
    print(" ".join(cmd))

    def child_preexec():
        # Start a new process group and set memory limits
        os.setsid()
        set_memory_limit(max_memory_gb)

    batch_start_time = time.perf_counter()

    proc = subprocess.Popen(
        cmd,
        preexec_fn=child_preexec,
    )

    try:
        proc.wait(timeout=batch_timeout_sec)
    except subprocess.TimeoutExpired:
        print(f"[WARN] Batch [{start}, {end}) timed out. Killing process group...")
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            time.sleep(3)
            if proc.poll() is None:
                print("[WARN] SIGTERM did not stop all processes, sending SIGKILL...")
                os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("[WARN] Process group still not fully reaped after SIGKILL.")
        raise

    batch_elapsed = time.perf_counter() - batch_start_time

    if proc.returncode != 0:
        raise RuntimeError(
            f"Worker script failed for batch [{start}, {end}) "
            f"with return code {proc.returncode}"
        )

    if not os.path.exists(output_jsonl):
        raise RuntimeError(
            f"Expected output file was not created: {output_jsonl}"
        )

    if os.path.getsize(output_jsonl) == 0:
        raise RuntimeError(
            f"Output file is empty: {output_jsonl}"
        )

    return batch_elapsed

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run geometry annotation in batches and clear cache after each batch."
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to cleaned_attribute.json",
    )
    parser.add_argument(
        "--worker_script",
        type=str,
        default="geometry_annotate_from_json.py",
        help="Path to the worker script",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save batch JSONL outputs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of UIDs per batch",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes passed to the worker script",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1,
        help="Chunksize passed to the worker script",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Logging frequency passed to the worker script",
    )
    parser.add_argument(
        "--start_batch",
        type=int,
        default=0,
        help="Batch index to start from",
    )
    parser.add_argument(
        "--end_batch",
        type=int,
        default=None,
        help="Batch index to end at (exclusive). Default: run all remaining batches",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.path.expanduser("~/.objaverse"),
        help="Objaverse cache directory to remove after each batch",
    )
    parser.add_argument(
        "--python_exec",
        type=str,
        default="python",
        help="Python executable used to call the worker script",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a batch if the output file already exists and is non-empty",
    )
    parser.add_argument(
        "--batch_timeout_sec",
        type=int,
        default=300,
        help="Timeout in seconds for one batch subprocess",
    )
    parser.add_argument(
        "--max_memory_gb",
        type=float,
        default=3.0,
        help="Optional memory limit in GB for the worker subprocess",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    total_start_time = time.perf_counter()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Reading input JSON: {args.input_json}")
    items = load_items_from_json(args.input_json)
    uids = extract_unique_uids(items)
    total_uids = len(uids)

    if total_uids == 0:
        raise RuntimeError("No valid UIDs found in input JSON.")

    total_batches = math.ceil(total_uids / args.batch_size)

    start_batch = max(0, args.start_batch)
    end_batch = total_batches if args.end_batch is None else min(args.end_batch, total_batches)

    batches_to_run = max(0, end_batch - start_batch)

    print(f"[INFO] Total unique UIDs: {total_uids}")
    print(f"[INFO] Batch size: {args.batch_size}")
    print(f"[INFO] Total batches: {total_batches}")
    print(f"[INFO] Running batches: [{start_batch}, {end_batch})")
    print(f"[INFO] Output directory: {args.output_dir}")
    print(f"[INFO] Worker script: {args.worker_script}")
    print(f"[INFO] Cache directory: {args.cache_dir}")
    print(f"[INFO] Batch timeout: {args.batch_timeout_sec}s")

    completed_batches = 0
    completed_uids = 0

    for batch_idx in range(start_batch, end_batch):
        start = batch_idx * args.batch_size
        end = min((batch_idx + 1) * args.batch_size, total_uids)
        current_batch_size = end - start

        output_jsonl = build_output_path(args.output_dir, start, end)

        print("=" * 80)
        print(f"[INFO] Batch {batch_idx + 1}/{total_batches}: UID range [{start}, {end})")
        print(f"[INFO] Output file: {output_jsonl}")

        if args.skip_existing and os.path.exists(output_jsonl) and os.path.getsize(output_jsonl) > 0:
            print("[INFO] Output already exists and is non-empty, skipping this batch.")
            clear_objaverse_cache(args.cache_dir)

            completed_batches += 1
            completed_uids += current_batch_size

            elapsed_total = time.perf_counter() - total_start_time
            avg_batch_time = elapsed_total / completed_batches if completed_batches > 0 else 0.0
            avg_uid_time = elapsed_total / completed_uids if completed_uids > 0 else 0.0
            remaining_batches = batches_to_run - completed_batches
            eta_seconds = avg_batch_time * remaining_batches

            print(f"[TIME] Total elapsed: {format_seconds(elapsed_total)}")
            print(f"[TIME] Avg per batch: {format_seconds(avg_batch_time)}")
            print(f"[TIME] Avg per UID: {avg_uid_time:.4f} s")
            print(f"[TIME] ETA remaining: {format_seconds(eta_seconds)}")
            continue

        batch_elapsed = None

        try:
            batch_elapsed = run_one_batch(
                worker_script=args.worker_script,
                input_json=args.input_json,
                output_jsonl=output_jsonl,
                start=start,
                end=end,
                num_workers=args.num_workers,
                chunksize=args.chunksize,
                log_every=args.log_every,
                python_exec=args.python_exec,
                batch_timeout_sec=args.batch_timeout_sec,
                max_memory_gb=args.max_memory_gb,
            )

        except subprocess.TimeoutExpired:
            print(
                f"[WARN] Batch [{start}, {end}) timed out after "
                f"{args.batch_timeout_sec} seconds. Skipping to next batch..."
            )
            # remove_output_if_exists(output_jsonl)

        except Exception as e:
            print(f"[WARN] Batch [{start}, {end}) failed: {e}")
            # remove_output_if_exists(output_jsonl)

        finally:
            gc.collect()

        if batch_elapsed is not None:
            print(
                f"[TIME] Batch elapsed: {format_seconds(batch_elapsed)} "
                f"({batch_elapsed / current_batch_size:.4f} s / UID)"
            )

        clear_objaverse_cache(args.cache_dir)

        completed_batches += 1
        completed_uids += current_batch_size

        elapsed_total = time.perf_counter() - total_start_time
        avg_batch_time = elapsed_total / completed_batches if completed_batches > 0 else 0.0
        avg_uid_time = elapsed_total / completed_uids if completed_uids > 0 else 0.0
        remaining_batches = batches_to_run - completed_batches
        eta_seconds = avg_batch_time * remaining_batches

        print(f"[TIME] Total elapsed: {format_seconds(elapsed_total)}")
        print(f"[TIME] Avg per batch: {format_seconds(avg_batch_time)}")
        print(f"[TIME] Avg per UID: {avg_uid_time:.4f} s")
        print(f"[TIME] ETA remaining: {format_seconds(eta_seconds)}")

    total_elapsed = time.perf_counter() - total_start_time

    print("=" * 80)
    print("[INFO] All requested batches finished.")
    print(f"[TIME] Final total elapsed: {format_seconds(total_elapsed)}")

    if completed_batches > 0:
        print(f"[TIME] Final avg per batch: {format_seconds(total_elapsed / completed_batches)}")
    if completed_uids > 0:
        print(f"[TIME] Final avg per UID: {total_elapsed / completed_uids:.4f} s")


if __name__ == "__main__":
    main()

# Example:
# python run_geometry_annotation_in_batches.py \
#   --input_json cleaned_attribute.json \
#   --worker_script geometry_annotate_from_json.py \
#   --output_dir geometry_parts \
#   --batch_size 100 \
#   --num_workers 10 \
#   --python_exec python \
#   --skip_existing \