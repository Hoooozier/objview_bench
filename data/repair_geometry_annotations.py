# -*- coding: utf-8 -*-
"""
Repair utility for geometry annotation JSONL shards.

Features
--------
1. Scan one directory of JSONL part files, summarize distinct `error` values.
2. Retry records that match selected error types or custom filters.
3. Update original JSONL records in place (atomically via temp file replace).
4. Optional backup before modification.
5. Supports exact error match / substring / regex / load_ok=false selection / UID list.
6. Retry each UID in an isolated subprocess with memory limit and timeout.
7. Supports controlled concurrency within each shard file.
"""

import os
import re
import gc
import json
import time
import shutil
import signal
import argparse
import subprocess
import resource
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


# ----------------------------------------------------------------------
# Subprocess retry helpers
# ----------------------------------------------------------------------

def set_memory_limit(max_memory_gb: float) -> None:
    """
    Set address-space memory limit for child process.
    Linux/Unix only.
    """
    limit_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))


def run_annotate_one_uid_subprocess(
    uid: str,
    worker_script: str,
    python_exec: str,
    uid_timeout_sec: int,
    max_memory_gb: float,
) -> Dict[str, Any]:
    """
    Run annotate_one_uid(uid) inside an isolated subprocess with:
      - process-group isolation
      - memory limit
      - wall-clock timeout

    The subprocess prints one JSON object to stdout.
    """
    worker_path = Path(worker_script).resolve()
    if not worker_path.exists():
        raise FileNotFoundError(f"Worker script not found: {worker_path}")

    child_code = r'''
import json
import importlib.util
import sys
from pathlib import Path

uid = sys.argv[1]
worker_script = sys.argv[2]

worker_path = Path(worker_script).resolve()
spec = importlib.util.spec_from_file_location("geometry_worker_module", str(worker_path))
if spec is None or spec.loader is None:
    raise ImportError(f"Failed to load module from: {worker_path}")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

if not hasattr(module, "annotate_one_uid"):
    raise AttributeError(f"`annotate_one_uid` not found in worker script: {worker_path}")

result = module.annotate_one_uid(uid)
print(json.dumps(result, ensure_ascii=False))
'''

    cmd = [
        python_exec,
        "-c",
        child_code,
        uid,
        str(worker_path),
    ]

    def child_preexec():
        os.setsid()
        set_memory_limit(max_memory_gb)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=child_preexec,
    )

    try:
        stdout, stderr = proc.communicate(timeout=uid_timeout_sec)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            time.sleep(2)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

        return {
            "UID": uid,
            "load_ok": False,
            "error": f"repair_subprocess_timeout: exceeded {uid_timeout_sec} seconds",
        }

    if proc.returncode != 0:
        stderr = (stderr or "").strip()
        return {
            "UID": uid,
            "load_ok": False,
            "error": f"repair_subprocess_nonzero_exit: returncode={proc.returncode} stderr={stderr}",
        }

    stdout = (stdout or "").strip()
    if not stdout:
        return {
            "UID": uid,
            "load_ok": False,
            "error": "repair_subprocess_empty_stdout",
        }

    last_line = stdout.splitlines()[-1].strip()
    try:
        rec = json.loads(last_line)
    except json.JSONDecodeError as e:
        return {
            "UID": uid,
            "load_ok": False,
            "error": f"repair_subprocess_bad_json: {type(e).__name__}: {e} | stdout_tail={last_line[:500]}",
        }

    if not isinstance(rec, dict):
        return {
            "UID": uid,
            "load_ok": False,
            "error": f"repair_subprocess_invalid_result_type: {type(rec).__name__}",
        }

    return rec


def retry_uid_with_attempts(
    uid: str,
    worker_script: str,
    python_exec: str,
    uid_timeout_sec: int,
    max_memory_gb: float,
    max_retries_per_uid: int,
    sleep_sec: float,
) -> Dict[str, Any]:
    """
    Retry one UID up to max_retries_per_uid times.
    Returns the latest record to write back.
    """
    last_exc = None
    last_rec = None

    for attempt in range(1, max_retries_per_uid + 1):
        try:
            rec = run_annotate_one_uid_subprocess(
                uid=uid,
                worker_script=worker_script,
                python_exec=python_exec,
                uid_timeout_sec=uid_timeout_sec,
                max_memory_gb=max_memory_gb,
            )
            last_rec = rec

            # Success: stop early
            if rec.get("load_ok", False):
                return rec

            # Still failed: if there are more tries, wait and retry
            if attempt < max_retries_per_uid and sleep_sec > 0:
                time.sleep(sleep_sec)

        except Exception as e:
            last_exc = e
            if attempt < max_retries_per_uid and sleep_sec > 0:
                time.sleep(sleep_sec)

    if last_rec is not None:
        return last_rec

    return {
        "UID": uid,
        "load_ok": False,
        "error": (
            f"repair_script_retry_failed: {type(last_exc).__name__}: {last_exc}"
            if last_exc is not None
            else "repair_script_retry_failed"
        ),
    }


# ----------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------

def iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield idx, json.loads(line)
            except json.JSONDecodeError as e:
                yield idx, {
                    "UID": None,
                    "load_ok": False,
                    "error": f"JSONDecodeError: {e}",
                    "_invalid_json_line": True,
                    "_raw_line": line,
                }


def write_jsonl_atomic(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def load_uid_file(uids_file: Optional[str]) -> Optional[set]:
    if not uids_file:
        return None
    p = Path(uids_file)
    if not p.exists():
        raise FileNotFoundError(f"UID file not found: {p}")
    uids = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            uid = line.strip()
            if uid:
                uids.add(uid)
    return uids


# ----------------------------------------------------------------------
# Scan / summary
# ----------------------------------------------------------------------

@dataclass
class MatchDecision:
    selected: bool
    reason: Optional[str] = None


def normalize_error(err: Any) -> Optional[str]:
    if err is None:
        return None
    return str(err).strip()


def collect_summary(files: List[Path]) -> Dict[str, Any]:
    error_counter: Counter = Counter()
    file_error_counter: Dict[str, Counter] = defaultdict(Counter)
    total_records = 0
    failed_records = 0
    success_records = 0

    for path in files:
        for _, rec in iter_jsonl(path):
            total_records += 1
            err = normalize_error(rec.get("error"))
            load_ok = bool(rec.get("load_ok", False))

            if load_ok:
                success_records += 1
            if err is not None:
                failed_records += 1
                error_counter[err] += 1
                file_error_counter[path.name][err] += 1

    return {
        "total_records": total_records,
        "success_records": success_records,
        "failed_records": failed_records,
        "distinct_error_count": len(error_counter),
        "error_counter": error_counter,
        "file_error_counter": file_error_counter,
    }


def print_summary(summary: Dict[str, Any], top_k: Optional[int] = None) -> None:
    print("=" * 80)
    print("[SUMMARY]")
    print(f"Total records        : {summary['total_records']}")
    print(f"Successful records   : {summary['success_records']}")
    print(f"Failed records       : {summary['failed_records']}")
    print(f"Distinct error types : {summary['distinct_error_count']}")
    print("-" * 80)
    print("Error counts:")

    items = summary["error_counter"].most_common(top_k)
    if not items:
        print("  (no errors found)")
        return

    for idx, (err, cnt) in enumerate(items, start=1):
        print(f"{idx:4d}. [{cnt:8d}] {err}")


# ----------------------------------------------------------------------
# Selection logic
# ----------------------------------------------------------------------

def build_matcher(
    retry_all_failed: bool,
    retry_error: List[str],
    retry_error_contains: List[str],
    retry_error_regex: Optional[str],
    uids_set: Optional[set],
    only_load_ok_false: bool,
):
    regex = re.compile(retry_error_regex) if retry_error_regex else None
    exact_errors = set(retry_error)
    contains_terms = list(retry_error_contains)

    def match(rec: Dict[str, Any]) -> MatchDecision:
        uid = rec.get("UID")
        err = normalize_error(rec.get("error"))
        load_ok = bool(rec.get("load_ok", False))

        if uids_set is not None and uid not in uids_set:
            return MatchDecision(False)

        if retry_all_failed and err is not None:
            return MatchDecision(True, "retry_all_failed")

        if only_load_ok_false and not load_ok:
            return MatchDecision(True, "only_load_ok_false")

        if err is None:
            return MatchDecision(False)

        if err in exact_errors:
            return MatchDecision(True, f"retry_error={err}")

        for term in contains_terms:
            if term in err:
                return MatchDecision(True, f"retry_error_contains={term}")

        if regex is not None and regex.search(err):
            return MatchDecision(True, f"retry_error_regex={retry_error_regex}")

        if uids_set is not None and uid in uids_set:
            return MatchDecision(True, "uids_file/manual_uid_selection")

        return MatchDecision(False)

    return match


# ----------------------------------------------------------------------
# Repair logic
# ----------------------------------------------------------------------

def backup_file(path: Path) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_suffix(path.suffix + f".bak.{ts}")
    shutil.copy2(path, backup_path)
    return backup_path


def repair_file(
    path: Path,
    matcher,
    dry_run: bool,
    do_backup: bool,
    sleep_sec: float,
    max_retries_per_uid: int,
    worker_script: str,
    python_exec: str,
    uid_timeout_sec: int,
    max_memory_gb: float,
    num_concurrent_retries: int,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    selected_positions: List[Tuple[int, Dict[str, Any], str]] = []

    for _, rec in iter_jsonl(path):
        records.append(rec)
        decision = matcher(rec)
        if decision.selected:
            selected_positions.append((len(records) - 1, rec, decision.reason or "matched"))

    result = {
        "file": str(path),
        "total_records": len(records),
        "selected_records": len(selected_positions),
        "updated_records": 0,
        "unchanged_records": 0,
        "selection_reasons": Counter(reason for _, _, reason in selected_positions),
        "examples": [],
    }

    if not selected_positions:
        return result

    if dry_run:
        for _, rec, reason in selected_positions[:5]:
            result["examples"].append({
                "UID": rec.get("UID"),
                "old_error": rec.get("error"),
                "reason": reason,
            })
        return result

    if do_backup:
        backup_path = backup_file(path)
        print(f"[BACKUP] {path.name} -> {backup_path.name}")

    # UID is None -> cannot retry
    valid_jobs: List[Tuple[int, Dict[str, Any], str]] = []
    for rec_idx, old_rec, reason in selected_positions:
        if old_rec.get("UID") is None:
            result["unchanged_records"] += 1
        else:
            valid_jobs.append((rec_idx, old_rec, reason))

    if not valid_jobs:
        return result

    max_workers = max(1, num_concurrent_retries)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {
            executor.submit(
                retry_uid_with_attempts,
                old_rec.get("UID"),
                worker_script,
                python_exec,
                uid_timeout_sec,
                max_memory_gb,
                max(1, max_retries_per_uid),
                sleep_sec,
            ): (rec_idx, old_rec, reason)
            for rec_idx, old_rec, reason in valid_jobs
        }

        finished = 0
        total = len(valid_jobs)

        for future in as_completed(future_to_job):
            rec_idx, old_rec, reason = future_to_job[future]
            uid = old_rec.get("UID")

            try:
                new_rec = future.result()
            except Exception as e:
                new_rec = dict(old_rec)
                new_rec["error"] = f"repair_parallel_future_failed: {type(e).__name__}: {e}"

            records[rec_idx] = new_rec

            old_error = old_rec.get("error")
            new_error = new_rec.get("error")
            old_load_ok = old_rec.get("load_ok")
            new_load_ok = new_rec.get("load_ok")

            changed = (old_rec != new_rec)
            if changed:
                result["updated_records"] += 1
            else:
                result["unchanged_records"] += 1

            finished += 1
            print(
                f"[RETRY {finished}/{total}] file={path.name} | UID={uid} | reason={reason} | "
                f"old_load_ok={old_load_ok} -> new_load_ok={new_load_ok} | "
                f"old_error={old_error!r} -> new_error={new_error!r}"
            )

            gc.collect()

    write_jsonl_atomic(path, records)
    return result


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize and repair geometry annotation JSONL part files."
    )
    parser.add_argument(
        "--parts_dir",
        type=str,
        required=True,
        help="Directory containing geometry_annotations_part_*.jsonl",
    )
    parser.add_argument(
        "--file_glob",
        type=str,
        default="geometry_annotations_part_*.jsonl",
        help="Glob pattern used to find shard files inside parts_dir",
    )
    parser.add_argument(
        "--worker_script",
        type=str,
        default="geometry_annotate_from_json.py",
        help="Path to the worker script that defines annotate_one_uid(uid)",
    )
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Only print unique error summary, do not modify files",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Only show top-k error types in summary",
    )

    # Selection options
    parser.add_argument(
        "--retry_all_failed",
        action="store_true",
        help="Retry every record whose error is not None",
    )
    parser.add_argument(
        "--retry_error",
        action="append",
        default=[],
        help="Retry records whose error exactly matches this string. Can be repeated.",
    )
    parser.add_argument(
        "--retry_error_contains",
        action="append",
        default=[],
        help="Retry records whose error contains this substring. Can be repeated.",
    )
    parser.add_argument(
        "--retry_error_regex",
        type=str,
        default=None,
        help="Retry records whose error matches this regex.",
    )
    parser.add_argument(
        "--only_load_ok_false",
        action="store_true",
        help="Retry records with load_ok == False, even if error is None.",
    )
    parser.add_argument(
        "--uids_file",
        type=str,
        default=None,
        help="Path to a text file containing one UID per line to retry.",
    )

    # Execution options
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a timestamped .bak copy before rewriting a shard.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be retried but do not modify any file.",
    )
    parser.add_argument(
        "--sleep_sec",
        type=float,
        default=0.0,
        help="Sleep between attempts for the same UID.",
    )
    parser.add_argument(
        "--max_retries_per_uid",
        type=int,
        default=1,
        help="How many times the repair script itself should retry one UID.",
    )

    # Per-UID subprocess protection
    parser.add_argument(
        "--python_exec",
        type=str,
        default="python",
        help="Python executable used to launch isolated per-UID retry subprocess.",
    )
    parser.add_argument(
        "--uid_timeout_sec",
        type=int,
        default=300,
        help="Timeout in seconds for one UID retry subprocess.",
    )
    parser.add_argument(
        "--max_memory_gb",
        type=float,
        default=3.0,
        help="Address-space memory limit in GB for one UID retry subprocess.",
    )

    # Concurrency
    parser.add_argument(
        "--num_concurrent_retries",
        type=int,
        default=1,
        help="Max number of UID retries running concurrently within one shard file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    parts_dir = Path(args.parts_dir)
    if not parts_dir.exists():
        raise FileNotFoundError(f"parts_dir not found: {parts_dir}")

    files = sorted(parts_dir.glob(args.file_glob))
    if not files:
        raise FileNotFoundError(
            f"No files matched: parts_dir={parts_dir}, glob={args.file_glob}"
        )

    print(f"[INFO] parts_dir: {parts_dir}")
    print(f"[INFO] matched files: {len(files)}")
    for p in files[:5]:
        print(f"  - {p.name}")
    if len(files) > 5:
        print(f"  ... ({len(files) - 5} more files)")

    summary = collect_summary(files)
    print_summary(summary, top_k=args.top_k)

    if args.summary_only:
        return

    has_selection = any([
        args.retry_all_failed,
        bool(args.retry_error),
        bool(args.retry_error_contains),
        bool(args.retry_error_regex),
        bool(args.uids_file),
        args.only_load_ok_false,
    ])
    if not has_selection:
        print("[INFO] No retry selector was provided. Summary finished; nothing to modify.")
        return

    matcher = build_matcher(
        retry_all_failed=args.retry_all_failed,
        retry_error=args.retry_error,
        retry_error_contains=args.retry_error_contains,
        retry_error_regex=args.retry_error_regex,
        uids_set=load_uid_file(args.uids_file),
        only_load_ok_false=args.only_load_ok_false,
    )

    grand_selected = 0
    grand_updated = 0
    grand_unchanged = 0

    print("=" * 80)
    print("[REPAIR START]")
    print(f"[INFO] num_concurrent_retries={max(1, args.num_concurrent_retries)}")
    print(f"[INFO] uid_timeout_sec={args.uid_timeout_sec}")
    print(f"[INFO] max_memory_gb={args.max_memory_gb}")
    print(f"[INFO] max_retries_per_uid={max(1, args.max_retries_per_uid)}")

    for path in files:
        res = repair_file(
            path=path,
            matcher=matcher,
            dry_run=args.dry_run,
            do_backup=args.backup,
            sleep_sec=args.sleep_sec,
            max_retries_per_uid=max(1, args.max_retries_per_uid),
            worker_script=args.worker_script,
            python_exec=args.python_exec,
            uid_timeout_sec=args.uid_timeout_sec,
            max_memory_gb=args.max_memory_gb,
            num_concurrent_retries=max(1, args.num_concurrent_retries),
        )

        if res["selected_records"] == 0:
            continue

        grand_selected += res["selected_records"]
        grand_updated += res["updated_records"]
        grand_unchanged += res["unchanged_records"]

        print("-" * 80)
        print(f"[FILE] {Path(res['file']).name}")
        print(f"  total_records   : {res['total_records']}")
        print(f"  selected_records: {res['selected_records']}")
        print(f"  updated_records : {res['updated_records']}")
        print(f"  unchanged       : {res['unchanged_records']}")
        if res["selection_reasons"]:
            print("  selection reasons:")
            for reason, cnt in res["selection_reasons"].most_common():
                print(f"    - [{cnt}] {reason}")
        if args.dry_run and res["examples"]:
            print("  examples:")
            for ex in res["examples"]:
                print(
                    f"    - UID={ex['UID']} | old_error={ex['old_error']!r} | reason={ex['reason']}"
                )

    print("=" * 80)
    print("[REPAIR DONE]")
    print(f"Selected records : {grand_selected}")
    print(f"Updated records  : {grand_updated}")
    print(f"Unchanged records: {grand_unchanged}")

    print("=" * 80)
    print("[POST-REPAIR SUMMARY]")
    post_summary = collect_summary(files)
    print_summary(post_summary, top_k=args.top_k)


if __name__ == "__main__":
    main()
    