#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FileJob:
    src: Path
    dst: Path
    size: int
    mtime_ns: int


class CopyStats:
    def __init__(self, total_files: int, total_bytes: int) -> None:
        self.total_files = total_files
        self.total_bytes = total_bytes

        self.copied_files = 0
        self.skipped_files = 0
        self.failed_files = 0
        self.copied_bytes = 0

        self._lock = threading.Lock()
        self._start_time = time.time()

    def add_copied(self, size: int) -> None:
        with self._lock:
            self.copied_files += 1
            self.copied_bytes += size

    def add_skipped(self) -> None:
        with self._lock:
            self.skipped_files += 1

    def add_failed(self) -> None:
        with self._lock:
            self.failed_files += 1

    def snapshot(self) -> tuple[int, int, int, int]:
        with self._lock:
            return (
                self.copied_files,
                self.skipped_files,
                self.failed_files,
                self.copied_bytes,
            )

    def print_progress(self, interval: float = 1.0) -> None:
        last_print = 0.0
        while True:
            now = time.time()
            copied_files, skipped_files, failed_files, copied_bytes = self.snapshot()
            done = copied_files + skipped_files + failed_files

            if now - last_print >= interval:
                speed = copied_bytes / max(now - self._start_time, 1e-6) / (1024 * 1024)
                pct = (done / self.total_files * 100.0) if self.total_files else 100.0
                print(
                    f"\rProgress: {done}/{self.total_files} ({pct:6.2f}%) | "
                    f"copied={copied_files}, skipped={skipped_files}, failed={failed_files} | "
                    f"written={copied_bytes / (1024**3):.2f} GiB | "
                    f"avg={speed:.2f} MiB/s",
                    end="",
                    flush=True,
                )
                last_print = now

            if done >= self.total_files:
                break
            time.sleep(0.1)
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively copy a folder to another location with multithreading."
    )
    parser.add_argument("src", type=Path, help="Source directory")
    parser.add_argument("dst", type=Path, help="Destination directory")
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=min(32, (os.cpu_count() or 8) * 4),
        help="Number of worker threads (default: min(32, cpu*4))",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files even if size and mtime match",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8 * 1024 * 1024,
        help="Copy chunk size in bytes (default: 8 MiB)",
    )
    parser.add_argument(
        "--max-pending",
        type=int,
        default=0,
        help="Maximum number of in-flight futures. Default: workers * 2",
    )
    return parser.parse_args()


def validate_paths(src: Path, dst: Path) -> tuple[Path, Path]:
    src = src.resolve()
    dst = dst.resolve()

    if not src.exists():
        raise FileNotFoundError(f"Source does not exist: {src}")
    if not src.is_dir():
        raise NotADirectoryError(f"Source is not a directory: {src}")

    # Prevent copying into itself
    try:
        dst.relative_to(src)
    except ValueError:
        pass
    else:
        raise ValueError("Destination cannot be inside source directory.")

    dst.mkdir(parents=True, exist_ok=True)
    return src, dst


def count_files_and_bytes(src_root: Path, dst_root: Path) -> tuple[int, int]:
    total_files = 0
    total_bytes = 0

    for root, _, files in os.walk(src_root):
        root_path = Path(root)
        rel_dir = root_path.relative_to(src_root)
        (dst_root / rel_dir).mkdir(parents=True, exist_ok=True)

        for name in files:
            src = root_path / name
            try:
                st = src.stat()
            except OSError as e:
                print(f"Warning: cannot stat {src}: {e}", file=sys.stderr)
                continue

            total_files += 1
            total_bytes += st.st_size

    return total_files, total_bytes


def iter_jobs(src_root: Path, dst_root: Path):
    for root, _, files in os.walk(src_root):
        root_path = Path(root)
        rel_dir = root_path.relative_to(src_root)
        (dst_root / rel_dir).mkdir(parents=True, exist_ok=True)

        for name in files:
            src = root_path / name
            try:
                st = src.stat()
            except OSError as e:
                print(f"Warning: cannot stat {src}: {e}", file=sys.stderr)
                continue

            dst = dst_root / rel_dir / name
            yield FileJob(
                src=src,
                dst=dst,
                size=st.st_size,
                mtime_ns=st.st_mtime_ns,
            )


def is_same_enough(job: FileJob) -> bool:
    if not job.dst.exists():
        return False
    try:
        st = job.dst.stat()
    except OSError:
        return False
    return st.st_size == job.size and st.st_mtime_ns == job.mtime_ns


def copy_file(job: FileJob, overwrite: bool, chunk_size: int) -> tuple[str, Path, int, str | None]:
    """
    Returns:
        (status, path, size, error_message)
        status in {"copied", "skipped", "failed"}
    """
    tmp_dst = job.dst.with_name(job.dst.name + ".part")

    try:
        job.dst.parent.mkdir(parents=True, exist_ok=True)

        if not overwrite and is_same_enough(job):
            return ("skipped", job.src, 0, None)

        with open(job.src, "rb") as fsrc, open(tmp_dst, "wb") as fdst:
            while True:
                buf = fsrc.read(chunk_size)
                if not buf:
                    break
                fdst.write(buf)

        shutil.copystat(job.src, tmp_dst, follow_symlinks=True)
        os.replace(tmp_dst, job.dst)

        return ("copied", job.src, job.size, None)

    except Exception as e:
        try:
            if tmp_dst.exists():
                tmp_dst.unlink()
        except Exception:
            pass
        return ("failed", job.src, 0, str(e))


def handle_completed_future(future, stats: CopyStats, failures: list[tuple[Path, str]]) -> None:
    status, path, size, err = future.result()
    if status == "copied":
        stats.add_copied(size)
    elif status == "skipped":
        stats.add_skipped()
    else:
        stats.add_failed()
        failures.append((path, err or "Unknown error"))


def main() -> int:
    args = parse_args()

    try:
        src_root, dst_root = validate_paths(args.src, args.dst)
    except Exception as e:
        print(f"Path error: {e}", file=sys.stderr)
        return 2

    max_pending = args.max_pending if args.max_pending > 0 else args.workers * 2

    print(f"Scanning files under: {src_root}")
    total_files, total_bytes = count_files_and_bytes(src_root, dst_root)

    print(f"Found {total_files} files, total size {total_bytes / (1024**3):.2f} GiB")
    print(f"Destination: {dst_root}")
    print(
        f"Workers: {args.workers}, chunk_size: {args.chunk_size / (1024**2):.1f} MiB, "
        f"max_pending: {max_pending}"
    )

    if total_files == 0:
        print("Nothing to copy.")
        return 0

    stats = CopyStats(total_files=total_files, total_bytes=total_bytes)
    progress_thread = threading.Thread(target=stats.print_progress, daemon=True)
    progress_thread.start()

    t0 = time.time()
    failures: list[tuple[Path, str]] = []

    job_iter = iter_jobs(src_root, dst_root)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = set()

        # Fill a bounded window first instead of submitting all tasks at once.
        while len(pending) < max_pending:
            try:
                job = next(job_iter)
            except StopIteration:
                break
            pending.add(executor.submit(copy_file, job, args.overwrite, args.chunk_size))

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)

            for future in done:
                handle_completed_future(future, stats, failures)

            while len(pending) < max_pending:
                try:
                    job = next(job_iter)
                except StopIteration:
                    break
                pending.add(executor.submit(copy_file, job, args.overwrite, args.chunk_size))

    progress_thread.join()

    elapsed = time.time() - t0
    copied_files, skipped_files, failed_files, copied_bytes = stats.snapshot()

    print("\nDone.")
    print(f"Elapsed: {elapsed:.2f} s")
    print(f"Copied : {copied_files} files")
    print(f"Skipped: {skipped_files} files")
    print(f"Failed : {failed_files} files")
    print(f"Written: {copied_bytes / (1024**3):.2f} GiB")
    print(f"Avg speed: {copied_bytes / max(elapsed, 1e-6) / (1024**2):.2f} MiB/s")

    if failures:
        print("\nSome files failed:", file=sys.stderr)
        for path, err in failures[:20]:
            print(f"  {path} -> {err}", file=sys.stderr)
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
