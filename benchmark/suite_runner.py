from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from launcher_algorithm import METHODS


@dataclass(frozen=True)
class ObjectRecord:
    uid: str
    obj_path: Path
    gt_pointcloud: Path


@dataclass(frozen=True)
class EpisodeJob:
    episode_id: str
    uid: str
    obj_path: Path
    gt_pointcloud: Path
    constraint: str
    feasibility_json: Path
    start_view_set: str
    start_view_id: int
    method_name: str
    method_display_name: str
    family: str
    execution_mode_compatibility: str
    execution_mode: str
    session_dir: Path
    summary_json: Path
    signal_path: Path
    stdout_log: Path
    stderr_log: Path


@dataclass
class PreparedProcess:
    job: EpisodeJob
    process: subprocess.Popen[Any]
    started_at: float
    stdout_handle: Any
    stderr_handle: Any


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a benchmark episode suite.")
    parser.add_argument("--objects-json", type=Path, required=True)
    parser.add_argument("--object-dataset-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--constraints", type=str, nargs="+", required=True)
    parser.add_argument("--methods", type=str, nargs="+", required=True)
    parser.add_argument("--start-view-ids", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--start-view-set", type=str, default="benchmark_start_3")
    parser.add_argument("--cache-index-json", type=Path, default=Path("render_cache/cache_index.json"))
    parser.add_argument("--feasibility-root", type=Path, default=Path("configs/feasibility"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--action-wait-timeout-sec", type=float, default=120.0)
    parser.add_argument("--action-poll-interval-sec", type=float, default=0.01)
    parser.add_argument("--algorithm-startup-ready-timeout-sec", type=float, default=300.0)
    parser.add_argument("--algorithm-startup-ready-poll-interval-sec", type=float, default=0.1)
    parser.add_argument("--algorithm-exit-timeout-sec", type=float, default=5.0)
    parser.add_argument(
        "--shape-completion-device",
        type=str,
        default=None,
        help="Device for shape completion service. Defaults to --device.",
    )
    parser.add_argument("--shape-completion-ready-timeout-sec", type=float, default=300.0)
    parser.add_argument("--shape-completion-ready-poll-interval-sec", type=float, default=0.05)
    parser.add_argument("--shape-completion-exit-timeout-sec", type=float, default=2.0)
    parser.add_argument(
        "--planning-network-command",
        action="append",
        default=[],
        help="Named planning-network service command to pass to launcher_episode, in name::command form.",
    )
    parser.add_argument("--planning-network-ready-timeout-sec", type=float, default=300.0)
    parser.add_argument("--planning-network-ready-poll-interval-sec", type=float, default=0.05)
    parser.add_argument("--planning-network-exit-timeout-sec", type=float, default=2.0)
    parser.add_argument(
        "--enable-iterative-enough-info-stop",
        dest="enable_iterative_enough_info_stop",
        action="store_true",
        help="Enable benchmark-side enough-info stopping for iterative methods.",
    )
    parser.add_argument(
        "--disable-iterative-enough-info-stop",
        dest="enable_iterative_enough_info_stop",
        action="store_false",
        help="Disable benchmark-side enough-info stopping for iterative methods.",
    )
    parser.set_defaults(enable_iterative_enough_info_stop=True)
    parser.add_argument("--max-visited-view-num", type=int, default=129)
    parser.add_argument(
        "--budget-checkpoints",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional K checkpoints to pass to launcher_episode. "
            "If omitted, launcher_episode uses method-family defaults."
        ),
    )
    parser.add_argument("--cleanup-session-on-success", action="store_true")
    parser.add_argument("--keep-session-observations", action="store_true")
    parser.add_argument("--cleanup-shape-completion-on-success", action="store_true")
    parser.add_argument("--cleanup-planning-network-on-success", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--prepare-next", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _read_objects(path: Path, *, object_dataset_root: Path) -> list[ObjectRecord]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        if "objects" in data:
            data = data["objects"]
        elif "uids" in data:
            data = data["uids"]
        else:
            raise ValueError(f"Unsupported objects json object keys: {sorted(data.keys())}")
    if not isinstance(data, list):
        raise ValueError("objects json must be a list, or a dict with objects/uids")

    records = []
    for item in data:
        if isinstance(item, str):
            uid = item
            obj_path = _default_obj_path(object_dataset_root, uid)
            gt_pointcloud = _default_gt_pointcloud_path(object_dataset_root, uid)
        elif isinstance(item, dict):
            uid = str(item["uid"])
            obj_path_raw = item.get("obj_path")
            gt_raw = item.get("gt_pointcloud") or item.get("gt_pointcloud_path") or item.get("pcd_path")
            obj_path = _resolve_dataset_path(object_dataset_root, obj_path_raw) if obj_path_raw else _default_obj_path(object_dataset_root, uid)
            gt_pointcloud = _resolve_dataset_path(object_dataset_root, gt_raw) if gt_raw else _default_gt_pointcloud_path(object_dataset_root, uid)
        else:
            raise ValueError(f"Unsupported object entry: {item!r}")
        records.append(ObjectRecord(uid=uid, obj_path=obj_path, gt_pointcloud=gt_pointcloud))
    records.sort(key=lambda record: record.uid)
    return records


def _resolve_dataset_path(root: Path, path_value: Any) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return root / path


def _default_obj_path(root: Path, uid: str) -> Path:
    return root / "geometry_sampled" / "obj_normalized" / uid / f"{uid}.obj"


def _default_gt_pointcloud_path(root: Path, uid: str) -> Path:
    return root / "geometry_sampled" / "pcd_normalized" / f"{uid}.pcd"


def _select_shard(records: list[ObjectRecord], *, shard_id: int, num_shards: int) -> list[ObjectRecord]:
    if num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError("--shard-id must satisfy 0 <= shard_id < num_shards")
    return [record for index, record in enumerate(records) if index % num_shards == shard_id]


def _build_jobs(args: argparse.Namespace, objects: list[ObjectRecord]) -> list[EpisodeJob]:
    jobs: list[EpisodeJob] = []
    summaries_dir = args.run_dir / "summaries"
    sessions_dir = args.run_dir / "sessions"
    signals_dir = args.run_dir / "signals"
    logs_dir = args.run_dir / "logs"
    for method in args.methods:
        if method not in METHODS:
            raise KeyError(f"Unknown method in launcher_algorithm registry: {method}")
    for obj in objects:
        for start_view_id in args.start_view_ids:
            for constraint in args.constraints:
                feasibility_json = args.feasibility_root / f"{constraint}.json"
                for method in args.methods:
                    spec = METHODS[method]
                    family = spec.family
                    episode_id = f"{obj.uid}__s{int(start_view_id)}__{constraint}__{method}"
                    jobs.append(
                        EpisodeJob(
                            episode_id=episode_id,
                            uid=obj.uid,
                            obj_path=obj.obj_path,
                            gt_pointcloud=obj.gt_pointcloud,
                            constraint=constraint,
                            feasibility_json=feasibility_json,
                            start_view_set=args.start_view_set,
                            start_view_id=int(start_view_id),
                            method_name=method,
                            method_display_name=spec.display_name or spec.method_name,
                            family=family,
                            execution_mode_compatibility=spec.execution_mode_compatibility,
                            execution_mode=spec.execution_mode,
                            session_dir=sessions_dir / episode_id,
                            summary_json=summaries_dir / f"{episode_id}.json",
                            signal_path=signals_dir / f"{episode_id}.start",
                            stdout_log=logs_dir / f"{episode_id}.stdout.log",
                            stderr_log=logs_dir / f"{episode_id}.stderr.log",
                        )
                    )
    if args.limit is not None:
        jobs = jobs[: max(0, int(args.limit))]
    return jobs


def _launcher_command(args: argparse.Namespace, job: EpisodeJob, *, wait_for_signal: bool) -> list[str]:
    shape_completion_device = args.shape_completion_device or args.device
    method_spec = METHODS[job.method_name]
    command = [
        sys.executable,
        "launcher_episode.py",
        "--episode-id",
        job.episode_id,
        "--uid",
        job.uid,
        "--obj-path",
        str(job.obj_path),
        "--gt-pointcloud",
        str(job.gt_pointcloud),
        "--cache-index-json",
        str(args.cache_index_json),
        "--feasibility-json",
        str(job.feasibility_json),
        "--viewspace-constraint-name",
        job.constraint,
        "--session-dir",
        str(job.session_dir),
        "--summary-json",
        str(job.summary_json),
        "--start-view-set",
        job.start_view_set,
        "--start-view-id",
        str(job.start_view_id),
        "--method-name",
        job.method_name,
        "--method-display-name",
        job.method_display_name,
        "--family",
        job.family,
        "--execution-mode-compatibility",
        job.execution_mode_compatibility,
        "--execution-mode",
        job.execution_mode,
        "--enable-iterative-enough-info-stop" if args.enable_iterative_enough_info_stop else "--disable-iterative-enough-info-stop",
        "--algorithm-command",
        f"{sys.executable} launcher_algorithm.py --method {job.method_name} --session-dir {{session_dir}}",
        "--device",
        args.device,
        "--max-visited-view-num",
        str(args.max_visited_view_num),
        "--action-wait-timeout-sec",
        str(args.action_wait_timeout_sec),
        "--action-poll-interval-sec",
        str(args.action_poll_interval_sec),
        "--algorithm-startup-ready-timeout-sec",
        str(args.algorithm_startup_ready_timeout_sec),
        "--algorithm-startup-ready-poll-interval-sec",
        str(args.algorithm_startup_ready_poll_interval_sec),
        "--algorithm-exit-timeout-sec",
        str(args.algorithm_exit_timeout_sec),
    ]
    if args.budget_checkpoints is not None:
        command.append("--budget-checkpoints")
        command.extend(str(int(k)) for k in args.budget_checkpoints)
    if bool(method_spec.metadata.get("requires_shape_completion", False)):
        command.extend(
            [
                "--shape-completion-command",
                f"{sys.executable} api_shape_completion.py --session-dir {{session_dir}} --device {shape_completion_device}",
                "--shape-completion-ready-timeout-sec",
                str(args.shape_completion_ready_timeout_sec),
                "--shape-completion-ready-poll-interval-sec",
                str(args.shape_completion_ready_poll_interval_sec),
                "--shape-completion-exit-timeout-sec",
                str(args.shape_completion_exit_timeout_sec),
            ]
        )
    planning_network_commands = list(args.planning_network_command)
    if method_spec.planning_network_command is not None:
        service_name = str(method_spec.metadata.get("planning_network_service_name", "default"))
        values = {
            "benchmark_root": str(Path.cwd()),
            "python": sys.executable,
            "session_dir": str(job.session_dir),
            "cache_index_json": str(args.cache_index_json),
            "view_set": str(args.start_view_set),
        }
        service_tokens = []
        for token in method_spec.planning_network_command:
            out = str(token)
            for key, value in values.items():
                out = out.replace("{" + key + "}", value)
            service_tokens.append(out)
        planning_network_commands.append(f"{service_name}::{shlex.join(service_tokens)}")
    for planning_network_command in planning_network_commands:
        command.extend(
            [
                "--planning-network-command",
                planning_network_command,
            ]
        )
    if planning_network_commands:
        command.extend(
            [
                "--planning-network-ready-timeout-sec",
                str(args.planning_network_ready_timeout_sec),
                "--planning-network-ready-poll-interval-sec",
                str(args.planning_network_ready_poll_interval_sec),
                "--planning-network-exit-timeout-sec",
                str(args.planning_network_exit_timeout_sec),
            ]
        )
    if args.cleanup_session_on_success:
        command.append("--cleanup-session-on-success")
    if args.keep_session_observations:
        command.append("--keep-session-observations")
    if args.cleanup_shape_completion_on_success:
        command.append("--cleanup-shape-completion-on-success")
    if args.cleanup_planning_network_on_success:
        command.append("--cleanup-planning-network-on-success")
    if wait_for_signal:
        command.extend(["--wait-for-start-signal", str(job.signal_path)])
    return command


def _prepare_job(args: argparse.Namespace, job: EpisodeJob, *, wait_for_signal: bool) -> PreparedProcess:
    job.stdout_log.parent.mkdir(parents=True, exist_ok=True)
    job.stderr_log.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = job.stdout_log.open("w", encoding="utf-8")
    stderr_handle = job.stderr_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        _launcher_command(args, job, wait_for_signal=wait_for_signal),
        cwd=str(Path.cwd()),
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
    )
    return PreparedProcess(
        job=job,
        process=process,
        started_at=time.perf_counter(),
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
    )


def _touch_signal(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _wait_prepared(prepared: PreparedProcess) -> dict[str, Any]:
    returncode = prepared.process.wait()
    finished_at = time.perf_counter()
    prepared.stdout_handle.close()
    prepared.stderr_handle.close()
    return _status_from_job(
        prepared.job,
        returncode=returncode,
        elapsed_sec=finished_at - prepared.started_at,
    )


def _status_from_job(job: EpisodeJob, *, returncode: int | None, elapsed_sec: float | None, skipped: bool = False) -> dict[str, Any]:
    status = {
        "episode_id": job.episode_id,
        "uid": job.uid,
        "start_view_id": job.start_view_id,
        "constraint": job.constraint,
        "method_name": job.method_name,
        "method_display_name": job.method_display_name,
        "family": job.family,
        "execution_mode_compatibility": job.execution_mode_compatibility,
        "execution_mode": job.execution_mode,
        "returncode": returncode,
        "status": "skipped" if skipped else ("success" if returncode == 0 else "failed"),
        "elapsed_sec": elapsed_sec,
        "summary_json": str(job.summary_json),
        "session_dir": str(job.session_dir),
        "stdout_log": str(job.stdout_log),
        "stderr_log": str(job.stderr_log),
    }
    if job.summary_json.exists():
        try:
            summary = json.loads(job.summary_json.read_text(encoding="utf-8"))
            for key in (
                "terminal_source",
                "terminal_reason",
                "algorithm_stop_reason",
                "benchmark_stop_reason",
                "method_display_name",
                "execution_mode_compatibility",
                "execution_mode",
                "visited_view_num",
                "accepted_nbv_action_num",
            ):
                status[key] = summary.get(key)
        except json.JSONDecodeError:
            status["summary_parse_error"] = True
    return status


def _summary_is_resume_success(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(summary, dict):
        return False
    terminal_source = summary.get("terminal_source")
    return terminal_source in {"algorithm", "benchmark"}


def _append_status(status_path: Path, status: dict[str, Any]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(status, ensure_ascii=True) + "\n")


def _write_suite_config(args: argparse.Namespace, *, total_objects: int, shard_objects: int, num_jobs: int) -> None:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "objects_json": str(args.objects_json),
        "object_dataset_root": str(args.object_dataset_root),
        "constraints": list(args.constraints),
        "methods": list(args.methods),
        "method_specs": {
            method: {
                "display_name": METHODS[method].display_name or METHODS[method].method_name,
                "family": METHODS[method].family,
                "execution_mode_compatibility": METHODS[method].execution_mode_compatibility,
                "execution_mode": METHODS[method].execution_mode,
                "metadata": dict(METHODS[method].metadata),
            }
            for method in args.methods
        },
        "start_view_ids": list(args.start_view_ids),
        "start_view_set": args.start_view_set,
        "cache_index_json": str(args.cache_index_json),
        "feasibility_root": str(args.feasibility_root),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "total_objects": total_objects,
        "shard_objects": shard_objects,
        "num_jobs": num_jobs,
        "prepare_next": bool(args.prepare_next),
        "enable_iterative_enough_info_stop": bool(args.enable_iterative_enough_info_stop),
        "shape_completion": {
            "device": args.shape_completion_device or args.device,
            "ready_timeout_sec": args.shape_completion_ready_timeout_sec,
            "ready_poll_interval_sec": args.shape_completion_ready_poll_interval_sec,
            "exit_timeout_sec": args.shape_completion_exit_timeout_sec,
            "cleanup_on_success": bool(args.cleanup_shape_completion_on_success),
            "methods_requiring_service": [
                method for method in args.methods if bool(METHODS[method].metadata.get("requires_shape_completion", False))
            ],
        },
        "planning_network": {
            "commands": list(args.planning_network_command),
            "method_commands": {
                method: None
                if METHODS[method].planning_network_command is None
                else list(METHODS[method].planning_network_command)
                for method in args.methods
            },
            "ready_timeout_sec": args.planning_network_ready_timeout_sec,
            "ready_poll_interval_sec": args.planning_network_ready_poll_interval_sec,
            "exit_timeout_sec": args.planning_network_exit_timeout_sec,
            "cleanup_on_success": bool(args.cleanup_planning_network_on_success),
        },
    }
    (args.run_dir / f"suite_config_shard{args.shard_id}.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(args.objects_json, args.run_dir / "objects_source.json")


def _run_jobs(args: argparse.Namespace, jobs: list[EpisodeJob]) -> None:
    status_path = args.run_dir / f"status_shard{args.shard_id}.jsonl"
    if args.dry_run:
        for job in jobs:
            print(shlex.join(_launcher_command(args, job, wait_for_signal=args.prepare_next)))
        return

    if not args.prepare_next:
        for job in jobs:
            if args.resume and _summary_is_resume_success(job.summary_json):
                _append_status(status_path, _status_from_job(job, returncode=0, elapsed_sec=0.0, skipped=True))
                continue
            prepared = _prepare_job(args, job, wait_for_signal=False)
            status = _wait_prepared(prepared)
            _append_status(status_path, status)
        return

    runnable_jobs = []
    for job in jobs:
        if args.resume and _summary_is_resume_success(job.summary_json):
            _append_status(status_path, _status_from_job(job, returncode=0, elapsed_sec=0.0, skipped=True))
        else:
            runnable_jobs.append(job)
    if not runnable_jobs:
        return

    current = _prepare_job(args, runnable_jobs[0], wait_for_signal=True)
    _touch_signal(runnable_jobs[0].signal_path)
    for index, job in enumerate(runnable_jobs):
        next_prepared = None
        if index + 1 < len(runnable_jobs):
            next_prepared = _prepare_job(args, runnable_jobs[index + 1], wait_for_signal=True)
        status = _wait_prepared(current)
        _append_status(status_path, status)
        if next_prepared is not None:
            _touch_signal(next_prepared.job.signal_path)
            current = next_prepared


def main() -> int:
    args = _build_argparser().parse_args()
    args.run_dir = args.run_dir.resolve()
    objects = _read_objects(args.objects_json, object_dataset_root=args.object_dataset_root)
    shard_objects = _select_shard(objects, shard_id=args.shard_id, num_shards=args.num_shards)
    jobs = _build_jobs(args, shard_objects)
    _write_suite_config(args, total_objects=len(objects), shard_objects=len(shard_objects), num_jobs=len(jobs))
    print(
        json.dumps(
            {
                "run_dir": str(args.run_dir),
                "total_objects": len(objects),
                "shard_objects": len(shard_objects),
                "num_jobs": len(jobs),
                "shard_id": args.shard_id,
                "num_shards": args.num_shards,
                "prepare_next": bool(args.prepare_next),
                "dry_run": bool(args.dry_run),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    _run_jobs(args, jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
python suite_runner.py \
  --objects-json final_clean_pool/final_clean_long_tail_pool.json \
  --object-dataset-root /mnt/d/ObjView-Bench/object_dataset \
  --constraints whole \
  --methods oracle_rollout_hybrid_nsc01 \
  --cache-index-json render_cache/cache_index.json \
  --action-wait-timeout-sec 120 \
  --prepare-next \
  --start-view-ids 0 \
  --device cuda:0 \
  --max-visited-view-num 129 \
  --num-shards 3 \
  --run-dir runs/final_clean_long_tail_oracle_rollout_hybrid_sorted_shard0of3 \
  --shard-id 0 \
  --resume

python suite_runner.py \
  --objects-json released_benchmark_splits/main_hidden_test.json \
  --object-dataset-root /mnt/d/ObjView-Bench/object_dataset \
  --constraints whole \
  --methods oracle_rollout_hybrid_nsc01 \
  --cache-index-json render_cache/cache_index.json \
  --action-wait-timeout-sec 120 \
  --prepare-next \
  --start-view-ids 0 \
  --device cuda:0 \
  --max-visited-view-num 129 \
  --num-shards 3 \
  --run-dir runs/main_hidden_oracle_rollout_hybrid_sorted_shard0of3 \
  --shard-id 0 \
  --resume

python suite_runner.py \
  --objects-json released_benchmark_splits/main_hidden_test.json \
  --object-dataset-root /mnt/d/ObjView-Bench/object_dataset \
  --constraints whole quarter \
  --methods simple_random_tsporder_5 simple_random_tsporder_10 simple_random_tsporder_30 \
  --cache-index-json render_cache/cache_index.json \
  --algorithm-startup-ready-timeout-sec 300 \
  --action-wait-timeout-sec 120 \
  --cleanup-session-on-success \
  --prepare-next \
  --start-view-ids 0 \
  --num-shards 3 \
  --run-dir runs/main_simple_random_tsporder_5_10_30_shard0of3 \
  --shard-id 0 \
  --resume

python suite_runner.py \
  --objects-json released_benchmark_splits/main_hidden_test.json \
  --object-dataset-root /mnt/d/ObjView-Bench/object_dataset \
  --constraints whole quarter \
  --methods classical_voxel_ig_rse \
  --cache-index-json render_cache/cache_index.json \
  --algorithm-startup-ready-timeout-sec 300 \
  --action-wait-timeout-sec 120 \
  --keep-session-observations \
  --prepare-next \
  --start-view-ids 0 \
  --device cuda:0 \
  --max-visited-view-num 129 \
  --num-shards 2 \
  --run-dir runs/main_classical_voxel_ig_rse_shard0of2\
  --shard-id 0 \
  --resume

python suite_runner.py \
  --objects-json released_benchmark_splits/main_hidden_test.json \
  --object-dataset-root /mnt/d/ObjView-Bench/object_dataset \
  --constraints whole quarter \
  --methods classical_voxel_ig_rse_mov \
  --cache-index-json render_cache/cache_index.json \
  --algorithm-startup-ready-timeout-sec 300 \
  --action-wait-timeout-sec 120 \
  --keep-session-observations \
  --prepare-next \
  --start-view-ids 0 \
  --device cuda:0 \
  --max-visited-view-num 129 \
  --run-dir runs/main_classical_voxel_ig_rse_mov_shard0of2 \
  --num-shards 2 \
  --shard-id 0 \
  --resume

python suite_runner.py \
  --objects-json released_benchmark_splits/main_hidden_test.json \
  --object-dataset-root /mnt/d/ObjView-Bench/object_dataset \
  --constraints whole quarter \
  --methods pointr_c_nbv \
  --cache-index-json render_cache/cache_index.json \
  --algorithm-startup-ready-timeout-sec 300 \
  --action-wait-timeout-sec 120 \
  --keep-session-observations \
  --cleanup-shape-completion-on-success \
  --prepare-next \
  --start-view-ids 0 \
  --device cuda:0 \
  --max-visited-view-num 129 \
  --run-dir runs/main_completion_planning_pointr_c_nbv_shard0of2 \
  --num-shards 2 \
  --shard-id 0 \
  --resume

python suite_runner.py \
  --objects-json released_benchmark_splits/main_hidden_test.json \
  --object-dataset-root /mnt/d/ObjView-Bench/object_dataset \
  --constraints whole quarter \
  --methods pointr_c_scp \
  --cache-index-json render_cache/cache_index.json \
  --algorithm-startup-ready-timeout-sec 300 \
  --action-wait-timeout-sec 120 \
  --keep-session-observations \
  --cleanup-shape-completion-on-success \
  --prepare-next \
  --start-view-ids 0 \
  --device cuda:0 \
  --max-visited-view-num 129 \
  --run-dir runs/main_completion_planning_pointr_c_scp_shard0of2 \
  --num-shards 2 \
  --shard-id 0 \
  --resume

python suite_runner.py \
  --objects-json released_benchmark_splits/main_hidden_test.json \
  --object-dataset-root /mnt/d/ObjView-Bench/object_dataset \
  --constraints whole quarter \
  --methods pointr_c_mcp_5 pointr_c_mcp_10 pointr_c_mcp_30 \
  --cache-index-json render_cache/cache_index.json \
  --algorithm-startup-ready-timeout-sec 300 \
  --action-wait-timeout-sec 120 \
  --keep-session-observations \
  --cleanup-shape-completion-on-success \
  --prepare-next \
  --start-view-ids 0 \
  --device cuda:0 \
  --max-visited-view-num 129 \
  --run-dir runs/main_completion_planning_pointr_c_mcp_shard0of2 \
  --num-shards 2 \
  --shard-id 0 \
  --resume

python suite_runner.py \
  --objects-json released_benchmark_splits/main_hidden_test.json \
  --object-dataset-root /mnt/d/ObjView-Bench/object_dataset \
  --constraints whole quarter \
  --methods mascvp_planning_network \
  --cache-index-json render_cache/cache_index.json \
  --algorithm-startup-ready-timeout-sec 300 \
  --planning-network-ready-timeout-sec 300 \
  --action-wait-timeout-sec 120 \
  --cleanup-planning-network-on-success \
  --prepare-next \
  --start-view-ids 0 \
  --device cuda:0 \
  --max-visited-view-num 129 \
  --run-dir runs/main_learned_set_cover_mascvp_shard0of2 \
  --num-shards 2 \
  --shard-id 0 \
  --resume

python suite_runner.py \
  --objects-json released_benchmark_splits/main_hidden_test.json \
  --object-dataset-root /mnt/d/ObjView-Bench/object_dataset \
  --constraints whole quarter \
  --methods benbv_planning_network \
  --cache-index-json render_cache/cache_index.json \
  --algorithm-startup-ready-timeout-sec 300 \
  --planning-network-ready-timeout-sec 300 \
  --action-wait-timeout-sec 120 \
  --cleanup-planning-network-on-success \
  --prepare-next \
  --start-view-ids 0 \
  --device cuda:0 \
  --max-visited-view-num 129 \
  --run-dir runs/main_learned_nbv_benbv_shard0of2 \
  --num-shards 2 \
  --shard-id 0 \
  --resume

python suite_runner.py \
  --objects-json released_benchmark_splits/main_hidden_test.json \
  --object-dataset-root /mnt/d/ObjView-Bench/object_dataset \
  --constraints whole quarter \
  --methods nbvnet_planning_network \
  --cache-index-json render_cache/cache_index.json \
  --algorithm-startup-ready-timeout-sec 300 \
  --planning-network-ready-timeout-sec 300 \
  --action-wait-timeout-sec 120 \
  --cleanup-planning-network-on-success \
  --prepare-next \
  --start-view-ids 0 \
  --device cuda:0 \
  --max-visited-view-num 129 \
  --run-dir runs/main_learned_nbv_nbvnet_shard0of2 \
  --num-shards 2 \
  --shard-id 0 \
  --resume

"""
