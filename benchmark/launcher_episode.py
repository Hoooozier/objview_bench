from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from api_evaluation import EvaluateAPI
from api_feasibility import FeasibilityAPI
from api_interaction import InteractionSession
from runner_episode import EpisodeConfig, EpisodeRunner, EpisodeRunnerError
from runner_episode import _load_cache_index, _resolve_start_pose_with_auto_fallback


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch one benchmark episode with an algorithm subprocess.")
    parser.add_argument("--episode-id", type=str, required=True)
    parser.add_argument("--uid", type=str, required=True)
    parser.add_argument("--obj-path", type=Path, required=True)
    parser.add_argument("--gt-pointcloud", type=Path, required=True)
    parser.add_argument("--cache-index-json", type=Path, default=Path("render_cache/cache_index.json"))
    parser.add_argument("--feasibility-json", type=Path, required=True)
    parser.add_argument("--viewspace-constraint-name", type=str, default=None)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--session-observations-root", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--start-pose", type=float, nargs=7, default=None)
    parser.add_argument("--start-view-set", type=str, default="benchmark_start_3")
    parser.add_argument("--start-view-id", type=int, default=0)
    parser.add_argument("--method-name", type=str, required=True)
    parser.add_argument("--method-display-name", type=str, default=None)
    parser.add_argument("--family", type=str, required=True)
    parser.add_argument("--execution-mode-compatibility", type=str, default="unknown")
    parser.add_argument("--execution-compatibility", type=str, default=None, help="Deprecated alias for --execution-mode-compatibility.")
    parser.add_argument("--execution-mode", type=str, default="unknown")
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
    parser.add_argument("--algorithm-command", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--max-visited-view-num",
        type=int,
        default=129,
        help="Safety cap on total acquired views, including the initial observation.",
    )
    parser.add_argument(
        "--budget-checkpoints",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Accepted NBV action counts at which to record K checkpoints. "
            "Defaults to 5/10/30/50, or 1/2/3/4/5/10/15/20/30/40/.../max_actions for oracle family."
        ),
    )
    parser.add_argument("--action-wait-timeout-sec", type=float, default=100.0)
    parser.add_argument("--action-poll-interval-sec", type=float, default=0.01)
    parser.add_argument(
        "--algorithm-exit-timeout-sec",
        type=float,
        default=5.0,
        help="Seconds to wait for the algorithm process after the runner finishes.",
    )
    parser.add_argument(
        "--no-clean-session",
        action="store_true",
        help="Do not remove the existing session directory before launch.",
    )
    parser.add_argument(
        "--keep-session-observations",
        action="store_true",
        help="Keep non-cache observation files under session_observations after launch.",
    )
    parser.add_argument(
        "--cleanup-shape-completion-on-success",
        action="store_true",
        help="Remove session/shape_completion after a successful launch while keeping the rest of the session.",
    )
    parser.add_argument(
        "--cleanup-session-on-success",
        action="store_true",
        help="Remove the whole session directory after a successful launch. Requires --summary-json outside session-dir.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print the full episode summary JSON instead of a compact launcher status.",
    )
    parser.add_argument(
        "--wait-for-start-signal",
        type=Path,
        default=None,
        help="Build the runner, then wait for this signal file before starting rollout.",
    )
    parser.add_argument("--start-signal-poll-interval-sec", type=float, default=0.1)
    parser.add_argument("--start-signal-timeout-sec", type=float, default=None)
    parser.add_argument(
        "--wait-for-algorithm-startup-ready",
        dest="wait_for_algorithm_startup_ready",
        action="store_true",
        help="Wait for the algorithm process to publish a startup-ready signal before starting rollout.",
    )
    parser.add_argument(
        "--no-wait-for-algorithm-startup-ready",
        dest="wait_for_algorithm_startup_ready",
        action="store_false",
        help="Do not wait for the algorithm startup-ready signal before starting rollout.",
    )
    parser.add_argument(
        "--algorithm-startup-ready-relpath",
        type=Path,
        default=Path("actions/algorithm_started"),
        help="Session-relative signal path written by the algorithm after startup/warmup completes.",
    )
    parser.add_argument("--algorithm-startup-ready-poll-interval-sec", type=float, default=0.1)
    parser.add_argument("--algorithm-startup-ready-timeout-sec", type=float, default=300.0)
    parser.add_argument(
        "--shape-completion-command",
        type=str,
        default=None,
        help="Optional command to launch a shape completion service subprocess.",
    )
    parser.add_argument(
        "--shape-completion-ready-relpath",
        type=Path,
        default=Path("shape_completion/service_ready"),
        help="Session-relative ready signal path written by the shape completion service after startup/warmup.",
    )
    parser.add_argument("--shape-completion-ready-poll-interval-sec", type=float, default=0.05)
    parser.add_argument("--shape-completion-ready-timeout-sec", type=float, default=300.0)
    parser.add_argument(
        "--shape-completion-exit-timeout-sec",
        type=float,
        default=2.0,
        help="Seconds to wait for the shape completion service after the runner finishes.",
    )
    parser.add_argument(
        "--planning-network-command",
        action="append",
        default=[],
        help=(
            "Named planning-network service command in the form name::command. "
            "The ready path is planning_network/<name>/service_ready."
        ),
    )
    parser.add_argument("--planning-network-ready-timeout-sec", type=float, default=300.0)
    parser.add_argument("--planning-network-ready-poll-interval-sec", type=float, default=0.05)
    parser.add_argument(
        "--planning-network-exit-timeout-sec",
        type=float,
        default=2.0,
        help="Seconds to wait for each planning network service after the runner finishes.",
    )
    parser.add_argument(
        "--cleanup-planning-network-on-success",
        action="store_true",
        help="Remove session/planning_network after a successful launch while keeping the rest of the session.",
    )
    parser.set_defaults(
        wait_for_algorithm_startup_ready=True,
        enable_iterative_enough_info_stop=True,
    )
    return parser


def _path_is_inside(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _clean_directory(path: Path, *, allowed_root: Path) -> None:
    resolved = path.resolve()
    root = allowed_root.resolve()
    if not _path_is_inside(resolved, root):
        raise ValueError(f"Refusing to clean path outside benchmark root: {resolved}")
    if resolved == root:
        raise ValueError(f"Refusing to clean benchmark root itself: {resolved}")
    if resolved.anchor == str(resolved):
        raise ValueError(f"Refusing to clean filesystem root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def _cleanup_shape_completion_dir(session_dir: Path, *, allowed_root: Path) -> None:
    shape_completion_dir = (session_dir / "shape_completion").resolve()
    root = allowed_root.resolve()
    if not _path_is_inside(shape_completion_dir, root):
        raise ValueError(f"Refusing to clean shape_completion outside benchmark root: {shape_completion_dir}")
    if shape_completion_dir.exists():
        shutil.rmtree(shape_completion_dir)


def _cleanup_planning_network_dir(session_dir: Path, *, allowed_root: Path) -> None:
    planning_network_dir = (session_dir / "planning_network").resolve()
    root = allowed_root.resolve()
    if not _path_is_inside(planning_network_dir, root):
        raise ValueError(f"Refusing to clean planning_network outside benchmark root: {planning_network_dir}")
    if planning_network_dir.exists():
        shutil.rmtree(planning_network_dir)


def _format_algorithm_command(template: str, *, args: argparse.Namespace, session_dir: Path) -> str:
    replacements = {
        "session_dir": str(session_dir),
        "episode_id": args.episode_id,
        "uid": args.uid,
        "method_name": args.method_name,
        "method_display_name": args.method_display_name or args.method_name,
        "family": args.family,
        "execution_mode_compatibility": args.execution_compatibility or args.execution_mode_compatibility,
        "execution_mode": args.execution_mode,
        "start_view_set": args.start_view_set,
        "start_view_id": str(args.start_view_id),
        "viewspace_constraint_name": args.viewspace_constraint_name or Path(args.feasibility_json).stem,
    }
    command = str(template)
    for key, value in replacements.items():
        command = command.replace("{" + key + "}", value)
    return command


def _parse_planning_network_commands(
    values: list[str],
    *,
    args: argparse.Namespace,
    session_dir: Path,
) -> list[dict[str, Any]]:
    services = []
    seen_names: set[str] = set()
    for raw in values:
        if "::" not in raw:
            raise ValueError(f"Expected planning network command in name::command form, got: {raw}")
        name, command_template = raw.split("::", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Planning network service name is empty in: {raw}")
        if any(ch in name for ch in "/\\"):
            raise ValueError(f"Planning network service name must not contain path separators: {name}")
        if name in seen_names:
            raise ValueError(f"Duplicate planning network service name: {name}")
        seen_names.add(name)
        command = _format_algorithm_command(command_template.strip(), args=args, session_dir=session_dir)
        ready_path = (session_dir / "planning_network" / name / "service_ready").resolve()
        services.append({"name": name, "command": command, "ready_path": ready_path})
    return services


def _default_budget_checkpoints(args: argparse.Namespace) -> tuple[int, ...]:
    if args.budget_checkpoints is not None:
        values = sorted({int(k) for k in args.budget_checkpoints if int(k) > 0})
        return tuple(values)

    if str(args.family) == "oracle":
        max_actions = max(0, int(args.max_visited_view_num) - 1)
        values = [k for k in (1, 2, 3, 4, 5, 10, 15, 20) if k <= max_actions]
        values.extend(range(30, max_actions + 1, 10))
        if max_actions > 0 and max_actions not in values:
            values.append(max_actions)
        return tuple(sorted(set(values)))

    return (5, 10, 30, 50)


def _launch_algorithm(command: str, *, cwd: Path) -> subprocess.Popen[Any]:
    return subprocess.Popen(
        command,
        cwd=str(cwd),
        shell=True,
        stdin=subprocess.DEVNULL,
        start_new_session=(os.name != "nt"),
    )


def _launch_subprocess(command: str, *, cwd: Path) -> subprocess.Popen[Any]:
    return subprocess.Popen(
        command,
        cwd=str(cwd),
        shell=True,
        stdin=subprocess.DEVNULL,
        start_new_session=(os.name != "nt"),
    )


def _terminate_process(process: subprocess.Popen[Any], *, kill: bool) -> None:
    if os.name != "nt":
        try:
            pgid = os.getpgid(process.pid)
        except ProcessLookupError:
            return
        sig = signal.SIGKILL if kill else signal.SIGTERM
        try:
            os.killpg(pgid, sig)
            return
        except ProcessLookupError:
            return
        except PermissionError:
            pass
    if kill:
        process.kill()
    else:
        process.terminate()


def _finish_algorithm_process(process: subprocess.Popen[Any], *, timeout_sec: float) -> dict[str, Any]:
    info: dict[str, Any] = {
        "pid": process.pid,
        "returncode": None,
        "terminated_by_launcher": False,
    }
    try:
        info["returncode"] = process.wait(timeout=max(0.0, float(timeout_sec)))
        return info
    except subprocess.TimeoutExpired:
        info["terminated_by_launcher"] = True
        _terminate_process(process, kill=False)
        try:
            info["returncode"] = process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            _terminate_process(process, kill=True)
            info["returncode"] = process.wait(timeout=2.0)
            info["killed_by_launcher"] = True
        return info


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _build_runner(args: argparse.Namespace, *, session_dir: Path, session_observations_root: Path) -> EpisodeRunner:
    from api_render import Render, intrinsics_from_dict

    cache_index = _load_cache_index(args.cache_index_json)
    intrinsics_data = cache_index.get("intrinsics")
    if not isinstance(intrinsics_data, dict):
        raise ValueError("cache index must contain intrinsics")
    intrinsics = intrinsics_from_dict(intrinsics_data)

    feasibility_api = FeasibilityAPI(args.feasibility_json)
    render_api = Render(
        uid=args.uid,
        obj_path=args.obj_path,
        intrinsics=intrinsics,
        device=args.device,
        mode="auto",
        cache_index_json=args.cache_index_json,
    )
    interaction = InteractionSession(session_dir=session_dir, feasibility_api=feasibility_api)
    evaluator = EvaluateAPI(
        uid=args.uid,
        gt_pointcloud_path=args.gt_pointcloud,
        render_api=render_api,
        feasibility_api=feasibility_api,
    )

    config = EpisodeConfig(
        episode_id=args.episode_id,
        uid=args.uid,
        obj_path=args.obj_path,
        gt_pointcloud_path=args.gt_pointcloud,
        start_pose=_resolve_start_pose_with_auto_fallback(
            cache_index=cache_index,
            uid=args.uid,
            start_view_set=args.start_view_set,
            start_view_id=args.start_view_id,
            explicit_start_pose=None if args.start_pose is None else [float(v) for v in args.start_pose],
        ),
        start_view_set=args.start_view_set,
        start_view_id=args.start_view_id,
        viewspace_constraint_name=args.viewspace_constraint_name or Path(args.feasibility_json).stem,
        method_name=args.method_name,
        method_display_name=args.method_display_name,
        family=args.family,
        execution_mode_compatibility=args.execution_compatibility or args.execution_mode_compatibility,
        execution_mode=args.execution_mode,
        budget_checkpoints=_default_budget_checkpoints(args),
        enable_iterative_enough_info_stop=bool(args.enable_iterative_enough_info_stop),
        max_visited_view_num=args.max_visited_view_num,
        action_wait_timeout_sec=args.action_wait_timeout_sec,
        action_poll_interval_sec=args.action_poll_interval_sec,
        extra={
            "launcher": {
                "algorithm_command_template": args.algorithm_command,
            }
        },
    )

    return EpisodeRunner(
        config=config,
        render_api=render_api,
        feasibility_api=feasibility_api,
        evaluator=evaluator,
        interaction=interaction,
        session_observations_root=session_observations_root,
    )


def _wait_for_start_signal(
    path: Path,
    *,
    poll_interval_sec: float,
    timeout_sec: float | None,
    process: subprocess.Popen[Any] | None = None,
    signal_name: str = "start signal",
    consume_signal: bool = True,
) -> None:
    path = path.resolve()
    start = time.perf_counter()
    while not path.exists():
        if process is not None:
            returncode = process.poll()
            if returncode is not None:
                raise RuntimeError(
                    f"Algorithm process exited with return code {returncode} before publishing {signal_name}: {path}"
                )
        if timeout_sec is not None and time.perf_counter() - start > float(timeout_sec):
            raise TimeoutError(f"Timed out waiting for {signal_name}: {path}")
        time.sleep(max(0.001, float(poll_interval_sec)))
    if consume_signal:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _clear_optional_signal(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _cleanup_session_observations(path: Path, *, allowed_root: Path) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    if not _path_is_inside(resolved, allowed_root.resolve()):
        raise ValueError(f"Refusing to clean observations outside benchmark root: {resolved}")
    if resolved == allowed_root.resolve():
        raise ValueError(f"Refusing to clean benchmark root itself: {resolved}")
    shutil.rmtree(resolved, ignore_errors=True)


def _validate_cleanup_paths(*, session_dir: Path, summary_json: Path, cleanup_session_on_success: bool) -> None:
    if not cleanup_session_on_success:
        return
    if _path_is_inside(summary_json.resolve(), session_dir.resolve()):
        raise ValueError(
            "Refusing --cleanup-session-on-success because summary_json is inside session_dir. "
            "Write --summary-json to a persistent results directory outside the session."
        )


def _cleanup_session_dir(path: Path, *, allowed_root: Path) -> None:
    resolved = path.resolve()
    root = allowed_root.resolve()
    if not _path_is_inside(resolved, root):
        raise ValueError(f"Refusing to clean session outside benchmark root: {resolved}")
    if resolved == root:
        raise ValueError(f"Refusing to clean benchmark root itself: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved, ignore_errors=True)


def main() -> int:
    args = _build_argparser().parse_args()
    benchmark_root = Path.cwd().resolve()
    session_dir = args.session_dir.resolve()
    session_observations_root = (
        args.session_observations_root.resolve()
        if args.session_observations_root is not None
        else session_dir / "session_observations"
    )
    summary_json = args.summary_json.resolve() if args.summary_json is not None else session_dir / "summary.json"
    algorithm_startup_ready_path = (
        (session_dir / args.algorithm_startup_ready_relpath).resolve()
        if args.wait_for_algorithm_startup_ready
        else None
    )
    shape_completion_ready_path = (
        (session_dir / args.shape_completion_ready_relpath).resolve()
        if args.shape_completion_command
        else None
    )
    planning_network_services = _parse_planning_network_commands(
        args.planning_network_command,
        args=args,
        session_dir=session_dir,
    )
    _validate_cleanup_paths(
        session_dir=session_dir,
        summary_json=summary_json,
        cleanup_session_on_success=args.cleanup_session_on_success,
    )

    if not args.no_clean_session:
        _clean_directory(session_dir, allowed_root=benchmark_root)
    else:
        session_dir.mkdir(parents=True, exist_ok=True)
    session_observations_root.mkdir(parents=True, exist_ok=True)
    _clear_optional_signal(algorithm_startup_ready_path)
    _clear_optional_signal(shape_completion_ready_path)
    for service in planning_network_services:
        _clear_optional_signal(service["ready_path"])

    algorithm_command = _format_algorithm_command(args.algorithm_command, args=args, session_dir=session_dir)
    shape_completion_command = (
        _format_algorithm_command(args.shape_completion_command, args=args, session_dir=session_dir)
        if args.shape_completion_command is not None
        else None
    )
    launcher_started_at = time.time()
    algorithm_process: subprocess.Popen[Any] | None = None
    shape_completion_process: subprocess.Popen[Any] | None = None
    planning_network_processes: list[dict[str, Any]] = []
    algorithm_launched_at: float | None = None
    shape_completion_launched_at: float | None = None
    algorithm_info: dict[str, Any] | None = None
    shape_completion_info: dict[str, Any] | None = None
    planning_network_infos: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    exit_code = 0

    try:
        runner = _build_runner(args, session_dir=session_dir, session_observations_root=session_observations_root)
        runner.publish_session_metadata()
        if args.wait_for_start_signal is not None:
            _wait_for_start_signal(
                args.wait_for_start_signal,
                poll_interval_sec=args.start_signal_poll_interval_sec,
                timeout_sec=args.start_signal_timeout_sec,
                signal_name="external start signal",
            )
        if shape_completion_command is not None:
            shape_completion_process = _launch_subprocess(shape_completion_command, cwd=benchmark_root)
            shape_completion_launched_at = time.time()
            if shape_completion_ready_path is not None:
                _wait_for_start_signal(
                    shape_completion_ready_path,
                    poll_interval_sec=args.shape_completion_ready_poll_interval_sec,
                    timeout_sec=args.shape_completion_ready_timeout_sec,
                    process=shape_completion_process,
                    signal_name="shape completion ready signal",
                    consume_signal=False,
                )
        for service in planning_network_services:
            process = _launch_subprocess(service["command"], cwd=benchmark_root)
            service["launched_at"] = time.time()
            service["process"] = process
            planning_network_processes.append(service)
            _wait_for_start_signal(
                service["ready_path"],
                poll_interval_sec=args.planning_network_ready_poll_interval_sec,
                timeout_sec=args.planning_network_ready_timeout_sec,
                process=process,
                signal_name=f"planning network service ready signal ({service['name']})",
                consume_signal=False,
            )
        algorithm_process = _launch_algorithm(algorithm_command, cwd=benchmark_root)
        algorithm_launched_at = time.time()
        if algorithm_startup_ready_path is not None:
            _wait_for_start_signal(
                algorithm_startup_ready_path,
                poll_interval_sec=args.algorithm_startup_ready_poll_interval_sec,
                timeout_sec=args.algorithm_startup_ready_timeout_sec,
                process=algorithm_process,
                signal_name="algorithm startup-ready signal",
            )
        summary = runner.run()
    except EpisodeRunnerError as exc:
        exit_code = 1
        summary = {
            "episode_id": args.episode_id,
            "uid": args.uid,
            "method_name": args.method_name,
            "method_display_name": args.method_display_name or args.method_name,
            "family": args.family,
            "execution_mode_compatibility": args.execution_compatibility or args.execution_mode_compatibility,
            "execution_mode": args.execution_mode,
            "viewspace_constraint_name": args.viewspace_constraint_name or Path(args.feasibility_json).stem,
            "has_started": False,
            "has_terminated": True,
            "terminal_source": "exception",
            "terminal_reason": exc.exception_info.get("type", "episode_runner_error"),
            "exception_info": exc.exception_info,
        }
    except Exception as exc:
        exit_code = 1
        summary = {
            "episode_id": args.episode_id,
            "uid": args.uid,
            "method_name": args.method_name,
            "method_display_name": args.method_display_name or args.method_name,
            "family": args.family,
            "execution_mode_compatibility": args.execution_compatibility or args.execution_mode_compatibility,
            "execution_mode": args.execution_mode,
            "viewspace_constraint_name": args.viewspace_constraint_name or Path(args.feasibility_json).stem,
            "has_started": False,
            "has_terminated": True,
            "terminal_source": "exception",
            "terminal_reason": type(exc).__name__,
            "exception_info": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
    finally:
        if algorithm_process is not None:
            algorithm_info = _finish_algorithm_process(
                algorithm_process,
                timeout_sec=args.algorithm_exit_timeout_sec,
            )
        if shape_completion_process is not None:
            shape_completion_info = _finish_algorithm_process(
                shape_completion_process,
                timeout_sec=args.shape_completion_exit_timeout_sec,
            )
        for service in planning_network_processes:
            process_info = _finish_algorithm_process(
                service["process"],
                timeout_sec=args.planning_network_exit_timeout_sec,
            )
            planning_network_infos.append(
                {
                    "name": service["name"],
                    "command": service["command"],
                    "process": process_info,
                    "launched_at": service.get("launched_at"),
                    "ready_path": str(service["ready_path"]),
                }
            )
        if summary is not None:
            summary["launcher_info"] = {
                "algorithm_command": algorithm_command,
                "shape_completion_command": shape_completion_command,
                "planning_network_services": planning_network_infos,
                "algorithm_process": algorithm_info,
                "shape_completion_process": shape_completion_info,
                "session_dir": str(session_dir),
                "session_observations_root": str(session_observations_root),
                "summary_json": str(summary_json),
                "started_at": launcher_started_at,
                "shape_completion_launched_at": shape_completion_launched_at,
                "algorithm_launched_at": algorithm_launched_at,
                "finished_at": time.time(),
                "cleaned_session_observations": False,
                "cleaned_shape_completion_dir": False,
                "cleaned_planning_network_dir": False,
                "wait_for_start_signal": None if args.wait_for_start_signal is None else str(args.wait_for_start_signal.resolve()),
                "algorithm_startup_ready_path": None if algorithm_startup_ready_path is None else str(algorithm_startup_ready_path),
                "shape_completion_ready_path": None if shape_completion_ready_path is None else str(shape_completion_ready_path),
            }
            if not args.keep_session_observations:
                _cleanup_session_observations(session_observations_root, allowed_root=benchmark_root)
                summary["launcher_info"]["cleaned_session_observations"] = True
            should_cleanup_session = bool(args.cleanup_session_on_success and exit_code == 0)
            should_cleanup_shape_completion = bool(
                args.cleanup_shape_completion_on_success and exit_code == 0 and not should_cleanup_session
            )
            should_cleanup_planning_network = bool(
                args.cleanup_planning_network_on_success and exit_code == 0 and not should_cleanup_session
            )
            summary["launcher_info"]["cleaned_session_dir"] = False
            _write_summary(summary_json, summary)
            if should_cleanup_shape_completion:
                _cleanup_shape_completion_dir(session_dir, allowed_root=benchmark_root)
                summary["launcher_info"]["cleaned_shape_completion_dir"] = True
                _write_summary(summary_json, summary)
            if should_cleanup_planning_network:
                _cleanup_planning_network_dir(session_dir, allowed_root=benchmark_root)
                summary["launcher_info"]["cleaned_planning_network_dir"] = True
                _write_summary(summary_json, summary)
            if should_cleanup_session:
                _cleanup_session_dir(session_dir, allowed_root=benchmark_root)
                summary["launcher_info"]["cleaned_session_dir"] = True
                _write_summary(summary_json, summary)

    if summary is None:
        return 1
    if args.print_summary:
        print(json.dumps(summary, indent=2, ensure_ascii=True))
    else:
        compact = {
            "episode_id": summary.get("episode_id"),
            "uid": summary.get("uid"),
            "method_name": summary.get("method_name"),
            "method_display_name": summary.get("method_display_name"),
            "execution_mode_compatibility": summary.get("execution_mode_compatibility"),
            "execution_mode": summary.get("execution_mode"),
            "terminal_source": summary.get("terminal_source"),
            "terminal_reason": summary.get("terminal_reason"),
            "algorithm_stop_reason": summary.get("algorithm_stop_reason"),
            "visited_view_num": summary.get("visited_view_num"),
            "accepted_nbv_action_num": summary.get("accepted_nbv_action_num"),
            "summary_json": str(summary_json),
            "algorithm_returncode": None if algorithm_info is None else algorithm_info.get("returncode"),
        }
        print(json.dumps(compact, indent=2, ensure_ascii=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
