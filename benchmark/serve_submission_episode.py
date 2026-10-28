"""One trusted benchmark episode. Does NOT build or execute submitted code."""
import argparse
import json
import math
import os
from pathlib import Path
import threading
import time


def validate_paths(private_root, observation_root, summary_path):
    private, public, summary = map(lambda p: Path(p).resolve(),
                                   (private_root, observation_root, summary_path))
    if private == public or private in public.parents or public in private.parents:
        raise ValueError('Private and released directories must be separate')
    if private not in summary.parents:
        raise ValueError('Summary must remain inside the private session directory')


def execute_episode(runner, protocol, *, startup_timeout_sec, cancelled=None):
    cancelled = cancelled or threading.Event()
    deadline = time.monotonic() + startup_timeout_sec
    try:
        while not protocol.is_ready():
            if cancelled.is_set():
                protocol.finish('EPISODE_FAILED')
                break
            if time.monotonic() >= deadline:
                protocol.finish('STARTUP_TIMEOUT')
                break
            cancelled.wait(.01)
        if protocol.status()['status'] == 'done':
            return {'status': 'failed', 'failure_code': protocol.status()['code'],
                    'official_timing': protocol.timing_summary()}
        result = runner.run()
        protocol.finish('COMPLETED')
        code = protocol.status()['code']
        return {**result, 'status': 'completed' if code == 'COMPLETED' else 'failed',
                **({'failure_code': code} if code != 'COMPLETED' else {}),
                'official_timing': protocol.timing_summary()}
    except Exception:
        # This summary stays private; failed episodes must never look like valid scores.
        protocol.finish('EPISODE_FAILED')
        return {'status': 'failed', 'failure_code': protocol.status()['code'],
                'official_timing': protocol.timing_summary()}


def positive_seconds(value):
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= 86400:
        raise argparse.ArgumentTypeError('Expected finite duration in (0, 86400]')
    return value


def persist_episode_result(result, summary_path):
    """Persist trusted output; callers must keep this path outside released observations."""
    Path(summary_path).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('uid',):
        parser.add_argument('--' + name, required=True)
    for name in ('obj-path', 'gt-pointcloud', 'cache-index-json', 'feasibility-json',
                 'private-dir', 'observation-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--start-pose', type=float, nargs=7)
    parser.add_argument('--start-view-id', type=int, default=0)
    parser.add_argument('--start-view-set', default='benchmark_start_3')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--max-visited-view-num', type=int, default=129)
    parser.add_argument('--host', default='127.0.0.1', help='Private worker network only; never publish to Internet')
    parser.add_argument('--port', type=int, default=8081)
    parser.add_argument('--action-timeout-sec', type=positive_seconds, default=120.)
    parser.add_argument('--startup-timeout-sec', type=positive_seconds, default=300.)
    parser.add_argument('--episode-timeout-sec', type=positive_seconds, default=3600.)
    parser.add_argument('--terminal-grace-sec', type=positive_seconds, default=2.)
    return parser


def create_runner(args):
    # Lazy imports keep --help usable without rendering/GPU dependencies.
    from api_render import Render, intrinsics_from_dict
    from api_evaluation import EvaluateAPI
    from api_feasibility import FeasibilityAPI
    from runner_episode import EpisodeConfig, EpisodeRunner, _load_cache_index, _resolve_start_pose_with_auto_fallback
    from submission_interaction import SubmissionInteraction
    from submission_protocol import SubmissionSession

    cache = _load_cache_index(args.cache_index_json)
    feasibility = FeasibilityAPI(args.feasibility_json)
    protocol = SubmissionSession(feasibility.is_feasible, action_timeout_sec=args.action_timeout_sec,
                                 max_steps=args.max_visited_view_num)
    interaction = SubmissionInteraction(args.private_dir, feasibility, protocol=protocol,
                                         observation_root=args.observation_root)
    renderer = Render(uid=args.uid, obj_path=args.obj_path, intrinsics=intrinsics_from_dict(cache['intrinsics']),
                      device=args.device, mode='auto', cache_index_json=args.cache_index_json)
    evaluator = EvaluateAPI(uid=args.uid, gt_pointcloud_path=args.gt_pointcloud,
                            render_api=renderer, feasibility_api=feasibility)
    config = EpisodeConfig(
        episode_id=protocol.episode_id, uid=args.uid, obj_path=args.obj_path,
        gt_pointcloud_path=args.gt_pointcloud,
        start_pose=_resolve_start_pose_with_auto_fallback(
            cache_index=cache, uid=args.uid, start_view_set=args.start_view_set,
            start_view_id=args.start_view_id, explicit_start_pose=args.start_pose),
        start_view_set=args.start_view_set, start_view_id=args.start_view_id,
        viewspace_constraint_name=args.feasibility_json.stem, method_name='submission',
        # No author-controlled family/oracle switch. Uniform cap for v1 submissions.
        family='submission', execution_mode='http', enable_iterative_enough_info_stop=False,
        max_visited_view_num=args.max_visited_view_num,
        action_wait_timeout_sec=args.action_timeout_sec + 1)
    runner = EpisodeRunner(config=config, render_api=renderer, feasibility_api=feasibility,
                           evaluator=evaluator, interaction=interaction,
                           session_observations_root=args.observation_root)
    runner.publish_session_metadata()
    return runner, protocol


def main():
    parser = build_parser()
    args = parser.parse_args()
    token = os.environ.get('OBJVIEW_EPISODE_TOKEN', '')
    if len(token) < 32 or not token.isascii():
        parser.error('OBJVIEW_EPISODE_TOKEN must contain at least 32 random ASCII characters')
    if not 1 <= args.max_visited_view_num <= 1024 or args.action_timeout_sec > 3600:
        parser.error('Invalid view cap or action timeout')
    summary_path = args.private_dir / 'summary.json'
    validate_paths(args.private_dir, args.observation_root, summary_path)
    if args.private_dir.exists() and any(args.private_dir.iterdir()):
        parser.error('Use a fresh private directory for each episode')
    runner, protocol = create_runner(args)
    import uvicorn
    from submission_gateway import create_app
    server = uvicorn.Server(uvicorn.Config(create_app(protocol, token, observation_root=args.observation_root), host=args.host, port=args.port,
                                           workers=1, access_log=False, proxy_headers=False,
                                           ws='none',
                                           limit_concurrency=32, timeout_keep_alive=2,
                                           timeout_graceful_shutdown=2, log_level='warning'))
    cancelled = threading.Event()
    completed = threading.Event()
    outcome = {'status': 'failed'}

    def work():
        try:
            while not server.started:
                if cancelled.wait(.01):
                    return
            result = execute_episode(runner, protocol, startup_timeout_sec=args.startup_timeout_sec,
                                     cancelled=cancelled)
            persist_episode_result(result, summary_path)
            outcome.update(result)
        finally:
            completed.set()
            cancelled.wait(args.terminal_grace_sec)
            server.should_exit = True

    def watchdog():
        if not completed.wait(args.episode_timeout_sec):
            protocol.finish('EPISODE_FAILED')
            cancelled.set()
            server.should_exit = True

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    threading.Thread(target=watchdog, daemon=True).start()
    try:
        server.run()
    finally:
        cancelled.set()
        worker.join(timeout=1)
    # Hard process/container deadlines remain the controller's responsibility,
    # especially for native GPU calls that cannot be interrupted by Python threads.
    return 0 if completed.is_set() and outcome['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
