"""Test-only benchmark container entrypoint. Synthetic outputs are NEVER official scores."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import sys
import threading
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode-root', type=Path, default=Path('/episode'))
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8081)
    args = parser.parse_args()
    token = os.environ.get('OBJVIEW_EPISODE_TOKEN', '')
    job_id = os.environ.get('OBJVIEW_JOB_ID', '')
    if len(token) < 32 or not token.isascii() or not re.fullmatch('[0-9a-f]{32}', job_id):
        parser.error('Controller token and job ID are required')
    root = args.episode_root.resolve()
    private, released = root / 'private', root / 'released'
    if not private.is_dir() or not released.is_dir() or any(private.iterdir()) or any(released.iterdir()):
        parser.error('Initializer must provide fresh private and released directories')
    # Dedicated smoke process only. Never change the official serving entrypoint.
    sys.modules['api_render'] = None
    import numpy as np
    import uvicorn
    import api_evaluation
    from api_feasibility import CameraPose, FeasibilityAPI
    from runner_episode import EpisodeConfig, EpisodeRunner
    from serve_submission_episode import execute_episode, persist_episode_result
    from smoke_fixtures import Renderer, coverage, chamfer
    from submission_gateway import create_app
    from submission_interaction import SubmissionInteraction
    from submission_protocol import SubmissionSession

    with ExitStack() as stack:
        for name, value in [('CameraPose', CameraPose), ('point_cloud_coverage', coverage),
                            ('chamfer_distance_np', chamfer)]:
            stack.enter_context(patch.object(api_evaluation, name, value))
        renderer = Renderer()
        feasibility = FeasibilityAPI(Path(__file__).parent / 'configs/feasibility/whole.json')
        protocol = SubmissionSession(feasibility.is_feasible, action_timeout_sec=5, max_steps=3)
        adapter = SubmissionInteraction(private, feasibility, protocol=protocol, observation_root=released)
        gt_path = private / 'synthetic-gt.npy'
        np.save(gt_path, np.array([[0., 0., 0.], [0.1, 0., 0.]]))
        evaluator = api_evaluation.EvaluateAPI(renderer.uid, gt_path, renderer,
                                              feasibility_api=feasibility, observable_cache_root=None)
        runner = EpisodeRunner(config=EpisodeConfig(
            'synthetic', renderer.uid, '/synthetic/not-a-real-mesh', gt_path,
            CameraPose((0., 0., 3.), (0., 0., 0.)), max_visited_view_num=3,
            enable_iterative_enough_info_stop=False, action_wait_timeout_sec=6),
            render_api=renderer, evaluator=evaluator, feasibility_api=feasibility,
            interaction=adapter, session_observations_root=released)
        runner.publish_session_metadata()
        server = uvicorn.Server(uvicorn.Config(create_app(protocol, token), host=args.host, port=args.port,
                                               ws='none', access_log=False, proxy_headers=False,
                                               limit_concurrency=32, timeout_graceful_shutdown=1,
                                               log_level='error'))
        cancelled = threading.Event()
        completed = threading.Event()
        outcome = {'status': 'failed'}

        def work():
            try:
                while not server.started:
                    if cancelled.wait(.01):
                        return
                port = server.servers[0].sockets[0].getsockname()[1]
                ready_tmp = private / 'gateway.tmp'
                ready_tmp.write_text(json.dumps({'port': port}), encoding='utf-8')
                ready_tmp.replace(private / 'gateway.ready')
                result = execute_episode(runner, protocol, startup_timeout_sec=10, cancelled=cancelled)
                result.update(mode='smoke_test', official_result=False, leaderboard_eligible=False,
                              algorithm_correctness_checked=False, job_id=job_id)
                persist_episode_result(result, private / 'summary.json')
                outcome.update(result)
            finally:
                completed.set()
                # Let the algorithm observe done before closing the private gateway.
                cancelled.wait(1)
                server.should_exit = True

        def watchdog():
            if not completed.wait(20):
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
            worker.join(timeout=2)
        return 0 if completed.is_set() and outcome['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
