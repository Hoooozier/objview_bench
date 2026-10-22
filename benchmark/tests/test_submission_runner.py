"""Real runner/evaluator, tiny synthetic images, NumPy substitutes for GPU kernels."""
import concurrent.futures
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import api_evaluation
from api_feasibility import CameraPose, FeasibilityAPI
from runner_episode import EpisodeConfig, EpisodeRunner, EpisodeRunnerError
from submission_client import SubmissionClient
from submission_interaction import SubmissionInteraction
from submission_protocol import ProtocolError, SubmissionSession
from test_submission_protocol import Clock


class Renderer:
    uid = 'SECRET_UID'
    source = 'cache'
    intrinsics = SimpleNamespace(image_width=2, image_height=2, fov_x_rad=1.,
                                 fov_y_rad=1., principal_x=1., principal_y=1.)
    _cache_index = {'objects': {uid: {'views': [{
        'view_set': 'tammes_360', 'rgb': '/SECRET_CACHE/image.png',
        'pose': {'camera_xyz': [0, 0, 3], 'lookat_xyz': [0, 0, 0], 'roll_rad': 0}}]}}}

    def render_loaded(self, pose):
        return SimpleNamespace(source=self.source, pose=pose, intrinsics=self.intrinsics,
                               rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                               depth=np.ones((2, 2), dtype=np.float32),
                               mask=np.ones((2, 2), dtype=bool),
                               points_world=np.array([[0., 0., 0.], [0.1, 0., 0.]]))


def coverage(gt, observed, *, threshold):
    distances = np.linalg.norm(gt[:, None] - observed[None, :], axis=2)
    return float((distances.min(axis=1) <= threshold).mean())


def chamfer(a, b, **kwargs):
    distances = ((a[:, None] - b[None, :]) ** 2).sum(axis=2)
    return float(distances.min(axis=0).mean() + distances.min(axis=1).mean())


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.clock = Clock()
        self.feasibility = FeasibilityAPI(Path(__file__).resolve().parents[1] / 'configs/feasibility/whole.json')
        self.protocol = SubmissionSession(self.feasibility.is_feasible, clock_ns=self.clock, max_steps=3)
        self.adapter = SubmissionInteraction(self.root / 'private', self.feasibility,
                                             protocol=self.protocol, observation_root=self.root / 'released')
        # The dependency-light modules have separate fallback pose dataclasses;
        # use the same one as production api_render would supply to both modules.
        for name, value in [('CameraPose', CameraPose), ('point_cloud_coverage', coverage),
                            ('chamfer_distance_np', chamfer)]:
            mocked = patch.object(api_evaluation, name, value)
            mocked.start()
            self.addCleanup(mocked.stop)
        np.save(self.root / 'gt.npy', np.array([[0., 0., 0.], [0.1, 0., 0.]]))
        self.renderer = Renderer()
        self.evaluator = api_evaluation.EvaluateAPI(
            'SECRET_UID', self.root / 'gt.npy', self.renderer,
            feasibility_api=self.feasibility, observable_cache_root=None)
        self.runner = EpisodeRunner(
            config=EpisodeConfig('SECRET_EPISODE', 'SECRET_UID', '/SECRET_OBJ', self.root / 'gt.npy',
                                 CameraPose((0., 0., 3.), (0., 0., 0.)), max_visited_view_num=3,
                                 enable_iterative_enough_info_stop=False, action_wait_timeout_sec=130),
            render_api=self.renderer, feasibility_api=self.feasibility, evaluator=self.evaluator,
            interaction=self.adapter, session_observations_root=self.root / 'released')
        self.runner.publish_session_metadata()
        self.protocol.ready()

    def wait_step(self, after=-1):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            step = self.protocol.next_step(after)
            if step:
                return step
            time.sleep(.005)
        self.fail('Runner did not publish observation')

    def test_move_stop_real_evaluator_uses_server_time_and_releases_cache_copy(self):
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(self.runner.run)
            first = self.wait_step()
            self.clock.now = 250_000_000
            action = SubmissionClient.make_action(first, 'move', pose=[3, 0, 0, 0, 0, 0, 0])
            self.protocol.submit(action)
            self.protocol.submit(action)
            second = self.wait_step(0)
            self.clock.now += 750_000_000
            self.protocol.submit(SubmissionClient.make_action(second, 'stop', stop_reason='plan_end'))
            result = future.result(timeout=3)
        self.assertEqual(result['evaluator_summary']['total_algorithm_runtime_sec'], 1.)
        self.assertEqual(result['official_timing']['total_decision_wall_time_ns'], 1_000_000_000)
        self.assertEqual(result['visited_view_num'], 2)
        self.assertEqual(result['terminal_reason'], 'algorithm_stop')
        self.assertEqual(self.protocol.status(), {'status': 'done', 'code': 'COMPLETED'})
        public_json = json.dumps([self.protocol.episode(), first, second])
        for path in (self.root / 'released').rglob('*.json'):
            public_json += path.read_text()
        self.assertNotIn('SECRET', public_json)
        for step in (first, second):
            for relative in step['observation'].values():
                self.assertTrue((self.root / 'released' / relative).is_file())
            with np.load(self.root / 'released' / step['observation']['depth_path']) as depth:
                np.testing.assert_array_equal(depth['depth'], np.ones((2, 2)))

    def test_private_file_actions_and_rpc_are_ignored(self):
        self.runner._publish_initial_step()
        self.adapter.submitted_action_path.write_text('{"action":"stop"}')
        self.adapter.ready_algorithm_path.touch()
        (self.adapter.requests_dir / 'oracle.json').write_text('{"method":"oracle"}')
        self.assertIsNone(self.adapter.read_submitted_action(expected_step_index=0,
                                                           expected_episode_id='SECRET_EPISODE'))
        self.assertEqual(self.adapter.process_pending_requests(), [])

    def test_view_cap_terminates_without_publishing_an_extra_decision(self):
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(self.runner.run)
            for index in range(2):
                step = self.wait_step(index - 1)
                self.clock.now += 250_000_000
                self.protocol.submit(SubmissionClient.make_action(step, 'move',
                                                                 pose=[3, index, 0, 0, 0, 0, 0]))
            result = future.result(timeout=3)
        self.assertEqual(result['visited_view_num'], 3)
        self.assertEqual(result['terminal_reason'], 'max_cap_reached')
        self.assertEqual(result['evaluator_summary']['total_algorithm_runtime_sec'], .5)
        self.assertEqual(len(result['official_timing']['steps']), 2)
        self.assertEqual(self.protocol.next_step(1), {'status': 'done', 'code': 'COMPLETED'})

    def test_cap_one_finishes_after_initial_observation_without_requesting_action(self):
        self.runner.config.max_visited_view_num = 1
        self.protocol.max_steps = 1
        with patch.object(self.runner, '_wait_for_action', side_effect=AssertionError('No action budget')), \
                patch.object(self.renderer, 'render_loaded', wraps=self.renderer.render_loaded) as render:
            result = self.runner.run()
        self.assertEqual(render.call_count, 1)
        self.assertEqual(result['visited_view_num'], 1)
        self.assertEqual(result['terminal_reason'], 'max_cap_reached')
        self.assertEqual(result['official_timing']['steps'], [])

    def test_sdk_over_real_tcp_to_runner(self):
        import uvicorn
        from submission_gateway import create_app
        token = 'local-test-token-' * 4
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(16)
        self.addCleanup(listener.close)
        server = uvicorn.Server(uvicorn.Config(create_app(self.protocol, token),
                                               access_log=False, log_level='error',
                                               ws='none',
                                               timeout_graceful_shutdown=1, proxy_headers=False))
        thread = threading.Thread(target=server.run, kwargs={'sockets': [listener]}, daemon=True)
        thread.start()
        client = SubmissionClient(f'http://127.0.0.1:{listener.getsockname()[1]}', token)
        try:
            deadline = time.monotonic() + 3
            while not server.started and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertTrue(server.started)
            client.ready()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(self.runner.run)
                step = client.next_step(wait=1)
                self.assertNotIn('SECRET', json.dumps([client.episode(), step]))
                self.clock.now = 250_000_000
                client.submit(client.make_action(step, 'stop', stop_reason='plan_end'))
                result = future.result(timeout=3)
                self.assertEqual(result['evaluator_summary']['total_algorithm_runtime_sec'], .25)
                self.assertEqual(client.next_step(0, wait=0)['status'], 'done')
        finally:
            client.close()
            self.protocol.finish('EPISODE_FAILED')
            server.should_exit = True
            thread.join(timeout=3)
        self.assertFalse(thread.is_alive())

    def test_dummy_process_http_move_stop_scoring_persistence_and_cleanup(self):
        """Smoke acceptance: assert pipeline behavior, never scientific metric values."""
        import uvicorn
        from serve_submission_episode import execute_episode, persist_episode_result
        from submission_gateway import create_app

        # Use the actual monotonic clock and require readiness from the child.
        protocol = SubmissionSession(self.feasibility.is_feasible, action_timeout_sec=5, max_steps=3)
        protocol.configure(self.runner._episode_config_dict())
        self.adapter.protocol = protocol
        token = secrets.token_hex(32)
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(16)
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(create_app(protocol, token), ws='none',
                                               access_log=False, log_level='error',
                                               timeout_graceful_shutdown=1, proxy_headers=False))
        server_thread = threading.Thread(target=server.run, kwargs={'sockets': [listener]}, daemon=True)
        server_thread.start()
        outcome = {}
        cancelled = threading.Event()

        def run_episode():
            outcome.update(execute_episode(self.runner, protocol, startup_timeout_sec=5, cancelled=cancelled))

        worker = threading.Thread(target=run_episode, daemon=True)
        worker.start()
        benchmark_root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(benchmark_root), PYTHONDONTWRITEBYTECODE='1',
                   BENCHMARK_URL=f'http://127.0.0.1:{port}', OBJVIEW_EPISODE_TOKEN=token,
                   OBSERVATION_ROOT=str(self.root / 'released'))
        try:
            deadline = time.monotonic() + 3
            while not server.started and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertTrue(server.started, 'Gateway failed to start')
            # Only our repository's trusted example is executed; no uploaded code.
            child = subprocess.run([sys.executable, '-B', str(benchmark_root / 'examples/submission/algorithm.py')],
                                   env=env, cwd=self.root, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=10, check=False)
            self.assertEqual(child.returncode, 0, 'Dummy process failed')
            worker.join(timeout=3)
            self.assertFalse(worker.is_alive(), 'Runner did not terminate')
            self.assertEqual(outcome['status'], 'completed')
            self.assertEqual(outcome['visited_view_num'], 2)
            self.assertEqual([x['action'] for x in outcome['algorithm_action_history']], ['move', 'stop'])
            self.assertIn('terminal_checkpoint', outcome)
            self.assertIn('evaluator_summary', outcome)
            timing = outcome['official_timing']
            self.assertEqual(timing['timing_source'], 'benchmark_monotonic_clock')
            self.assertEqual(len(timing['steps']), 2)
            self.assertGreater(timing['total_decision_wall_time_ns'], 0)
            # Keep synthetic summaries unmistakably separate from official scores.
            outcome.update(mode='smoke_test', official_result=False, leaderboard_eligible=False,
                           algorithm_correctness_checked=False)
            summary_path = self.root / 'private' / 'smoke-summary.json'
            persist_episode_result(outcome, summary_path)
            persisted = json.loads(summary_path.read_text())
            self.assertEqual(persisted['status'], 'completed')
            self.assertIs(persisted['leaderboard_eligible'], False)
            for index in (0, 1):
                for name in ('rgb.png', 'depth.npz', 'mask.png', 'frame_meta.json'):
                    self.assertTrue((self.root / 'released' / f'step_{index:03d}' / name).is_file())
        finally:
            cancelled.set()
            protocol.finish('EPISODE_FAILED')
            server.should_exit = True
            worker.join(timeout=3)
            server_thread.join(timeout=3)
            listener.close()
        self.assertFalse(worker.is_alive())
        self.assertFalse(server_thread.is_alive())
        self.tmp.cleanup()
        self.assertFalse(self.root.exists(), 'Episode files not cleaned')

    def test_infeasible_action_finalizes_with_measured_time(self):
        self.runner._publish_initial_step()
        self.clock.now = 250_000_000
        with self.assertRaises(ProtocolError):
            self.protocol.submit(SubmissionClient.make_action(self.wait_step(), 'move',
                                                             pose=[0, 0, .5, 0, 0, 0, 0]))
        action = self.adapter.read_submitted_action(expected_step_index=0, expected_episode_id='SECRET_EPISODE')
        self.assertEqual(action['action'], 'invalid')
        self.assertEqual(action['algorithm_runtime_sec'], .25)

    def test_timeout_expires_without_an_algorithm_request(self):
        self.runner._publish_initial_step()
        self.clock.now = 120_000_000_000
        with self.assertRaises(EpisodeRunnerError):
            self.runner._wait_for_action()
        self.assertEqual(self.protocol.status()['code'], 'ACTION_TIMEOUT')
        self.assertEqual(self.protocol.timing_summary()['total_decision_wall_time_ns'], self.clock.now)

    def test_no_private_or_reused_directory_can_be_released(self):
        for root in (self.root / 'private', self.root, self.root / 'private' / 'nested'):
            with self.assertRaises(ValueError):
                SubmissionInteraction(self.root / 'private', self.feasibility,
                                      protocol=self.protocol, observation_root=root)
        (self.root / 'released' / 'leftover.txt').touch()
        with self.assertRaises(ValueError):
            SubmissionInteraction(self.root / 'private2', self.feasibility,
                                  protocol=self.protocol, observation_root=self.root / 'released')

    def test_online_and_cached_observations_have_same_public_shape(self):
        pose = self.runner.config.start_pose
        cached = self.adapter.materialize_frame(step_index=0, pose=pose, frame=self.renderer.render_loaded(pose))
        cached_root = self.adapter.session_dir / 'staging'
        self.assertEqual(list((self.root / 'released').iterdir()), [])
        self.renderer.source = 'online'
        online = self.adapter.materialize_frame(step_index=1, pose=pose, frame=self.renderer.render_loaded(pose))
        self.assertEqual(set(cached), set(online))
        for key in cached:
            self.assertEqual((cached_root / cached[key]).read_bytes(), (cached_root / online[key]).read_bytes())


if __name__ == '__main__':
    unittest.main()
