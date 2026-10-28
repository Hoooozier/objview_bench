import concurrent.futures
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submission_protocol import ProtocolError, SubmissionSession


class Clock:
    now = 0

    def __call__(self):
        return self.now


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.session = SubmissionSession(lambda pose: pose[2] >= 0, clock_ns=self.clock)
        self.session.configure({'uid': 'SECRET', 'episode_id': 'SECRET__whole',
                                'extra': {'password': 'SECRET'},
                                'camera_intrinsics': {'image_width': 2, 'secret': 'SECRET'},
                                'viewspace_constraint': {'name': 'whole'}})
        self.session.ready()
        self.session.publish(0, [0, 0, 3, 0, 0, 0, 0], {'rgb_path': 'step_000/rgb.png'})

    def action(self, **changes):
        value = dict(episode_id=self.session.episode_id, step_index=0, request_id='r1',
                     action='move', pose=[0, 0, 3, 0, 0, 0, 0])
        value.update(changes)
        return value

    def error(self, code, callback):
        with self.assertRaises(ProtocolError) as caught:
            callback()
        self.assertEqual(caught.exception.code, code)

    def test_time_is_server_measured_and_retry_not_recounted(self):
        self.clock.now = 250_000_000
        receipt = self.session.submit(self.action())
        self.clock.now = 999_000_000
        self.assertEqual(self.session.submit(self.action()), receipt)
        self.assertEqual(self.session.take_action()['algorithm_runtime_sec'], 0.25)
        self.assertIsNone(self.session.take_action())
        self.assertEqual(self.session.timing_summary()['total_decision_wall_time_ns'], 250_000_000)

    def test_spoofed_time_rejected(self):
        self.error('INVALID_ACTION', lambda: self.session.submit(self.action(algorithm_runtime_sec=0)))

    def test_huge_json_integer_is_invalid_action_not_internal_failure(self):
        self.error('INVALID_ACTION', lambda: self.session.submit(
            self.action(pose=[0, 0, 10 ** 400, 0, 0, 0, 0])))
        self.assertEqual(self.session.status()['status'], 'running')

    def test_timing_starts_before_files_become_visible(self):
        self.session.submit(self.action())
        self.session.take_action()
        def release():
            self.clock.now += 250_000_000
        self.session.publish(1, [0, 0, 3, 0, 0, 0, 0], {}, release_observation=release)
        self.session.submit(self.action(step_index=1, request_id='r2'))
        self.assertEqual(self.session.take_action()['algorithm_runtime_sec'], .25)

    def test_anonymous_allowlist_and_copy(self):
        result = self.session.episode()
        self.assertNotIn('SECRET', json.dumps(result))
        result['camera_intrinsics']['image_width'] = 999
        self.assertEqual(self.session.episode()['camera_intrinsics']['image_width'], 2)
        self.assertNotEqual(self.session.episode_id, SubmissionSession(lambda p: True).episode_id)

    def test_finite_pose_and_strict_integer(self):
        for value in [float('nan'), float('inf'), True, '3', 1e300]:
            pose = [0, 0, value, 0, 0, 0, 0]
            self.error('INVALID_ACTION', lambda: self.session.submit(self.action(pose=pose)))
        self.error('INVALID_ACTION', lambda: self.session.submit(self.action(step_index=True)))
        self.error('INVALID_ACTION', lambda: self.session.submit(self.action(pose=[0]*7)))

    def test_wrong_episode_step_and_conflicting_retry(self):
        self.error('WRONG_EPISODE', lambda: self.session.submit(self.action(episode_id='other')))
        self.error('WRONG_STEP', lambda: self.session.submit(self.action(step_index=1)))
        self.session.submit(self.action())
        self.error('REQUEST_CONFLICT', lambda: self.session.submit(self.action(pose=[0, 0, 4, 0, 0, 0, 0])))
        self.error('ACTION_ALREADY_SUBMITTED', lambda: self.session.submit(self.action(request_id='r2')))

    def test_deadline_includes_late_action_even_without_runner_poll(self):
        self.clock.now = 120_000_000_000
        self.error('ACTION_TIMEOUT', lambda: self.session.submit(self.action()))
        self.assertEqual(self.session.status()['status'], 'done')
        self.assertEqual(self.session.timing_summary()['total_decision_wall_time_ns'], 120_000_000_000)

    def test_feasibility_quota_and_invalid_pose(self):
        self.error('INVALID_PARAMS', lambda: self.session.check_feasibility([[0]*7]))
        poses = [[0, 0, 3, 0, 0, 0, 0]] * 512
        self.error('QUERY_LIMIT', lambda: self.session.check_feasibility(poses + poses))
        for _ in range(8):
            self.assertTrue(all(self.session.check_feasibility(poses)['feasible']))
        self.error('QUERY_LIMIT', lambda: self.session.check_feasibility(poses))

    def test_infeasible_action_is_terminal_and_counted(self):
        self.clock.now = 100_000_000
        self.error('INFEASIBLE_ACTION', lambda: self.session.submit(self.action(pose=[0, 0, -3, 0, 0, 0, 0])))
        self.assertEqual(self.session.status()['status'], 'done')
        self.assertEqual(self.session.timing_summary()['total_decision_wall_time_ns'], 100_000_000)

    def test_publish_requires_consumption_and_strict_sequence(self):
        self.error('INVALID_STATE', lambda: self.session.publish(1, [0, 0, 3, 0, 0, 0, 0], {}))
        self.session.submit(self.action())
        self.session.take_action()
        self.session.publish(1, [0, 0, 3, 0, 0, 0, 0], {})
        self.error('WRONG_STEP', lambda: self.session.submit(self.action(request_id='late')))
        self.assertEqual(self.session.next_step(0)['step_index'], 1)

    def test_concurrent_requests_accept_exactly_one(self):
        def send(index):
            try:
                return self.session.submit(self.action(request_id=f'r{index}'))['accepted']
            except ProtocolError:
                return False
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            self.assertEqual(sum(pool.map(send, range(8))), 1)

    def test_done_has_no_internal_details(self):
        self.session.finish('exception: /SECRET/uid')
        self.assertEqual(self.session.status(), {'status': 'done', 'code': 'EPISODE_FAILED'})
        self.error('EPISODE_DONE', lambda: self.session.submit(self.action()))

    def test_observation_paths_cannot_escape(self):
        session = SubmissionSession(lambda p: True)
        session.configure({})
        session.ready()
        for path in ['/SECRET/a.png', '../SECRET', 'step_000/../../secret']:
            self.error('INVALID_OBSERVATION', lambda: session.publish(0, [0, 0, 3, 0, 0, 0, 0], {'rgb_path': path}))


if __name__ == '__main__':
    unittest.main()
