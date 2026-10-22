import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve_submission_episode import execute_episode, validate_paths
from submission_protocol import SubmissionSession


class ServiceTests(unittest.TestCase):
    def test_startup_timeout_does_not_run_renderer(self):
        protocol = SubmissionSession(lambda pose: True)
        runner = Mock()
        result = execute_episode(runner, protocol, startup_timeout_sec=.01)
        runner.run.assert_not_called()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(protocol.status()['code'], 'STARTUP_TIMEOUT')

    def test_runner_exception_is_not_exposed_to_algorithm(self):
        protocol = SubmissionSession(lambda pose: True)
        protocol.ready()
        runner = Mock()
        runner.run.side_effect = RuntimeError('/SECRET/path')
        result = execute_episode(runner, protocol, startup_timeout_sec=1)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(protocol.status(), {'status': 'done', 'code': 'EPISODE_FAILED'})
        self.assertNotIn('SECRET', str(result))

    def test_success_is_marked_and_has_authoritative_timing(self):
        protocol = SubmissionSession(lambda pose: True)
        protocol.ready()
        runner = Mock()
        runner.run.return_value = {'visited_view_num': 1}
        result = execute_episode(runner, protocol, startup_timeout_sec=1)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['official_timing']['timing_source'], 'benchmark_monotonic_clock')

    def test_summary_must_be_private(self):
        root = Path('/tmp/submission-path-validation')
        for summary in (root / 'released' / 'result.json', root / 'result.json'):
            with self.assertRaises(ValueError):
                validate_paths(root / 'private', root / 'released', summary)
        validate_paths(root / 'private', root / 'released', root / 'private' / 'result.json')

    def test_invalid_episode_is_not_a_successful_score(self):
        protocol = SubmissionSession(lambda pose: True)
        protocol.ready()
        runner = Mock()
        def invalid_run():
            protocol.finish('INVALID_ACTION')
            return {'terminal_reason': 'invalid_action'}
        runner.run.side_effect = invalid_run
        result = execute_episode(runner, protocol, startup_timeout_sec=1)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['failure_code'], 'INVALID_ACTION')


if __name__ == '__main__':
    unittest.main()
