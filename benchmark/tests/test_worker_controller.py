"""Stateful Docker contract double. This is NOT live container/isolation evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from submission_worker.controller import run_smoke
from submission_worker.docker_runtime import LABEL, RuntimeFailure

IMAGE = 'sha256:' + 'a' * 64


class FakeDocker:
    def __init__(self, *, failure=None):
        self.failure = failure
        self.resources = {}
        self.calls = []
        self.token = None
        self.job_id = None
        self.preparation_timeouts = []

    def preflight(self, *, timeout=30):
        self.preparation_timeouts.append(timeout)
        return {'server_version': 'contract-double'}

    def resolve_image(self, reference, role, *, timeout=30):
        self.preparation_timeouts.append(timeout)
        return IMAGE

    def lookup(self, kind, name, **kwargs):
        if name not in self.resources:
            return None
        label = 'foreign' if self.failure == 'foreign' and name.endswith('-data') else self.job_id
        return {'Labels': {LABEL: label}, 'Config': {'Labels': {LABEL: label}}}

    def call(self, args, **kwargs):
        self.calls.append(args)
        if args[:2] in (['network', 'create'], ['volume', 'create']):
            name = args[-1]
            self.job_id = name.split('-')[1]
            self.resources[name] = args[0]
            if self.failure == 'create' and args[0] == 'volume':
                raise RuntimeFailure()  # create may have succeeded before response loss
            return name
        if args[0] == 'create':
            name = args[args.index('--name') + 1]
            self.resources[name] = 'container'
            if '--env-file' in args:
                content = Path(args[args.index('--env-file') + 1]).read_text()
                self.token = content.split('OBJVIEW_EPISODE_TOKEN=')[1].splitlines()[0]
            return name
        if args[0] == 'start':
            return args[-1]
        if args[:2] == ['container', 'inspect']:
            name = args[2]
            kind = name.rsplit('-', 1)[1]
            running = self.failure == 'timeout' and kind == 'init'
            failed = self.failure == kind
            state = {'Running': running, 'ExitCode': 1 if failed else 0,
                     'OOMKilled': self.failure == 'oom' and kind == 'algorithm'}
            if kind == 'benchmark' and not any(n.endswith('-algorithm') for n in self.resources):
                state['Running'] = True
            if self.failure == 'benchmark_early' and any(n.endswith('-algorithm') for n in self.resources):
                state['Running'] = kind == 'algorithm'
                state['ExitCode'] = 1 if kind == 'benchmark' else 0
            return json.dumps([{'State': state}])
        if args[0] == 'exec':
            return 'ready'
        if args[0] == 'cp':
            summary = {'mode': 'smoke_test', 'official_result': False, 'leaderboard_eligible': False,
                       'algorithm_correctness_checked': False, 'status': 'completed', 'job_id': self.job_id,
                       'visited_view_num': 2, 'official_timing': {'timing_source': 'benchmark_monotonic_clock',
                           'total_decision_wall_time_ns': 1, 'steps': [
                               {'step_index': 0, 'decision_wall_time_ns': 1},
                               {'step_index': 1, 'decision_wall_time_ns': 0}]}}
            if self.failure == 'summary':
                summary['leaderboard_eligible'] = True
            if self.failure == 'job_id':
                summary['job_id'] = 'wrong'
            if self.failure == 'timing':
                summary['official_timing']['total_decision_wall_time_ns'] = -1
            if self.failure == 'extra':
                summary['official_timing']['private_details'] = 'SECRET_HIDDEN_PATH'
                summary['official_timing']['steps'][0]['hidden_score'] = 'SECRET_SCORE'
            Path(args[-1]).write_text(json.dumps(summary))
            return ''
        if args[0] == 'rm' or args[:2] in (['network', 'rm'], ['volume', 'rm']):
            if self.failure == 'cleanup' and args[-1].endswith('-data'):
                raise RuntimeFailure()
            self.resources.pop(args[-1], None)
            return ''
        raise AssertionError(f'Unexpected Docker call: {args}')


class ControllerTests(unittest.TestCase):
    def execute(self, failure=None, **kwargs):
        fake = FakeDocker(failure=failure)
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / 'new-job'
            result = run_smoke(fake, IMAGE, IMAGE, directory, trusted_smoke=True,
                               timeout_sec=.05 if failure == 'timeout' else 5, poll_sec=.001, **kwargs)
            self.assertEqual(json.loads((directory / 'report.json').read_text()), result)
            self.assertFalse((directory / 'episode.env').exists())
            self.assertFalse((directory / 'summary.json').exists())
            if fake.token:
                self.assertNotIn(fake.token, json.dumps(result))
            return result, fake

    def test_success_and_cleanup(self):
        result, fake = self.execute()
        self.assertEqual(result['status'], 'SUCCEEDED')
        self.assertIs(result['leaderboard_eligible'], False)
        self.assertEqual(result['states'], ['PREPARING', 'STARTING', 'RUNNING', 'SUCCEEDED'])
        self.assertEqual(fake.resources, {})
        self.assertTrue(result['cleanup']['complete'])
        self.assertGreater(result['timing']['total_job_wall_time_ns'], 0)

    def test_failure_classification_and_cleanup(self):
        for failure, expected in [('init', 'INFRASTRUCTURE_FAILED'), ('create', 'INFRASTRUCTURE_FAILED'),
                                  ('algorithm', 'ALGORITHM_CRASHED'), ('benchmark', 'BENCHMARK_FAILED'),
                                  ('oom', 'ALGORITHM_OUT_OF_MEMORY'), ('timeout', 'JOB_TIMEOUT'),
                                  ('summary', 'INVALID_SUMMARY'), ('job_id', 'INVALID_SUMMARY'),
                                  ('timing', 'INVALID_SUMMARY')]:
            with self.subTest(failure=failure):
                result, fake = self.execute(failure)
                self.assertEqual(result['status'], 'FAILED')
                self.assertEqual(result['failure_code'], expected)
                self.assertEqual(fake.resources, {})
                starts = [tuple(x) for x in fake.calls if x[0] == 'start']
                self.assertEqual(len(starts), len(set(starts)), 'No automatic retry')

    def test_cleanup_failure_or_foreign_label_never_reports_success(self):
        for failure in ('cleanup', 'foreign'):
            with self.subTest(failure=failure):
                result, fake = self.execute(failure)
                self.assertEqual(result['status'], 'FAILED')
                self.assertEqual(result['failure_code'], 'CLEANUP_FAILED')
                self.assertFalse(result['cleanup']['complete'])
                self.assertEqual(len(fake.resources), 1)
                self.assertEqual(len(result['cleanup']['unconfirmed_resources']), 1)

    def test_requires_explicit_trusted_smoke_and_fresh_directory(self):
        fake = FakeDocker()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                run_smoke(fake, IMAGE, IMAGE, Path(root) / 'new')
            with self.assertRaises(FileExistsError):
                run_smoke(fake, IMAGE, IMAGE, Path(root), trusted_smoke=True)
        self.assertEqual(fake.calls, [])

    def test_benchmark_crash_is_detected_while_algorithm_waits(self):
        result, fake = self.execute('benchmark_early')
        self.assertEqual(result['failure_code'], 'BENCHMARK_FAILED')
        self.assertEqual(fake.resources, {})

    def test_report_only_copies_allowlisted_timing_fields(self):
        result, _ = self.execute('extra')
        self.assertEqual(result['status'], 'SUCCEEDED')
        self.assertNotIn('SECRET', json.dumps(result))

    def test_preparation_calls_share_total_job_deadline(self):
        _, fake = self.execute()
        self.assertEqual(len(fake.preparation_timeouts), 3)
        self.assertTrue(all(0 < timeout <= 5 for timeout in fake.preparation_timeouts))


if __name__ == '__main__':
    unittest.main()
