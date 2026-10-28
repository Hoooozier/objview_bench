import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smoke_submission import run_checks


class SmokeReportTests(unittest.TestCase):
    def run_case(self, body):
        with tempfile.TemporaryDirectory() as root:
            report = Path(root) / 'report.json'
            suite = unittest.TestSuite([unittest.FunctionTestCase(body)])
            code = run_checks(suite, report, stream=io.StringIO())
            return code, json.loads(report.read_text())

    def test_success_report_is_never_an_official_score(self):
        code, report = self.run_case(lambda: None)
        self.assertEqual(code, 0)
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['mode'], 'smoke_test')
        for key in ('official_result', 'leaderboard_eligible', 'algorithm_correctness_checked'):
            self.assertIs(report[key], False)
        self.assertEqual(report['tests_run'], 1)
        self.assertEqual(len(report['executed_test_ids']), 1)
        self.assertNotIn('metrics', report)

    def test_failure_does_not_report_success_or_echo_exception(self):
        def fail():
            raise RuntimeError('SECRET_TOKEN_AND_PATH')
        code, report = self.run_case(fail)
        self.assertEqual(code, 1)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['errors'], 1)
        self.assertNotIn('SECRET', json.dumps(report))

    def test_skipped_required_case_is_not_success(self):
        def skip():
            raise unittest.SkipTest('Missing component')
        code, report = self.run_case(skip)
        self.assertEqual(code, 1)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['skipped'], 1)

    def test_empty_suite_is_not_success(self):
        with tempfile.TemporaryDirectory() as root:
            report = Path(root) / 'report.json'
            self.assertEqual(run_checks(unittest.TestSuite(), report, stream=io.StringIO()), 1)
            self.assertEqual(json.loads(report.read_text())['status'], 'failed')

    def test_interrupted_run_cannot_leave_an_old_passed_status(self):
        def interrupt():
            raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as root:
            report = Path(root) / 'report.json'
            report.write_text('{"status":"passed"}')
            with self.assertRaises(KeyboardInterrupt):
                run_checks(unittest.TestSuite([unittest.FunctionTestCase(interrupt)]), report,
                           stream=io.StringIO())
            self.assertEqual(json.loads(report.read_text())['status'], 'running')

    def test_cli_refuses_to_overwrite_existing_report(self):
        with tempfile.TemporaryDirectory() as root:
            report = Path(root) / 'report.json'
            report.write_text('existing evidence')
            cli = Path(__file__).resolve().parents[1] / 'smoke_submission.py'
            result = subprocess.run([sys.executable, '-B', str(cli), '--report', str(report)],
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(report.read_text(), 'existing evidence')


if __name__ == '__main__':
    unittest.main()
