"""CPU-only development acceptance. Never produces an official benchmark score."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest


class SmokeResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executed_test_ids = []

    def startTest(self, test):
        self.executed_test_ids.append(test.id())
        super().startTest(test)


def run_checks(suite, report_path, *, stream=None):
    """Run real checks and persist only acceptance metadata, never synthetic scores."""
    report_path = Path(report_path)
    # Invalidate an old result before any test can hang or fail.
    report = {
        'schema_version': 'objview-smoke-report/v1', 'mode': 'smoke_test',
        'official_result': False, 'leaderboard_eligible': False,
        'algorithm_correctness_checked': False, 'status': 'running',
        'started_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'implemented_benchmark_chain',
        'test_doubles': ['synthetic_rgbd', 'numpy_distance_kernels', 'fallback_camera_pose',
                         'docker_command_boundary'],
        'not_verified': ['algorithm_quality', 'scientific_metric_correctness', 'gpu_execution',
                         'docker_build_and_isolation', 'website_upload_and_oauth',
                         'controller_queue', 'leaderboard_publication'],
    }
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    started = time.monotonic()
    result = unittest.TextTestRunner(stream=stream or sys.stderr, verbosity=2,
                                     resultclass=SmokeResult).run(suite)
    passed = (result.testsRun > 0 and result.wasSuccessful() and not result.skipped
              and not result.expectedFailures)
    report.update(status='passed' if passed else 'failed', tests_run=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors),
                  skipped=len(result.skipped), expected_failures=len(result.expectedFailures),
                  unexpected_successes=len(result.unexpectedSuccesses),
                  executed_test_ids=result.executed_test_ids,
                  elapsed_sec=time.monotonic() - started)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, help='New report path; existing files are never overwritten')
    args = parser.parse_args()
    report_path = args.report
    if report_path is None:
        report_path = Path(tempfile.mkdtemp(prefix='objview-smoke-report-')) / 'report.json'
    # Reserve the output before running; prevents accidentally clobbering an old report.
    try:
        with report_path.open('x', encoding='utf-8'):
            pass
    except OSError:
        parser.error('Report must be a new writable file in an existing directory')
    print(f'SMOKE TEST ONLY — not eligible for official leaderboard. Report: {report_path}', flush=True)
    # Test-process-only dependency override. Existing modules already support this
    # fallback. Do not import Torch/PyTorch3D even if partially installed locally.
    sys.modules['api_render'] = None
    tests = Path(__file__).resolve().parent / 'tests'
    sys.path.insert(0, str(tests))
    suite = unittest.defaultTestLoader.discover(str(tests), pattern='test_submission_*.py')
    suite.addTests(unittest.defaultTestLoader.discover(str(tests), pattern='test_worker_*.py'))
    return run_checks(suite, report_path)


if __name__ == '__main__':
    raise SystemExit(main())
