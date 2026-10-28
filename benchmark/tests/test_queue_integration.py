"""Docker-free end-to-end trusted dummy queue acceptance."""
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from objview_submit.store import Store
from submission_worker.controller import validate_summary
from submission_worker.queue_worker import QueueWorker


class QueueIntegrationTests(unittest.TestCase):
    def test_queued_job_runs_repository_trusted_dummy_and_persists_safe_result(self):
        benchmark = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as state:
            state = Path(state)
            now = [1000.0]
            store = Store(state / 'control', clock=lambda: now[0])
            invite = store.create_invitation('integration', 3600)
            owner = store.redeem(invite['token'])['user']['id']
            method = store.create_method(owner, 'trusted integration')
            manifest = {'method_id': method['id'], 'repository': 'https://github.com/example/algorithm',
                        'commit_sha': 'a' * 40, 'dockerfile': 'Dockerfile',
                        'entrypoint': '/opt/objview/run.sh', 'protocol_version': 'objview-interaction/v1'}
            submission = store.submit(owner, manifest, 'integration-1')
            store.review(submission['id'], 'start', 0)
            store.review(submission['id'], 'approve', 1)
            store.release_batch()

            def executor(claim):
                root = state / ('episode-' + claim['job_id'])
                (root / 'private').mkdir(parents=True)
                (root / 'released').mkdir()
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(benchmark),
                           OBJVIEW_EPISODE_TOKEN=secrets.token_hex(32), OBJVIEW_JOB_ID=claim['job_id'])
                server = subprocess.Popen(
                    [sys.executable, '-B', str(benchmark / 'smoke_container_service.py'),
                     '--episode-root', str(root), '--host', '127.0.0.1', '--port', '0'],
                    env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                try:
                    ready = root / 'private' / 'gateway.ready'
                    deadline = time.monotonic() + 8
                    while not ready.exists() and server.poll() is None and time.monotonic() < deadline:
                        time.sleep(.01)
                    if not ready.exists():
                        return {'status': 'FAILED', 'error_code': 'BENCHMARK_FAILED'}
                    port = json.loads(ready.read_text())['port']
                    env.update(BENCHMARK_URL='http://127.0.0.1:%s' % port,
                               OBSERVATION_ROOT='/not-mounted', OBJVIEW_RANDOM_SEED='queue-test')
                    algorithm = subprocess.run(
                        [sys.executable, '-B', str(benchmark / 'examples/submission/random_algorithm.py')],
                        env=env, cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
                    if algorithm.returncode != 0 or server.wait(timeout=8) != 0:
                        return {'status': 'FAILED', 'error_code': 'ALGORITHM_CRASHED'}
                    summary = json.loads((root / 'private' / 'summary.json').read_text())
                    return {'status': 'SUCCEEDED', 'result': validate_summary(summary, claim['job_id'])}
                except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError):
                    return {'status': 'FAILED', 'error_code': 'INFRASTRUCTURE_FAILED'}
                finally:
                    if server.poll() is None:
                        server.kill()
                    server.wait(timeout=5)

            result = QueueWorker(store, 'integration-worker', executor, lease_seconds=60).run_once()
            self.assertEqual(result['state'], 'SUCCEEDED')
            self.assertFalse(result['official_result'])
            self.assertFalse(result['leaderboard_eligible'])
            self.assertEqual(result['result']['timing_source'], 'benchmark_monotonic_clock')
            self.assertEqual(store.get_job(owner, submission['id'])['state'], 'SUCCEEDED')


if __name__ == '__main__':
    unittest.main()
