"""Run the actual smoke-image entrypoint as a local process, without Docker."""
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time
import unittest


class ContainerServiceTests(unittest.TestCase):
    def test_smoke_entrypoint_and_dummy_process_persist_nonofficial_result(self):
        benchmark = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / 'private').mkdir()
            (root / 'released').mkdir()
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(benchmark),
                       OBJVIEW_EPISODE_TOKEN=secrets.token_hex(32), OBJVIEW_JOB_ID='c' * 32)
            server = subprocess.Popen([sys.executable, '-B', str(benchmark / 'smoke_container_service.py'),
                                       '--episode-root', str(root), '--host', '127.0.0.1', '--port', '0'],
                                      env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                ready = root / 'private' / 'gateway.ready'
                deadline = time.monotonic() + 8
                while not ready.exists() and server.poll() is None and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(ready.exists(), 'Smoke benchmark failed to start')
                port = json.loads(ready.read_text())['port']
                env.update(BENCHMARK_URL=f'http://127.0.0.1:{port}', OBSERVATION_ROOT='/not-mounted')
                algorithm = subprocess.run([sys.executable, '-B', str(benchmark / 'examples/submission/algorithm.py')],
                                           env=env, cwd=root, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL, timeout=10)
                self.assertEqual(algorithm.returncode, 0)
                self.assertEqual(server.wait(timeout=8), 0)
                summary = json.loads((root / 'private' / 'summary.json').read_text())
                self.assertEqual(summary['status'], 'completed')
                self.assertEqual(summary['mode'], 'smoke_test')
                self.assertEqual(summary['job_id'], 'c' * 32)
                self.assertIs(summary['leaderboard_eligible'], False)
                self.assertEqual(summary['visited_view_num'], 2)
                self.assertEqual([a['action'] for a in summary['algorithm_action_history']], ['move', 'stop'])
            finally:
                if server.poll() is None:
                    server.kill()
                server.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
