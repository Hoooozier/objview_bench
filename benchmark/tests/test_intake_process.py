"""Launch the actual localhost CLI twice; persistence cannot be an in-memory fake."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

import httpx

ROOT = Path(__file__).resolve().parents[2]
ADMIN = 'process-test-admin-0123456789-0123456789'


class IntakeProcessTests(unittest.TestCase):
    def test_cli_requires_development_flag_and_private_admin_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / 'state'
            command = [sys.executable, '-m', 'objview_submit', '--state-dir', str(state)]
            env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
            env.pop('OBJVIEW_SUBMIT_ADMIN_TOKEN', None)
            for args, token in [(command, ADMIN), (command + ['--development-only'], ''),
                                (command + ['--development-only'], 'SECRET_WEAK')]:
                result = subprocess.run(args, cwd=ROOT, env={**env, 'OBJVIEW_SUBMIT_ADMIN_TOKEN': token},
                                        capture_output=True, text=True, timeout=5)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn('Traceback', result.stderr)
                self.assertNotIn('SECRET_WEAK', result.stderr)
                self.assertFalse(state.exists())

    def test_real_http_service_restart_retains_submission_and_session(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryFile(mode='w+b') as logs:
            state = Path(tmp) / 'state'
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                port = sock.getsockname()[1]
            env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OBJVIEW_SUBMIT_ADMIN_TOKEN': ADMIN}
            args = [sys.executable, '-m', 'objview_submit', '--development-only', '--state-dir', str(state), '--port', str(port)]
            session = None
            submission = None
            invitation_token = None
            for run in range(2):
                process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=logs, stderr=logs)
                try:
                    with httpx.Client(base_url=f'http://127.0.0.1:{port}', trust_env=False, timeout=2) as client:
                        deadline = time.monotonic() + 8
                        while time.monotonic() < deadline:
                            self.assertIsNone(process.poll(), 'service exited before readiness')
                            try:
                                response = client.get('/healthz')
                                if response.status_code == 200:
                                    break
                            except httpx.TransportError:
                                pass
                            time.sleep(0.03)
                        else:
                            self.fail('service never became ready')
                        self.assertFalse(response.json()['execution_enabled'])
                        if run == 0:
                            invite = client.post('/v1/admin/invitations', headers={'Authorization': 'Bearer ' + ADMIN},
                                                 json={'label': 'process-researcher', 'ttl_seconds': 3600})
                            self.assertEqual(invite.status_code, 201)
                            invitation_token = invite.json()['token']
                            session = client.post('/v1/invitations/redeem', json={'token': invitation_token}).json()
                            headers = {'Authorization': 'Bearer ' + session['access_token']}
                            method = client.post('/v1/methods', headers=headers, json={'name': 'Pipeline demo'}).json()
                            manifest = {'method_id': method['id'], 'repository': 'https://github.com/example/dummy',
                                        'commit_sha': 'a' * 40, 'dockerfile': 'Dockerfile',
                                        'entrypoint': '/opt/objview/run.sh', 'protocol_version': 'objview-interaction/v1'}
                            response = client.post('/v1/submissions', headers={**headers, 'Idempotency-Key': 'tcp-test'}, json=manifest)
                            self.assertEqual(response.status_code, 201)
                            submission = response.json()
                        else:
                            headers = {'Authorization': 'Bearer ' + session['access_token']}
                            self.assertEqual(client.get('/v1/me', headers=headers).json()['label'], 'process-researcher')
                            self.assertEqual(client.get('/v1/submissions/' + submission['id'], headers=headers).json(), submission)
                            response = client.post('/v1/admin/submissions/' + submission['id'] + '/review',
                                                   headers={'Authorization': 'Bearer ' + ADMIN},
                                                   json={'decision': 'start', 'expected_version': 0})
                            self.assertEqual(response.status_code, 200)
                            self.assertEqual(response.json()['status'], 'ADMIN_REVIEW')
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            logs.seek(0)
            output = logs.read().decode('utf-8', errors='replace')
            for token in [ADMIN, session['access_token'], invitation_token]:
                self.assertNotIn(token, output)


if __name__ == '__main__':
    unittest.main()
