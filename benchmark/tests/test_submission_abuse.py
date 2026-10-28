"""Gateway abuse boundaries and API-only observation delivery (no Docker claims)."""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from submission_client import SubmissionClient, SubmissionError
from submission_gateway import create_app, GatewayGuard, MAX_OBSERVATION_BYTES, ObservationReader
from submission_protocol import SubmissionSession, ProtocolError

TOKEN = 'abuse-regression-token-0123456789abcdef'


class AbuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'released'
        (self.root / 'step_000').mkdir(parents=True)
        self.file = self.root / 'step_000/rgb.png'
        self.file.write_bytes(b'released-png')
        self.secret = Path(self.temp.name) / 'private-secret'
        self.secret.write_bytes(b'SECRET-HIDDEN-GEOMETRY')
        self.session = SubmissionSession(lambda pose: True, max_steps=3)
        self.session.configure({})
        self.session.ready()
        self.session.publish(0, [0, 0, 3, 0, 0, 0, 0], {'rgb_path': 'step_000/rgb.png'})
        self.now = 0.
        self.client = TestClient(create_app(self.session, TOKEN, observation_root=self.root,
                                            clock=lambda: self.now), raise_server_exceptions=False)
        self.client.headers['Authorization'] = 'Bearer ' + TOKEN
        self.addCleanup(self.client.close)

    def test_only_current_published_assets_are_accessible(self):
        sdk = SubmissionClient('http://testserver', TOKEN, transport=self.client)
        step = sdk.next_step(wait=0)
        self.assertEqual(sdk.observation(step, 'rgb'), b'released-png')
        for path in ['/v1/observations/1/rgb', '/v1/observations/0/depth', '/v1/observations/0/summary',
                     '/v1/observations/0/..%2f..%2fprivate-secret', '/observations/step_000/rgb.png', '/v1/summary']:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 404)
            self.assertNotIn('SECRET', response.text)
        self.assertEqual(self.client.get('/v1/observations/0/rgb', headers={'Authorization': 'Bearer wrong'}).status_code, 401)
        sdk.submit(sdk.make_action(step, 'stop', stop_reason='plan_end'))
        self.assertEqual(self.client.get('/v1/observations/0/rgb').status_code, 404)

    def test_symlinks_hardlinks_special_files_and_oversized_assets_are_rejected(self):
        self.file.unlink()
        self.file.symlink_to(self.secret)
        self.assertEqual(self.client.get('/v1/observations/0/rgb').status_code, 404)
        self.file.unlink()
        os.link(self.secret, self.file)
        self.assertEqual(self.client.get('/v1/observations/0/rgb').status_code, 404)
        self.file.unlink()
        os.mkfifo(self.file)
        self.assertEqual(self.client.get('/v1/observations/0/rgb').status_code, 404)
        self.file.unlink()
        with self.file.open('wb') as stream:
            stream.truncate(MAX_OBSERVATION_BYTES + 1)
        self.assertEqual(self.client.get('/v1/observations/0/rgb').status_code, 413)
        self.file.unlink()
        self.file.parent.rmdir()
        self.file.parent.symlink_to(self.secret.parent, target_is_directory=True)
        self.assertEqual(self.client.get('/v1/observations/0/rgb').status_code, 404)

    def test_download_count_and_byte_budgets_are_enforced(self):
        for _ in range(10):
            self.assertEqual(self.client.get('/v1/observations/0/rgb').status_code, 200)
        response = self.client.get('/v1/observations/0/rgb')
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()['error'], 'OBSERVATION_BUDGET_EXHAUSTED')
        for limit in ('MAX_STEP_BYTES', 'MAX_EPISODE_BYTES'):
            with patch('submission_gateway.' + limit, 20):
                reader = ObservationReader(self.root)
                reader.read('step_000/rgb.png', 0)
                with self.assertRaises(ProtocolError) as caught:
                    reader.read('step_000/rgb.png', 0)
                self.assertEqual(caught.exception.code, 'OBSERVATION_BUDGET_EXHAUSTED')

    def test_rate_limit_counts_invalid_requests_and_ignores_spoofed_client_keys(self):
        for index in range(40):
            self.assertEqual(self.client.get('/v1/status', headers={
                'Authorization': 'Bearer wrong-' + str(index), 'X-Forwarded-For': str(index)}).status_code, 401)
        response = self.client.get('/v1/status')
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers['Retry-After'], '1')
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.now += 1
        self.assertEqual(self.client.get('/v1/status').status_code, 200)

    def test_episode_request_budget_terminates_only_its_session(self):
        other = SubmissionSession(lambda p: True)
        for _ in range(128 + 64 * self.session.max_steps):
            self.now += 1
            self.assertEqual(self.client.get('/v1/status').status_code, 200)
        response = self.client.get('/v1/status')
        self.assertEqual(response.json()['error'], 'REQUEST_BUDGET_EXHAUSTED')
        self.assertEqual(self.session.status(), {'status': 'done', 'code': 'EPISODE_FAILED'})
        self.assertNotEqual(other.status()['status'], 'done')

    def test_concurrency_limits_remain_held_until_body_is_sent(self):
        async def check(path, maximum):
            release = asyncio.Event()
            entered = 0
            async def app(scope, receive, send):
                nonlocal entered
                await send({'type': 'http.response.start', 'status': 200, 'headers': []})
                entered += 1
                await release.wait()
                await send({'type': 'http.response.body', 'body': b'ok'})
            guard = GatewayGuard(app, session=self.session, token=TOKEN, clock=lambda: 0.)
            async def request():
                messages = []
                async def send(message):
                    messages.append(message)
                async def receive():
                    return {'type': 'http.request', 'body': b''}
                await guard({'type': 'http', 'path': path, 'headers': [(b'authorization', ('Bearer ' + TOKEN).encode())]}, receive, send)
                return messages[0]['status']
            tasks = [asyncio.create_task(request()) for _ in range(maximum)]
            try:
                for _ in range(100):
                    if entered == maximum:
                        break
                    await asyncio.sleep(0)
                self.assertEqual(entered, maximum)
                self.assertEqual(await request(), 429)
            finally:
                release.set()
                self.assertEqual(await asyncio.gather(*tasks), [200] * maximum)
            self.assertEqual(await request(), 200)
        for path, maximum in [('/v1/status', 8), ('/v1/steps/next', 1), ('/v1/observations/0/rgb', 2)]:
            asyncio.run(check(path, maximum))

    def test_sdk_backoff_is_bounded_and_preserves_action_identity(self):
        import httpx
        seen = []
        def handler(request):
            seen.append(json.loads(request.content))
            if len(seen) < 3:
                return httpx.Response(429, json={'error': 'RATE_LIMITED'})
            return httpx.Response(200, json={'accepted': True})
        with httpx.Client(base_url='http://testserver', transport=httpx.MockTransport(handler)) as transport:
            sdk = SubmissionClient('http://testserver', TOKEN, transport=transport)
            action = sdk.make_action(self.session.next_step(), 'stop', stop_reason='plan_end')
            with patch('submission_client.time.sleep') as sleep:
                self.assertEqual(sdk.submit(action), {'accepted': True})
                self.assertEqual(sleep.call_count, 2)
            self.assertEqual(seen, [action] * 3)
        for code, calls in [('RATE_LIMITED', 3), ('REQUEST_BUDGET_EXHAUSTED', 1)]:
            with httpx.Client(base_url='http://testserver', transport=httpx.MockTransport(
                    lambda request: httpx.Response(429, json={'error': code}))) as transport:
                sdk = SubmissionClient('http://testserver', TOKEN, transport=transport)
                with patch('submission_client.time.sleep') as sleep, self.assertRaises(SubmissionError):
                    sdk.ready()
                self.assertEqual(sleep.call_count, calls - 1)


if __name__ == '__main__':
    unittest.main()
