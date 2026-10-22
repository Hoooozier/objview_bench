import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from submission_gateway import create_app
from submission_client import SubmissionClient
from submission_protocol import SubmissionSession

TOKEN = 'test-token-with-at-least-32-characters'


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.session = SubmissionSession(lambda p: True)
        self.session.configure({'uid': 'SECRET'})
        self.client = TestClient(create_app(self.session, TOKEN), raise_server_exceptions=False)
        self.client.headers['Authorization'] = f'Bearer {TOKEN}'

    def test_authentication_applies_to_all_paths(self):
        for path in ['/v1/interface', '/v1/episode', '/v1/status', '/docs', '/openapi.json', '/oracle']:
            response = self.client.get(path, headers={'Authorization': 'Bearer wrong'})
            self.assertEqual(response.status_code, 401)
        self.assertEqual(self.client.get('/v1/interface').status_code, 200)
        self.assertEqual(self.client.get('/docs').status_code, 404)

    def test_body_limits_malformed_json_and_errors_do_not_echo(self):
        for body in ['{SECRET', '{"action":"SECRET","action":"move"}', '{"x":NaN}']:
            response = self.client.post('/v1/actions', content=body, headers={'Content-Type': 'application/json'})
            self.assertEqual(response.status_code, 400)
            self.assertNotIn('SECRET', response.text)
        response = self.client.post('/v1/actions', content='x' * 65537, headers={'Content-Type': 'application/json'})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.client.post('/v1/actions', json={'uid': 'SECRET'}).json(), {'error': 'INVALID_ACTION'})

    def test_routes_do_not_expose_files_or_oracle(self):
        for path in ['/v1/oracle_best_nsc_gain', '/observations/step_000/rgb.png', '/v1/summary']:
            self.assertEqual(self.client.get(path).status_code, 404)
        self.assertNotIn('SECRET', json.dumps(self.client.get('/v1/episode').json()))

    def test_internal_exception_is_sanitized(self):
        def fail(p):
            raise ValueError('/hidden/SECRET.pcd')
        self.session.feasible = fail
        self.session.ready()
        self.session.publish(0, [0, 0, 3, 0, 0, 0, 0], {})
        result = self.client.post('/v1/feasibility/check', json={'poses': [[0, 0, 3, 0, 0, 0, 0]]})
        self.assertEqual(result.status_code, 500)
        self.assertEqual(result.json(), {'error': 'INTERNAL_ERROR'})

    def test_poll_validation(self):
        for query in ['after=bad', 'after=-2', 'wait=inf', 'wait=11']:
            self.assertEqual(self.client.get('/v1/steps/next?' + query).status_code, 422)

    def test_huge_numeric_pose_returns_422_without_ending_session(self):
        self.session.ready()
        self.session.publish(0, [0, 0, 3, 0, 0, 0, 0], {})
        pose = [0, 0, 10 ** 400, 0, 0, 0, 0]
        step = self.session.next_step()
        action = SubmissionClient.make_action(step, 'move', pose=pose)
        for endpoint, data in [('/v1/actions', action), ('/v1/feasibility/check', {'poses': [pose]})]:
            response = self.client.post(endpoint, json=data)
            self.assertEqual(response.status_code, 422)
            self.assertEqual(self.session.status()['status'], 'running')

    def test_sdk_real_round_trip_retry_and_terminal(self):
        sdk = SubmissionClient('http://testserver', TOKEN, transport=self.client)
        sdk.ready()
        self.session.publish(0, [0, 0, 3, 0, 0, 0, 0], {'rgb_path': 'step_000/rgb.png'})
        step = sdk.next_step(-1, wait=0)
        self.assertEqual(step['step_index'], 0)
        self.assertEqual(sdk.feasibility([[0, 0, 3, 0, 0, 0, 0]]), [True])
        action = sdk.make_action(step, 'move', pose=[0, 3, 0, 0, 0, 0, 0])
        self.assertEqual(sdk.submit(action), sdk.submit(action))
        self.assertEqual(self.session.take_action()['pose'], [0, 3, 0, 0, 0, 0, 0])
        self.session.publish(1, [0, 3, 0, 0, 0, 0, 0], {})
        sdk.submit(sdk.make_action(sdk.next_step(0, wait=0), 'stop', stop_reason='plan_end'))
        self.assertEqual(self.session.take_action()['action'], 'stop')
        self.session.finish()
        self.assertEqual(sdk.next_step(1, wait=0), {'status': 'done', 'code': 'COMPLETED'})


if __name__ == '__main__':
    unittest.main()
