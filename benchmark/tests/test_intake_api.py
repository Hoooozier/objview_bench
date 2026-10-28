"""Real HTTP/SQLite intake workflow and trust-boundary regressions."""
from pathlib import Path
import json
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastapi.testclient import TestClient
from objview_submit.app import create_app
from objview_submit.store import Store

ADMIN = 'test-admin-token-not-for-deployment-0123456789'


def auth(token):
    return {'Authorization': 'Bearer ' + token}


class IntakeAPITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'state'
        self.store = Store(self.path)
        self.client = TestClient(create_app(self.store, ADMIN, development_mode=True), raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def register(self, label):
        invite = self.client.post('/v1/admin/invitations', headers=auth(ADMIN),
                                  json={'label': label, 'ttl_seconds': 3600})
        self.assertEqual(invite.status_code, 201)
        response = self.client.post('/v1/invitations/redeem', json={'token': invite.json()['token']})
        self.assertEqual(response.status_code, 200)
        return response.json()

    def submission(self, token):
        headers = auth(token)
        method = self.client.post('/v1/methods', headers=headers, json={'name': 'Dummy'})
        self.assertEqual(method.status_code, 201)
        manifest = {'method_id': method.json()['id'], 'repository': 'https://github.com/example/algorithm',
                    'commit_sha': 'a' * 40, 'dockerfile': 'docker/Dockerfile',
                    'entrypoint': '/opt/objview/run.sh', 'protocol_version': 'objview-interaction/v1'}
        response = self.client.post('/v1/submissions', headers={**headers, 'Idempotency-Key': 'first'}, json=manifest)
        self.assertEqual(response.status_code, 201)
        return response.json(), manifest

    def test_http_invitation_submission_review_and_restart(self):
        session = self.register('assigned')
        token = session['access_token']
        result, manifest = self.submission(token)
        self.assertEqual(result['status'], 'SUBMITTED')
        sid = result['id']
        for decision, version, status in [('start', 0, 'ADMIN_REVIEW'), ('approve', 1, 'APPROVED_FOR_PREFLIGHT')]:
            response = self.client.post(f'/v1/admin/submissions/{sid}/review', headers=auth(ADMIN),
                                        json={'decision': decision, 'expected_version': version})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['status'], status)
        restarted = TestClient(create_app(Store(self.path), ADMIN, development_mode=True))
        self.addCleanup(restarted.close)
        row = restarted.get(f'/v1/submissions/{sid}', headers=auth(token)).json()
        self.assertEqual(row['status'], 'APPROVED_FOR_PREFLIGHT')
        self.assertEqual(row['version'], 2)
        self.assertFalse(row['official_result'])
        self.assertFalse(row['leaderboard_eligible'])
        self.assertEqual(restarted.get('/v1/submissions', headers=auth(token)).json()['items'], [row])
        self.assertEqual(restarted.get('/v1/methods', headers=auth(token)).json()['items'][0]['id'], manifest['method_id'])
        self.assertEqual(restarted.get('/v1/admin/submissions', headers=auth(ADMIN)).json()['items'], [row])
        audit = restarted.get('/v1/admin/audit', headers=auth(ADMIN)).json()['items']
        self.assertEqual(audit[0]['event'], 'REVIEW_APPROVE')

    def test_authentication_and_cross_role_isolation(self):
        for path in ['/v1/me', '/v1/submissions', '/v1/admin/audit', '/docs', '/openapi.json']:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.headers['cache-control'], 'no-store')
        user = self.register('one')
        token = user['access_token']
        self.assertEqual(self.client.get('/v1/admin/audit', headers=auth(token)).status_code, 403)
        self.assertEqual(self.client.get('/v1/me', headers=auth(ADMIN)).status_code, 403)
        for path in ['/docs', '/openapi.json', '/hidden', '/v1/results', '/v1/worker/claim']:
            self.assertEqual(self.client.get(path, headers=auth(token)).status_code, 404)
        response = self.client.get('/v1/me', headers=auth(token))
        self.assertEqual(response.json()['label'], 'one')
        self.assertNotIn('access-control-allow-origin', response.headers)
        self.assertEqual(response.headers['x-content-type-options'], 'nosniff')

    def test_cross_owner_reads_and_method_references_are_denied(self):
        first = self.register('one')['access_token']
        second = self.register('two')['access_token']
        row, manifest = self.submission(first)
        response = self.client.get('/v1/submissions/' + row['id'], headers=auth(second))
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.client.get('/v1/submissions', headers=auth(second)).json(), {'items': []})
        response = self.client.post('/v1/submissions', json=manifest,
                                    headers={**auth(second), 'Idempotency-Key': 'other'})
        self.assertEqual(response.status_code, 404)

    def test_strict_json_limits_and_no_sensitive_validation_echo(self):
        token = self.register('one')['access_token']
        for raw in ['{SECRET', '{"name":"SECRET","name":"duplicate"}', '{"name":NaN}', '[' * 1100]:
            response = self.client.post('/v1/methods', headers={**auth(token), 'Content-Type': 'application/json'}, content=raw)
            self.assertEqual(response.status_code, 400)
            self.assertNotIn('SECRET', response.text)
        response = self.client.post('/v1/methods', headers={**auth(token), 'Content-Type': 'application/json'}, content='x' * 16385)
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.client.post('/v1/methods', headers=auth(token), content='{}').status_code, 415)
        for value in [{'name': 'name', 'role': 'admin'}, {'name': {'SECRET': 1}}, [], None]:
            response = self.client.post('/v1/methods', headers={**auth(token), 'Content-Type': 'application/json'},
                                        content='null' if value is None else json.dumps(value))
            self.assertEqual(response.status_code, 422)
            self.assertNotIn('SECRET', response.text)

    def test_tokens_revoke_and_identity_cannot_be_claimed_at_redemption(self):
        invite = self.client.post('/v1/admin/invitations', headers=auth(ADMIN), json={'label': 'one', 'ttl_seconds': 60}).json()
        self.assertEqual(self.client.post('/v1/invitations/redeem', json={'token': invite['token'], 'role': 'admin'}).status_code, 422)
        self.assertEqual(self.client.post(f"/v1/admin/invitations/{invite['id']}/revoke", headers=auth(ADMIN), json={}).status_code, 200)
        self.assertEqual(self.client.post('/v1/invitations/redeem', json={'token': invite['token']}).status_code, 401)
        user = self.register('two')
        token = user['access_token']
        self.assertEqual(self.client.post('/v1/session/logout', headers=auth(token), json={}).status_code, 200)
        self.assertEqual(self.client.get('/v1/me', headers=auth(token)).status_code, 401)
        user = self.register('three')
        response = self.client.post(f"/v1/admin/users/{user['user']['id']}/disable", headers=auth(ADMIN), json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get('/v1/me', headers=auth(user['access_token'])).status_code, 401)

    def test_idempotency_version_and_input_validation_reach_store(self):
        token = self.register('one')['access_token']
        row, manifest = self.submission(token)
        headers = {**auth(token), 'Idempotency-Key': 'first'}
        self.assertEqual(self.client.post('/v1/submissions', headers=headers, json=manifest).json(), row)
        self.assertEqual(self.client.post('/v1/submissions', headers=headers, json={**manifest, 'commit_sha': 'b'*40}).status_code, 409)
        self.assertEqual(self.client.post('/v1/submissions', headers=auth(token), json=manifest).status_code, 422)
        for field in ['metrics', 'elapsed_ns', 'owner_id', 'official_result']:
            self.assertEqual(self.client.post('/v1/submissions', headers=headers, json={**manifest, field: 'SECRET'}).status_code, 422)
        path = '/v1/admin/submissions/' + row['id'] + '/review'
        self.assertEqual(self.client.post(path, headers=auth(ADMIN), json={'decision': 'start', 'expected_version': 0}).status_code, 200)
        self.assertEqual(self.client.post(path, headers=auth(ADMIN), json={'decision': 'approve', 'expected_version': 0}).status_code, 409)

    def test_development_guard_health_and_redemption_rate_limit(self):
        with self.assertRaises(ValueError):
            create_app(self.store, ADMIN)
        with self.assertRaises(ValueError):
            create_app(self.store, 'weak', development_mode=True)
        response = self.client.get('/healthz')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['mode'], 'development_intake')
        for _ in range(60):
            response = self.client.post('/v1/invitations/redeem', json={'token': 'x' * 43})
            self.assertEqual(response.status_code, 401)
        response = self.client.post('/v1/invitations/redeem', json={'token': 'x' * 43})
        self.assertEqual(response.status_code, 429)

    def test_storage_failures_do_not_return_raw_errors_or_corrupt_metadata(self):
        token = self.register('one')['access_token']
        row, _ = self.submission(token)
        with sqlite3.connect(self.path / 'control.sqlite3') as db:
            db.execute('UPDATE submissions SET manifest_json=? WHERE id=?', ('{SECRET', row['id']))
        response = self.client.get('/v1/submissions/' + row['id'], headers=auth(token))
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {'error': 'INTERNAL_ERROR'})
        with sqlite3.connect(self.path / 'control.sqlite3') as db:
            db.execute('DROP TABLE methods')
        response = self.client.get('/v1/methods', headers=auth(token))
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {'error': 'STORAGE_UNAVAILABLE'})
        self.assertEqual(response.headers['cache-control'], 'no-store')


if __name__ == '__main__':
    unittest.main()
