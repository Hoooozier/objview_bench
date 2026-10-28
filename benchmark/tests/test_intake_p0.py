"""P0 pagination and browser-page regression coverage."""
from pathlib import Path
import tempfile
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastapi.testclient import TestClient
from objview_submit.app import create_app
from objview_submit.store import Store
from objview_submit.validation import IntakeError


ADMIN = 'test-admin-token-not-for-deployment-0123456789'


class IntakeP0Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'state'
        self.now = 1000.0
        self.store = Store(self.path, clock=lambda: self.now)
        self.client = TestClient(create_app(self.store, ADMIN, development_mode=True),
                                 raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def _owner(self):
        invite = self.store.create_invitation('pager', 3600)
        return self.store.redeem(invite['token'])['user']['id']

    def _manifest(self, owner):
        method = self.store.create_method(owner, 'P0 method')
        return {'method_id': method['id'], 'repository': 'https://github.com/example/algorithm',
                'commit_sha': 'a' * 40, 'dockerfile': 'Dockerfile',
                'entrypoint': '/opt/objview/run.sh', 'protocol_version': 'objview-interaction/v1'}

    def test_keyset_pages_are_complete_and_scope_bound(self):
        owner = self._owner()
        manifest = self._manifest(owner)
        for index in range(7):
            if index == 5:
                self.now += 86401
            row = self.store.submit(owner, manifest, 'page-%d' % index)
            self.store.review(row['id'], 'start', 0)
            self.store.review(row['id'], 'reject', 1)

        seen = []
        cursor = None
        while True:
            page, cursor = self.store.list_submissions_page(owner, limit=2, cursor=cursor)
            seen.extend(row['id'] for row in page)
            if cursor is None:
                break
        self.assertEqual(len(seen), 7)
        self.assertEqual(len(set(seen)), 7)
        rejected, _ = self.store.list_submissions_page(owner, limit=3, status='REJECTED')
        self.assertEqual(len(rejected), 3)
        with self.assertRaises(IntakeError) as raised:
            self.store.list_submissions_page(owner, cursor='tampered')
        self.assertEqual(raised.exception.code, 'INVALID_CURSOR')

        audit_seen = []
        cursor = None
        while True:
            page, cursor = self.store.audit_events_page(limit=4, cursor=cursor)
            audit_seen.extend(event['id'] for event in page)
            if cursor is None:
                break
        self.assertEqual(len(audit_seen), len(set(audit_seen)))
        self.assertGreater(len(audit_seen), 20)

    def test_api_pagination_status_filter_and_ui_cookie_flow(self):
        invite = self.client.post('/v1/admin/invitations', headers={'Authorization': 'Bearer ' + ADMIN},
                                  json={'label': 'browser', 'ttl_seconds': 3600}).json()
        response = self.client.post('/ui/redeem', data={'token': invite['token']}, follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers['location'], '/ui/researcher')
        self.assertNotIn(invite['token'], response.headers['location'])
        self.assertIn('HttpOnly', response.headers['set-cookie'])
        self.assertIn('SameSite=strict', response.headers['set-cookie'])
        page = self.client.get('/ui/researcher')
        self.assertEqual(page.status_code, 200)
        self.assertIn('Create method', page.text)
        self.assertIn("form-action 'self'", page.headers['content-security-policy'])
        self.assertEqual(page.headers['referrer-policy'], 'no-referrer')
        method = self.client.post('/ui/methods', data={'name': '<b>safe</b>'}, follow_redirects=False)
        self.assertEqual(method.status_code, 303)
        page = self.client.get('/ui/researcher')
        self.assertIn('&lt;b&gt;safe&lt;/b&gt;', page.text)

        # Browser submission uses the method visible in the store, while its
        # credentials remain in the HttpOnly cookie rather than the URL.
        # Find the active browser owner without exposing a token to the page.
        with self.store._connect() as db:
            owner_id = db.execute("SELECT id FROM users WHERE label='browser'").fetchone()[0]
        method_id = self.store.list_methods(owner_id)[0]['id']
        form = {'method_id': method_id, 'repository': 'https://github.com/example/algorithm',
                'commit_sha': 'a' * 40, 'dockerfile': 'Dockerfile',
                'entrypoint': '/opt/objview/run.sh', 'protocol_version': 'objview-interaction/v1',
                'idempotency_key': 'browser-1'}
        response = self.client.post('/ui/submissions', data=form, follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        sid = self.store.list_submissions(owner_id)[0]['id']
        self.assertIn(sid, self.client.get('/ui/researcher').text)

        login = self.client.post('/ui/admin/login', data={'token': ADMIN}, follow_redirects=False)
        self.assertEqual(login.status_code, 303)
        admin_page = self.client.get('/ui/admin')
        self.assertIn(sid, admin_page.text)
        reviewed = self.client.post('/ui/admin/submissions/%s/review' % sid,
                                    data={'decision': 'start', 'expected_version': '0'},
                                    follow_redirects=False)
        self.assertEqual(reviewed.status_code, 303)
        api = self.client.get('/v1/admin/submissions?limit=1&status=ADMIN_REVIEW',
                              headers={'Authorization': 'Bearer ' + ADMIN})
        self.assertEqual(api.status_code, 200)
        self.assertEqual(api.json()['items'][0]['status'], 'ADMIN_REVIEW')
        self.assertNotIn('access_token', admin_page.text)
        notice = 'At this stage, Dockerfiles are collected for scheduled batch validation.'
        self.assertIn(notice, self.client.get('/ui').text)
        self.assertIn(notice, self.client.get('/ui/researcher').text)
        approved = self.client.post('/ui/admin/submissions/%s/review' % sid,
                                    data={'decision': 'approve', 'expected_version': '1'},
                                    follow_redirects=False)
        self.assertEqual(approved.status_code, 303)
        self.assertIsNone(self.store.claim_job('browser-test'))
        self.assertIn('Collecting; awaiting batch validation', self.client.get('/ui/submissions/' + sid).text)
        released = self.client.post('/ui/admin/evaluation-batches/release', content='',
                                    headers={'Content-Type': 'application/x-www-form-urlencoded'},
                                    follow_redirects=False)
        self.assertEqual(released.status_code, 303)
        job = self.store.get_job(owner_id, sid)
        self.assertEqual(job['state'], 'QUEUED')
        self.assertIn(job['batch_id'], self.client.get('/ui/submissions/' + sid).text)
        self.assertIn(job['batch_id'], self.client.get('/ui/admin').text)
        logout = self.client.post('/ui/logout', follow_redirects=False)
        self.assertEqual(logout.status_code, 303)
        self.assertIn('objview_session=""', logout.headers['set-cookie'])

    def test_batch_api_requires_admin_and_rejects_client_selected_jobs(self):
        path = '/v1/admin/evaluation-batches/release'
        self.assertEqual(self.client.post(path, json={}).status_code, 401)
        invite = self.store.create_invitation('batch-api', 3600)
        session = self.store.redeem(invite['token'])
        researcher = {'Authorization': 'Bearer ' + session['access_token']}
        admin = {'Authorization': 'Bearer ' + ADMIN}
        self.assertEqual(self.client.post(path, headers=researcher, json={}).status_code, 403)
        self.assertEqual(self.client.get('/v1/admin/evaluation-batches', headers=researcher).status_code, 403)
        owner = session['user']['id']
        row = self.store.submit(owner, self._manifest(owner), 'batch-api')
        self.store.review(row['id'], 'start', 0)
        self.store.review(row['id'], 'approve', 1)
        self.assertEqual(self.client.post(path, headers=admin, json={'command': 'run'}).status_code, 422)
        self.assertEqual(self.client.get('/v1/submissions/' + row['id'] + '/evaluation',
                                         headers=researcher).json()['state'], 'COLLECTING')
        response = self.client.post(path, headers=admin, json={})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()['job_count'], 1)
        self.assertEqual(self.client.post(path, headers=admin, json={}).status_code, 409)
        self.assertEqual(len(self.client.get('/v1/admin/evaluation-batches', headers=admin).json()['items']), 1)
        self.client.cookies.clear()
        self.assertEqual(self.client.post('/ui/admin/evaluation-batches/release').status_code, 401)


if __name__ == '__main__':
    unittest.main()
