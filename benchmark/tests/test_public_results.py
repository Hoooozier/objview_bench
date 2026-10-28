"""Paper provenance/semantics and anonymous-browsing permission regressions."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastapi.testclient import TestClient
from objview_submit.app import create_app
from objview_submit.paper_results import load_results, select_results
from objview_submit.store import Store


ADMIN = 'public-results-test-admin-secret-0123456789'


class PublicResultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name) / 'state')
        self.client = TestClient(create_app(self.store, ADMIN, development_mode=True))
        self.addCleanup(self.client.close)

    def test_anonymous_home_styles_and_paper_download(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.url.path, '/ui')
        for text in ('Paper baseline results', 'NSC@0.02', 'SC@0.02', 'Path Cost', 'PR bin',
                     'Real RGB-D', 'CRC@1cm', 'Algorithm-reported', 'Dockerfiles are collected'):
            self.assertIn(text, response.text)
        self.assertNotIn('name="token"', response.text)
        self.assertNotIn('set-cookie', response.headers)
        self.assertIn("style-src 'self'", response.headers['content-security-policy'])
        css = self.client.get('/ui/assets/site.css')
        self.assertEqual(css.status_code, 200)
        self.assertIn('text/css', css.headers['content-type'])
        self.assertIn('@media(max-width:700px)', css.text)
        snapshot = self.client.get('/ui/paper-results.json')
        self.assertEqual(snapshot.status_code, 200)
        self.assertEqual(snapshot.json()['kind'], 'paper_reported_results')
        self.assertIn('attachment', snapshot.headers['content-disposition'])
        self.assertNotIn(ADMIN, response.text + snapshot.text)

    def test_curated_snapshot_preserves_reported_values_and_precision(self):
        data = load_results()
        self.assertEqual(len(data['synthetic']), 76)
        self.assertEqual(len(data['real_rgbd']), 14)
        self.assertEqual(len({r['method'] for r in data['synthetic']}), 9)
        self.assertTrue(data['source']['file'].endswith('main_post_rebuttal.tex'))
        self.assertRegex(data['source']['sha256'], r'^[0-9a-f]{64}$')
        # Independent anchors from manuscript main/appendix tables, not computed
        # using the extractor. Includes asymmetric W/Q columns and runtime bins.
        anchors = [
            ('RSE', 'whole', 'K=5', '0.980', '25.66', '6.0', '0.5–1'),
            ('PoinTr-C+MCP', 'whole', 'K=30', '0.998', '51.35', '31.0', '1–3'),
            ('Random+TSP', 'quarter', 'K=30', '0.997', '24.34', '31.0', '<0.1'),
            ('RSE-Mov', 'whole', 'MS@0.02', '0.985', '31.92', '14.1', '3–5'),
            ('NBV-Net', 'quarter', 'K=30', '0.472', '0.90', '1.2', '0.1–0.5'),
            ('BENBV-Net', 'whole', 'MS@0.02', '0.986', '27.49', '10.4', '1–3'),
            ('MA-SCVP', 'whole', 'plan_end', '0.990', '57.19', '49.5', '0.1–0.5'),
            ('PoinTr-C+SCP', 'whole', 'plan_end', '0.993', '30.17', '15.5', '0.5–1'),
            ('PoinTr-C+NBV', 'quarter', 'K=30', '0.998', '71.71', '31.0', '>5'),
        ]
        for method, space, checkpoint, nsc, path, views, time_bin in anchors:
            row = next(r for r in data['synthetic'] if (r['method'], r['space'], r['checkpoint']) ==
                       (method, space, checkpoint))
            self.assertEqual((row['nsc02'], row['path'], row['views'], row['time_bin']),
                             (nsc, path, views, time_bin))
        real = next(r for r in data['real_rgbd'] if r['method'] == 'PoinTr-C+MCP' and r['checkpoint'] == 'K=30')
        self.assertEqual((real['views'], real['crc02'], real['path']), ('30.52', '0.997', '35.96'))

    def test_filters_respect_execution_modes_and_path_quality_threshold(self):
        small = select_results()
        self.assertEqual(len(small), 7)
        self.assertEqual(small[0]['method'], 'RSE')
        self.assertNotIn('MA-SCVP', {r['method'] for r in small})
        automatic = select_results(setting='automatic')
        self.assertEqual(len(automatic), 7)
        self.assertEqual(automatic[0]['method'], 'PoinTr-C+SCP')
        self.assertNotIn('Random+TSP', {r['method'] for r in automatic})
        self.assertEqual({r['checkpoint'] for r in automatic}, {'plan_end', 'MS@0.02'})
        path_rows = select_results(order='path')
        self.assertEqual([r['method'] for r in path_rows[:3]], ['BENBV-Net', 'PoinTr-C+MCP', 'NBV-Net'])
        self.assertNotIn('RSE-Mov', {r['method'] for r in path_rows})
        self.assertEqual(len(select_results(setting='MS@0.01')), 5)
        response = self.client.get('/ui', params={'space': 'quarter', 'setting': 'automatic', 'order': 'path'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('value="quarter" selected', response.text)
        self.assertIn('value="automatic" selected', response.text)
        for key in ('space', 'setting', 'order'):
            invalid = self.client.get('/ui', params={key: '<script>alert(1)</script>'})
            self.assertEqual(invalid.status_code, 422)
            self.assertNotIn('<script>', invalid.text)

    def test_browsing_never_exposes_private_submissions_or_requires_session(self):
        invite = self.store.create_invitation('private-owner-name', 3600)
        session = self.store.redeem(invite['token'])
        owner = session['user']['id']
        method = self.store.create_method(owner, 'private-method-name')
        row = self.store.submit(owner, {
            'method_id': method['id'], 'repository': 'https://github.com/private/research',
            'commit_sha': 'b' * 40, 'dockerfile': 'private/Dockerfile',
            'entrypoint': '/opt/objview/run.sh', 'protocol_version': 'objview-interaction/v1'}, 'private-key')
        for cookie in (None, session['access_token'], 'expired-or-invalid-cookie'):
            self.client.cookies.clear()
            if cookie:
                self.client.cookies.set('objview_session', cookie)
            response = self.client.get('/ui', follow_redirects=False)
            self.assertEqual(response.status_code, 200)
            for secret in (row['id'], 'private-owner-name', 'private-method-name', 'private/research', session['access_token']):
                self.assertNotIn(secret, response.text)
                self.assertNotIn(secret, self.client.get('/ui/paper-results.json').text)
        self.client.cookies.clear()
        self.assertEqual(self.client.get('/ui/submissions/' + row['id']).status_code, 401)
        self.assertEqual(self.client.get('/v1/submissions').status_code, 401)
        self.assertEqual(self.client.get('/v1/admin/submissions').status_code, 401)

    def test_token_is_requested_only_on_submission_entry_and_write_is_guarded(self):
        entry = self.client.get('/ui/submit')
        self.assertEqual(entry.status_code, 200)
        self.assertIn('name="token"', entry.text)
        self.assertIn('type="password"', entry.text)
        self.assertIn('Dockerfile file uploads are not available yet', entry.text)
        self.assertEqual(self.client.get('/ui/researcher', follow_redirects=False).headers['location'], '/ui/submit')
        self.assertEqual(self.client.post('/ui/submissions', data={'dockerfile': 'Dockerfile'}).status_code, 401)
        self.assertEqual(self.client.post('/ui/methods', data={'name': 'unauthorized'}).status_code, 401)
        self.assertEqual(self.client.post('/v1/submissions', json={}).status_code, 401)
        self.assertEqual(self.client.post('/ui/admin/evaluation-batches/release').status_code, 401)
        self.assertEqual(self.store.list_submissions(), [])

    def test_public_to_authenticated_submission_still_waits_for_batch(self):
        self.assertEqual(self.client.get('/ui').status_code, 200)
        invite = self.store.create_invitation('new-submitter', 3600)
        response = self.client.post('/ui/redeem', data={'token': invite['token']}, follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        self.assertIn('HttpOnly', response.headers['set-cookie'])
        self.assertNotIn(invite['token'], response.text + response.headers['location'])
        self.assertIn('Continue submission', self.client.get('/ui/submit').text)
        self.assertEqual(self.client.get('/ui', follow_redirects=False).status_code, 200)
        self.client.post('/ui/methods', data={'name': 'new method'})
        with self.store._connect() as db:
            owner = db.execute("SELECT id FROM users WHERE label='new-submitter'").fetchone()[0]
        method = self.store.list_methods(owner)[0]
        response = self.client.post('/ui/submissions', data={
            'method_id': method['id'], 'repository': 'https://github.com/example/algorithm',
            'commit_sha': 'a' * 40, 'dockerfile': 'Dockerfile', 'entrypoint': '/opt/objview/run.sh',
            'protocol_version': 'objview-interaction/v1', 'idempotency_key': 'public-flow'}, follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        row = self.store.list_submissions(owner)[0]
        self.assertEqual(row['status'], 'SUBMITTED')
        self.assertIsNone(self.store.claim_job('public-flow-worker'))
        self.assertEqual(self.client.get('/ui/submissions/' + row['id']).status_code, 200)
        self.client.post('/ui/logout')
        self.assertEqual(self.client.get('/ui').status_code, 200)
        self.assertEqual(self.client.get('/ui/submissions/' + row['id']).status_code, 401)

    def test_english_guide_and_downloads_use_the_tested_api_only_example(self):
        for path in ('/ui', '/ui/submit', '/ui/guide', '/ui/admin'):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertIn('lang="en"', response.text)
            self.assertNotRegex(response.text, r'[\u3400-\u9fff]')
        guide = self.client.get('/ui/guide').text
        for text in ('Producer–consumer', '/v1/observations/{step_index}/{asset}',
                     '20 requests/second', 'burst of 40', '512 MiB per episode',
                     'Arbitrary submitted Dockerfiles are not built or executed'):
            self.assertIn(text, guide)
        repo = Path(__file__).resolve().parents[2]
        for name, source in [('algorithm.py', 'benchmark/examples/submission/algorithm.py'),
                             ('run.sh', 'benchmark/examples/submission/run.sh'),
                             ('submission_client.py', 'benchmark/submission_client.py')]:
            response = self.client.get('/ui/examples/' + name)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.text, (repo / source).read_text())
            self.assertIn('attachment', response.headers['content-disposition'])
            self.assertEqual(response.headers['x-content-type-options'], 'nosniff')
        dockerfile = self.client.get('/ui/examples/Dockerfile').text
        for text in ('USER 65532:65532', '/opt/objview/run.sh', 'COPY submission_client.py algorithm.py run.sh'):
            self.assertIn(text, dockerfile)
        self.assertNotIn('OBJVIEW_EPISODE_TOKEN=', dockerfile)
        self.assertEqual(self.client.get('/ui/examples/control.sqlite3').status_code, 404)
        self.assertEqual(self.client.get('/ui/examples/..%2Fstore.py').status_code, 404)
        self.assertEqual(self.client.post('/ui/examples/Dockerfile').status_code, 405)
        self.assertIn('Example Dockerfile', self.client.get('/ui/submit').text)

    def test_login_paths_share_throttle_and_forms_are_size_bounded(self):
        oversized = self.client.post('/ui/redeem', content='token=' + 'a' * 16384,
                                     headers={'Content-Type': 'application/x-www-form-urlencoded'})
        self.assertEqual(oversized.status_code, 413)
        for index in range(59):
            path = '/ui/redeem' if index % 2 else '/ui/admin/login'
            response = self.client.post(path, data={'token': 'invalid'})
            self.assertNotEqual(response.status_code, 429)
        response = self.client.post('/v1/invitations/redeem', json={'token': 'invalid'})
        self.assertEqual(response.status_code, 429)
        self.assertEqual(self.client.get('/ui').status_code, 200)


if __name__ == '__main__':
    unittest.main()
