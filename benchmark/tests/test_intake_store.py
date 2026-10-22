"""Real SQLite boundaries; no mocked database or user-supplied identity."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from objview_submit.store import Store
from objview_submit.validation import IntakeError


class IntakeStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'state'
        self.now = 1000.0
        self.store = Store(self.path, clock=lambda: self.now)

    def user(self, label='researcher-a'):
        invitation = self.store.create_invitation(label, 3600)
        session = self.store.redeem(invitation['token'])
        return session['user']['id'], session['access_token']

    def manifest(self, owner):
        method = self.store.create_method(owner, 'Example method')
        return {'method_id': method['id'], 'repository': 'https://github.com/example/algorithm',
                'commit_sha': 'a' * 40, 'dockerfile': 'Dockerfile',
                'entrypoint': '/opt/objview/run.sh', 'protocol_version': 'objview-interaction/v1'}

    def error(self, code, function, *args):
        with self.assertRaises(IntakeError) as raised:
            function(*args)
        self.assertEqual(raised.exception.code, code)

    def test_single_use_invitation_assigns_identity_and_hashes_secrets(self):
        invite = self.store.create_invitation('assigned-identity', 60)
        result = self.store.redeem(invite['token'])
        self.assertEqual(result['user']['label'], 'assigned-identity')
        self.assertEqual(result['user']['role'], 'researcher')
        self.error('INVALID_INVITATION', self.store.redeem, invite['token'])
        self.assertEqual(self.store.authenticate(result['access_token'])['id'], result['user']['id'])
        with sqlite3.connect(self.path / 'control.sqlite3') as db:
            dump = '\n'.join(db.iterdump())
        for secret in [invite['token'], result['access_token']]:
            self.assertNotIn(secret, dump)
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o700)
        self.assertEqual(os.stat(self.path / 'control.sqlite3').st_mode & 0o777, 0o600)

    def test_invite_and_session_expiry_logout_and_disable(self):
        invitation = self.store.create_invitation('expired', 60)
        self.now += 60
        self.error('INVALID_INVITATION', self.store.redeem, invitation['token'])
        owner, token = self.user()
        self.now += 86400
        self.error('UNAUTHORIZED', self.store.authenticate, token)
        owner2, token2 = self.user('second')
        self.store.logout(token2)
        self.error('UNAUTHORIZED', self.store.authenticate, token2)
        owner3, token3 = self.user('third')
        self.store.disable_user(owner3)
        self.error('UNAUTHORIZED', self.store.authenticate, token3)
        self.error('UNAUTHORIZED', self.store.create_method, owner3, 'disabled')

    def test_invitation_revocation_and_concurrent_one_time_redemption(self):
        invite = self.store.create_invitation('revoked', 60)
        self.store.revoke_invitation(invite['id'])
        self.error('INVALID_INVITATION', self.store.redeem, invite['token'])
        invite = self.store.create_invitation('racing', 60)
        def redeem(_):
            try:
                return Store(self.path, clock=lambda: self.now).redeem(invite['token'])['user']['id']
            except IntakeError as exc:
                return exc.code
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(redeem, range(4)))
        self.assertEqual(results.count('INVALID_INVITATION'), 3)

    def test_restart_retains_ownership_submission_and_audit(self):
        owner, token = self.user()
        manifest = self.manifest(owner)
        submission = self.store.submit(owner, manifest, 'request-1')
        restarted = Store(self.path, clock=lambda: self.now)
        self.assertEqual(restarted.authenticate(token)['id'], owner)
        self.assertEqual(restarted.get_submission(owner, submission['id']), submission)
        self.assertEqual(restarted.list_methods(owner)[0]['id'], manifest['method_id'])
        self.assertEqual(restarted.list_submissions(owner), [submission])
        self.assertEqual(submission['status'], 'SUBMITTED')
        self.assertFalse(submission['official_result'])
        events = restarted.audit_events()
        self.assertIn('SUBMISSION_CREATED', [row['event'] for row in events])
        serialized = json.dumps(events)
        self.assertNotIn(token, serialized)
        self.assertNotIn('github.com', serialized)

    def test_other_owner_cannot_read_or_submit_using_foreign_method(self):
        owner, _ = self.user()
        other, _ = self.user('other')
        manifest = self.manifest(owner)
        submission = self.store.submit(owner, manifest, 'key')
        self.error('NOT_FOUND', self.store.get_submission, other, submission['id'])
        self.error('NOT_FOUND', self.store.submit, other, manifest, 'key')
        self.assertEqual(self.store.list_submissions(other), [])
        self.assertEqual(self.store.list_methods(other), [])

    def test_manifest_rejects_mutable_sources_paths_and_client_scores(self):
        owner, _ = self.user()
        valid = self.manifest(owner)
        for field, value in [('repository', 'http://github.com/a/b'),
                             ('repository', 'https://github.com.evil/a/b'),
                             ('repository', 'https://user:pass@github.com/a/b'),
                             ('repository', 'file:///hidden'), ('commit_sha', 'main'),
                             ('dockerfile', '../Dockerfile'), ('dockerfile', '/Dockerfile'),
                             ('dockerfile', 'a\\Dockerfile'), ('dockerfile', 'a//Dockerfile'),
                             ('entrypoint', 'sh -c anything'), ('protocol_version', 'v1'),
                             ('score', 100), ('inference_time_ns', 1), ('status', 'PUBLISHED')]:
            with self.subTest(field=field, value=value):
                self.error('INVALID_PARAMS', self.store.submit, owner, {**valid, field: value}, 'key')
        self.assertEqual(self.store.list_submissions(owner), [])

    def test_idempotency_survives_restart_and_rejects_payload_reuse(self):
        owner, _ = self.user()
        manifest = self.manifest(owner)
        result = self.store.submit(owner, manifest, 'retry')
        restarted = Store(self.path, clock=lambda: self.now)
        self.assertEqual(restarted.submit(owner, manifest, 'retry'), result)
        self.error('IDEMPOTENCY_CONFLICT', restarted.submit, owner,
                   {**manifest, 'commit_sha': 'b' * 40}, 'retry')
        self.assertEqual(len(restarted.list_submissions(owner)), 1)

    def test_daily_quota_counts_rejected_submissions_not_retry(self):
        owner, _ = self.user()
        manifest = self.manifest(owner)
        for i in range(5):
            result = self.store.submit(owner, manifest, f'key-{i}')
            self.store.review(result['id'], 'start', 0)
            self.store.review(result['id'], 'reject', 1)
        self.error('INTAKE_QUOTA_EXCEEDED', self.store.submit, owner, manifest, 'sixth')
        self.assertEqual(len(self.store.list_submissions(owner)), 5)
        self.assertEqual(self.store.submit(owner, manifest, 'key-0')['status'], 'REJECTED')
        self.now += 86400
        self.assertEqual(self.store.submit(owner, manifest, 'sixth')['status'], 'SUBMITTED')

    def test_pending_quota_is_atomic_under_concurrent_submissions(self):
        owner, _ = self.user()
        manifest = self.manifest(owner)
        def submit(index):
            try:
                return Store(self.path, clock=lambda: self.now).submit(owner, manifest, f'key-{index}')['status']
            except IntakeError as exc:
                return exc.code
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(submit, range(6)))
        self.assertEqual(results.count('SUBMITTED'), 3)
        self.assertEqual(results.count('PENDING_LIMIT_EXCEEDED'), 3)

    def test_review_requires_expected_version_and_finite_transitions(self):
        owner, _ = self.user()
        result = self.store.submit(owner, self.manifest(owner), 'key')
        sid = result['id']
        self.error('INVALID_TRANSITION', self.store.review, sid, 'approve', 0)
        self.assertEqual(self.store.review(sid, 'start', 0)['status'], 'ADMIN_REVIEW')
        self.error('STALE_VERSION', self.store.review, sid, 'approve', 0)
        approved = self.store.review(sid, 'approve', 1)
        self.assertEqual(approved['status'], 'APPROVED_FOR_PREFLIGHT')
        self.assertFalse(approved['leaderboard_eligible'])
        self.error('INVALID_TRANSITION', self.store.review, sid, 'reject', 2)
        self.error('INVALID_PARAMS', self.store.review, sid, 'publish', 2)
        self.error('INVALID_PARAMS', self.store.review, sid, 'start', True)

    def test_method_limit_and_input_bounds(self):
        owner, _ = self.user()
        for i in range(10):
            self.store.create_method(owner, f'method-{i}')
        self.error('METHOD_LIMIT_EXCEEDED', self.store.create_method, owner, 'extra')
        for value in ['', 'x' * 81, '\nmethod', 123]:
            self.error('INVALID_PARAMS', self.store.create_method, owner, value)
        for ttl in [0, -1, True, 604801]:
            self.error('INVALID_PARAMS', self.store.create_invitation, 'label', ttl)

    def test_renewed_invitation_keeps_identity_and_does_not_reset_quota(self):
        owner, old_token = self.user('stable-identity')
        manifest = self.manifest(owner)
        for i in range(3):
            self.store.submit(owner, manifest, f'pending-{i}')
        renewed_owner, renewed_token = self.user('stable-identity')
        self.assertEqual(renewed_owner, owner)
        self.assertNotEqual(renewed_token, old_token)
        self.error('PENDING_LIMIT_EXCEEDED', self.store.submit, renewed_owner, manifest, 'extra')
        self.store.disable_user(owner)
        self.error('UNAUTHORIZED', self.store.authenticate, old_token)
        self.error('UNAUTHORIZED', self.store.authenticate, renewed_token)

    def test_competing_reviews_commit_only_one_transition_and_audit(self):
        owner, _ = self.user()
        sid = self.store.submit(owner, self.manifest(owner), 'first')['id']
        self.store.review(sid, 'start', 0)
        def review(decision):
            try:
                return Store(self.path, clock=lambda: self.now).review(sid, decision, 1)['status']
            except IntakeError as exc:
                return exc.code
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(review, ['approve', 'reject']))
        self.assertEqual(results.count('STALE_VERSION'), 1)
        self.assertEqual(self.store.get_submission(owner, sid)['version'], 2)
        committed = [e for e in self.store.audit_events() if e['event'] in {'REVIEW_APPROVE', 'REVIEW_REJECT'}]
        self.assertEqual(len(committed), 1)

    def test_refuses_shared_state_directory_symlinks_and_unknown_schema(self):
        shared = Path(self.tmp.name) / 'shared'
        shared.mkdir(mode=0o755)
        with self.assertRaises(ValueError):
            Store(shared)
        link = Path(self.tmp.name) / 'link'
        link.symlink_to(self.path)
        with self.assertRaises(ValueError):
            Store(link)
        with sqlite3.connect(self.path / 'control.sqlite3') as db:
            db.execute('PRAGMA user_version=99')
        with self.assertRaises(ValueError):
            Store(self.path)


if __name__ == '__main__':
    unittest.main()
