"""Persistent trusted-smoke queue contract tests."""
from pathlib import Path
import tempfile
import unittest
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from objview_submit.store import Store
from objview_submit.validation import IntakeError
from submission_worker.queue_worker import QueueWorker


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now = 1000.0
        self.store = Store(Path(self.tmp.name) / 'state', clock=lambda: self.now)
        invite = self.store.create_invitation('queue-user', 3600)
        self.owner = self.store.redeem(invite['token'])['user']['id']
        method = self.store.create_method(self.owner, 'trusted dummy')
        self.manifest = {'method_id': method['id'], 'repository': 'https://github.com/example/algorithm',
                         'commit_sha': 'a' * 40, 'dockerfile': 'Dockerfile',
                         'entrypoint': '/opt/objview/run.sh', 'protocol_version': 'objview-interaction/v1'}

    def _approved_job(self):
        row = self.store.submit(self.owner, self.manifest, 'queue-key')
        self.store.review(row['id'], 'start', 0)
        self.store.review(row['id'], 'approve', 1)
        self.assertEqual(self.store.get_job(self.owner, row['id'])['state'], 'COLLECTING')
        self.store.release_batch()
        return self.store.get_job(self.owner, row['id'])

    @staticmethod
    def result(total=3):
        return {'timing_source': 'benchmark_monotonic_clock', 'total_decision_wall_time_ns': total,
                'steps': [{'step_index': 0, 'decision_wall_time_ns': total}]}

    def test_batch_release_enqueues_fixed_contract_and_persists_restart(self):
        job = self._approved_job()
        self.assertEqual(job['state'], 'QUEUED')
        claim = self.store.claim_job('worker-a', lease_seconds=10)
        self.assertEqual(claim['kind'], 'trusted_smoke_v1')
        self.assertNotIn('repository', claim)
        restarted = Store(Path(self.tmp.name) / 'state', clock=lambda: self.now)
        self.assertEqual(restarted.get_job(self.owner, claim['submission_id'])['state'], 'RUNNING')

    def test_expired_claim_is_recovered_and_stale_worker_cannot_finish(self):
        job = self._approved_job()
        first = self.store.claim_job('worker-a', lease_seconds=10)
        self.now += 11
        second = self.store.claim_job('worker-b', lease_seconds=10)
        self.assertEqual(second['job_id'], first['job_id'])
        with self.assertRaises(IntakeError) as raised:
            self.store.finish_job(first['job_id'], 'worker-a', success=False,
                                  error_code='JOB_TIMEOUT')
        self.assertEqual(raised.exception.code, 'STALE_WORKER')
        done = self.store.finish_job(second['job_id'], 'worker-b', success=True,
                                     result=self.result())
        self.assertEqual(done['state'], 'SUCCEEDED')
        self.assertEqual(self.store.finish_job(second['job_id'], 'worker-b', success=True,
                                               result=self.result())['state'], 'SUCCEEDED')
        with self.assertRaises(IntakeError) as raised:
            self.store.finish_job(second['job_id'], 'worker-a', success=False,
                                  error_code='JOB_TIMEOUT')
        self.assertEqual(raised.exception.code, 'STALE_WORKER')

    def test_invalid_or_untrusted_result_fails_closed(self):
        job = self._approved_job()
        claim = self.store.claim_job('worker-a', lease_seconds=10)
        with self.assertRaises(IntakeError) as raised:
            self.store.finish_job(claim['job_id'], 'worker-a', success=True,
                                  result={'total_decision_wall_time_ns': 1})
        self.assertEqual(raised.exception.code, 'INVALID_RESULT')
        with self.assertRaises(IntakeError) as raised:
            self.store.finish_job(claim['job_id'], 'worker-a', success=False,
                                  error_code='USER_SUPPLIED_SCORE')
        self.assertEqual(raised.exception.code, 'INVALID_RESULT')

    def test_worker_adapter_runs_only_fixed_trusted_kind(self):
        self._approved_job()
        seen = []
        def executor(claim):
            seen.append(claim)
            return {'status': 'SUCCEEDED', 'result': self.result(7)}
        worker = QueueWorker(self.store, 'worker-a', executor, lease_seconds=10)
        result = worker.run_once()
        self.assertEqual(result['state'], 'SUCCEEDED')
        self.assertEqual(seen[0]['kind'], 'trusted_smoke_v1')

    def test_approval_waits_until_one_atomic_batch_release(self):
        # Build two approved submissions before releasing; both receive the same
        # batch timestamp and neither is claimable during collection.
        rows = []
        for key in ('batch-a', 'batch-b'):
            row = self.store.submit(self.owner, self.manifest, key)
            self.store.review(row['id'], 'start', 0)
            self.store.review(row['id'], 'approve', 1)
            rows.append(row)
        self.assertEqual(self.store.get_job(self.owner, rows[0]['id'])['state'], 'COLLECTING')
        self.assertIsNone(self.store.claim_job('worker-a'))
        batch = self.store.release_batch()
        self.assertEqual(batch['job_count'], 2)
        self.assertEqual(len(self.store.list_batches()), 1)
        claims = [self.store.claim_job('worker-a'), self.store.claim_job('worker-b')]
        self.assertEqual({claim['batch_id'] for claim in claims}, {batch['id']})
        with self.assertRaises(IntakeError) as raised:
            self.store.release_batch()
        self.assertEqual(raised.exception.code, 'NO_ELIGIBLE_SUBMISSIONS')

    def test_concurrent_release_and_later_approval_use_distinct_batches(self):
        rows = [self.store.submit(self.owner, self.manifest, key) for key in ('early', 'later')]
        for row in rows:
            self.store.review(row['id'], 'start', 0)
        self.store.review(rows[0]['id'], 'approve', 1)
        def release():
            store = Store(Path(self.tmp.name) / 'state', clock=lambda: self.now)
            try:
                return store.release_batch()['id']
            except IntakeError as exc:
                return exc.code
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: release(), range(2)))
        self.assertEqual(results.count('NO_ELIGIBLE_SUBMISSIONS'), 1)
        self.store.review(rows[1]['id'], 'approve', 1)
        self.assertEqual(self.store.get_job(self.owner, rows[1]['id'])['state'], 'COLLECTING')
        batch = self.store.release_batch()
        self.assertNotIn(batch['id'], results)
        self.assertEqual(batch['job_count'], 1)

    def test_legacy_unbatched_job_is_preserved_but_requires_release(self):
        row = self.store.submit(self.owner, self.manifest, 'legacy')
        self.store.review(row['id'], 'start', 0)
        self.store.review(row['id'], 'approve', 1)
        with self.store.transaction() as db:
            db.execute('DROP TABLE evaluation_jobs')
            db.execute('''CREATE TABLE evaluation_jobs (
                id TEXT PRIMARY KEY, submission_id TEXT UNIQUE, state TEXT,
                worker_id TEXT, lease_expires_at REAL, attempt INTEGER DEFAULT 0,
                result_json TEXT, error_code TEXT, created_at REAL, updated_at REAL)''')
            db.execute("INSERT INTO evaluation_jobs (id,submission_id,state,created_at,updated_at) VALUES (?,?,'QUEUED',?,?)",
                       ('legacy-id', row['id'], self.now, self.now))
        restarted = Store(Path(self.tmp.name) / 'state', clock=lambda: self.now)
        self.assertIsNone(restarted.claim_job('worker-a'))
        self.assertEqual(restarted.get_job(self.owner, row['id'])['state'], 'COLLECTING')
        batch = restarted.release_batch()
        claim = restarted.claim_job('worker-a')
        self.assertEqual(claim['job_id'], 'legacy-id')
        self.assertEqual(claim['batch_id'], batch['id'])


if __name__ == '__main__':
    unittest.main()
