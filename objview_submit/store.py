"""Local SQLite adapter. Transactions own identity, quotas, state and audit together."""
from contextlib import contextmanager
import hashlib
import base64
import binascii
import json
import os
from pathlib import Path
import secrets
import sqlite3
import stat
import time
import uuid

from .validation import IntakeError, manifest_fields, request_key, text_field


SCHEMA = '''
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY, label TEXT NOT NULL UNIQUE, status TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS invitations (
    id TEXT PRIMARY KEY, token_hash TEXT NOT NULL UNIQUE, label TEXT NOT NULL,
    expires_at REAL NOT NULL, used_at REAL, revoked INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES users(id),
    expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS methods (
    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES users(id),
    name TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS submissions (
    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES users(id),
    method_id TEXT NOT NULL REFERENCES methods(id), request_key TEXT NOT NULL,
    manifest_json TEXT NOT NULL, status TEXT NOT NULL, version INTEGER NOT NULL,
    created_at REAL NOT NULL, updated_at REAL NOT NULL, UNIQUE(owner_id, request_key)
);
CREATE INDEX IF NOT EXISTS submissions_owner_time ON submissions(owner_id, created_at);
CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY, actor_id TEXT NOT NULL, event TEXT NOT NULL,
    subject_id TEXT NOT NULL, created_at REAL NOT NULL
);
PRAGMA user_version=1;
COMMIT;
'''

JOB_SCHEMA = '''
CREATE TABLE IF NOT EXISTS evaluation_batches (
    id TEXT PRIMARY KEY, state TEXT NOT NULL, created_by TEXT NOT NULL,
    created_at REAL NOT NULL, released_at REAL
);
CREATE INDEX IF NOT EXISTS evaluation_batches_time ON evaluation_batches(created_at DESC, id DESC);
CREATE TABLE IF NOT EXISTS evaluation_jobs (
    id TEXT PRIMARY KEY, submission_id TEXT NOT NULL UNIQUE REFERENCES submissions(id),
    batch_id TEXT REFERENCES evaluation_batches(id),
    state TEXT NOT NULL, worker_id TEXT, lease_expires_at REAL,
    attempt INTEGER NOT NULL DEFAULT 0, result_json TEXT, error_code TEXT,
    created_at REAL NOT NULL, updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS evaluation_jobs_claim ON evaluation_jobs(state, lease_expires_at, created_at);
'''


def _digest(token):
    if not isinstance(token, str) or not 32 <= len(token) <= 256 or not token.isascii():
        raise IntakeError('UNAUTHORIZED', 401)
    return hashlib.sha256(token.encode('ascii')).hexdigest()


def _id():
    return uuid.uuid4().hex


def _user(row):
    return {'id': row['id'], 'label': row['label'], 'role': 'researcher', 'status': row['status']}


def _submission(row):
    return {'id': row['id'], 'owner_id': row['owner_id'], 'manifest': json.loads(row['manifest_json']),
            'status': row['status'], 'version': row['version'], 'created_at': row['created_at'],
            'updated_at': row['updated_at'], 'mode': 'development_intake',
            'official_result': False, 'leaderboard_eligible': False}


def _job(row):
    result = None if row['result_json'] is None else json.loads(row['result_json'])
    value = {'id': row['id'], 'submission_id': row['submission_id'], 'batch_id': row['batch_id'],
             'state': row['state'],
             'attempt': row['attempt'], 'created_at': row['created_at'], 'updated_at': row['updated_at'],
             'error_code': row['error_code'], 'official_result': False,
             'leaderboard_eligible': False}
    if result is not None:
        value['result'] = result
    return value


def _batch(row):
    return {'id': row['id'], 'state': row['state'], 'created_by': row['created_by'],
            'created_at': row['created_at'], 'released_at': row['released_at'],
            'official_result': False, 'leaderboard_eligible': False}


def _safe_worker_result(result):
    """Allow only benchmark-owned timing fields into the persistent result."""
    if not isinstance(result, dict) or set(result) != {'timing_source', 'total_decision_wall_time_ns', 'steps'}:
        raise IntakeError('INVALID_RESULT')
    total = result['total_decision_wall_time_ns']
    steps = result['steps']
    if (result['timing_source'] != 'benchmark_monotonic_clock' or type(total) is not int or total < 0
            or not isinstance(steps, list) or not 1 <= len(steps) <= 32):
        raise IntakeError('INVALID_RESULT')
    clean_steps = []
    for index, step in enumerate(steps):
        if (not isinstance(step, dict) or set(step) != {'step_index', 'decision_wall_time_ns'}
                or step['step_index'] != index or type(step['decision_wall_time_ns']) is not int
                or step['decision_wall_time_ns'] < 0):
            raise IntakeError('INVALID_RESULT')
        clean_steps.append({'step_index': index, 'decision_wall_time_ns': step['decision_wall_time_ns']})
    if sum(step['decision_wall_time_ns'] for step in clean_steps) != total:
        raise IntakeError('INVALID_RESULT')
    return {'timing_source': 'benchmark_monotonic_clock',
            'total_decision_wall_time_ns': total, 'steps': clean_steps}


def _encode_cursor(kind, values, scope=''):
    """Encode a keyset cursor without exposing a second authentication token.

    Cursors are deliberately only navigation hints.  Every query still applies
    its owner/status filter, so a copied cursor cannot grant access to another
    account's rows.
    """
    payload = json.dumps({'v': 1, 'kind': kind, 'scope': scope, 'values': values},
                         separators=(',', ':'), sort_keys=True).encode('ascii')
    return base64.urlsafe_b64encode(payload).rstrip(b'=').decode('ascii')


def _decode_cursor(cursor, kind, scope=''):
    if cursor is None:
        return None
    if not isinstance(cursor, str) or not 1 <= len(cursor) <= 512:
        raise IntakeError('INVALID_CURSOR')
    try:
        padded = cursor + '=' * (-len(cursor) % 4)
        payload = base64.urlsafe_b64decode(padded.encode('ascii'))
        value = json.loads(payload.decode('ascii'))
    except (ValueError, UnicodeError, json.JSONDecodeError, binascii.Error):
        raise IntakeError('INVALID_CURSOR') from None
    if (not isinstance(value, dict) or value.get('v') != 1 or value.get('kind') != kind
            or value.get('scope') != scope or not isinstance(value.get('values'), list)):
        raise IntakeError('INVALID_CURSOR')
    return value['values']


class Store:
    def __init__(self, state_dir, clock=time.time):
        self.clock = clock
        directory = Path(state_dir)
        directory.mkdir(mode=0o700, parents=False, exist_ok=True)
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError('State directory must be owned by this user with mode 0700')
        self.path = directory / 'control.sqlite3'
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise ValueError('Database must be a private regular file')
        finally:
            os.close(fd)
        db = self._connect()
        try:
            version = db.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0, 1):
                raise ValueError('Unsupported database schema; no automatic downgrade')
            if version == 0:
                db.executescript(SCHEMA)
            # Jobs were added after schema v1.  This additive table is safe for
            # existing local databases; unknown future schema versions still
            # fail closed above.
            db.executescript(JOB_SCHEMA)
            columns = {row['name'] for row in db.execute('PRAGMA table_info(evaluation_jobs)')}
            if 'batch_id' not in columns:
                db.execute('ALTER TABLE evaluation_jobs ADD COLUMN batch_id TEXT REFERENCES evaluation_batches(id)')
            # Preserve legacy job IDs/history. Unbatched queued jobs are inert
            # until an operator explicitly releases them into a batch.
        finally:
            db.close()

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        return db

    @contextmanager
    def transaction(self):
        db = self._connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _audit(self, db, actor, event, subject):
        db.execute('INSERT INTO audit_events(actor_id,event,subject_id,created_at) VALUES (?,?,?,?)',
                   (actor, event, subject, self.clock()))

    @staticmethod
    def _active(db, owner):
        row = db.execute('SELECT * FROM users WHERE id=? AND status=?', (owner, 'active')).fetchone()
        if row is None:
            raise IntakeError('UNAUTHORIZED', 401)
        return row

    def create_invitation(self, label, ttl_seconds):
        text_field(label)
        if type(ttl_seconds) is not int or not 1 <= ttl_seconds <= 604800:
            raise IntakeError('INVALID_PARAMS')
        token, iid = secrets.token_urlsafe(32), _id()
        expires = self.clock() + ttl_seconds
        with self.transaction() as db:
            count = db.execute('SELECT COUNT(*) FROM invitations WHERE used_at IS NULL AND revoked=0 AND expires_at>?',
                               (self.clock(),)).fetchone()[0]
            if count >= 100:
                raise IntakeError('INVITATION_LIMIT_EXCEEDED', 429)
            db.execute('INSERT INTO invitations(id,token_hash,label,expires_at) VALUES (?,?,?,?)',
                       (iid, _digest(token), label, expires))
            self._audit(db, 'admin', 'INVITATION_CREATED', iid)
        return {'id': iid, 'token': token, 'expires_at': expires}

    def redeem(self, token):
        try:
            digest = _digest(token)
        except IntakeError:
            raise IntakeError('INVALID_INVITATION', 401) from None
        with self.transaction() as db:
            now = self.clock()
            invite = db.execute('SELECT * FROM invitations WHERE token_hash=? AND used_at IS NULL AND revoked=0 AND expires_at>?',
                                (digest, now)).fetchone()
            if invite is None:
                raise IntakeError('INVALID_INVITATION', 401)
            row = db.execute('SELECT * FROM users WHERE label=?', (invite['label'],)).fetchone()
            if row is None:
                owner = _id()
                db.execute('INSERT INTO users VALUES (?,?,?,?)', (owner, invite['label'], 'active', now))
                row = self._active(db, owner)
            elif row['status'] != 'active':
                raise IntakeError('INVALID_INVITATION', 401)
            token = secrets.token_urlsafe(32)
            db.execute('INSERT INTO sessions VALUES (?,?,?)', (_digest(token), row['id'], now + 86400))
            db.execute('UPDATE invitations SET used_at=? WHERE id=?', (now, invite['id']))
            self._audit(db, row['id'], 'INVITATION_REDEEMED', invite['id'])
            return {'access_token': token, 'token_type': 'Bearer', 'expires_at': now + 86400, 'user': _user(row)}

    def authenticate(self, token):
        digest = _digest(token)
        with self.transaction() as db:
            session = db.execute('SELECT * FROM sessions WHERE token_hash=? AND expires_at>?',
                                 (digest, self.clock())).fetchone()
            if session is None:
                raise IntakeError('UNAUTHORIZED', 401)
            return _user(self._active(db, session['owner_id']))

    def logout(self, token):
        digest = _digest(token)
        with self.transaction() as db:
            row = db.execute('SELECT * FROM sessions WHERE token_hash=?', (digest,)).fetchone()
            if row is not None:
                db.execute('DELETE FROM sessions WHERE token_hash=?', (digest,))
                self._audit(db, row['owner_id'], 'SESSION_REVOKED', row['owner_id'])
        return {'status': 'revoked'}

    def revoke_invitation(self, iid):
        with self.transaction() as db:
            if not db.execute('SELECT 1 FROM invitations WHERE id=?', (iid,)).fetchone():
                raise IntakeError('NOT_FOUND', 404)
            db.execute('UPDATE invitations SET revoked=1 WHERE id=?', (iid,))
            self._audit(db, 'admin', 'INVITATION_REVOKED', iid)
        return {'status': 'revoked'}

    def disable_user(self, owner):
        with self.transaction() as db:
            if not db.execute('SELECT 1 FROM users WHERE id=?', (owner,)).fetchone():
                raise IntakeError('NOT_FOUND', 404)
            db.execute('UPDATE users SET status=? WHERE id=?', ('disabled', owner))
            db.execute('DELETE FROM sessions WHERE owner_id=?', (owner,))
            self._audit(db, 'admin', 'USER_DISABLED', owner)
        return {'status': 'disabled'}

    def create_method(self, owner, name):
        text_field(name)
        with self.transaction() as db:
            self._active(db, owner)
            if db.execute('SELECT COUNT(*) FROM methods WHERE owner_id=?', (owner,)).fetchone()[0] >= 10:
                raise IntakeError('METHOD_LIMIT_EXCEEDED', 429)
            mid = _id()
            db.execute('INSERT INTO methods VALUES (?,?,?,?)', (mid, owner, name, self.clock()))
            self._audit(db, owner, 'METHOD_CREATED', mid)
        return {'id': mid, 'name': name}

    def list_methods(self, owner):
        with self.transaction() as db:
            self._active(db, owner)
            return [dict(r) for r in db.execute('SELECT id,name FROM methods WHERE owner_id=? ORDER BY created_at,id', (owner,))]

    def submit(self, owner, manifest, key):
        data = manifest_fields(manifest)
        request_key(key)
        canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
        with self.transaction() as db:
            self._active(db, owner)
            old = db.execute('SELECT * FROM submissions WHERE owner_id=? AND request_key=?', (owner, key)).fetchone()
            if old is not None:
                if old['manifest_json'] != canonical:
                    raise IntakeError('IDEMPOTENCY_CONFLICT', 409)
                return _submission(old)
            if not db.execute('SELECT 1 FROM methods WHERE id=? AND owner_id=?', (data['method_id'], owner)).fetchone():
                raise IntakeError('NOT_FOUND', 404)
            now = self.clock()
            if db.execute('SELECT COUNT(*) FROM submissions WHERE owner_id=? AND created_at>?', (owner, now - 86400)).fetchone()[0] >= 5:
                raise IntakeError('INTAKE_QUOTA_EXCEEDED', 429)
            if db.execute("SELECT COUNT(*) FROM submissions WHERE owner_id=? AND status!='REJECTED'", (owner,)).fetchone()[0] >= 3:
                raise IntakeError('PENDING_LIMIT_EXCEEDED', 429)
            sid = _id()
            db.execute('INSERT INTO submissions VALUES (?,?,?,?,?,?,?,?,?)',
                       (sid, owner, data['method_id'], key, canonical, 'SUBMITTED', 0, now, now))
            self._audit(db, owner, 'SUBMISSION_CREATED', sid)
            return _submission(db.execute('SELECT * FROM submissions WHERE id=?', (sid,)).fetchone())

    def get_submission(self, owner, sid):
        with self.transaction() as db:
            self._active(db, owner)
            row = db.execute('SELECT * FROM submissions WHERE id=? AND owner_id=?', (sid, owner)).fetchone()
            if row is None:
                raise IntakeError('NOT_FOUND', 404)
            return _submission(row)

    @staticmethod
    def _page_limit(limit):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise IntakeError('INVALID_PARAMS')
        return limit

    def list_submissions_page(self, owner=None, *, limit=50, cursor=None, status=None):
        """Return one stable keyset page and an opaque continuation cursor."""
        limit = self._page_limit(limit)
        if status is not None and (not isinstance(status, str) or status not in {
                'SUBMITTED', 'ADMIN_REVIEW', 'APPROVED_FOR_PREFLIGHT', 'REJECTED'}):
            raise IntakeError('INVALID_PARAMS')
        scope = owner or 'admin'
        values = _decode_cursor(cursor, 'submissions', scope + '|' + (status or ''))
        with self.transaction() as db:
            if owner is not None:
                self._active(db, owner)
            clauses, params = [], []
            if owner is not None:
                clauses.append('owner_id=?')
                params.append(owner)
            if status is not None:
                clauses.append('status=?')
                params.append(status)
            if values is not None:
                if (len(values) != 2 or not isinstance(values[0], (int, float))
                        or isinstance(values[0], bool) or not isinstance(values[1], str)):
                    raise IntakeError('INVALID_CURSOR')
                clauses.append('(created_at < ? OR (created_at = ? AND id < ?))')
                params.extend((values[0], values[0], values[1]))
            where = (' WHERE ' + ' AND '.join(clauses)) if clauses else ''
            rows = db.execute('SELECT * FROM submissions' + where
                              + ' ORDER BY created_at DESC,id DESC LIMIT ?', (*params, limit + 1)).fetchall()
            page = rows[:limit]
            next_cursor = None
            if len(rows) > limit:
                last = page[-1]
                next_cursor = _encode_cursor('submissions', [last['created_at'], last['id']],
                                             scope + '|' + (status or ''))
            return [_submission(row) for row in page], next_cursor

    def list_submissions(self, owner=None):
        # Compatibility helper for local callers; HTTP clients use the paged API.
        return self.list_submissions_page(owner, limit=100)[0]

    @staticmethod
    def _worker_id(worker_id):
        if (not isinstance(worker_id, str) or not 1 <= len(worker_id) <= 80
                or not all(char.isalnum() or char in '._-' for char in worker_id)):
            raise IntakeError('INVALID_PARAMS')
        return worker_id

    def get_job(self, owner, submission_id):
        with self.transaction() as db:
            self._active(db, owner)
            row = db.execute('''SELECT j.* FROM evaluation_jobs j
                                JOIN submissions s ON s.id=j.submission_id
                                WHERE j.submission_id=? AND s.owner_id=?''',
                             (submission_id, owner)).fetchone()
            if row is None:
                submission = db.execute('SELECT status FROM submissions WHERE id=? AND owner_id=?',
                                         (submission_id, owner)).fetchone()
                if submission is None:
                    raise IntakeError('NOT_FOUND', 404)
                if submission['status'] == 'APPROVED_FOR_PREFLIGHT':
                    return {'submission_id': submission_id, 'batch_id': None, 'state': 'COLLECTING',
                            'attempt': 0, 'error_code': None, 'official_result': False,
                            'leaderboard_eligible': False}
                raise IntakeError('NOT_FOUND', 404)
            value = _job(row)
            if row['batch_id'] is None and row['state'] == 'QUEUED':
                value['state'] = 'COLLECTING'
            return value

    def list_batches(self, *, limit=50):
        limit = self._page_limit(limit)
        with self.transaction() as db:
            rows = db.execute('SELECT * FROM evaluation_batches ORDER BY created_at DESC,id DESC LIMIT ?',
                              (limit,)).fetchall()
            return [_batch(row) for row in rows]

    def release_batch(self, actor='admin'):
        if not isinstance(actor, str) or actor != 'admin':
            raise IntakeError('INVALID_PARAMS')
        with self.transaction() as db:
            now = self.clock()
            eligible = db.execute('''SELECT s.id, j.id AS job_id FROM submissions s
                                     LEFT JOIN evaluation_jobs j ON j.submission_id=s.id
                                     WHERE s.status='APPROVED_FOR_PREFLIGHT'
                                       AND (j.id IS NULL OR (j.batch_id IS NULL AND j.state='QUEUED'))
                                     ORDER BY s.created_at ASC,s.id ASC''').fetchall()
            if not eligible:
                raise IntakeError('NO_ELIGIBLE_SUBMISSIONS', 409)
            batch_id = _id()
            db.execute('''INSERT INTO evaluation_batches
                          (id,state,created_by,created_at,released_at)
                          VALUES (?, 'RELEASED', ?, ?, ?)''',
                       (batch_id, actor, now, now))
            for row in eligible:
                if row['job_id'] is not None:
                    db.execute('UPDATE evaluation_jobs SET batch_id=?,updated_at=? WHERE id=?',
                               (batch_id, now, row['job_id']))
                    continue
                db.execute('''INSERT INTO evaluation_jobs
                              (id,submission_id,batch_id,state,attempt,created_at,updated_at)
                              VALUES (?, ?, ?, 'QUEUED', 0, ?, ?)''',
                           (_id(), row['id'], batch_id, now, now))
            self._audit(db, actor, 'BATCH_RELEASED', batch_id)
            batch = _batch(db.execute('SELECT * FROM evaluation_batches WHERE id=?', (batch_id,)).fetchone())
            batch['job_count'] = len(eligible)
            return batch

    def _requeue_expired(self, db, now):
        db.execute('''UPDATE evaluation_jobs
                      SET state='QUEUED', worker_id=NULL, lease_expires_at=NULL,
                          updated_at=?
                      WHERE state='RUNNING' AND lease_expires_at <= ?''', (now, now))

    def claim_job(self, worker_id, lease_seconds=60):
        worker_id = self._worker_id(worker_id)
        if type(lease_seconds) is not int or not 1 <= lease_seconds <= 3600:
            raise IntakeError('INVALID_PARAMS')
        with self.transaction() as db:
            now = self.clock()
            self._requeue_expired(db, now)
            row = db.execute('''SELECT j.id, j.submission_id, j.batch_id, j.attempt
                                FROM evaluation_jobs j
                                JOIN evaluation_batches b ON b.id=j.batch_id AND b.state='RELEASED'
                                WHERE j.state='QUEUED'
                                ORDER BY j.created_at ASC, j.id ASC LIMIT 1''').fetchone()
            if row is None:
                return None
            updated = db.execute('''UPDATE evaluation_jobs
                                    SET state='RUNNING', worker_id=?, lease_expires_at=?,
                                        attempt=attempt+1, updated_at=?
                                    WHERE id=? AND state='QUEUED' ''',
                                 (worker_id, now + lease_seconds, now, row['id']))
            if updated.rowcount != 1:
                return None
            claimed = db.execute('SELECT * FROM evaluation_jobs WHERE id=?', (row['id'],)).fetchone()
            # The worker receives only a fixed trusted smoke contract.  Intake
            # repository/command fields never cross this boundary.
            return {'job_id': claimed['id'], 'submission_id': claimed['submission_id'],
                    'batch_id': claimed['batch_id'],
                    'kind': 'trusted_smoke_v1', 'attempt': claimed['attempt'],
                    'lease_expires_at': claimed['lease_expires_at']}

    def heartbeat_job(self, job_id, worker_id, lease_seconds=60):
        worker_id = self._worker_id(worker_id)
        if not isinstance(job_id, str) or not 1 <= len(job_id) <= 64:
            raise IntakeError('INVALID_PARAMS')
        if type(lease_seconds) is not int or not 1 <= lease_seconds <= 3600:
            raise IntakeError('INVALID_PARAMS')
        with self.transaction() as db:
            now = self.clock()
            updated = db.execute('''UPDATE evaluation_jobs
                                    SET lease_expires_at=?, updated_at=?
                                    WHERE id=? AND state='RUNNING' AND worker_id=?
                                      AND lease_expires_at > ?''',
                                 (now + lease_seconds, now, job_id, worker_id, now))
            if updated.rowcount != 1:
                raise IntakeError('STALE_WORKER', 409)
            return {'status': 'alive', 'lease_expires_at': now + lease_seconds}

    def finish_job(self, job_id, worker_id, *, success, result=None, error_code=None):
        worker_id = self._worker_id(worker_id)
        if not isinstance(job_id, str) or not 1 <= len(job_id) <= 64 or type(success) is not bool:
            raise IntakeError('INVALID_PARAMS')
        if success:
            clean = _safe_worker_result(result)
            state, error_code = 'SUCCEEDED', None
        else:
            clean = None
            if (not isinstance(error_code, str)
                    or error_code not in {'JOB_TIMEOUT', 'BENCHMARK_FAILED', 'ALGORITHM_CRASHED',
                                          'INFRASTRUCTURE_FAILED', 'INVALID_SUMMARY', 'CLEANUP_FAILED'}):
                raise IntakeError('INVALID_RESULT')
            state = 'FAILED'
        encoded = None if clean is None else json.dumps(clean, sort_keys=True, separators=(',', ':'))
        with self.transaction() as db:
            now = self.clock()
            row = db.execute('SELECT * FROM evaluation_jobs WHERE id=?', (job_id,)).fetchone()
            if row is None:
                raise IntakeError('NOT_FOUND', 404)
            if row['state'] in {'SUCCEEDED', 'FAILED'}:
                if row['worker_id'] != worker_id:
                    raise IntakeError('STALE_WORKER', 409)
                if row['result_json'] == encoded and row['error_code'] == error_code:
                    return _job(row)
                raise IntakeError('JOB_TERMINAL', 409)
            if (row['state'] != 'RUNNING' or row['worker_id'] != worker_id
                    or row['lease_expires_at'] is None or row['lease_expires_at'] <= now):
                raise IntakeError('STALE_WORKER', 409)
            db.execute('''UPDATE evaluation_jobs
                          SET state=?, result_json=?, error_code=?, lease_expires_at=NULL, updated_at=?
                          WHERE id=? AND state='RUNNING' AND worker_id=?''',
                       (state, encoded, error_code, now, job_id, worker_id))
            return _job(db.execute('SELECT * FROM evaluation_jobs WHERE id=?', (job_id,)).fetchone())

    def recover_jobs(self):
        with self.transaction() as db:
            now = self.clock()
            self._requeue_expired(db, now)
            return db.execute("""SELECT COUNT(*) FROM evaluation_jobs j
                                JOIN evaluation_batches b ON b.id=j.batch_id AND b.state='RELEASED'
                                WHERE j.state='QUEUED'""").fetchone()[0]

    def review(self, sid, decision, expected_version):
        transitions = {'start': ('SUBMITTED', 'ADMIN_REVIEW'),
                       'approve': ('ADMIN_REVIEW', 'APPROVED_FOR_PREFLIGHT'),
                       'reject': ('ADMIN_REVIEW', 'REJECTED')}
        if (not isinstance(decision, str) or decision not in transitions
                or type(expected_version) is not int or not 0 <= expected_version <= 100):
            raise IntakeError('INVALID_PARAMS')
        with self.transaction() as db:
            row = db.execute('SELECT * FROM submissions WHERE id=?', (sid,)).fetchone()
            if row is None:
                raise IntakeError('NOT_FOUND', 404)
            if row['version'] != expected_version:
                raise IntakeError('STALE_VERSION', 409)
            before, after = transitions[decision]
            if row['status'] != before:
                raise IntakeError('INVALID_TRANSITION', 409)
            db.execute('UPDATE submissions SET status=?,version=version+1,updated_at=? WHERE id=?',
                       (after, self.clock(), sid))
            self._audit(db, 'admin', 'REVIEW_' + decision.upper(), sid)
            return _submission(db.execute('SELECT * FROM submissions WHERE id=?', (sid,)).fetchone())

    def audit_events_page(self, *, limit=50, cursor=None):
        limit = self._page_limit(limit)
        values = _decode_cursor(cursor, 'audit')
        with self.transaction() as db:
            params = []
            where = ''
            if values is not None:
                if len(values) != 1 or type(values[0]) is not int or values[0] < 1:
                    raise IntakeError('INVALID_CURSOR')
                where = ' WHERE id < ?'
                params.append(values[0])
            rows = db.execute('SELECT id,actor_id,event,subject_id,created_at FROM audit_events'
                              + where + ' ORDER BY id DESC LIMIT ?', (*params, limit + 1)).fetchall()
            page = rows[:limit]
            next_cursor = None
            if len(rows) > limit:
                next_cursor = _encode_cursor('audit', [page[-1]['id']])
            return [dict(row) for row in page], next_cursor

    def audit_events(self):
        return self.audit_events_page(limit=100)[0]
