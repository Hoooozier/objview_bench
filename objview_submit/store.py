"""Local SQLite adapter. Transactions own identity, quotas, state and audit together."""
from contextlib import contextmanager
import hashlib
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

    def list_submissions(self, owner=None):
        with self.transaction() as db:
            if owner is None:
                rows = db.execute('SELECT * FROM submissions ORDER BY created_at DESC,id DESC LIMIT 100')
            else:
                self._active(db, owner)
                rows = db.execute('SELECT * FROM submissions WHERE owner_id=? ORDER BY created_at DESC,id DESC LIMIT 100', (owner,))
            return [_submission(row) for row in rows]

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

    def audit_events(self):
        with self.transaction() as db:
            return [dict(row) for row in db.execute('SELECT * FROM audit_events ORDER BY id DESC LIMIT 100')]
