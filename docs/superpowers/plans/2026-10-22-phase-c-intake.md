# Phase C1: invited submission intake implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement task-by-task. Track the checkboxes below.

**Goal:** Run a Docker-free, persistent, invited-user submission/review API without executing submitted code.

**Architecture:** Independent `objview_submit` package, no benchmark/worker imports. SQLite is a local-development adapter, not a replacement for the planned production PostgreSQL service. A manually delivered one-time invitation creates an expiring bearer session for an administrator-assigned identity; this is explicitly not GitHub OAuth.

**Tech Stack:** Python 3.10, sqlite3, FastAPI, Uvicorn, unittest, httpx (existing development environment).

**Spec:** `docs/superpowers/specs/2026-10-21-objview-submission-platform-design.md`, sections 5, 6, 9, 11–15, 17 (Phase C).

## Global constraints and staged scope

- Keep github.io unchanged; API has no Docker socket, hidden dataset, shell execution or repository fetching.
- Local development only, bind CLI to 127.0.0.1; no public registration. No fake GitHub identity or implicit administrator account.
- No official evaluations, results or leaderboard publication. Intake success is not build/run success.
- Store only SHA256 digests of invitation/session credentials; raw tokens returned once, no request-body logging. Private state directory 0700, DB 0600.
- Submitted source is metadata: HTTPS github.com owner/repository, immutable 40-hex commit, relative Dockerfile, fixed `/opt/objview/run.sh`, protocol `objview-interaction/v1`. Reject additional manifest fields including user-provided timing/results.
- Submission states: `SUBMITTED -> ADMIN_REVIEW -> APPROVED_FOR_PREFLIGHT | REJECTED`. The approved state is a development holding state, not an executing job. No automatic builds or retries.
- Development intake limits: 5 new submissions per account in a rolling 24h, at most 3 non-rejected submissions, 10 methods/account. Transactions include checks and inserts. These are NOT the formal per-image/method-lineage hidden-test quotas; hidden execution remains unavailable.
- Client retries use owner-scoped idempotency keys. Administrator transitions use expected-version compare-and-swap. Audit events contain IDs and fixed event codes only, never raw credentials/manifests/logs.
- Genuine GitHub OAuth, PostgreSQL migrations, archive/object storage, build queue, hidden evaluation quotas, frontend and public deployment are separate milestones. No claims that this increment implements them.

## Task 1: persistent identity, intake and review core

**Files:** create `objview_submit/{__init__,store,validation}.py`; test `benchmark/tests/test_intake_store.py`.

**Interfaces:** `Store(state_dir, clock=time.time)`; `create_invitation(label, ttl_seconds)`, `redeem(token)`, `authenticate(token)`, `logout(token)`, `revoke_invitation(id)`, `disable_user(id)`; `create_method(owner, name)`, `list_methods(owner)`; `submit(owner, manifest, key)`, `list_submissions(owner=None)`, `get_submission(owner, id)`, `review(id, decision, expected_version)`; `audit_events()`. Methods return JSON-safe allowlisted dictionaries; failures are `IntakeError(code, status)`.

- [x] Write real SQLite tests for one-use concurrent redemption, expiry, restart, hashing, revocation, owner isolation, immutable metadata, daily/pending limits, retry conflict and concurrent review.
- [x] Run `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s benchmark/tests -p 'test_intake_store.py' -v`; observe missing implementation.
- [x] Implement private directory/database setup, schema version check, `BEGIN IMMEDIATE` transactions; validate before insert and roll back on errors. Example atomic shape:

```python
with self.transaction() as db:
    existing = db.execute('SELECT * FROM submissions WHERE owner_id=? AND request_key=?', (owner, key)).fetchone()
    if existing is not None:
        if existing['manifest_json'] != canonical_manifest:
            raise IntakeError('IDEMPOTENCY_CONFLICT', 409)
        return public_submission(existing)
```

- [x] Run store tests; include actual independent Store instances/threads to exercise DB locking, not mocked persistence.

## Task 2: isolated local HTTP service

**Files:** create `objview_submit/{app,__main__}.py`, `objview_submit/requirements.txt`; test `benchmark/tests/test_intake_api.py`.

**Interfaces:** `create_app(store, admin_token, *, development_mode=False)` must refuse operation unless development mode explicitly enabled. CLI: `python3 -m objview_submit --development-only --state-dir PATH --port 8090`; administrator token only from `OBJVIEW_SUBMIT_ADMIN_TOKEN`. Uvicorn localhost, no proxy-header trust/access logs.

Routes: unauthenticated `GET /healthz`, `POST /v1/invitations/redeem`; researcher `/v1/me`, `/v1/session/logout`, `/v1/methods`, `/v1/submissions[/{id}]`; administrator `/v1/admin/invitations`, `/v1/admin/invitations/{id}/revoke`, `/v1/admin/users/{id}/disable`, `/v1/admin/submissions`, `/v1/admin/submissions/{id}/review`, `/v1/admin/audit`.

- [x] Write HTTP tests for real invite→redeem→method→submit→review→restart chain and unauthenticated/cross-owner/cross-role denial. Test bounded strict JSON, invalid tokens, safe errors, disabled docs, rejection of scores/timing/unknown keys.
- [x] Run `... unittest discover -s benchmark/tests -p 'test_intake_api.py' -v`; observe missing app.
- [x] Implement explicit authentication, strict JSON body max16KiB/read5s, no CORS, no docs, sanitized error envelopes, no-store/security headers. Bound active requests; redemption rate-limit is only a process-local development backstop, not distributed admission protection.
- [x] Run HTTP tests plus process-level CLI acceptance: loopback-only Uvicorn, metadata persists across service restart, terminal-safe stdout without credentials. Use temporary state and terminate only test-owned processes.

## Task 3: document and verify the increment

**Files:** create `objview_submit/README.md`; update design appendix and `docs/superpowers/progress.md`. Existing benchmark smoke report remains scoped to benchmark/worker, intake tests run via full unittest suite.

- [x] Document launch, token generation, API payloads, persistence/backup limitations, revocation, holding-state meaning and exact missing production controls. Do not put real secrets in examples or repo.
- [x] Run full `PYTHONDONTWRITEBYTECODE=1 timeout 90s python3 -m unittest discover -s benchmark/tests -v` and existing `... python3 benchmark/smoke_submission.py` independently; run CLI help and `git diff --check`.
- [x] Request independent read-only code review; fix substantive findings with failing regression tests first.
- [x] Record actual evidence and remaining gaps. Keep changes on `feat/submission-protocol`; no merge, push, deployment or Docker installation.
