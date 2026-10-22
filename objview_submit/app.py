"""Development intake HTTP boundary. No uploads, execution, metrics or OAuth claims."""
from collections import deque
import hmac
import json
import re
import sqlite3
import time

import anyio
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from .validation import IntakeError


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate key')
        result[key] = value
    return result


def _invalid_number(value):
    raise ValueError('Non-finite number')


async def body(request: Request):
    if request.headers.get('content-type', '').split(';')[0].strip().lower() != 'application/json':
        raise IntakeError('JSON_REQUIRED', 415)
    data = bytearray()
    try:
        with anyio.fail_after(5):
            async for chunk in request.stream():
                if len(data) + len(chunk) > 16384:
                    raise IntakeError('BODY_TOO_LARGE', 413)
                data.extend(chunk)
    except TimeoutError:
        raise IntakeError('BODY_TIMEOUT', 408) from None
    try:
        return json.loads(data, object_pairs_hook=_unique, parse_constant=_invalid_number)
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise IntakeError('INVALID_JSON', 400) from None


def fields(data, expected):
    if not isinstance(data, dict) or set(data) != set(expected):
        raise IntakeError('INVALID_PARAMS')
    return data


def create_app(store, admin_token, *, development_mode=False):
    if development_mode is not True:
        raise ValueError('This adapter is development-only; explicitly opt in')
    if not isinstance(admin_token, str) or not re.fullmatch(r'[A-Za-z0-9_-]{32,128}', admin_token):
        raise ValueError('Supply a random administrator token of at least 32 URL-safe characters')
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, redirect_slashes=False)
    active = 0
    redemptions = deque()

    @app.middleware('http')
    async def boundary(request, call_next):
        nonlocal active
        try:
            if active >= 16:
                raise IntakeError('TOO_MANY_REQUESTS', 429)
            active += 1
            try:
                path = request.url.path
                if path == '/v1/invitations/redeem' and request.method == 'POST':
                    now = time.monotonic()
                    while redemptions and redemptions[0] <= now - 60:
                        redemptions.popleft()
                    if len(redemptions) >= 60:
                        raise IntakeError('TOO_MANY_REQUESTS', 429)
                    redemptions.append(now)
                elif not (path == '/healthz' and request.method == 'GET'):
                    header = request.headers.get('authorization', '')
                    if not header.startswith('Bearer ') or len(header) > 263:
                        raise IntakeError('UNAUTHORIZED', 401)
                    token = header[7:]
                    is_admin = hmac.compare_digest(token.encode(), admin_token.encode())
                    if is_admin:
                        if not path.startswith('/v1/admin/'):
                            raise IntakeError('FORBIDDEN', 403)
                    else:
                        request.state.user = await anyio.to_thread.run_sync(store.authenticate, token)
                        if path.startswith('/v1/admin/'):
                            raise IntakeError('FORBIDDEN', 403)
                    request.state.token = token
                response = await call_next(request)
            finally:
                active -= 1
        except IntakeError as exc:
            response = JSONResponse({'error': exc.code}, status_code=exc.status)
        except sqlite3.OperationalError:
            response = JSONResponse({'error': 'STORAGE_UNAVAILABLE'}, status_code=503)
        except Exception:
            response = JSONResponse({'error': 'INTERNAL_ERROR'}, status_code=500)
        response.headers.update({'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
                                 'Content-Security-Policy': "default-src 'none'; frame-ancestors 'none'"})
        return response

    @app.exception_handler(IntakeError)
    async def intake_error(request, exc):
        return JSONResponse({'error': exc.code}, status_code=exc.status)

    @app.get('/healthz')
    def health():
        return {'status': 'ok', 'mode': 'development_intake', 'execution_enabled': False}

    @app.post('/v1/invitations/redeem')
    def redeem(data=Depends(body)):
        return store.redeem(fields(data, {'token'})['token'])

    @app.get('/v1/me')
    def me(request: Request):
        return request.state.user

    @app.post('/v1/session/logout')
    def logout(request: Request, data=Depends(body)):
        fields(data, set())
        return store.logout(request.state.token)

    @app.post('/v1/methods', status_code=201)
    def method(request: Request, data=Depends(body)):
        return store.create_method(request.state.user['id'], fields(data, {'name'})['name'])

    @app.get('/v1/methods')
    def methods(request: Request):
        return {'items': store.list_methods(request.state.user['id'])}

    @app.post('/v1/submissions', status_code=201)
    def submit(request: Request, data=Depends(body)):
        return store.submit(request.state.user['id'], data, request.headers.get('idempotency-key'))

    @app.get('/v1/submissions')
    def submissions(request: Request):
        return {'items': store.list_submissions(request.state.user['id'])}

    @app.get('/v1/submissions/{sid}')
    def submission(sid: str, request: Request):
        return store.get_submission(request.state.user['id'], sid)

    @app.post('/v1/admin/invitations', status_code=201)
    def invitation(data=Depends(body)):
        fields(data, {'label', 'ttl_seconds'})
        return store.create_invitation(data['label'], data['ttl_seconds'])

    @app.post('/v1/admin/invitations/{iid}/revoke')
    def revoke(iid: str, data=Depends(body)):
        fields(data, set())
        return store.revoke_invitation(iid)

    @app.post('/v1/admin/users/{uid}/disable')
    def disable(uid: str, data=Depends(body)):
        fields(data, set())
        return store.disable_user(uid)

    @app.get('/v1/admin/submissions')
    def review_list():
        return {'items': store.list_submissions()}

    @app.post('/v1/admin/submissions/{sid}/review')
    def review(sid: str, data=Depends(body)):
        fields(data, {'decision', 'expected_version'})
        return store.review(sid, data['decision'], data['expected_version'])

    @app.get('/v1/admin/audit')
    def audit():
        return {'items': store.audit_events()}

    return app
