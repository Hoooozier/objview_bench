"""Public paper results and private development intake; no uploaded code execution."""
from collections import deque
import html
import hmac
import json
import re
import sqlite3
import time
from pathlib import Path
from urllib.parse import parse_qs

import anyio
from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from .validation import IntakeError
from .paper_results import load_results, render_results
from .submission_guide import dockerfile_example, example_file, render_guide


SITE_CSS = (Path(__file__).parent / 'site.css').read_text()


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


async def ui_form(request: Request):
    """Small, bounded urlencoded form parser; no multipart/uploads are accepted."""
    if request.headers.get('content-type', '').split(';')[0].strip().lower() != 'application/x-www-form-urlencoded':
        raise IntakeError('FORM_REQUIRED', 415)
    raw = bytearray()
    try:
        with anyio.fail_after(5):
            async for chunk in request.stream():
                if len(raw) + len(chunk) > 16384:
                    raise IntakeError('BODY_TOO_LARGE', 413)
                raw.extend(chunk)
    except TimeoutError:
        raise IntakeError('BODY_TIMEOUT', 408) from None
    try:
        values = parse_qs(raw.decode('utf-8'), keep_blank_values=True, strict_parsing=True)
    except (UnicodeDecodeError, ValueError):
        raise IntakeError('INVALID_FORM', 400) from None
    if any(len(v) != 1 for v in values.values()):
        raise IntakeError('INVALID_FORM', 400)
    return {key: vals[0] for key, vals in values.items()}


def _escape(value):
    return html.escape(str(value), quote=True)


def _page(title, content):
    return HTMLResponse('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                        '<meta name="viewport" content="width=device-width,initial-scale=1">'
                        '<title>' + _escape(title) + ' · ObjView-Bench</title>'
                        '<link rel="stylesheet" href="/ui/assets/site.css"></head><body>'
                        '<header class="site-header"><div class="header-inner">'
                        '<a class="brand" href="/ui">ObjView-Bench</a><nav aria-label="Main navigation">'
                        '<a href="/ui#results">Paper results</a><a href="/ui#protocol">Protocol</a>'
                        '<a href="/ui/guide">Submission guide</a><a href="/ui/submit">Submit Dockerfile</a><a href="/ui/researcher">My submissions</a>'
                        '<a href="/ui/admin">Admin</a></nav></div></header><main>' + content +
                        '</main><footer>ObjView-Bench · Paper baselines and batch validation</footer></body></html>')


def _link(href, label):
    return '<a href="' + _escape(href) + '">' + _escape(label) + '</a>'


def _form(action, fields_html, label='Submit'):
    return '<form method="post" action="' + _escape(action) + '">' + fields_html + \
        '<button type="submit">' + _escape(label) + '</button></form>'


def _text_input(name, value='', label=None, required=True):
    label = label or name
    req = ' required' if required else ''
    kind = ' type="password" autocomplete="off"' if name == 'token' else ''
    return '<label>' + _escape(label) + ': <input name="' + _escape(name) + '" value="' + _escape(value) + '"' + kind + req + '></label><br>'


COLLECTION_NOTICE = ('<p class="collection-notice" role="status"><strong>At this stage, Dockerfiles are collected for scheduled batch validation. '
                     '</strong>Submission does not trigger an immediate build or run. '
                     'Each submission runs independently. Batch dates will be announced by the administrators.</p>')


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
                if path in {'/v1/invitations/redeem', '/ui/redeem', '/ui/admin/login'} and request.method == 'POST':
                    now = time.monotonic()
                    while redemptions and redemptions[0] <= now - 60:
                        redemptions.popleft()
                    if len(redemptions) >= 60:
                        raise IntakeError('TOO_MANY_REQUESTS', 429)
                    redemptions.append(now)
                elif path.startswith('/ui/') or path == '/ui':
                    # Browser pages authenticate their HttpOnly cookie in the
                    # route itself; API bearer authentication remains mandatory.
                    pass
                elif not (path in {'/', '/healthz'} and request.method == 'GET'):
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
                                 'Referrer-Policy': 'no-referrer',
                                 'Content-Security-Policy': "default-src 'none'; style-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"})
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
    def submissions(request: Request, limit: int = 50, cursor: str = None):
        items, next_cursor = store.list_submissions_page(request.state.user['id'], limit=limit, cursor=cursor)
        result = {'items': items}
        if next_cursor is not None:
            result['next_cursor'] = next_cursor
        return result

    @app.get('/v1/submissions/{sid}')
    def submission(sid: str, request: Request):
        return store.get_submission(request.state.user['id'], sid)

    @app.get('/v1/submissions/{sid}/evaluation')
    def evaluation(sid: str, request: Request):
        return store.get_job(request.state.user['id'], sid)

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
    def review_list(limit: int = 50, cursor: str = None, status: str = None):
        items, next_cursor = store.list_submissions_page(limit=limit, cursor=cursor, status=status)
        result = {'items': items}
        if next_cursor is not None:
            result['next_cursor'] = next_cursor
        return result

    @app.get('/v1/admin/evaluation-batches')
    def evaluation_batches(limit: int = 50):
        return {'items': store.list_batches(limit=limit)}

    @app.post('/v1/admin/evaluation-batches/release', status_code=201)
    def release_evaluation_batch(data=Depends(body)):
        fields(data, set())
        return store.release_batch()

    @app.post('/v1/admin/submissions/{sid}/review')
    def review(sid: str, data=Depends(body)):
        fields(data, {'decision', 'expected_version'})
        return store.review(sid, data['decision'], data['expected_version'])

    @app.get('/v1/admin/audit')
    def audit(limit: int = 50, cursor: str = None):
        items, next_cursor = store.audit_events_page(limit=limit, cursor=cursor)
        result = {'items': items}
        if next_cursor is not None:
            result['next_cursor'] = next_cursor
        return result

    def ui_token(request):
        token = request.cookies.get('objview_session')
        if not token:
            raise IntakeError('UNAUTHORIZED', 401)
        return token

    def ui_user(request):
        return store.authenticate(ui_token(request))

    def ui_admin_token(request):
        token = request.cookies.get('objview_admin')
        if not token or not hmac.compare_digest(token.encode(), admin_token.encode()):
            raise IntakeError('UNAUTHORIZED', 401)
        return token

    def ui_error(title, exc):
        code = exc.code if isinstance(exc, IntakeError) else 'INTERNAL_ERROR'
        response = _page(title, '<p>Request failed: ' + _escape(code) + '</p>' + _link('/ui', 'Back to results'))
        response.status_code = exc.status if isinstance(exc, IntakeError) else 500
        return response

    @app.get('/')
    def home():
        return RedirectResponse('/ui', status_code=303)

    @app.get('/ui/assets/site.css')
    def ui_styles():
        return Response(SITE_CSS, media_type='text/css')

    @app.get('/ui/paper-results.json')
    def ui_paper_data():
        return JSONResponse(load_results(), headers={
            'Content-Disposition': 'attachment; filename="objview-paper-results.json"'})

    @app.get('/ui/guide')
    def ui_guide():
        return _page('Submission guide', COLLECTION_NOTICE + render_guide())

    @app.get('/ui/examples/{name}')
    def ui_example(name: str):
        content = example_file(name)
        if content is None:
            return Response('Example not found', status_code=404, media_type='text/plain')
        return Response(content, media_type='text/plain', headers={
            'Content-Disposition': 'attachment; filename="' + name + '"'})

    @app.get('/ui')
    def ui_home(space: str = 'whole', setting: str = 'K=5', order: str = 'quality'):
        try:
            content = render_results(space, setting, order)
        except IntakeError as exc:
            return ui_error('Invalid result filters', exc)
        return _page('Paper results', COLLECTION_NOTICE + content)

    @app.get('/ui/submit')
    def ui_submit_entry(request: Request):
        content = '<h1>Submit Dockerfile</h1>' + COLLECTION_NOTICE
        content += ('<p>Browse results without a token. Use an administrator-issued submission invitation when you are ready to submit.</p>'
                    '<p>Prepare a repository URL, an immutable commit, a Dockerfile path and a launch script. '
                    'This preview collects source references; Dockerfile file uploads are not available yet.</p>')
        if request.cookies.get('objview_session'):
            try:
                ui_user(request)
                content += '<p><a class="button" href="/ui/researcher">Continue submission / View my records</a></p>'
                return _page('Submit Dockerfile', content + dockerfile_example())
            except IntakeError:
                pass
        content += _form('/ui/redeem',
                         '<label>Submission invitation token<br><input class="token-input" type="password" '
                         'name="token" required autocomplete="off"></label>', 'Verify and start submission')
        content += '<p class="muted">Verification creates a short-lived browser session. Your submission records remain private.</p>'
        return _page('Submit Dockerfile', content + dockerfile_example())

    @app.post('/ui/redeem')
    async def ui_redeem(request: Request):
        try:
            data = await ui_form(request)
            result = store.redeem(data.get('token'))
        except IntakeError as exc:
            return ui_error('Invitation verification failed', exc)
        except Exception:
            return ui_error('Invitation verification failed', IntakeError('INVALID_INVITATION', 401))
        response = RedirectResponse('/ui/researcher', status_code=303)
        response.set_cookie('objview_session', result['access_token'], httponly=True,
                           secure=False, samesite='strict', max_age=86400, path='/ui')
        return response

    @app.post('/ui/logout')
    def ui_logout(request: Request):
        response = RedirectResponse('/ui', status_code=303)
        token = request.cookies.get('objview_session')
        if token:
            try:
                store.logout(token)
            except IntakeError:
                pass
        response.delete_cookie('objview_session', path='/ui', samesite='strict')
        response.delete_cookie('objview_admin', path='/ui', samesite='strict')
        return response

    @app.get('/ui/researcher')
    def ui_researcher(request: Request, cursor: str = None):
        try:
            user = ui_user(request)
            methods = store.list_methods(user['id'])
            items, next_cursor = store.list_submissions_page(user['id'], limit=20, cursor=cursor)
        except IntakeError as exc:
            if exc.status == 401:
                return RedirectResponse('/ui/submit', status_code=303)
            return ui_error('Researcher workspace', exc)
        content = '<h1>My submissions</h1>' + COLLECTION_NOTICE + '<p>Account: ' + _escape(user['label']) + '</p>'
        content += '<p><a href="/ui/guide">Dockerfile example and API integration guide</a></p>'
        content += _form('/ui/methods', _text_input('name', label='New method name'), 'Create method')
        if methods:
            options = ''.join('<option value="' + _escape(m['id']) + '">' + _escape(m['name']) + '</option>' for m in methods)
            fields_html = '<label>Method: <select name="method_id">' + options + '</select></label><br>'
            for name in ('repository', 'commit_sha', 'dockerfile', 'entrypoint', 'protocol_version'):
                fields_html += _text_input(name)
            fields_html += _text_input('idempotency_key', label='Idempotency key')
            content += _form('/ui/submissions', fields_html, 'Submit source references')
        else:
            content += '<p>Create a method to begin.</p>'
        content += '<h2>Submission history</h2><ul>'
        for item in items:
            content += '<li>' + _link('/ui/submissions/' + item['id'], item['id']) + ': ' + _escape(item['status']) + '</li>'
        content += '</ul>'
        if next_cursor:
            content += _link('/ui/researcher?cursor=' + _escape(next_cursor), 'Next page')
        content += _form('/ui/logout', '', 'Sign out')
        return _page('My submissions', content)

    @app.post('/ui/methods')
    async def ui_method(request: Request):
        try:
            user = ui_user(request)
            data = await ui_form(request)
            store.create_method(user['id'], data.get('name'))
        except IntakeError as exc:
            return ui_error('Could not create method', exc)
        return RedirectResponse('/ui/researcher', status_code=303)

    @app.post('/ui/submissions')
    async def ui_submission(request: Request):
        try:
            user = ui_user(request)
            data = await ui_form(request)
            key = data.pop('idempotency_key', None)
            store.submit(user['id'], data, key)
        except IntakeError as exc:
            return ui_error('Submission failed', exc)
        return RedirectResponse('/ui/researcher', status_code=303)

    @app.get('/ui/submissions/{sid}')
    def ui_submission_detail(sid: str, request: Request):
        try:
            item = store.get_submission(ui_user(request)['id'], sid)
        except IntakeError as exc:
            return ui_error('Submission details', exc)
        manifest = item['manifest']
        content = '<h1>Submission details</h1><p>ID: ' + _escape(item['id']) + '</p><p>Status: ' + _escape(item['status']) + '</p>'
        try:
            job = store.get_job(ui_user(request)['id'], sid)
        except IntakeError:
            job = None
        if job is not None:
            state_label = 'Collecting; awaiting batch validation' if job['state'] == 'COLLECTING' else job['state']
            content += '<p>Validation status: ' + _escape(state_label) + '</p>'
            if job.get('batch_id'):
                content += '<p>Batch: ' + _escape(job['batch_id']) + '</p>'
            if job.get('error_code'):
                content += '<p>Preflight failed: ' + _escape(job['error_code']) + '</p>'
        if item['status'] == 'APPROVED_FOR_PREFLIGHT':
            content += COLLECTION_NOTICE + '<p>This is not an official result.</p>'
        content += '<dl>' + ''.join('<dt>' + _escape(k) + '</dt><dd>' + _escape(v) + '</dd>' for k, v in manifest.items()) + '</dl>'
        content += _link('/ui/researcher', 'Back to my submissions')
        return _page('Submission details', content)

    @app.get('/ui/admin')
    def ui_admin(request: Request, cursor: str = None, status: str = None):
        if not request.cookies.get('objview_admin'):
            content = '<h1>Administrator access</h1>' + _form('/ui/admin/login', _text_input('token', label='Administrator token'), 'Sign in')
            return _page('Administrator access', content)
        try:
            ui_admin_token(request)
            items, next_cursor = store.list_submissions_page(limit=20, cursor=cursor, status=status)
            audit, audit_cursor = store.audit_events_page(limit=20)
        except IntakeError as exc:
            return ui_error('Administrator workspace', exc)
        content = '<h1>Submission review</h1>' + COLLECTION_NOTICE + '<p>Approval only adds a submission to the collection pool. It does not create a runnable job.</p>'
        content += '<p>Release includes all approved, unbatched submissions. Later approvals wait for the next batch. Jobs run independently as resources become available.</p>'
        content += _form('/ui/admin/evaluation-batches/release', '', 'Release current collection batch') + '<ul>'
        for item in items:
            content += '<li>' + _escape(item['id']) + ': ' + _escape(item['status']) + ' ' + _form(
                '/ui/admin/submissions/' + item['id'] + '/review',
                '<input type="hidden" name="expected_version" value="' + _escape(item['version']) + '">' +
                '<button name="decision" value="start">Start review</button><button name="decision" value="approve">Approve</button><button name="decision" value="reject">Reject</button>', '') + '</li>'
        content += '</ul>'
        if next_cursor:
            content += _link('/ui/admin?cursor=' + _escape(next_cursor), 'Next submissions')
        content += '<h2>Recently released batches</h2><ul>' + ''.join(
            '<li>' + _escape(batch['id']) + ': Released (jobs may still be pending)</li>'
            for batch in store.list_batches(limit=20)) + '</ul>'
        content += '<h2>Audit log</h2><ul>' + ''.join('<li>' + _escape(e['event']) + ': ' + _escape(e['subject_id']) + '</li>' for e in audit) + '</ul>'
        if audit_cursor:
            content += '<p>More audit records are available through API pagination.</p>'
        content += _form('/ui/logout', '', 'Sign out')
        return _page('Administrator review', content)

    @app.post('/ui/admin/login')
    async def ui_admin_login(request: Request):
        try:
            data = await ui_form(request)
            token = data.get('token', '')
            if not hmac.compare_digest(token.encode(), admin_token.encode()):
                raise IntakeError('UNAUTHORIZED', 401)
        except IntakeError as exc:
            return ui_error('Administrator sign-in failed', exc)
        response = RedirectResponse('/ui/admin', status_code=303)
        response.set_cookie('objview_admin', token, httponly=True, secure=False,
                           samesite='strict', max_age=86400, path='/ui')
        return response

    @app.post('/ui/admin/submissions/{sid}/review')
    async def ui_admin_review(sid: str, request: Request):
        try:
            ui_admin_token(request)
            data = await ui_form(request)
            store.review(sid, data.get('decision'), int(data.get('expected_version', '-1')))
        except (IntakeError, ValueError) as exc:
            if not isinstance(exc, IntakeError):
                exc = IntakeError('INVALID_PARAMS')
            return ui_error('Review failed', exc)
        return RedirectResponse('/ui/admin', status_code=303)

    @app.post('/ui/admin/evaluation-batches/release')
    async def ui_release_batch(request: Request):
        try:
            ui_admin_token(request)
            data = await ui_form(request)
            fields(data, set())
            store.release_batch()
        except IntakeError as exc:
            return ui_error('Batch release failed', exc)
        return RedirectResponse('/ui/admin', status_code=303)

    return app
