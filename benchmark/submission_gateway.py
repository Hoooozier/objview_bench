"""Private single-session HTTP transport; deploy only behind worker isolation."""
import asyncio
import hmac
import json
import math
import os
from pathlib import Path
import stat
import threading
import time
import anyio

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from submission_protocol import ProtocolError

MAX_BODY_BYTES = 65536
REQUEST_RATE = 20
REQUEST_BURST = 40
MAX_ACTIVE_REQUESTS = 8
MAX_OBSERVATION_BYTES = 8 * 1024 * 1024
MAX_STEP_BYTES = 32 * 1024 * 1024
MAX_EPISODE_BYTES = 512 * 1024 * 1024


class GatewayGuard:
    """One gateway per episode, one ASGI worker; keys cannot be rotated by clients.

    Counters are checked without awaits and remain held until the response body
    is sent, including slow readers. Transport/header limits remain a worker duty.
    """
    def __init__(self, app, *, session, token, clock=time.monotonic):
        self.app, self.session, self.token, self.clock = app, session, token, clock
        self.tokens, self.updated = float(REQUEST_BURST), clock()
        self.total = self.active = self.polls = self.downloads = 0

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        async def reject(code, status, retry=False):
            headers = {'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'}
            if retry:
                headers['Retry-After'] = '1'
            await JSONResponse({'error': code}, status_code=status, headers=headers)(scope, receive, send)
        # Rejected requests also consume the finite episode request budget.
        self.total += 1
        if self.total > 128 + 64 * self.session.max_steps:
            self.session.finish('EPISODE_FAILED')
            return await reject('REQUEST_BUDGET_EXHAUSTED', 429)
        now = self.clock()
        self.tokens = min(REQUEST_BURST, self.tokens + max(0, now - self.updated) * REQUEST_RATE)
        self.updated = now
        if self.tokens < 1:
            return await reject('RATE_LIMITED', 429, True)
        self.tokens -= 1
        headers = dict(scope.get('headers', []))
        expected = ('Bearer ' + self.token).encode('ascii')
        if not hmac.compare_digest(headers.get(b'authorization', b''), expected):
            return await reject('UNAUTHORIZED', 401)
        poll = scope['path'] == '/v1/steps/next'
        download = scope['path'].startswith('/v1/observations/')
        if self.active >= MAX_ACTIVE_REQUESTS or (poll and self.polls >= 1) or (download and self.downloads >= 2):
            return await reject('TOO_MANY_REQUESTS', 429, True)
        self.active += 1
        self.polls += int(poll)
        self.downloads += int(download)
        try:
            await self.app(scope, receive, send)
        finally:
            self.active -= 1
            self.polls -= int(poll)
            self.downloads -= int(download)


class ObservationReader:
    """Fixed published assets only; no symlinks, special files or arbitrary paths."""
    def __init__(self, root):
        self.root = None if root is None else Path(root).absolute()
        self.lock = threading.Lock()
        self.total_bytes = 0
        self.steps = {}

    def read(self, relative, step_index):
        if self.root is None:
            raise ProtocolError('OBSERVATION_NOT_AVAILABLE', 404)
        # relative is produced by SubmissionSession, never a client filesystem path.
        directory, filename = relative.split('/')
        root_fd = step_fd = fd = None
        try:
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            root_fd = os.open(self.root, flags)
            step_fd = os.open(directory, flags, dir_fd=root_fd)
            fd = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=step_fd)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ProtocolError('OBSERVATION_NOT_AVAILABLE', 404)
            if info.st_size > MAX_OBSERVATION_BYTES:
                raise ProtocolError('OBSERVATION_TOO_LARGE', 413)
            with self.lock:
                count, size = self.steps.get(step_index, (0, 0))
                if count >= 10 or size + info.st_size > MAX_STEP_BYTES or self.total_bytes + info.st_size > MAX_EPISODE_BYTES:
                    raise ProtocolError('OBSERVATION_BUDGET_EXHAUSTED', 429)
                self.steps[step_index] = (count + 1, size + info.st_size)
                self.total_bytes += info.st_size
            # Published files are immutable, benchmark-owned. Check size again
            # after reading to reject unexpected changes rather than undercharge.
            with os.fdopen(fd, 'rb') as stream:
                fd = None
                content = stream.read(info.st_size + 1)
            if len(content) != info.st_size:
                raise ProtocolError('OBSERVATION_NOT_AVAILABLE', 404)
            return content
        except OSError:
            raise ProtocolError('OBSERVATION_NOT_AVAILABLE', 404) from None
        finally:
            for handle in (fd, step_fd, root_fd):
                if handle is not None:
                    os.close(handle)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate key')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError('Non-finite JSON')


async def _body(request):
    if request.headers.get('content-type', '').split(';')[0] != 'application/json':
        raise ProtocolError('JSON_REQUIRED', 415)
    chunks = bytearray()
    try:
        with anyio.fail_after(5):
            async for chunk in request.stream():
                if len(chunks) + len(chunk) > MAX_BODY_BYTES:
                    raise ProtocolError('BODY_TOO_LARGE', 413)
                chunks.extend(chunk)
    except TimeoutError:
        raise ProtocolError('BODY_TIMEOUT', 408)
    try:
        return json.loads(chunks, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise ProtocolError('INVALID_JSON', 400)


def create_app(session, token, *, observation_root=None, clock=time.monotonic):
    if not isinstance(token, str) or len(token) < 32 or not token.isascii():
        raise ValueError('Use a random ASCII token of at least 32 characters')
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, redirect_slashes=False)
    reader = ObservationReader(observation_root)

    @app.middleware('http')
    async def boundary(request, call_next):
        try:
            response = await call_next(request)
        except Exception:
            session.finish('EPISODE_FAILED')
            response = JSONResponse({'error': 'INTERNAL_ERROR'}, status_code=500)
        response.headers.update({'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'})
        return response

    app.add_middleware(GatewayGuard, session=session, token=token, clock=clock)

    @app.exception_handler(ProtocolError)
    async def protocol_error(request, exc):
        return JSONResponse({'error': exc.code}, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        return JSONResponse({'error': 'INVALID_PARAMS'}, status_code=422)

    @app.get('/v1/interface')
    async def interface():
        return {**session.interface(), 'observation_transport': 'http',
                'limits': {'requests_per_second': REQUEST_RATE, 'request_burst': REQUEST_BURST,
                           'requests_per_episode': 128 + 64 * session.max_steps,
                           'concurrent_requests': MAX_ACTIVE_REQUESTS, 'concurrent_polls': 1,
                           'concurrent_downloads': 2, 'observation_bytes': MAX_OBSERVATION_BYTES,
                           'observation_requests_per_step': 10, 'observation_bytes_per_step': MAX_STEP_BYTES,
                           'observation_bytes_per_episode': MAX_EPISODE_BYTES}}

    @app.get('/v1/observations/{step_index}/{asset}')
    async def observation(step_index: int, asset: str):
        relative = session.observation_path(step_index, asset)
        content = await anyio.to_thread.run_sync(reader.read, relative, step_index)
        media = {'rgb': 'image/png', 'depth': 'application/octet-stream', 'mask': 'image/png',
                 'frame_meta': 'application/json', 'points': 'application/octet-stream'}
        return Response(content, media_type=media[asset])

    @app.get('/v1/episode')
    async def episode():
        return session.episode()

    @app.post('/v1/ready')
    async def ready(request: Request):
        if await _body(request) != {}:
            raise ProtocolError('INVALID_PARAMS', 422)
        return session.ready()

    @app.get('/v1/status')
    async def status():
        return session.status()

    @app.get('/v1/steps/next')
    async def next_step(request: Request):
        try:
            after = int(request.query_params.get('after', '-1'))
            wait = float(request.query_params.get('wait', '10'))
        except ValueError:
            raise ProtocolError('INVALID_PARAMS', 422)
        if not math.isfinite(wait) or not 0 <= wait <= 10:
            raise ProtocolError('INVALID_PARAMS', 422)
        deadline = asyncio.get_running_loop().time() + wait
        while True:
            result = session.next_step(after)
            if result is not None:
                return result
            if asyncio.get_running_loop().time() >= deadline:
                return Response(status_code=204)
            await asyncio.sleep(0.01)

    @app.post('/v1/feasibility/check')
    async def feasibility(request: Request):
        data = await _body(request)
        if not isinstance(data, dict) or set(data) != {'poses'}:
            raise ProtocolError('INVALID_PARAMS', 422)
        return session.check_feasibility(data['poses'])

    @app.post('/v1/actions')
    async def actions(request: Request):
        return session.submit(await _body(request))

    return app
