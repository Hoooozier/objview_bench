"""Private single-session HTTP transport; deploy only behind worker isolation."""
import asyncio
import hmac
import json
import math
import anyio

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from submission_protocol import ProtocolError

MAX_BODY_BYTES = 65536


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


def create_app(session, token):
    if not isinstance(token, str) or len(token) < 32 or not token.isascii():
        raise ValueError('Use a random ASCII token of at least 32 characters')
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    active_requests = 0

    @app.middleware('http')
    async def boundary(request, call_next):
        nonlocal active_requests
        auth = request.headers.get('authorization', '')
        if not hmac.compare_digest(auth.encode(), ('Bearer ' + token).encode()):
            return JSONResponse({'error': 'UNAUTHORIZED'}, status_code=401)
        if active_requests >= 16:
            return JSONResponse({'error': 'TOO_MANY_REQUESTS'}, status_code=429)
        active_requests += 1
        try:
            try:
                response = await call_next(request)
            except Exception:
                session.finish('EPISODE_FAILED')
                response = JSONResponse({'error': 'INTERNAL_ERROR'}, status_code=500)
            response.headers['Cache-Control'] = 'no-store'
            return response
        finally:
            active_requests -= 1

    @app.exception_handler(ProtocolError)
    async def protocol_error(request, exc):
        return JSONResponse({'error': exc.code}, status_code=exc.status)

    @app.get('/v1/interface')
    async def interface():
        return session.interface()

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
