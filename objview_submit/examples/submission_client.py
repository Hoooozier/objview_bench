"""Small Python submission SDK. Retry submit with the SAME action dictionary."""
import secrets
import time

import httpx


class SubmissionError(RuntimeError):
    pass


class SubmissionClient:
    def __init__(self, base_url, token, *, transport=None):
        self._owns_transport = transport is None
        self.http = transport or httpx.Client(base_url=base_url, timeout=15, trust_env=False,
                                             follow_redirects=False)
        self.headers = {'Authorization': 'Bearer ' + token}

    def _response(self, method, path, **kwargs):
        # Bounded backoff only for temporary gateway pressure. Keep the same
        # action/request_id; a new ID could duplicate an ambiguously accepted action.
        for attempt in range(3):
            response = self.http.request(method, path, headers=self.headers, **kwargs)
            if response.status_code != 429 or attempt == 2:
                break
            try:
                code = response.json().get('error')
            except (ValueError, AttributeError):
                break
            if code not in {'RATE_LIMITED', 'TOO_MANY_REQUESTS'}:
                break
            time.sleep(1)
        if response.status_code not in (200, 204):
            raise SubmissionError(f'Benchmark request failed (HTTP {response.status_code})')
        return response

    def _request(self, method, path, **kwargs):
        response = self._response(method, path, **kwargs)
        if response.status_code == 204:
            return None
        return response.json()

    def observation(self, step, asset):
        if asset not in {'rgb', 'depth', 'mask', 'frame_meta', 'points'}:
            raise ValueError('Unsupported observation asset')
        index = step['step_index']
        if type(index) is not int or index < 0:
            raise ValueError('Invalid step index')
        return self._response('GET', f'/v1/observations/{index}/{asset}').content

    def ready(self):
        return self._request('POST', '/v1/ready', json={})

    def episode(self):
        return self._request('GET', '/v1/episode')

    def next_step(self, after=-1, *, wait=10):
        return self._request('GET', '/v1/steps/next', params={'after': after, 'wait': wait})

    def feasibility(self, poses):
        return self._request('POST', '/v1/feasibility/check', json={'poses': poses})['feasible']

    @staticmethod
    def make_action(step, action, **payload):
        return dict(episode_id=step['episode_id'], step_index=step['step_index'],
                    request_id=secrets.token_hex(16), action=action, **payload)

    def submit(self, action):
        return self._request('POST', '/v1/actions', json=action)

    def close(self):
        if self._owns_transport:
            self.http.close()
