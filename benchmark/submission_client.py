"""Small Python submission SDK. Retry submit with the SAME action dictionary."""
import secrets

import httpx


class SubmissionError(RuntimeError):
    pass


class SubmissionClient:
    def __init__(self, base_url, token, *, transport=None):
        self._owns_transport = transport is None
        self.http = transport or httpx.Client(base_url=base_url, timeout=15, trust_env=False,
                                             follow_redirects=False)
        self.headers = {'Authorization': 'Bearer ' + token}

    def _request(self, method, path, **kwargs):
        response = self.http.request(method, path, headers=self.headers, **kwargs)
        if response.status_code == 204:
            return None
        if response.status_code != 200:
            # Do not include raw HTML, paths or credentials in client exception text.
            raise SubmissionError(f'Benchmark request failed (HTTP {response.status_code})')
        return response.json()

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
