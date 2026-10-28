"""Persistent trusted-smoke queue adapter.

This worker deliberately accepts a fixed ``trusted_smoke_v1`` claim from the
local control plane.  It never receives a repository, Dockerfile or command
from a submission.  Production deployment will replace the injected executor
with the already-reviewed controller and keep the same lease/result contract.
"""
import threading
import time

from objview_submit.validation import IntakeError


class QueueWorker:
    def __init__(self, store, worker_id, executor, *, lease_seconds=60, poll_seconds=1):
        self.store = store
        self.worker_id = worker_id
        self.executor = executor
        if type(lease_seconds) is not int or not 1 <= lease_seconds <= 3600:
            raise ValueError('Invalid lease')
        if not isinstance(poll_seconds, (int, float)) or not 0 < poll_seconds <= 60:
            raise ValueError('Invalid poll interval')
        self.lease_seconds = lease_seconds
        self.poll_seconds = poll_seconds

    def run_once(self):
        claim = self.store.claim_job(self.worker_id, self.lease_seconds)
        if claim is None:
            return None
        try:
            outcome = self.executor(claim)
            if not isinstance(outcome, dict) or outcome.get('status') not in {'SUCCEEDED', 'FAILED'}:
                raise ValueError('Invalid executor outcome')
            if outcome['status'] == 'SUCCEEDED':
                return self.store.finish_job(claim['job_id'], self.worker_id, success=True,
                                             result=outcome.get('result'))
            return self.store.finish_job(claim['job_id'], self.worker_id, success=False,
                                         error_code=outcome.get('error_code'))
        except IntakeError:
            raise
        except (ValueError, TypeError, KeyError):
            return self.store.finish_job(claim['job_id'], self.worker_id, success=False,
                                         error_code='INFRASTRUCTURE_FAILED')

    def run_forever(self, stop_event=None):
        stop_event = stop_event or threading.Event()
        while not stop_event.is_set():
            self.run_once()
            stop_event.wait(self.poll_seconds)


def trusted_smoke_executor(run_controller):
    """Adapt a reviewed controller callback to the queue result contract.

    ``run_controller`` receives only the claim dictionary and must return the
    controller's private allowlisted result.  This helper does not catch
    arbitrary exceptions or turn user data into a successful result.
    """
    def execute(claim):
        if claim.get('kind') != 'trusted_smoke_v1':
            return {'status': 'FAILED', 'error_code': 'INFRASTRUCTURE_FAILED'}
        report = run_controller(claim)
        if not isinstance(report, dict):
            return {'status': 'FAILED', 'error_code': 'INFRASTRUCTURE_FAILED'}
        if report.get('status') == 'SUCCEEDED':
            return {'status': 'SUCCEEDED', 'result': report.get('result')}
        return {'status': 'FAILED', 'error_code': report.get('failure_code', 'INFRASTRUCTURE_FAILED')}
    return execute
