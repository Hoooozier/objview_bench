"""Single-episode submission boundary. No renderer, filesystem or HTTP dependencies."""
from __future__ import annotations

import copy
import math
import re
import secrets
import threading
import time
from typing import Callable

VERSION = 'objview-interaction/v1'
STOP_REASONS = {'plan_end', 'candidate_exhausted', 'next_view_visited', 'next_view_infeasible'}


class ProtocolError(Exception):
    def __init__(self, code: str, status: int = 409):
        super().__init__(code)
        self.code, self.status = code, status


def validated_pose(value, code='INVALID_ACTION'):
    if not isinstance(value, list) or len(value) != 7:
        raise ProtocolError(code, 422)
    if any(type(x) not in (int, float) or abs(x) > 1e6 or not math.isfinite(x) for x in value):
        raise ProtocolError(code, 422)
    if sum((value[i] - value[i + 3]) ** 2 for i in range(3)) < 1e-12:
        raise ProtocolError(code, 422)
    return [float(x) for x in value]


class SubmissionSession:
    """Trusted controller methods publish/take/finish are never exposed as routes.

    One lock orders publication, action acceptance, quotas and deadline expiration.
    Only accepted requests are cached, bounded by the maximum episode view count.
    """

    def __init__(self, feasible: Callable, *, clock_ns=time.perf_counter_ns,
                 action_timeout_sec=120.0, max_steps=129):
        if not 0 < action_timeout_sec <= 3600 or type(max_steps) is not int or not 1 <= max_steps <= 1024:
            raise ValueError('Invalid session limits')
        self.episode_id = secrets.token_hex(16)
        self.feasible = feasible
        self.clock = clock_ns
        self.timeout_ns = int(action_timeout_sec * 1e9)
        self.max_steps = max_steps
        self.lock = threading.RLock()
        self._config = None
        self._ready = False
        self._created_ns = self.clock()
        self._ready_ns = None
        self._step = None
        self._started_ns = None
        self._awaiting = False
        self._pending = None
        self._receipts = {}
        self._queries = 0
        self._timings = []
        self._done = None
        self._last_action = None

    def configure(self, config):
        # Reconstruct public nested fields, never recursively copy user/runner extras.
        with self.lock:
            if self._config is not None:
                raise ProtocolError('INVALID_STATE')
            intrinsics = {}
            for key in ('image_width', 'image_height', 'fov_x_rad', 'fov_y_rad', 'principal_x', 'principal_y'):
                value = config.get('camera_intrinsics', {}).get(key)
                if type(value) in (int, float) and math.isfinite(value):
                    intrinsics[key] = value
            constraint = config.get('viewspace_constraint', {}).get('name', 'whole')
            if constraint not in {'whole', 'hemi', 'quarter', 'eighth'}:
                constraint = 'custom'
            self._config = {
                'protocol_version': VERSION, 'episode_id': self.episode_id,
                'coordinate_frame': 'object_normalized_frame',
                'pose_format': 'camera_lookat_roll',
                'camera_model': {'type': 'pinhole', 'distortion': 'none'},
                'depth_semantics': {'type': 'camera_z', 'unit': 'object_normalized_unit', 'invalid_value': 0},
                'object_normalization': {'object_center': [0., 0., 0.], 'unit_sphere_radius': 1.},
                'camera_intrinsics': intrinsics,
                'viewspace_constraint': {'name': constraint},
                'max_visited_view_num': self.max_steps,
            }

    def interface(self):
        return {'protocol_version': VERSION, 'pose_format': 'camera_lookat_roll',
                'actions': ['move', 'stop'], 'stop_reasons': sorted(STOP_REASONS),
                'action_timeout_sec': self.timeout_ns / 1e9,
                'feasibility_request_limit': 512, 'feasibility_step_limit': 4096,
                'timing_source': 'benchmark_monotonic_clock'}

    def episode(self):
        with self.lock:
            if self._config is None:
                raise ProtocolError('NOT_CONFIGURED')
            return copy.deepcopy(self._config)

    def ready(self):
        with self.lock:
            if self._done:
                raise ProtocolError('EPISODE_DONE')
            if not self._ready:
                self._ready, self._ready_ns = True, self.clock()
            return {'ready': True}

    def is_ready(self):
        with self.lock:
            return self._ready

    def publish(self, step_index, pose, manifest, *, release_observation=None):
        with self.lock:
            expected = 0 if self._step is None else self._step['step_index'] + 1
            if (not self._ready or self._config is None or self._done or self._awaiting
                    or self._pending is not None or step_index != expected
                    or step_index >= self.max_steps or self._last_action == 'stop'):
                raise ProtocolError('INVALID_STATE')
            public_manifest = {}
            names = {'rgb_path': 'rgb.png', 'depth_path': 'depth.npz',
                     'mask_path': 'mask.png', 'frame_meta_path': 'frame_meta.json', 'points_path': 'points.pcd'}
            for key, value in manifest.items():
                if key not in names or value != f'step_{step_index:03d}/{names[key]}':
                    raise ProtocolError('INVALID_OBSERVATION', 422)
                public_manifest[key] = value
            self._step = {'episode_id': self.episode_id, 'step_index': step_index,
                          'visited_view_num': step_index + 1, 'current_pose': validated_pose(pose),
                          'observation': public_manifest}
            self._queries = 0
            self._started_ns = self.clock()
            # Trusted callback atomically exposes staged files. Start the clock
            # first: filesystem watchers must not gain unmetered inference time.
            if release_observation is not None:
                release_observation()
            self._awaiting = True

    def _expire(self):
        if self._awaiting and self.clock() - self._started_ns >= self.timeout_ns:
            self._timings.append({'step_index': self._step['step_index'], 'decision_wall_time_ns': self.timeout_ns})
            self._awaiting = False
            self._done = 'ACTION_TIMEOUT'

    def next_step(self, after=-1):
        with self.lock:
            if type(after) is not int or not -1 <= after < self.max_steps:
                raise ProtocolError('INVALID_PARAMS', 422)
            self._expire()
            if self._done:
                return {'status': 'done', 'code': self._done}
            if self._step is not None and self._step['step_index'] > after and self._awaiting:
                return copy.deepcopy(self._step)
            return None

    def status(self):
        with self.lock:
            self._expire()
            if self._done:
                return {'status': 'done', 'code': self._done}
            return {'status': 'running' if self._step else ('ready' if self._ready else 'starting')}

    def observation_path(self, step_index, asset):
        """Resolve only an asset already published for the current pending step."""
        names = {'rgb': 'rgb_path', 'depth': 'depth_path', 'mask': 'mask_path',
                 'frame_meta': 'frame_meta_path', 'points': 'points_path'}
        with self.lock:
            self._expire()
            if (self._done or not self._awaiting or self._step is None
                    or type(step_index) is not int or step_index != self._step['step_index']
                    or asset not in names or names[asset] not in self._step['observation']):
                raise ProtocolError('OBSERVATION_NOT_AVAILABLE', 404)
            return self._step['observation'][names[asset]]

    def _active(self):
        self._expire()
        if self._done:
            raise ProtocolError('ACTION_TIMEOUT' if self._done == 'ACTION_TIMEOUT' else 'EPISODE_DONE')
        if self._step is None:
            raise ProtocolError('NOT_READY')

    def submit(self, data):
        with self.lock:
            if not isinstance(data, dict):
                raise ProtocolError('INVALID_ACTION', 422)
            fields = {'episode_id', 'step_index', 'request_id', 'action'}
            action = data.get('action')
            if action == 'move':
                fields.add('pose')
            elif action == 'stop':
                fields.add('stop_reason')
            else:
                raise ProtocolError('INVALID_ACTION', 422)
            if set(data) != fields or type(data.get('step_index')) is not int:
                raise ProtocolError('INVALID_ACTION', 422)
            request_id = data.get('request_id')
            if not isinstance(request_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', request_id):
                raise ProtocolError('INVALID_ACTION', 422)
            if action == 'move':
                validated_pose(data['pose'])
            elif not isinstance(data['stop_reason'], str) or data['stop_reason'] not in STOP_REASONS:
                raise ProtocolError('INVALID_ACTION', 422)
            if data['episode_id'] != self.episode_id:
                raise ProtocolError('WRONG_EPISODE')
            if request_id in self._receipts:
                old, receipt = self._receipts[request_id]
                if old != data:
                    raise ProtocolError('REQUEST_CONFLICT')
                return copy.deepcopy(receipt)
            self._active()
            if data['step_index'] != self._step['step_index']:
                raise ProtocolError('WRONG_STEP')
            if not self._awaiting:
                raise ProtocolError('ACTION_ALREADY_SUBMITTED')
            # Receipt time is taken after syntax/identity validation, before feasibility work.
            elapsed = self.clock() - self._started_ns
            feasible = action != 'move' or self.feasible(data['pose'])
            self._timings.append({'step_index': data['step_index'], 'decision_wall_time_ns': elapsed})
            self._awaiting = False
            if not feasible:
                self._done = 'INVALID_ACTION'
                raise ProtocolError('INFEASIBLE_ACTION', 422)
            self._last_action = action
            self._pending = copy.deepcopy(data)
            self._pending['algorithm_runtime_sec'] = elapsed / 1e9
            self._pending['decision_wall_time_ns'] = elapsed
            receipt = {'accepted': True, 'step_index': data['step_index'], 'request_id': request_id}
            self._receipts[request_id] = (copy.deepcopy(data), receipt)
            return copy.deepcopy(receipt)

    def take_action(self):
        with self.lock:
            self._expire()
            result, self._pending = self._pending, None
            return result

    def check_feasibility(self, poses):
        with self.lock:
            self._active()
            if not self._awaiting:
                raise ProtocolError('ACTION_ALREADY_SUBMITTED')
            if not isinstance(poses, list) or not poses:
                raise ProtocolError('INVALID_PARAMS', 422)
            if len(poses) > 512 or self._queries + len(poses) > 4096:
                raise ProtocolError('QUERY_LIMIT', 429)
            checked = [validated_pose(p, 'INVALID_PARAMS') for p in poses]
            self._queries += len(checked)
            return {'feasible': [bool(self.feasible(p)) for p in checked]}

    def finish(self, reason='COMPLETED'):
        with self.lock:
            if self._done is None:
                self._done = reason if reason in {'COMPLETED', 'INVALID_ACTION', 'STARTUP_TIMEOUT', 'ACTION_TIMEOUT'} else 'EPISODE_FAILED'
            self._awaiting = False

    def timing_summary(self):
        with self.lock:
            return {'timing_source': 'benchmark_monotonic_clock',
                    'model_ready_time_ns': None if self._ready_ns is None else self._ready_ns - self._created_ns,
                    'total_decision_wall_time_ns': sum(x['decision_wall_time_ns'] for x in self._timings),
                    'steps': copy.deepcopy(self._timings)}
