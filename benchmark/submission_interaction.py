"""Trusted bridge: private legacy records, anonymous read-only observation export."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from api_interaction import InteractionSession, pose_to_extrinsics


class SubmissionInteraction(InteractionSession):
    def __init__(self, session_dir, feasibility_api, *, protocol, observation_root):
        private = Path(session_dir).resolve()
        public = Path(observation_root).resolve()
        if private == public or private in public.parents or public in private.parents:
            raise ValueError('Observation directory must be separate from private session')
        if public.exists() and any(public.iterdir()):
            raise ValueError('Observation directory must be empty and unique per episode')
        self.protocol = protocol
        self.observation_root = public
        public.mkdir(parents=True, exist_ok=True)
        self._manifest = None
        self._current = None
        self._terminal_consumed = False
        super().__init__(private, feasibility_api)
        self.staging_root = private / 'staging'
        self.staging_root.mkdir(exist_ok=True)
        if self.staging_root.stat().st_dev != public.stat().st_dev:
            raise ValueError('Private staging and released observations must share a filesystem for atomic rename')

    def publish_episode_config(self, episode_config):
        self.protocol.configure(episode_config)
        return super().publish_episode_config(episode_config)

    def materialize_frame(self, *, step_index, pose, frame):
        """Copy only acquired RGBD; never copy cache metadata or preserve source paths.

        The controller must ensure the algorithm cannot write this directory.
        Files stay private until publication. No prior step is reused.
        """
        step_name = f'step_{step_index:03d}'
        target = self.staging_root / step_name
        target.mkdir(exist_ok=False)
        Image.fromarray(np.asarray(frame.rgb, dtype=np.uint8)).save(target / 'rgb.png')
        np.savez_compressed(target / 'depth.npz', depth=np.asarray(frame.depth, dtype=np.float32))
        Image.fromarray(np.asarray(frame.mask, dtype=bool).astype(np.uint8) * 255).save(target / 'mask.png')
        c2w, w2c = pose_to_extrinsics(pose)
        intrinsics = {name: float(getattr(frame.intrinsics, name)) for name in
                      ('fov_x_rad', 'fov_y_rad', 'principal_x', 'principal_y')}
        intrinsics.update({name: int(getattr(frame.intrinsics, name))
                           for name in ('image_width', 'image_height')})
        metadata = {'source': 'released_observation', 'intrinsics': intrinsics,
                    'pose': {'camera_xyz': list(pose.camera_xyz), 'lookat_xyz': list(pose.lookat_xyz),
                             'roll_rad': float(pose.roll_rad)},
                    'camera_to_world': c2w.tolist(), 'world_to_camera': w2c.tolist()}
        (target / 'frame_meta.json').write_text(json.dumps(metadata, allow_nan=False) + '\n', encoding='utf-8')
        return {key: f'{step_name}/{name}' for key, name in
                [('rgb_path', 'rgb.png'), ('depth_path', 'depth.npz'),
                 ('mask_path', 'mask.png'), ('frame_meta_path', 'frame_meta.json')]}

    def publish_observation_manifest(self, *, step_index, observation_manifest):
        self._manifest = dict(observation_manifest)
        return super().publish_observation_manifest(step_index=step_index, observation_manifest=observation_manifest)

    def publish_current_step(self, *, step_index, current_step_record):
        self._current = dict(current_step_record)
        return super().publish_current_step(step_index=step_index, current_step_record=current_step_record)

    def publish_benchmark_ready(self, *, episode_id, step_index, visited_view_num):
        record = super().publish_benchmark_ready(episode_id=episode_id, step_index=step_index,
                                                 visited_view_num=visited_view_num)
        name = f'step_{step_index:03d}'
        self.protocol.publish(step_index, self._current['current_pose'], self._manifest,
                              release_observation=lambda: (self.staging_root / name).rename(self.observation_root / name))
        return record

    def process_pending_requests(self):
        # Legacy oracle/file RPC is trusted-only, never consumed in submission mode.
        return []

    def read_submitted_action(self, *, expected_step_index=None, expected_episode_id=None):
        action = self.protocol.take_action()
        if action is not None:
            action['episode_id'] = expected_episode_id
            return action
        status = self.protocol.status()
        if status['status'] == 'done' and not self._terminal_consumed:
            self._terminal_consumed = True
            if status['code'] == 'INVALID_ACTION':
                timing = self.protocol.timing_summary()['steps'][-1]
                return {'episode_id': expected_episode_id, 'step_index': expected_step_index,
                        'action': 'invalid', 'algorithm_runtime_sec': timing['decision_wall_time_ns'] / 1e9}
            from runner_episode import EpisodeRunnerError
            raise EpisodeRunnerError('Submission session terminated', exception_info={
                'type': status['code'], 'step_index': expected_step_index})
        return None

    def publish_episode_done(self, **record):
        if record['terminal_source'] == 'exception':
            code = 'EPISODE_FAILED'
        elif record['terminal_reason'] == 'invalid_action':
            code = 'INVALID_ACTION'
        else:
            code = 'COMPLETED'
        self.protocol.finish(code)
        return super().publish_episode_done(**record)

    def timing_summary(self):
        return self.protocol.timing_summary()
