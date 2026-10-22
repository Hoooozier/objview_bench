"""Synthetic test fixtures only. Never import in official benchmark entrypoints."""
from types import SimpleNamespace

import numpy as np


class Renderer:
    uid = 'SECRET_UID'
    source = 'cache'
    intrinsics = SimpleNamespace(image_width=2, image_height=2, fov_x_rad=1.,
                                 fov_y_rad=1., principal_x=1., principal_y=1.)
    _cache_index = {'objects': {uid: {'views': [{
        'view_set': 'tammes_360', 'rgb': '/SECRET_CACHE/image.png',
        'pose': {'camera_xyz': [0, 0, 3], 'lookat_xyz': [0, 0, 0], 'roll_rad': 0}}]}}}

    def render_loaded(self, pose):
        return SimpleNamespace(source=self.source, pose=pose, intrinsics=self.intrinsics,
                               rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                               depth=np.ones((2, 2), dtype=np.float32),
                               mask=np.ones((2, 2), dtype=bool),
                               points_world=np.array([[0., 0., 0.], [0.1, 0., 0.]]))


def coverage(gt, observed, *, threshold):
    distances = np.linalg.norm(gt[:, None] - observed[None, :], axis=2)
    return float((distances.min(axis=1) <= threshold).mean())


def chamfer(a, b, **kwargs):
    distances = ((a[:, None] - b[None, :]) ** 2).sum(axis=2)
    return float(distances.min(axis=0).mean() + distances.min(axis=1).mean())
