from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
from PIL import Image

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    AmbientLights,
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    RasterizationSettings,
    SoftPhongShader,
)


RenderMode = Literal["cache", "online", "auto"]


@dataclass(frozen=True)
class CameraIntrinsics:
    """
    Camera intrinsics specification for benchmark rendering.

    Notes:
    - image_width / image_height are in pixels
    - fov_x_rad / fov_y_rad are in radians
    - principal_x / principal_y are in pixel coordinates
    - This intrinsics spec is fixed for a Render instance
    """
    image_width: int
    image_height: int
    fov_x_rad: float
    fov_y_rad: float
    principal_x: float
    principal_y: float


@dataclass(frozen=True)
class CameraPose:
    """
    Equivalent 6DoF camera pose specification using an over-parameterized form:

      - camera_xyz : camera center in the object/world frame
      - lookat_xyz : target point in the same frame, defining the viewing direction
      - roll_rad   : in-plane image rotation around the viewing axis
                     camera_xyz -> lookat_xyz, with camera_xyz and lookat_xyz fixed

    Important:
    - Render does NOT perform additional centering or scaling.
    - In the benchmark paper / official benchmark assets, objects may already be
      normalized beforehand, but this renderer itself does not assume or enforce that.
    - For fixed camera_xyz and lookat_xyz, changing roll_rad should only rotate
      the image plane around the principal point; it should not move the camera
      center or change the viewing direction itself.
    """
    camera_xyz: tuple[float, float, float]
    lookat_xyz: tuple[float, float, float]
    roll_rad: float = 0.0


@dataclass
class RGBDFrame:
    rgb: np.ndarray                    # uint8, (H, W, 3)
    depth: np.ndarray                  # float32, (H, W), standard RGB-D camera Z depth
    mask: np.ndarray                   # bool,   (H, W)
    intrinsics: CameraIntrinsics       # protocol-level camera intrinsics for this frame
    pose: CameraPose                   # protocol-level camera pose for this frame
    source: str                        # "cache" | "online"
    points_world: Optional[np.ndarray] # float32, (H, W, 3), optional
    points_world_from_depth: Optional[np.ndarray] = None  # float32, (H, W, 3), optional


def intrinsics_to_dict(intr: CameraIntrinsics) -> dict[str, float | int]:
    return {
        "image_width": int(intr.image_width),
        "image_height": int(intr.image_height),
        "fov_x_rad": float(intr.fov_x_rad),
        "fov_y_rad": float(intr.fov_y_rad),
        "principal_x": float(intr.principal_x),
        "principal_y": float(intr.principal_y),
    }


def intrinsics_from_dict(data: dict[str, Any]) -> CameraIntrinsics:
    return CameraIntrinsics(
        image_width=int(data["image_width"]),
        image_height=int(data["image_height"]),
        fov_x_rad=float(data["fov_x_rad"]),
        fov_y_rad=float(data["fov_y_rad"]),
        principal_x=float(data["principal_x"]),
        principal_y=float(data["principal_y"]),
    )


def pose_to_dict(pose: CameraPose) -> dict[str, Any]:
    return {
        "camera_xyz": [float(x) for x in pose.camera_xyz],
        "lookat_xyz": [float(x) for x in pose.lookat_xyz],
        "roll_rad": float(pose.roll_rad),
    }


def frame_meta_dict(
    frame: RGBDFrame,
    *,
    source: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    camera_to_world, world_to_camera = pose_to_extrinsics(frame.pose)
    meta = {
        "source": source or frame.source,
        "intrinsics": intrinsics_to_dict(frame.intrinsics),
        "pose": pose_to_dict(frame.pose),
        "camera_to_world": camera_to_world.tolist(),
        "world_to_camera": world_to_camera.tolist(),
    }
    if extra:
        meta.update(extra)
    return meta


def pose_to_extrinsics(pose: CameraPose) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert CameraPose into rigid extrinsics for the exported RGB-D camera frame.

    Exported camera convention:
    - +X right
    - +Y down
    - +Z forward

    Pose semantics:
    - camera_xyz is the camera center in object/world coordinates
    - lookat_xyz defines the forward viewing direction
    - roll_rad rotates the image plane around +Z forward while keeping the camera
      center and viewing direction fixed

    Returned transforms are 4x4 homogeneous matrices for column-vector usage:
    - camera_to_world: p_world_h = camera_to_world @ p_camera_h
    - world_to_camera: p_camera_h = world_to_camera @ p_world_h
    """
    camera_pos = np.asarray(pose.camera_xyz, dtype=np.float32)
    look_at = np.asarray(pose.lookat_xyz, dtype=np.float32)

    cam_z = look_at - camera_pos
    cam_z = cam_z / max(np.linalg.norm(cam_z), 1e-12)
    if np.linalg.norm(cam_z - np.array([0.0, 0.0, -1.0], dtype=np.float32)) < 1e-6:
        cam_z = np.array([1e-8, 1e-8, -1.0], dtype=np.float32)
        cam_z = cam_z / max(np.linalg.norm(cam_z), 1e-12)
    if np.linalg.norm(cam_z - np.array([0.0, 0.0, 1.0], dtype=np.float32)) < 1e-6:
        cam_z = np.array([1e-8, 1e-8, 1.0], dtype=np.float32)
        cam_z = cam_z / max(np.linalg.norm(cam_z), 1e-12)

    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    cam_x_left = np.cross(-cam_z, world_up)
    cam_x_left = cam_x_left / max(np.linalg.norm(cam_x_left), 1e-12)
    cam_y_up = np.cross(cam_x_left, -cam_z)
    cam_y_up = cam_y_up / max(np.linalg.norm(cam_y_up), 1e-12)

    c = np.cos(float(pose.roll_rad)).astype(np.float32)
    s = np.sin(float(pose.roll_rad)).astype(np.float32)
    cam_x_left_0 = cam_x_left
    cam_y_up_0 = cam_y_up
    cam_x_left = c * cam_x_left_0 + s * cam_y_up_0
    cam_y_up = -s * cam_x_left_0 + c * cam_y_up_0

    cam_x_right = -cam_x_left
    cam_y_down = -cam_y_up

    camera_to_world = np.eye(4, dtype=np.float32)
    camera_to_world[:3, :3] = np.stack([cam_x_right, cam_y_down, cam_z], axis=1)
    camera_to_world[:3, 3] = camera_pos

    world_to_camera = np.eye(4, dtype=np.float32)
    world_to_camera[:3, :3] = camera_to_world[:3, :3].T
    world_to_camera[:3, 3] = -world_to_camera[:3, :3] @ camera_pos
    return camera_to_world, world_to_camera


def make_camera_axes_pointcloud(
    pose: CameraPose,
    *,
    axis_length: float = 0.5,
    samples_per_axis: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample 3 colored axis segments for the exported RGB-D camera frame.

    Colors follow the conventional RGB axis coding:
    - +X : red
    - +Y : green
    - +Z : blue

    Returned points are in object/world coordinates.
    """
    if axis_length <= 0.0:
        raise ValueError(f"axis_length must be positive, got {axis_length}")
    if samples_per_axis < 2:
        raise ValueError(f"samples_per_axis must be >= 2, got {samples_per_axis}")

    camera_to_world, _ = pose_to_extrinsics(pose)
    R_c2w = camera_to_world[:3, :3]
    t_c2w = camera_to_world[:3, 3]

    axis_dirs_camera = np.array(
        [
            [1.0, 0.0, 0.0],  # +X right
            [0.0, 1.0, 0.0],  # +Y down
            [0.0, 0.0, 1.0],  # +Z forward
        ],
        dtype=np.float32,
    )
    axis_colors = np.array(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ],
        dtype=np.uint8,
    )

    ts = np.linspace(0.0, float(axis_length), int(samples_per_axis), dtype=np.float32)

    all_points = []
    all_colors = []
    for axis_dir_cam, axis_color in zip(axis_dirs_camera, axis_colors):
        pts_cam = ts[:, None] * axis_dir_cam[None, :]
        pts_world = pts_cam @ R_c2w.T + t_c2w[None, :]
        cols = np.repeat(axis_color[None, :], len(ts), axis=0)
        all_points.append(pts_world.astype(np.float32))
        all_colors.append(cols.astype(np.uint8))

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    return points, colors


def rgbd_to_colored_pointcloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    pose: CameraPose | None = None,
    renderer: "Render" | None = None,
    frame: Literal["camera", "world"] = "world",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert exported benchmark RGB-D into a colored point cloud.

    Exported RGB-D convention:
    - rgb / depth / mask share the same pixel grid
    - depth stores standard pinhole Z depth in the exported camera frame
      (+X right, +Y down, +Z forward)
    - for fixed camera_xyz and lookat_xyz, changing roll_rad corresponds to an
      in-plane image rotation around the principal point

    Args:
        rgb:
            uint8 image of shape (H, W, 3)
        depth:
            float32 depth map of shape (H, W)
        mask:
            bool visibility mask of shape (H, W)
        intrinsics:
            camera intrinsics used for rendering
        pose:
            required when frame == "world"; interpreted in object/world frame
        renderer:
            optional Render instance. When provided, we reuse the renderer's
            exact camera transform to map points into the world frame.
        frame:
            "camera": return points in exported RGB-D camera frame
            "world" : return points in object/world frame

    Returns:
        points:
            float32 array of shape (N, 3)
        colors:
            uint8 array of shape (N, 3)
    """
    h = intrinsics.image_height
    w = intrinsics.image_width

    if rgb.shape != (h, w, 3):
        raise ValueError(f"rgb shape mismatch: expected {(h, w, 3)}, got {rgb.shape}")
    if depth.shape != (h, w):
        raise ValueError(f"depth shape mismatch: expected {(h, w)}, got {depth.shape}")
    if mask.shape != (h, w):
        raise ValueError(f"mask shape mismatch: expected {(h, w)}, got {mask.shape}")

    fx = 0.5 * float(intrinsics.image_width) / math.tan(0.5 * float(intrinsics.fov_x_rad))
    fy = 0.5 * float(intrinsics.image_height) / math.tan(0.5 * float(intrinsics.fov_y_rad))

    u, v = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    u = u + 0.5
    v = v + 0.5

    z = depth.astype(np.float32)
    x = (u - float(intrinsics.principal_x)) * z / float(fx)
    y = (v - float(intrinsics.principal_y)) * z / float(fy)
    points_camera = np.stack([x, y, z], axis=-1).astype(np.float32)

    valid = mask.astype(bool)
    colors = rgb[valid].astype(np.uint8)

    if frame == "camera":
        return points_camera[valid], colors

    if frame != "world":
        raise ValueError(f"frame must be 'camera' or 'world', got {frame}")

    if pose is None:
        raise ValueError("pose is required when frame='world'")

    if renderer is not None:
        points_world = renderer._points_world_from_depth(depth, mask, pose)
        return points_world[valid], colors

    camera_to_world, _ = pose_to_extrinsics(pose)
    R_c2w = camera_to_world[:3, :3]
    t_c2w = camera_to_world[:3, 3]
    points_world = (
        points_camera.reshape(-1, 3) @ R_c2w.T + t_c2w[None, :]
    ).reshape(h, w, 3).astype(np.float32)
    return points_world[valid], colors


class Render:
    """
    Stateful benchmark render backend.

    Design choices:
    - One Render instance is bound to one OBJ asset.
    - Camera intrinsics are fixed at initialization.
    - render_loaded() only consumes CameraPose.
    - mode controls whether observations come from cache, online rendering, or fallback logic.

    Cache behavior:
    - Cache is indexed by pose values, not by integer view id.
    - Matching uses tolerance-based numeric comparison (default atol = 1e-4).

    Depth convention:
    - depth[y, x] stores the visible point's +Z coordinate in a standard RGB-D
      camera frame: +X right, +Y down, +Z from camera to scene.
    - This is the exported user-facing convention, so depth can be back-projected
      with standard pinhole intrinsics and aligned with rgb[y, x].
    - Background pixels are assigned depth = 0.0 and mask = False.
    - For fixed camera_xyz and lookat_xyz, changing roll_rad should behave like
      an in-plane rotation of rgb/depth/mask around the principal point.
    """

    def __init__(
        self,
        uid: str,
        obj_path: str | Path,
        intrinsics: CameraIntrinsics,
        *,
        device: str | torch.device = "cuda:0" if torch.cuda.is_available() else "cpu",
        mode: RenderMode = "auto",
        cache_index_json: str | Path | None = None,
        pose_match_atol: float = 1e-4,
        znear: float = 0.01,
        zfar: float = 10.0,
        background_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        ambient_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        cull_backfaces: bool = False,
    ) -> None:
        self.uid = str(uid)
        self.obj_path = Path(obj_path)
        self.intrinsics = intrinsics
        self.device = torch.device(device)
        self.mode = mode
        self.pose_match_atol = float(pose_match_atol)
        self.znear = float(znear)
        self.zfar = float(zfar)
        self.background_color = background_color
        self.ambient_color = ambient_color
        self.cull_backfaces = cull_backfaces

        if not self.obj_path.exists():
            raise FileNotFoundError(f"OBJ not found: {self.obj_path}")

        self._validate_intrinsics(self.intrinsics)

        self._renderer = self._make_renderer()
        self._mesh = self._load_mesh_with_texture(self.obj_path)

        self._cache_index: dict[str, Any] | None = None
        self._cache_index_path: Path | None = None
        if cache_index_json is not None:
            cache_index_path = Path(cache_index_json)
            if not cache_index_path.exists():
                raise FileNotFoundError(f"cache_index_json not found: {cache_index_path}")
            self._cache_index = self._load_json(cache_index_path)
            self._cache_index_path = cache_index_path.resolve()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_loaded(self, pose: CameraPose) -> RGBDFrame:
        """
        Main benchmark render API.

        mode == "cache":
            return cached RGBD if a numerically matching pose exists; fail otherwise

        mode == "online":
            always render online from the loaded OBJ mesh

        mode == "auto":
            try cache first, fall back to online rendering on miss
        """
        if self.mode == "cache":
            return self._render_from_cache(pose)

        if self.mode == "online":
            return self._render_online(pose)

        cached = self._try_render_from_cache(pose)
        if cached is not None:
            return cached
        return self._render_online(pose)

    # ------------------------------------------------------------------
    # Cache branch
    # ------------------------------------------------------------------

    def _try_render_from_cache(self, pose: CameraPose) -> Optional[RGBDFrame]:
        try:
            return self._render_from_cache(pose)
        except Exception:
            return None

    def _render_from_cache(self, pose: CameraPose) -> RGBDFrame:
        if self._cache_index is None:
            raise RuntimeError("No cache index loaded")

        cache_intrinsics = self._cache_index.get("intrinsics")
        if not isinstance(cache_intrinsics, dict):
            raise ValueError("cache index must contain a top-level 'intrinsics' dict")
        cached_intr = intrinsics_from_dict(cache_intrinsics)
        if not self._intrinsics_equal(self.intrinsics, cached_intr, atol=self.pose_match_atol):
            raise ValueError(
                "cache intrinsics mismatch between current Render instance and cache index"
            )

        objects = self._cache_index.get("objects")
        if not isinstance(objects, dict):
            raise ValueError("cache index must contain a top-level 'objects' dict")

        uid_entry = objects.get(self.uid)
        if uid_entry is None:
            raise KeyError(f"uid={self.uid} not found in cache index")

        views = uid_entry.get("views")
        if not isinstance(views, list):
            raise ValueError(f"cache entry for uid={self.uid} must contain a list field 'views'")

        matched_entry = None
        for entry in views:
            entry_pose = self._parse_pose_entry(entry)
            if self._pose_equal(pose, entry_pose, atol=self.pose_match_atol):
                matched_entry = entry
                break

        if matched_entry is None:
            raise KeyError(f"No cached pose matched for uid={self.uid}")

        rgb_path = self._resolve_cache_path(matched_entry["rgb"])
        depth_path = self._resolve_cache_path(matched_entry["depth"])
        mask_path = self._resolve_cache_path(matched_entry["mask"])

        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        depth = self._read_depth(depth_path)
        mask = self._read_mask(mask_path)

        points_world = None
        if "points_world" in matched_entry:
            points_path = self._resolve_cache_path(matched_entry["points_world"])
            if points_path.suffix.lower() == ".npy":
                points_world = np.load(points_path).astype(np.float32)
            elif points_path.suffix.lower() == ".npz":
                data = np.load(points_path)
                if "points_world" in data:
                    points_world = data["points_world"].astype(np.float32)
                elif "points" in data:
                    points_world = data["points"].astype(np.float32)

        self._validate_frame_shapes(rgb, depth, mask)
        points_world_from_depth = self._points_world_from_depth(depth, mask, pose)

        return RGBDFrame(
            rgb=rgb,
            depth=depth.astype(np.float32),
            mask=mask.astype(bool),
            intrinsics=cached_intr,
            pose=pose,
            source="cache",
            points_world=points_world,
            points_world_from_depth=points_world_from_depth,
        )

    def _parse_pose_entry(self, entry: dict[str, Any]) -> CameraPose:
        pose_dict = entry.get("pose")
        if not isinstance(pose_dict, dict):
            raise ValueError("Each cache view entry must contain a dict field 'pose'")

        camera_xyz = tuple(float(x) for x in pose_dict["camera_xyz"])
        lookat_xyz = tuple(float(x) for x in pose_dict["lookat_xyz"])
        roll_rad = float(pose_dict["roll_rad"])
        return CameraPose(
            camera_xyz=camera_xyz,
            lookat_xyz=lookat_xyz,
            roll_rad=roll_rad,
        )

    @staticmethod
    def _read_depth(path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return np.load(path).astype(np.float32)
        if suffix == ".npz":
            data = np.load(path)
            if "depth" not in data:
                raise KeyError(f"'depth' not found in {path}")
            return data["depth"].astype(np.float32)

        arr = np.asarray(Image.open(path))
        return arr.astype(np.float32)

    @staticmethod
    def _read_mask(path: Path) -> np.ndarray:
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return np.load(path).astype(bool)
        if suffix == ".npz":
            data = np.load(path)
            if "mask" in data:
                return data["mask"].astype(bool)
            if "valid" in data:
                return data["valid"].astype(bool)
            raise KeyError(f"'mask'/'valid' not found in {path}")

        arr = np.asarray(Image.open(path))
        return arr > 0

    # ------------------------------------------------------------------
    # Online branch
    # ------------------------------------------------------------------

    def _render_online(self, pose: CameraPose) -> RGBDFrame:
        rgb, depth, mask, points_world, points_world_from_depth = self._render_rgbd(self._mesh, pose)
        return RGBDFrame(
            rgb=rgb,
            depth=depth,
            mask=mask,
            intrinsics=self.intrinsics,
            pose=pose,
            source="online",
            points_world=points_world,
            points_world_from_depth=points_world_from_depth,
        )

    def _render_rgbd(
        self,
        mesh,
        pose: CameraPose,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        roll_rad = float(pose.roll_rad)
        camera_pos = np.asarray(pose.camera_xyz, dtype=np.float32)
        look_at = np.asarray(pose.lookat_xyz, dtype=np.float32)
        R, T = self._build_camera_R_T(camera_pos, look_at, roll_rad)
        R = R.to(self.device)
        T = T.to(self.device)

        cameras = self._make_cameras(R=R, T=T)

        rgba = self._renderer(mesh, cameras=cameras)
        rgba_np = (torch.clamp(rgba, 0, 1).detach().cpu().numpy()[0] * 255).astype(np.uint8)
        rgb = rgba_np[..., :3]

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=self._raster_settings(),
        )
        fragments = rasterizer(mesh)

        pix_to_face = fragments.pix_to_face
        bary_coords = fragments.bary_coords

        verts = mesh.verts_packed()
        faces = mesh.faces_packed()
        faces_verts = verts[faces]

        points_3d = interpolate_face_attributes(pix_to_face, bary_coords, faces_verts)
        points_world = points_3d[0, ..., 0, :].detach().cpu().numpy().astype(np.float32)

        face_id = pix_to_face[0, ..., 0]
        mask = (face_id >= 0).detach().cpu().numpy().astype(bool)
        points_world[~mask] = np.nan

        points_world_t = torch.from_numpy(points_world.reshape(1, -1, 3)).to(
            device=self.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            points_view_t = cameras.get_world_to_view_transform().transform_points(points_world_t)
        points_view = points_view_t[0].reshape(points_world.shape[0], points_world.shape[1], 3)
        # Convert PyTorch3D view coordinates (+X left, +Y up, +Z forward)
        # into a standard RGB-D camera frame (+X right, +Y down, +Z forward).
        points_cam = points_view.detach().cpu().numpy().astype(np.float32)
        points_cam[..., 0] *= -1.0
        points_cam[..., 1] *= -1.0

        depth = points_cam[..., 2].astype(np.float32)
        depth[~mask] = 0.0
        depth = np.maximum(depth, 0.0)
        points_world_from_depth = self._points_world_from_depth(
            depth,
            mask,
            pose,
            cameras=cameras,
        )

        self._validate_frame_shapes(rgb, depth, mask)
        return rgb, depth, mask, points_world, points_world_from_depth

    def _points_world_from_depth(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        pose: CameraPose,
        *,
        cameras: PerspectiveCameras | None = None,
    ) -> np.ndarray:
        """
        Reconstruct world-space points from standard RGB-D depth under the
        exported camera convention: +X right, +Y down, +Z forward.

        Important:
        - We convert the exported RGB-D camera coordinates back into PyTorch3D
          view coordinates (+X left, +Y up, +Z forward) first.
        - Then we apply the inverse of the exact world->view transform used by
          the renderer. This avoids subtle basis mismatches between a manually
          reconstructed camera frame and the actual camera convention inside
          PyTorch3D.
        """
        h = self.intrinsics.image_height
        w = self.intrinsics.image_width

        if depth.shape != (h, w):
            raise ValueError(f"depth shape mismatch: expected {(h, w)}, got {depth.shape}")
        if mask.shape != (h, w):
            raise ValueError(f"mask shape mismatch: expected {(h, w)}, got {mask.shape}")

        fx = self._fov_to_focal_px(
            fov_rad=self.intrinsics.fov_x_rad,
            image_extent_px=self.intrinsics.image_width,
        )
        fy = self._fov_to_focal_px(
            fov_rad=self.intrinsics.fov_y_rad,
            image_extent_px=self.intrinsics.image_height,
        )

        camera_pos = np.asarray(pose.camera_xyz, dtype=np.float32)
        look_at = np.asarray(pose.lookat_xyz, dtype=np.float32)

        u, v = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )
        # Use pixel centers for standard pinhole back-projection.
        u = u + 0.5
        v = v + 0.5

        z_cam = depth.astype(np.float32)
        x_cam = (u - float(self.intrinsics.principal_x)) * z_cam / float(fx)
        y_cam = (v - float(self.intrinsics.principal_y)) * z_cam / float(fy)

        if cameras is None:
            R, T = self._build_camera_R_T(camera_pos, look_at, float(pose.roll_rad))
            R = R.to(self.device)
            T = T.to(self.device)
            cameras = self._make_cameras(R=R, T=T)

        # Exported standard RGB-D frame -> PyTorch3D view frame.
        points_view = np.stack(
            [
                -x_cam,
                -y_cam,
                z_cam,
            ],
            axis=-1,
        )

        points_view_t = torch.from_numpy(points_view.reshape(1, -1, 3)).to(
            device=self.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            view_to_world = cameras.get_world_to_view_transform().inverse()
            points_world_t = view_to_world.transform_points(points_view_t)

        points_world = (
            points_world_t[0]
            .reshape(h, w, 3)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        points_world[~mask] = np.nan
        return points_world

    # ------------------------------------------------------------------
    # Camera + renderer helpers
    # ------------------------------------------------------------------

    def _make_renderer(self) -> MeshRenderer:
        blend_params = BlendParams(background_color=self.background_color)
        lights = AmbientLights(
            device=self.device,
            ambient_color=(self.ambient_color,),
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=None,
                raster_settings=self._raster_settings(),
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=None,
                lights=lights,
                blend_params=blend_params,
            ),
        )
        return renderer

    def _raster_settings(self) -> RasterizationSettings:
        return RasterizationSettings(
            image_size=(self.intrinsics.image_height, self.intrinsics.image_width),
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=self.cull_backfaces,
            bin_size=0,
        )

    def _make_cameras(self, R: torch.Tensor, T: torch.Tensor) -> PerspectiveCameras:
        fx = self._fov_to_focal_px(
            fov_rad=self.intrinsics.fov_x_rad,
            image_extent_px=self.intrinsics.image_width,
        )
        fy = self._fov_to_focal_px(
            fov_rad=self.intrinsics.fov_y_rad,
            image_extent_px=self.intrinsics.image_height,
        )

        focal_length = torch.tensor(
            [[fx, fy]],
            dtype=torch.float32,
            device=self.device,
        )
        principal_point = torch.tensor(
            [[self.intrinsics.principal_x, self.intrinsics.principal_y]],
            dtype=torch.float32,
            device=self.device,
        )
        image_size = torch.tensor(
            [[self.intrinsics.image_height, self.intrinsics.image_width]],
            dtype=torch.float32,
            device=self.device,
        )

        return PerspectiveCameras(
            device=self.device,
            R=R,
            T=T,
            focal_length=focal_length,
            principal_point=principal_point,
            in_ndc=False,
            image_size=image_size,
        )

    @staticmethod
    def _fov_to_focal_px(fov_rad: float, image_extent_px: int) -> float:
        if fov_rad <= 0.0 or fov_rad >= math.pi:
            raise ValueError(f"fov_rad must be in (0, pi), got {fov_rad}")
        return 0.5 * float(image_extent_px) / math.tan(0.5 * float(fov_rad))

    def _load_mesh_with_texture(self, obj_path: Path):
        """
        Load OBJ with texture.

        Notes:
        - No extra centering/scaling is performed here.
        - If the benchmark officially uses normalized assets, that should be enforced
          by the benchmark data pipeline, not by this renderer.
        """
        mesh = load_objs_as_meshes(
            [str(obj_path)],
            device=self.device,
            load_textures=True,
        )
        if mesh.isempty():
            raise RuntimeError(f"Loaded empty mesh: {obj_path}")
        return mesh

    # ------------------------------------------------------------------
    # Pose math
    # ------------------------------------------------------------------

    def _build_camera_R_T(
        self,
        camera_pos: np.ndarray,
        look_at: np.ndarray,
        roll_rad: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build world-to-camera extrinsics from:
        - camera position
        - look-at target
        - in-plane roll around the viewing axis

        The resulting camera frame is first constructed in the PyTorch3D view
        convention (+X left, +Y up, +Z forward), then roll is applied around the
        forward viewing axis. The exported user-facing RGB-D convention is derived
        from this by flipping X/Y into (+X right, +Y down, +Z forward).
        """
        if camera_pos.shape != (3,) or look_at.shape != (3,):
            raise ValueError("camera_pos and look_at must both be shape (3,)")

        cam_x_rolled, cam_y_rolled, cam_z = self._camera_axes(camera_pos, look_at, roll_rad)

        # PyTorch3D uses row-vector world->view transforms:
        #   X_cam = X_world R + T
        # Therefore R should store the camera basis vectors as columns in world
        # coordinates, and T = -C R where C is the camera center in world space.
        R_w2c = np.stack([cam_x_rolled, cam_y_rolled, cam_z], axis=1).astype(np.float32)
        T_w2c = (-camera_pos.reshape(1, 3) @ R_w2c).astype(np.float32)

        R = torch.from_numpy(R_w2c).float().unsqueeze(0)
        T = torch.from_numpy(T_w2c[0]).float().unsqueeze(0)
        return R, T

    def _camera_axes(
        self,
        camera_pos: np.ndarray,
        look_at: np.ndarray,
        roll_rad: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return camera axes in the PyTorch3D view convention:
        - cam_x : +X left
        - cam_y : +Y up
        - cam_z : +Z forward

        roll_rad rotates the image plane around cam_z while keeping camera_pos
        and look_at fixed.
        """
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        cam_z = self._normalize_np(look_at - camera_pos).astype(np.float32)
        if np.linalg.norm(cam_z) < 1e-12:
            raise ValueError("camera_xyz and lookat_xyz must not be identical")

        # Keep the original singularity handling logic unchanged
        if np.linalg.norm(cam_z - np.array([0.0, 0.0, -1.0], dtype=np.float32)) < 1e-6:
            cam_z = self._normalize_np(np.array([1e-8, 1e-8, -1.0], dtype=np.float32))

        if np.linalg.norm(cam_z - np.array([0.0, 0.0, 1.0], dtype=np.float32)) < 1e-6:
            cam_z = self._normalize_np(np.array([1e-8, 1e-8, 1.0], dtype=np.float32))

        cam_x = self._normalize_np(np.cross(-cam_z, world_up))
        cam_y = self._normalize_np(np.cross(cam_x, -cam_z))

        # Apply in-plane roll around the forward viewing axis cam_z.
        c = np.cos(roll_rad).astype(np.float32)
        s = np.sin(roll_rad).astype(np.float32)

        cam_x_rolled = c * cam_x + s * cam_y
        cam_y_rolled = -s * cam_x + c * cam_y
        return cam_x_rolled, cam_y_rolled, cam_z

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_np(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        n = np.linalg.norm(v)
        if n < eps:
            return v
        return v / n

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_cache_path(self, path_str: str) -> Path:
        path = Path(path_str)
        if path.is_absolute():
            return path
        if self._cache_index_path is None:
            return path
        return (self._cache_index_path.parent / path).resolve()

    @staticmethod
    def _validate_intrinsics(intr: CameraIntrinsics) -> None:
        if intr.image_width <= 0 or intr.image_height <= 0:
            raise ValueError("image_width and image_height must be positive")
        if not (0.0 < intr.fov_x_rad < math.pi):
            raise ValueError("fov_x_rad must be in (0, pi)")
        if not (0.0 < intr.fov_y_rad < math.pi):
            raise ValueError("fov_y_rad must be in (0, pi)")

    @staticmethod
    def _intrinsics_equal(a: CameraIntrinsics, b: CameraIntrinsics, atol: float) -> bool:
        return (
            abs(float(a.image_width) - float(b.image_width)) <= atol
            and abs(float(a.image_height) - float(b.image_height)) <= atol
            and abs(float(a.fov_x_rad) - float(b.fov_x_rad)) <= atol
            and abs(float(a.fov_y_rad) - float(b.fov_y_rad)) <= atol
            and abs(float(a.principal_x) - float(b.principal_x)) <= atol
            and abs(float(a.principal_y) - float(b.principal_y)) <= atol
        )

    def _validate_frame_shapes(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        h = self.intrinsics.image_height
        w = self.intrinsics.image_width

        if rgb.shape != (h, w, 3):
            raise ValueError(f"rgb shape mismatch: expected {(h, w, 3)}, got {rgb.shape}")
        if depth.shape != (h, w):
            raise ValueError(f"depth shape mismatch: expected {(h, w)}, got {depth.shape}")
        if mask.shape != (h, w):
            raise ValueError(f"mask shape mismatch: expected {(h, w)}, got {mask.shape}")

    @staticmethod
    def _pose_equal(a: CameraPose, b: CameraPose, atol: float) -> bool:
        return (
            np.allclose(np.asarray(a.camera_xyz), np.asarray(b.camera_xyz), atol=atol, rtol=0.0)
            and np.allclose(np.asarray(a.lookat_xyz), np.asarray(b.lookat_xyz), atol=atol, rtol=0.0)
            and abs(float(a.roll_rad) - float(b.roll_rad)) <= atol
        )


def make_depth_vis(
    depth: np.ndarray,
    mask: np.ndarray,
    *,
    near_is_white: bool = False,
    qmin: float = 1.0,
    qmax: float = 99.0,
) -> np.ndarray:
    """
    Create a uint8 grayscale visualization for depth.

    Args:
        depth: (H, W) float32 view-space Z depth
        mask:  (H, W) bool valid mask
        near_is_white:
            - False: near -> dark, far -> bright
            - True : near -> bright, far -> dark
        qmin, qmax:
            percentile range used for robust normalization

    Returns:
        vis: (H, W) uint8 grayscale image
    """
    vis = np.zeros(depth.shape, dtype=np.uint8)
    valid = mask.astype(bool)

    if not np.any(valid):
        return vis

    d = depth[valid].astype(np.float32)

    lo = float(np.percentile(d, qmin))
    hi = float(np.percentile(d, qmax))

    if hi - lo < 1e-12:
        vis[valid] = 255
        return vis

    x = np.clip((depth[valid] - lo) / (hi - lo), 0.0, 1.0)

    if near_is_white:
        x = 1.0 - x

    vis[valid] = (x * 255.0).astype(np.uint8)
    return vis  


def save_pcd(points: np.ndarray, path: str | Path, colors: np.ndarray | None = None) -> None:
    import open3d as o3d

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    if colors is not None:
        cols = np.asarray(colors, dtype=np.float64)
        if cols.shape != pts.shape:
            raise ValueError(f"colors must have shape {pts.shape}, got {cols.shape}")
        if cols.max() > 1.0:
            cols = cols / 255.0
        pcd.colors = o3d.utility.Vector3dVector(cols)

    o3d.io.write_point_cloud(str(path), pcd, write_ascii=True)


def append_camera_axes_to_pointcloud(
    points: np.ndarray,
    colors: np.ndarray,
    pose: CameraPose,
    *,
    axis_length: float = 0.5,
    samples_per_axis: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    axis_points, axis_colors = make_camera_axes_pointcloud(
        pose,
        axis_length=axis_length,
        samples_per_axis=samples_per_axis,
    )
    return (
        np.concatenate([points.astype(np.float32), axis_points.astype(np.float32)], axis=0),
        np.concatenate([colors.astype(np.uint8), axis_colors.astype(np.uint8)], axis=0),
    )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal test entry for Render")

    parser.add_argument("--uid", type=str, default="debug_uid")
    parser.add_argument("--obj-path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="online", choices=["cache", "online", "auto"])
    parser.add_argument("--cache-index-json", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--fov-x-deg", type=float, default=45.0)
    parser.add_argument("--fov-y-deg", type=float, default=45.0)
    parser.add_argument("--principal-x", type=float, default=256.0)
    parser.add_argument("--principal-y", type=float, default=256.0)

    parser.add_argument("--camera-x", type=float, default=3.0)
    parser.add_argument("--camera-y", type=float, default=0.0)
    parser.add_argument("--camera-z", type=float, default=0.0)

    parser.add_argument("--lookat-x", type=float, default=0.0)
    parser.add_argument("--lookat-y", type=float, default=0.0)
    parser.add_argument("--lookat-z", type=float, default=0.0)

    parser.add_argument("--roll-deg", type=float, default=0.0)

    parser.add_argument("--out-rgb", type=str, default="debug_rgb.png")
    parser.add_argument("--out-depth", type=str, default="debug_depth.npy")
    parser.add_argument("--out-mask", type=str, default="debug_mask.png")
    parser.add_argument("--out-frame-meta", type=str, default="debug_frame_meta.json")
    parser.add_argument("--out-depth-vis", type=str, default=None)
    parser.add_argument("--out-pcd", type=str, default=None)
    parser.add_argument("--out-pcd-depth", type=str, default=None)
    parser.add_argument("--out-pcd-world", type=str, default=None)
    parser.add_argument("--out-pcd-rgbd", type=str, default=None)
    parser.add_argument("--debug-camera-axes", action="store_true")
    parser.add_argument("--debug-camera-axes-len", type=float, default=0.5)
    parser.add_argument("--debug-camera-axes-samples", type=int, default=64)

    return parser


def main() -> int:
    parser = _build_argparser()
    args = parser.parse_args()

    intr = CameraIntrinsics(
        image_width=args.image_width,
        image_height=args.image_height,
        fov_x_rad=np.deg2rad(args.fov_x_deg),
        fov_y_rad=np.deg2rad(args.fov_y_deg),
        principal_x=args.principal_x,
        principal_y=args.principal_y,
    )

    pose = CameraPose(
        camera_xyz=(args.camera_x, args.camera_y, args.camera_z),
        lookat_xyz=(args.lookat_x, args.lookat_y, args.lookat_z),
        roll_rad=np.deg2rad(args.roll_deg),
    )

    renderer = Render(
        uid=args.uid,
        obj_path=args.obj_path,
        intrinsics=intr,
        device=args.device,
        mode=args.mode,
        cache_index_json=args.cache_index_json,
    )

    frame = renderer.render_loaded(pose)

    Image.fromarray(frame.rgb).save(args.out_rgb)
    np.save(args.out_depth, frame.depth)

    mask_png = (frame.mask.astype(np.uint8) * 255)
    Image.fromarray(mask_png).save(args.out_mask)

    frame_meta = frame_meta_dict(frame)
    with Path(args.out_frame_meta).open("w", encoding="utf-8") as f:
        json.dump(frame_meta, f, indent=2)

    if args.out_depth_vis is not None:
        depth_vis = make_depth_vis(frame.depth, frame.mask)
        Image.fromarray(depth_vis).save(args.out_depth_vis)

    out_pcd_world = args.out_pcd_world or args.out_pcd
    if out_pcd_world is not None:
        if frame.points_world is None:
            print("[WARN] points_world is unavailable in this frame; skip world PCD export.")
        else:
            pts = frame.points_world[frame.mask]
            cols = frame.rgb[frame.mask]
            if args.debug_camera_axes:
                pts, cols = append_camera_axes_to_pointcloud(
                    pts,
                    cols,
                    pose,
                    axis_length=args.debug_camera_axes_len,
                    samples_per_axis=args.debug_camera_axes_samples,
                )
            save_pcd(pts, out_pcd_world, colors=cols)

    if args.out_pcd_depth is not None:
        if frame.points_world_from_depth is None:
            print("[WARN] points_world_from_depth is unavailable in this frame; skip depth PCD export.")
        else:
            pts = frame.points_world_from_depth[frame.mask]
            cols = frame.rgb[frame.mask]
            if args.debug_camera_axes:
                pts, cols = append_camera_axes_to_pointcloud(
                    pts,
                    cols,
                    pose,
                    axis_length=args.debug_camera_axes_len,
                    samples_per_axis=args.debug_camera_axes_samples,
                )
            save_pcd(pts, args.out_pcd_depth, colors=cols)

    if args.out_pcd_rgbd is not None:
        pts, cols = rgbd_to_colored_pointcloud(
            frame.rgb,
            frame.depth,
            frame.mask,
            intr,
            pose=pose,
            renderer=renderer,
            frame="world",
        )
        if args.debug_camera_axes:
            pts, cols = append_camera_axes_to_pointcloud(
                pts,
                cols,
                pose,
                axis_length=args.debug_camera_axes_len,
                samples_per_axis=args.debug_camera_axes_samples,
            )
        save_pcd(pts, args.out_pcd_rgbd, colors=cols)

    print("[DONE]")
    print(f"source    : {frame.source}")
    print(f"rgb       : {args.out_rgb}")
    print(f"depth     : {args.out_depth}")
    print(f"mask      : {args.out_mask}")
    print(f"frame_meta: {args.out_frame_meta}")
    if args.out_depth_vis is not None:
        print(f"depth_vis : {args.out_depth_vis}")
    if out_pcd_world is not None:
        print(f"pcd_world : {out_pcd_world}")
    if args.out_pcd_depth is not None:
        print(f"pcd_depth : {args.out_pcd_depth}")
    if args.out_pcd_rgbd is not None:
        print(f"pcd_rgbd  : {args.out_pcd_rgbd}")
    if args.debug_camera_axes:
        print(
            "camera_axes: enabled "
            f"(len={args.debug_camera_axes_len:.4f}, samples={args.debug_camera_axes_samples})"
        )
    print(f"rgb shape : {frame.rgb.shape}")

    valid_depth = frame.depth[frame.mask]
    print("depth min/max (valid only): ", end="")
    if len(valid_depth) == 0:
        print("no valid pixels")
    else:
        print(f"{valid_depth.min():.6f} / {valid_depth.max():.6f}")

    if frame.points_world is not None and frame.points_world_from_depth is not None:
        diff = frame.points_world_from_depth[frame.mask] - frame.points_world[frame.mask]
        dist = np.linalg.norm(diff, axis=1)
        print("pcd delta (depth vs world): ", end="")
        if len(dist) == 0:
            print("no valid points")
        else:
            print(f"mean={dist.mean():.6e}, max={dist.max():.6e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
