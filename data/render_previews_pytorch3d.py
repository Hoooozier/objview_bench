import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import numpy as np
from PIL import Image

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    SoftPhongShader,
    AmbientLights,
    BlendParams,
    TexturesUV,
)


BASE_DIR = Path("geometry_sampled")
INPUT_JSON_PATH = BASE_DIR / "manual_review_candidates_with_risk.json"
OUTPUT_PREVIEW_DIR = BASE_DIR / "previews"

IMAGE_SIZE = 512
CAMERA_DISTANCE = 3.0
FOV_DEG = 45.0

VIEW_SPECS = {
    "front": np.array([0.0, 0.0, CAMERA_DISTANCE], dtype=np.float32),
    "side":  np.array([CAMERA_DISTANCE, 0.0, 0.0], dtype=np.float32),
    "top":   np.array([0.0, CAMERA_DISTANCE, 0.0], dtype=np.float32),
    "iso":   np.array([CAMERA_DISTANCE, CAMERA_DISTANCE, CAMERA_DISTANCE], dtype=np.float32),
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_np(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def build_camera_R_T(camera_pos: np.ndarray, look_at: np.ndarray) -> (torch.Tensor, torch.Tensor):
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    cam_z = normalize_np(look_at - camera_pos).astype(np.float32)

    if np.linalg.norm(cam_z - np.array([0.0, 0.0, -1.0], dtype=np.float32)) < 1e-6:
        cam_z = normalize_np(np.array([1e-8, 1e-8, -1.0], dtype=np.float32))

    if np.linalg.norm(cam_z - np.array([0.0, 0.0, 1.0], dtype=np.float32)) < 1e-6:
        cam_z = normalize_np(np.array([1e-8, 1e-8, 1.0], dtype=np.float32))

    cam_x = normalize_np(np.cross(-cam_z, world_up))
    cam_y = normalize_np(np.cross(cam_x, -cam_z))

    R_c2w = np.stack([cam_x, cam_y, cam_z], axis=1)
    R_w2c = R_c2w.T
    T_w2c = -R_w2c @ camera_pos.reshape(3, 1)

    R = torch.from_numpy(R_w2c).float().unsqueeze(0)
    T = torch.from_numpy(T_w2c[:, 0]).float().unsqueeze(0)
    return R, T


def make_renderer(device: torch.device) -> MeshRenderer:
    raster_settings = RasterizationSettings(
        image_size=IMAGE_SIZE,
        blur_radius=0.0,
        faces_per_pixel=1,
        cull_backfaces=False,
        bin_size=0
    )

    blend_params = BlendParams(background_color=(1.0, 1.0, 1.0))

    lights = AmbientLights(device=device, ambient_color=((1.0, 1.0, 1.0),))

    # Cameras are provided dynamically at each render call.
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=None,
            raster_settings=raster_settings,
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=None,
            lights=lights,
            blend_params=blend_params,
        ),
    )
    return renderer


def render_one_view(mesh, renderer: MeshRenderer, device: torch.device, eye: np.ndarray) -> np.ndarray:
    look_at = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    R, T = build_camera_R_T(eye.astype(np.float32), look_at)

    R = R.to(device)
    T = T.to(device)

    cameras = FoVPerspectiveCameras(
        device=device,
        R=R,
        T=T,
        fov=FOV_DEG,
        znear=0.01,
        zfar=20.0,
    )

    images = renderer(mesh, cameras=cameras)
    image = images[0, ..., :3].detach().cpu().numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image


def load_mesh_with_texture(obj_path: Path, device: torch.device):
    """
    Load OBJ + MTL + UV textures directly with PyTorch3D.
    If texture assets exist, they are automatically constructed as TexturesUV.
    """
    mesh = load_objs_as_meshes([str(obj_path)], device=device, load_textures=True)

    if mesh.isempty():
        raise RuntimeError(f"Loaded empty mesh: {obj_path}")

    return mesh


def render_uid_previews(
    uid: str,
    obj_path: Path,
    output_dir: Path,
    renderer: MeshRenderer,
    device: torch.device,
) -> bool:
    if not obj_path.exists():
        print(f"[WARN] OBJ not found for uid={uid}: {obj_path}")
        return False

    ensure_dir(output_dir)

    try:
        mesh = load_mesh_with_texture(obj_path, device)

        for view_name, eye in VIEW_SPECS.items():
            rgb = render_one_view(mesh, renderer, device, eye)
            out_path = output_dir / f"{view_name}.png"
            Image.fromarray(rgb).save(out_path)

        return True

    except Exception as e:
        print(f"[WARN] Failed rendering uid={uid}: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=str(INPUT_JSON_PATH),
        help="Path to manual_review_candidates_with_risk.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_PREVIEW_DIR),
        help="Directory to save previews",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index in input list",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="End index (exclusive), -1 means all",
    )
    parser.add_argument(
        "--only-priority-ge",
        type=int,
        default=None,
        help="Only render samples with manual_priority_score >= this value",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing preview images",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Rendering device, e.g. cuda:0 or cpu",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[INFO] using device: {device}")

    input_path = Path(args.input)
    output_root = Path(args.output_dir)

    rows: List[Dict[str, Any]] = load_json(input_path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in {input_path}, got {type(rows)}")

    if args.only_priority_ge is not None:
        rows = [
            r for r in rows
            if int(r.get("manual_priority_score", 0)) >= args.only_priority_ge
        ]

    total_before_slice = len(rows)

    start = max(0, args.start)
    end = len(rows) if args.end < 0 else min(args.end, len(rows))
    rows = rows[start:end]

    print(f"[INFO] total rows before slice/filter: {total_before_slice}")
    print(f"[INFO] rendering rows in range [{start}, {end}) -> {len(rows)} samples")
    print(f"[INFO] output dir: {output_root}")

    renderer = make_renderer(device)

    success = 0
    skipped = 0
    failed = 0

    for i, row in enumerate(rows, start=1):
        uid = str(row["uid"])
        obj_path = Path(row["obj_path"])
        uid_out_dir = output_root / uid

        expected_files = [
            uid_out_dir / "front.png",
            uid_out_dir / "side.png",
            uid_out_dir / "top.png",
            uid_out_dir / "iso.png",
        ]

        if (not args.overwrite) and all(p.exists() for p in expected_files):
            skipped += 1
            print(f"[{i}/{len(rows)}] skip uid={uid} (already exists)")
            continue

        ok = render_uid_previews(uid, obj_path, uid_out_dir, renderer, device)
        if ok:
            success += 1
            print(f"[{i}/{len(rows)}] ok   uid={uid}")
        else:
            failed += 1
            print(f"[{i}/{len(rows)}] fail uid={uid}")

    print("[DONE]")
    print(f"  success: {success}")
    print(f"  skipped: {skipped}")
    print(f"  failed:  {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
python render_previews_pytorch3d.py \
  --input geometry_sampled/manual_review_candidates_with_risk.json \
  --output-dir geometry_sampled/previews \
  --device cuda:0 \
  --start 0 --end 8000

python render_previews_pytorch3d.py \
  --input geometry_sampled/manual_review_candidates_with_risk.json \
  --output-dir geometry_sampled/previews \
  --device cuda:0 \
  --start 8000 --end 12000
"""