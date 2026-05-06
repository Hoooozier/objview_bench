import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict

ROOT = Path(__file__).resolve().parent
import sys

sys.path.append(str(ROOT))

from datasets.io import IO
from models import build_model_from_cfg

def parse_args():
    parser = argparse.ArgumentParser(description="Run a single PoinTr completion inference.")
    parser.add_argument("--pc", required=True, help="Input partial point cloud (.npy/.ply/.pcd/.txt).")
    parser.add_argument(
        "--output-pc",
        required=True,
        help="Output completed point cloud path (.npy/.txt/.ply/.pcd).",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "cfgs" / "ShapeNet55_models" / "PoinTr.yaml"),
        help="Model config yaml.",
    )
    parser.add_argument(
        "--ckpt",
        default=str(ROOT / "PoinTr-C" / "ckpt-best.pth"),
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device. PoinTr in this repo expects CUDA.",
    )
    parser.add_argument(
        "--fuse-farthest-tammes128",
        action="store_true",
        help="Fuse the input view with the farthest Tammes-128 view from the same cache directory before inference.",
    )
    parser.add_argument(
        "--tammes128",
        default=str(ROOT.parent / "Tammes_sphere" / "128_xyz.txt"),
        help="Tammes-128 xyz file used to find the farthest companion view.",
    )
    return parser.parse_args()


def _merge_config(config: EasyDict, new_config: dict, config_root: Path) -> EasyDict:
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == "_base_":
                base_path = Path(val)
                if not base_path.is_absolute():
                    candidate_local = (config_root / base_path).resolve()
                    candidate_repo = (ROOT / base_path).resolve()
                    if candidate_local.exists():
                        base_path = candidate_local
                    elif candidate_repo.exists():
                        base_path = candidate_repo
                    else:
                        base_path = candidate_local
                with open(base_path, "r", encoding="utf-8") as f:
                    base_cfg = yaml.load(f, Loader=yaml.FullLoader)
                config[key] = EasyDict()
                _merge_config(config[key], base_cfg, base_path.parent)
            else:
                config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        _merge_config(config[key], val, config_root)
    return config


def load_config(cfg_path: Path) -> EasyDict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        root_cfg = yaml.load(f, Loader=yaml.FullLoader)
    return _merge_config(EasyDict(), root_cfg, cfg_path.parent)


def upsample_or_downsample_to_2048(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected point cloud shape [N, 3], got {points.shape}")

    n_points = points.shape[0]
    target = 2048
    if n_points == target:
        return points.astype(np.float32, copy=False)
    if n_points > target:
        choice = rng.permutation(n_points)[:target]
        return points[choice].astype(np.float32, copy=False)

    out = points.astype(np.float32, copy=True)
    curr = out.shape[0]
    need = target - curr
    while curr <= need:
        out = np.tile(out, (2, 1))
        need -= curr
        curr *= 2
    choice = rng.permutation(curr)[:need]
    out = np.concatenate([out, out[choice]], axis=0)
    return out.astype(np.float32, copy=False)


def infer_view_index_from_filename(pc_path: Path) -> int:
    stem = pc_path.stem
    if not stem.startswith("view_"):
        raise ValueError(f"Expected input file name like view_000.pcd, got {pc_path.name}")
    return int(stem.split("_", 1)[1])


def load_tammes_xyz(path: Path) -> np.ndarray:
    xyz = np.loadtxt(path, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected Tammes xyz shape [N,3], got {xyz.shape}")
    return xyz


def farthest_tammes_index(tammes_xyz: np.ndarray, source_index: int) -> int:
    if source_index < 0 or source_index >= tammes_xyz.shape[0]:
        raise IndexError(f"source_index={source_index} out of range for Tammes set size {tammes_xyz.shape[0]}")
    source = tammes_xyz[source_index]
    d2 = np.sum((tammes_xyz - source[None, :]) ** 2, axis=1)
    d2[source_index] = -1.0
    return int(np.argmax(d2))


def maybe_fuse_with_farthest_tammes_view(points: np.ndarray, input_path: Path, tammes_path: Path) -> tuple[np.ndarray, dict]:
    source_index = infer_view_index_from_filename(input_path)
    tammes_xyz = load_tammes_xyz(tammes_path)
    companion_index = farthest_tammes_index(tammes_xyz, source_index)
    companion_path = input_path.with_name(f"view_{companion_index:03d}{input_path.suffix}")
    if not companion_path.exists():
        raise FileNotFoundError(f"Farthest companion view file not found: {companion_path}")
    companion_points = IO.get(str(companion_path)).astype(np.float32)
    fused = np.concatenate([points, companion_points], axis=0)
    return fused, {
        "enabled": True,
        "source_view_index": source_index,
        "companion_view_index": companion_index,
        "companion_path": str(companion_path),
        "num_points_primary": int(points.shape[0]),
        "num_points_companion": int(companion_points.shape[0]),
        "num_points_fused": int(fused.shape[0]),
    }


def maybe_normalize_shapenet(points: np.ndarray, config) -> tuple[np.ndarray, dict]:
    dataset_name = str(config.dataset.train._base_.get("NAME", ""))
    if dataset_name != "ShapeNet":
        return points, {"applied": False}

    centroid = np.mean(points, axis=0)
    centered = points - centroid
    scale = np.max(np.sqrt(np.sum(centered**2, axis=1)))
    if scale <= 0:
        scale = 1.0
    normalized = centered / scale
    return normalized.astype(np.float32, copy=False), {
        "applied": True,
        "centroid": centroid.astype(np.float32),
        "scale": float(scale),
    }


def maybe_denormalize_shapenet(points: np.ndarray, norm_meta: dict) -> np.ndarray:
    if not norm_meta.get("applied", False):
        return points
    return points * norm_meta["scale"] + norm_meta["centroid"]


def load_weights(model: torch.nn.Module, ckpt_path: Path) -> None:
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    if checkpoint.get("model") is not None:
        state_dict = checkpoint["model"]
    elif checkpoint.get("base_model") is not None:
        state_dict = checkpoint["base_model"]
    else:
        raise RuntimeError(f"Could not find model/base_model weights in checkpoint: {ckpt_path}")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)


def save_point_cloud(points: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".npy":
        np.save(output_path, points.astype(np.float32))
        return
    if suffix == ".txt":
        np.savetxt(output_path, points.astype(np.float32), fmt="%.8f")
        return
    if suffix in {".ply", ".pcd"}:
        import open3d as o3d

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if not o3d.io.write_point_cloud(str(output_path), pc):
            raise RuntimeError(f"Failed to write point cloud to {output_path}")
        return
    raise ValueError(f"Unsupported output suffix: {suffix}")


def main():
    args = parse_args()
    if not args.device.lower().startswith("cuda"):
        raise RuntimeError("This PoinTr variant currently expects a CUDA device.")

    config_path = Path(args.config).resolve()
    ckpt_path = Path(args.ckpt).resolve()
    input_path = Path(args.pc).resolve()
    output_path = Path(args.output_pc).resolve()

    rng = np.random.default_rng(0)

    config = load_config(config_path)
    t0 = time.perf_counter()
    raw_points = IO.get(str(input_path)).astype(np.float32)
    fuse_meta = {"enabled": False}
    if args.fuse_farthest_tammes128:
        raw_points, fuse_meta = maybe_fuse_with_farthest_tammes_view(
            raw_points,
            input_path=input_path,
            tammes_path=Path(args.tammes128).resolve(),
        )
    sampled_points = upsample_or_downsample_to_2048(raw_points, rng)
    normalized_points, norm_meta = maybe_normalize_shapenet(sampled_points, config)
    preprocess_sec = time.perf_counter() - t0

    device = torch.device(args.device.lower())

    model = build_model_from_cfg(config.model)
    load_weights(model, ckpt_path)
    model = model.to(device)
    model.eval()

    inference_start = time.perf_counter()
    input_tensor = torch.from_numpy(normalized_points.copy()).float().unsqueeze(0).to(device)
    with torch.no_grad():
        ret = model(input_tensor)
        dense_points = ret[-1].squeeze(0).detach().cpu().numpy()
    inference_sec = time.perf_counter() - inference_start

    post_start = time.perf_counter()
    dense_points = maybe_denormalize_shapenet(dense_points, norm_meta).astype(np.float32, copy=False)
    save_point_cloud(dense_points, output_path)
    postprocess_sec = time.perf_counter() - post_start

    result = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "config_path": str(config_path),
        "ckpt_path": str(ckpt_path),
        "device": str(device),
        "num_input_points_raw": int(raw_points.shape[0]),
        "num_input_points_model": int(sampled_points.shape[0]),
        "num_output_points": int(dense_points.shape[0]),
        "fused_init": fuse_meta,
        "runtime": {
            "preprocess_sec": preprocess_sec,
            "inference_sec": inference_sec,
            "postprocess_sec": postprocess_sec,
            "total_sec": preprocess_sec + inference_sec + postprocess_sec,
        },
    }
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()

"""
python PoinTr/infer_once.py \
  --pc /mnt/d/ObjView-Bench/object_dataset/clean_pool/view_pcd_cache/ec730e3a13c846c8a92ba85b49c8f179/view_000.pcd \
  --output-pc /mnt/d/ObjView-Bench/benchmark/PoinTr/tmp_pred_smoke/ec730e3a13c846c8a92ba85b49c8f179_view_000_completed.pcd \
  --device cuda:0

python PoinTr/infer_once.py \
  --pc /mnt/d/ObjView-Bench/object_dataset/clean_pool/view_pcd_cache/ec730e3a13c846c8a92ba85b49c8f179/view_000.pcd \
  --output-pc /mnt/d/ObjView-Bench/benchmark/PoinTr/tmp_pred_smoke/ec730e3a13c846c8a92ba85b49c8f179_view_000_fps2_completed.pcd \
  --fuse-farthest-tammes128 \
  --device cuda:0
"""