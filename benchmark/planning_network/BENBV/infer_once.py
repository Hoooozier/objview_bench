import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import PointCloudNet


def load_model(ckpt_path: Path, *, device: torch.device) -> PointCloudNet:
    model = PointCloudNet().to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value

    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model


def build_random_inputs(
    *,
    batch_size: int,
    num_partial_points: int,
    num_candidates: int,
    device: torch.device,
    zero_partial: bool,
    zero_candidates: bool,
    zero_context: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if zero_partial:
        partial = torch.zeros(batch_size, num_partial_points, 6, device=device)
    else:
        partial = torch.rand(batch_size, num_partial_points, 6, device=device)
        partial[:, :, :3] = partial[:, :, :3] * 2.0 - 1.0
        partial[:, :, 3:] = partial[:, :, 3:] * 2.0 - 1.0

    if zero_candidates:
        candidates = torch.zeros(batch_size, num_candidates, 6, device=device)
    else:
        candidates = torch.rand(batch_size, num_candidates, 6, device=device)
        candidates[:, :, :3] = candidates[:, :, :3] * 2.0 - 1.0
        candidates[:, :, 3:] = candidates[:, :, 3:] * 2.0 - 1.0

    if zero_context:
        context = torch.zeros(batch_size, num_candidates + 1, 1, device=device)
    else:
        density = torch.rand(batch_size, num_candidates, 1, device=device)
        step_order = torch.randint(low=0, high=32, size=(batch_size, 1, 1), device=device).float()
        context = torch.cat([density, step_order], dim=1)

    return partial, candidates, context


def _as_ascii_meta(data: np.lib.npyio.NpzFile, meta: dict, key: str) -> None:
    if key not in data.files:
        return
    arr = data[key]
    try:
        meta[key.replace("_ascii", "")] = bytes(arr.tolist()).decode("utf-8").rstrip("\x00")
    except Exception:
        meta[key] = arr.tolist()


def build_inputs_from_npz(npz_path: Path, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    data = np.load(npz_path, allow_pickle=True)

    required = ("P", "S", "C")
    for key in required:
        if key not in data.files:
            raise ValueError(f"{npz_path} does not contain required key {key!r}")

    partial = torch.from_numpy(data["P"]).float()
    candidates = torch.from_numpy(data["S"]).float()
    context = torch.from_numpy(data["C"]).float()

    if partial.ndim == 2:
        partial = partial.unsqueeze(0)
    if candidates.ndim == 2:
        candidates = candidates.unsqueeze(0)
    if context.ndim == 2:
        context = context.unsqueeze(0)

    if partial.ndim != 3 or partial.shape[-1] != 6:
        raise ValueError(f"P should become (B,N,6), got {tuple(partial.shape)} from {npz_path}")
    if candidates.ndim != 3 or candidates.shape[-1] != 6:
        raise ValueError(f"S should become (B,20,6), got {tuple(candidates.shape)} from {npz_path}")
    if context.ndim != 3 or context.shape[-1] != 1:
        raise ValueError(f"C should become (B,21,1), got {tuple(context.shape)} from {npz_path}")

    meta = {"npz_path": str(npz_path)}
    for key in ("uid_ascii", "views_path_ascii", "constraint_name_ascii"):
        _as_ascii_meta(data, meta, key)
    for key in ("start_view_id", "step_id", "view_origin", "bbox_min", "bbox_max"):
        if key in data.files:
            meta[key] = data[key].tolist()

    return partial.to(device), candidates.to(device), context.to(device), meta


def summarize_inputs(partial: torch.Tensor, candidates: torch.Tensor, context: torch.Tensor) -> dict:
    p_cpu = partial.detach().cpu()
    s_cpu = candidates.detach().cpu()
    c_cpu = context.detach().cpu()
    return {
        "partial_shape": list(p_cpu.shape),
        "partial_dtype": str(p_cpu.dtype),
        "partial_xyz_min": float(p_cpu[:, :, :3].min().item()),
        "partial_xyz_max": float(p_cpu[:, :, :3].max().item()),
        "partial_nonzero_count": int(torch.count_nonzero(p_cpu).item()),
        "candidate_shape": list(s_cpu.shape),
        "candidate_dtype": str(s_cpu.dtype),
        "candidate_xyz_min": float(s_cpu[:, :, :3].min().item()),
        "candidate_xyz_max": float(s_cpu[:, :, :3].max().item()),
        "context_shape": list(c_cpu.shape),
        "context_dtype": str(c_cpu.dtype),
        "density_min": float(c_cpu[:, :-1, :].min().item()),
        "density_max": float(c_cpu[:, :-1, :].max().item()),
        "step_feature_values": c_cpu[:, -1, 0].tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal one-shot inference for BENBV with random input or a single npz case.")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("planning_network/BENBV/best_val_model.pth"),
        help="Checkpoint path.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--input-npz", type=Path, default=None, help="Optional npz containing P/S/C tensors.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-partial-points", type=int, default=4096)
    parser.add_argument("--num-candidates", type=int, default=20)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zero-partial", action="store_true")
    parser.add_argument("--zero-candidates", action="store_true")
    parser.add_argument("--zero-context", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print(f"[WARN] CUDA unavailable, falling back to CPU from {args.device}.")
        device = torch.device("cpu")
    else:
        device = requested_device

    model = load_model(args.ckpt, device=device)

    case_meta = None
    if args.input_npz is not None:
        partial, candidates, context, case_meta = build_inputs_from_npz(args.input_npz, device=device)
    else:
        partial, candidates, context = build_random_inputs(
            batch_size=args.batch_size,
            num_partial_points=args.num_partial_points,
            num_candidates=args.num_candidates,
            device=device,
            zero_partial=args.zero_partial,
            zero_candidates=args.zero_candidates,
            zero_context=args.zero_context,
        )

    with torch.no_grad():
        scores = model(partial, candidates, context)

    num_candidates = int(scores.shape[1])
    topk = min(args.topk, num_candidates)
    topk_scores, topk_indices = torch.topk(scores.squeeze(-1), k=topk, dim=1)
    best_scores, best_indices = torch.max(scores.squeeze(-1), dim=1)

    result = {
        "device": str(device),
        "checkpoint": str(args.ckpt),
        "input_npz": str(args.input_npz) if args.input_npz is not None else None,
        "batch_size": int(scores.shape[0]),
        "num_candidates": num_candidates,
        "input_summary": summarize_inputs(partial, candidates, context),
        "scores": scores.squeeze(-1).cpu().tolist(),
        "best_index": best_indices.cpu().tolist(),
        "best_score": best_scores.cpu().tolist(),
        "topk_indices": topk_indices.cpu().tolist(),
        "topk_scores": topk_scores.cpu().tolist(),
    }
    if case_meta is not None:
        result["case_meta"] = case_meta

    print(json.dumps(result, ensure_ascii=True, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=True, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
python planning_network/BENBV/infer_once.py \
  --input-npz /mnt/d/ObjView-Bench/object_dataset/clean_pool/benbv_training_npz/0af36b8200b541ee91e63fa3d0866b2e/start000.npz \
  --device cuda:0 \
  --topk 5
"""
