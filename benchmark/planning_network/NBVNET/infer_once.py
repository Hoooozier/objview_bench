#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from train_nbvnet_5k_position import NBVNet5KPosition


def load_view_positions(path: Path, *, radius: float = 1.0, expected_views: int | None = None) -> torch.Tensor:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Invalid view position line: {line!r}")
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not rows:
        raise ValueError(f"No view positions found in {path}")
    view_positions = torch.tensor(rows, dtype=torch.float32)
    if expected_views is not None and int(view_positions.shape[0]) != int(expected_views):
        raise ValueError(f"Expected {expected_views} view positions, got {view_positions.shape[0]} from {path}")
    norms = torch.linalg.norm(view_positions, dim=1, keepdim=True)
    if torch.any(norms <= 0):
        raise ValueError(f"Encountered zero-length view direction in {path}")
    return view_positions / norms * float(radius)


def _state_dict_from_checkpoint(ckpt: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "model_state", "state_dict"):
            if key in ckpt:
                return ckpt[key]
        return ckpt
    return ckpt


def _clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("backbone."):
            key = key[len("backbone.") :]
        cleaned[key] = value
    return cleaned


def load_model(
    ckpt_path: Path,
    view_positions: torch.Tensor,
    *,
    device: torch.device,
    grid_size: int | None = None,
    num_classes: int | None = None,
    dropout_prob: float | None = None,
    context_channels: int | None = None,
    position_hidden_dim: int | None = None,
) -> NBVNet5KPosition:
    ckpt = torch.load(ckpt_path, map_location=device)
    params = ckpt.get("parameters", {}) if isinstance(ckpt, dict) else {}
    model_grid_size = int(grid_size if grid_size is not None else ckpt.get("grid_size", 64))
    model_num_classes = int(num_classes if num_classes is not None else ckpt.get("num_classes", view_positions.shape[0]))
    model_dropout = float(dropout_prob if dropout_prob is not None else params.get("dropout_prob", 0.3))
    model_context_channels = int(context_channels if context_channels is not None else params.get("context_channels", 4))
    model_position_hidden_dim = int(
        position_hidden_dim if position_hidden_dim is not None else params.get("position_hidden_dim", 64)
    )

    model = NBVNet5KPosition(
        view_positions=view_positions,
        dropout_prob=model_dropout,
        grid_size=model_grid_size,
        num_classes=model_num_classes,
        context_channels=model_context_channels,
        position_hidden_dim=model_position_hidden_dim,
    ).to(device)

    state_dict = _clean_state_dict(_state_dict_from_checkpoint(ckpt))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _normalize_grid(grid: torch.Tensor, *, npz_path: Path | None = None) -> torch.Tensor:
    if grid.ndim == 1:
        grid_size = int(round(float(grid.numel()) ** (1.0 / 3.0)))
        expected = grid_size**3
        if int(grid.numel()) != expected:
            raise ValueError(f"Flat grid size={grid.numel()} is not cubic from {npz_path}")
        grid = grid.reshape(1, 1, grid_size, grid_size, grid_size)
    elif grid.ndim == 3:
        grid = grid.unsqueeze(0).unsqueeze(0)
    elif grid.ndim == 4:
        if grid.shape[0] == 1:
            grid = grid.unsqueeze(0)
        elif grid.shape[-1] == 1:
            grid = grid.permute(3, 0, 1, 2).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported 4D grid shape={tuple(grid.shape)} from {npz_path}")
    if grid.ndim != 5 or grid.shape[1] != 1:
        raise ValueError(f"grid should become (B,1,D,D,D), got {tuple(grid.shape)} from {npz_path}")
    return grid.contiguous()


def build_inputs_from_npz(npz_path: Path, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
    data = np.load(npz_path, allow_pickle=True)
    if "grid" not in data.files:
        raise ValueError(f"{npz_path} does not contain required key 'grid'")

    grid = _normalize_grid(torch.from_numpy(data["grid"]).float(), npz_path=npz_path)
    view_state = None
    if "vs" in data.files:
        view_state = torch.from_numpy(data["vs"]).float()
        if view_state.ndim == 1:
            view_state = view_state.unsqueeze(0)
        elif view_state.ndim == 3 and view_state.shape[1] == 1:
            view_state = view_state.squeeze(1)
        if view_state.ndim != 2:
            raise ValueError(f"vs should become (B,V), got {tuple(view_state.shape)} from {npz_path}")

    meta = {"npz_path": str(npz_path)}
    for key in ("start_view_id", "step_id", "grid_size", "bbox_min", "bbox_max", "view_origin"):
        if key in data.files:
            meta[key] = data[key].tolist()
    for key in ("uid_ascii", "views_path_ascii", "constraint_name_ascii"):
        if key in data.files:
            arr = data[key]
            try:
                meta[key.replace("_ascii", "")] = bytes(arr.tolist()).decode("utf-8").rstrip("\x00")
            except Exception:
                meta[key] = arr.tolist()

    return grid.to(device), None if view_state is None else view_state.to(device), meta


def build_random_inputs(
    *,
    batch_size: int,
    grid_size: int,
    num_views: int,
    random_grid: bool,
    random_view_state: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if random_grid:
        grid = torch.rand(batch_size, 1, grid_size, grid_size, grid_size, device=device)
    else:
        grid = torch.zeros(batch_size, 1, grid_size, grid_size, grid_size, device=device)
    if random_view_state:
        view_state = torch.randint(0, 2, (batch_size, num_views), device=device).float()
    else:
        view_state = torch.zeros(batch_size, num_views, device=device)
    return grid, view_state


def summarize_inputs(grid: torch.Tensor, view_state: torch.Tensor | None) -> dict:
    grid_cpu = grid.detach().cpu()
    out = {
        "grid_shape": list(grid_cpu.shape),
        "grid_dtype": str(grid_cpu.dtype),
        "grid_min": float(grid_cpu.min().item()),
        "grid_max": float(grid_cpu.max().item()),
        "grid_mean": float(grid_cpu.mean().item()),
        "grid_nonzero_count": int(torch.count_nonzero(grid_cpu).item()),
    }
    if view_state is not None:
        vs_cpu = view_state.detach().cpu()
        out.update(
            {
                "view_state_shape": list(vs_cpu.shape),
                "view_state_dtype": str(vs_cpu.dtype),
                "view_state_min": float(vs_cpu.min().item()),
                "view_state_max": float(vs_cpu.max().item()),
                "view_state_mean": float(vs_cpu.mean().item()),
                "view_state_nonzero_count": int(torch.count_nonzero(vs_cpu).item()),
                "view_state_active_indices": torch.nonzero(vs_cpu[0] > 0, as_tuple=False).flatten().tolist()
                if vs_cpu.ndim == 2 and vs_cpu.shape[0] > 0
                else [],
                "view_state_used_by_model": False,
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="One-shot inference for NBVNET 5k position model.")
    parser.add_argument("--ckpt", type=Path, default=Path("planning_network/NBVNET/best_val_loss.pt"))
    parser.add_argument("--views", type=Path, default=Path("Tammes_sphere/128_xyz.txt"))
    parser.add_argument("--view-position-radius", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-npz", type=Path, default=None, help="Optional MASCVP-style NPZ containing at least grid.")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--include-scores",
        action="store_true",
        help="Include raw logits/softmax scores for debugging. Algorithm semantics should use only class ranking.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zero-grid", action="store_true")
    parser.add_argument("--zero-view-state", action="store_true")
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

    ckpt = torch.load(args.ckpt, map_location=device)
    num_classes = int(ckpt.get("num_classes", 128)) if isinstance(ckpt, dict) else 128
    ckpt_grid_size = int(ckpt.get("grid_size", 64)) if isinstance(ckpt, dict) else 64
    grid_size = int(args.grid_size) if args.grid_size is not None else ckpt_grid_size
    view_positions = load_view_positions(args.views, radius=args.view_position_radius, expected_views=num_classes)

    model = load_model(
        args.ckpt,
        view_positions,
        device=device,
        grid_size=grid_size,
        num_classes=num_classes,
    )

    case_meta = None
    if args.input_npz is not None:
        grid, view_state, case_meta = build_inputs_from_npz(args.input_npz, device=device)
    else:
        grid, view_state = build_random_inputs(
            batch_size=args.batch_size,
            grid_size=grid_size,
            num_views=num_classes,
            random_grid=not args.zero_grid,
            random_view_state=not args.zero_view_state,
            device=device,
        )

    with torch.no_grad():
        logits = model(grid)
        scores = torch.softmax(logits, dim=1)

    topk = min(int(args.topk), int(scores.shape[1]))
    _, topk_indices = torch.topk(logits, k=topk, dim=1)
    best_indices = torch.argmax(logits, dim=1)

    result = {
        "device": str(device),
        "checkpoint": str(args.ckpt),
        "views": str(args.views),
        "view_position_radius": args.view_position_radius,
        "grid_size": grid_size,
        "batch_size": int(grid.shape[0]),
        "num_candidates": int(scores.shape[1]),
        "input_npz": str(args.input_npz) if args.input_npz is not None else None,
        "random_grid": args.input_npz is None and (not args.zero_grid),
        "random_view_state": args.input_npz is None and (not args.zero_view_state),
        "input_summary": summarize_inputs(grid, view_state),
        "best_index": best_indices.cpu().tolist(),
        "topk_indices": topk_indices.cpu().tolist(),
        "score_semantics": "single-class classification; logits/softmax are uncalibrated and should be used only for ranking",
    }
    if args.include_scores:
        topk_scores = torch.gather(scores, dim=1, index=topk_indices)
        best_scores = torch.gather(scores, dim=1, index=best_indices.view(-1, 1)).squeeze(1)
        result.update(
            {
                "logits": logits.cpu().tolist(),
                "scores": scores.cpu().tolist(),
                "best_score": best_scores.cpu().tolist(),
                "topk_scores": topk_scores.cpu().tolist(),
            }
        )
    if case_meta is not None:
        result["case_meta"] = case_meta

    text = json.dumps(result, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
python planning_network/NBVNET/infer_once.py \
  --ckpt planning_network/NBVNET/best_val_loss.pt \
  --views Tammes_sphere/128_xyz.txt \
  --input-npz /mnt/d/ObjView-Bench/object_dataset/clean_pool/mascvp_offline_cases/1dac1b1bc87f4b7fb30a096dc704d75a/1dac1b1bc87f4b7fb30a096dc704d75a__view_000__step_000.npz \
  --device cuda:0
"""
