import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model_position import MASCVPPositionNet


def load_view_positions(path: Path, *, radius: float = 1.0) -> torch.Tensor:
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
    norms = torch.linalg.norm(view_positions, dim=1, keepdim=True)
    if torch.any(norms <= 0):
        raise ValueError(f"Encountered zero-length view direction in {path}")
    view_positions = view_positions / norms
    view_positions = view_positions * float(radius)
    return view_positions


def load_model(
    ckpt_path: Path,
    view_positions: torch.Tensor,
    *,
    device: torch.device,
    grid_size: int,
    output_views: int,
) -> MASCVPPositionNet:
    model = MASCVPPositionNet(
        view_positions=view_positions,
        grid_size=grid_size,
        output_views=output_views,
    ).to(device)

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
        if key.startswith("backbone."):
            key = key[len("backbone.") :]
        cleaned[key] = value

    model.load_state_dict(cleaned, strict=True)

    model.eval()
    return model


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


def build_inputs_from_npz(npz_path: Path, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict]:
    data = np.load(npz_path, allow_pickle=True)

    if "grid" not in data.files:
        raise ValueError(f"{npz_path} does not contain required key 'grid'")
    if "vs" not in data.files:
        raise ValueError(f"{npz_path} does not contain required key 'vs'")

    grid = torch.from_numpy(data["grid"]).float()
    vs = torch.from_numpy(data["vs"]).float()

    if grid.ndim == 3:
        grid = grid.unsqueeze(0).unsqueeze(0)
    elif grid.ndim == 4:
        grid = grid.unsqueeze(0)
    if grid.ndim != 5:
        raise ValueError(f"grid should become (B,1,64,64,64), got {tuple(grid.shape)} from {npz_path}")

    if vs.ndim == 1:
        vs = vs.unsqueeze(0)
    elif vs.ndim == 3 and vs.shape[1] == 1:
        vs = vs.squeeze(1)
    if vs.ndim != 2:
        raise ValueError(f"vs should become (B,128), got {tuple(vs.shape)} from {npz_path}")

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

    return grid.to(device), vs.to(device), meta


def summarize_inputs(grid: torch.Tensor, view_state: torch.Tensor) -> dict:
    grid_cpu = grid.detach().cpu()
    vs_cpu = view_state.detach().cpu()
    return {
        "grid_shape": list(grid_cpu.shape),
        "grid_dtype": str(grid_cpu.dtype),
        "grid_min": float(grid_cpu.min().item()),
        "grid_max": float(grid_cpu.max().item()),
        "grid_mean": float(grid_cpu.mean().item()),
        "grid_nonzero_count": int(torch.count_nonzero(grid_cpu).item()),
        "view_state_shape": list(vs_cpu.shape),
        "view_state_dtype": str(vs_cpu.dtype),
        "view_state_min": float(vs_cpu.min().item()),
        "view_state_max": float(vs_cpu.max().item()),
        "view_state_mean": float(vs_cpu.mean().item()),
        "view_state_nonzero_count": int(torch.count_nonzero(vs_cpu).item()),
        "view_state_active_indices": torch.nonzero(vs_cpu[0] > 0, as_tuple=False).flatten().tolist()
        if vs_cpu.ndim == 2 and vs_cpu.shape[0] > 0
        else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal one-shot inference for MASCVP with synthetic inputs.")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("planning_network/MASCVP/main_train_public/best_f1.pt"),
        help="Checkpoint path. Can be overridden to test a different trained model.",
    )
    parser.add_argument(
        "--views",
        type=Path,
        default=Path("Tammes_sphere/128_xyz.txt"),
        help="Path to 128-view xyz file.",
    )
    parser.add_argument(
        "--view-position-radius",
        type=float,
        default=1.0,
        help="Radius used to scale normalized Tammes view directions before feeding them to the network.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-npz", type=Path, default=None, help="Optional offline case NPZ containing grid and vs.")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.5, help="Selection threshold used to form the output index set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zero-grid", action="store_true", help="Use an all-zero grid instead of random values.")
    parser.add_argument(
        "--zero-view-state",
        action="store_true",
        help="Use an all-zero visited-state vector instead of random binary values.",
    )
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

    view_positions = load_view_positions(args.views, radius=args.view_position_radius)
    num_views = int(view_positions.shape[0])
    if num_views != 128:
        raise ValueError(f"Expected 128 view positions, got {num_views} from {args.views}")

    model = load_model(
        args.ckpt,
        view_positions,
        device=device,
        grid_size=args.grid_size,
        output_views=num_views,
    )

    case_meta = None
    if args.input_npz is not None:
        grid, view_state, case_meta = build_inputs_from_npz(args.input_npz, device=device)
    else:
        grid, view_state = build_random_inputs(
            batch_size=args.batch_size,
            grid_size=args.grid_size,
            num_views=num_views,
            random_grid=not args.zero_grid,
            random_view_state=not args.zero_view_state,
            device=device,
        )

    with torch.no_grad():
        logits = model(grid, view_state)
        scores = torch.sigmoid(logits)

    topk = min(args.topk, num_views)
    topk_scores, topk_indices = torch.topk(scores, k=topk, dim=1)
    selected_mask = scores >= args.gamma
    selected_indices = []
    selected_scores = []
    for batch_idx in range(scores.shape[0]):
        batch_indices = torch.nonzero(selected_mask[batch_idx], as_tuple=False).flatten()
        selected_indices.append(batch_indices.cpu().tolist())
        selected_scores.append(scores[batch_idx, batch_indices].cpu().tolist())

    result = {
        "device": str(device),
        "checkpoint": str(args.ckpt),
        "views": str(args.views),
        "view_position_radius": args.view_position_radius,
        "grid_size": args.grid_size,
        "batch_size": int(grid.shape[0]),
        "input_npz": str(args.input_npz) if args.input_npz is not None else None,
        "random_grid": args.input_npz is None and (not args.zero_grid),
        "random_view_state": args.input_npz is None and (not args.zero_view_state),
        "gamma": args.gamma,
        "input_summary": summarize_inputs(grid, view_state),
        "scores": scores.cpu().tolist(),
        "selected_indices": selected_indices,
        "selected_scores": selected_scores,
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
python planning_network/MASCVP/infer_once.py \
  --ckpt planning_network/MASCVP/Raw_1K/best_f1.pt \
  --views Tammes_sphere/128_xyz.txt \
  --input-npz /mnt/d/ObjView-Bench/object_dataset/clean_pool/mascvp_offline_cases/0ee443fd9cf041e08b708b338be5ffe2/0ee443fd9cf041e08b708b338be5ffe2__view_000__step_001.npz \
  --device cuda:0 \
  --gamma 0.5 \
  --view-position-radius 1.0
"""
