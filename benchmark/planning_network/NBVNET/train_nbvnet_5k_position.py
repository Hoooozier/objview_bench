#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


NUM_VIEWS = 128


class Public5KNBVPositionDataset(Dataset):
    """Lazy grid.npy dataset with single-class labels cached from id.txt."""

    def __init__(
        self,
        split_dir: str | Path,
        cache_dir: str | Path,
        num_views: int = NUM_VIEWS,
        max_samples: int | None = None,
        rebuild_label_cache: bool = False,
    ):
        self.split_dir = Path(split_dir)
        self.cache_dir = Path(cache_dir)
        self.num_views = int(num_views)
        self.index_path = self.split_dir / "sample_index.json"
        if not self.index_path.exists():
            raise FileNotFoundError(f"Missing sample index: {self.index_path}")

        with self.index_path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        if max_samples is not None:
            records = records[: int(max_samples)]
        self.records = records
        self.length = len(records)
        if self.length == 0:
            raise ValueError(f"Empty dataset: {self.split_dir}")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"{self.split_dir.name}_{self.length}_views{self.num_views}"
        self.labels_cache = self.cache_dir / f"nbvnet_5k_position_{suffix}_labels.npy"
        self.cache_meta = self.cache_dir / f"nbvnet_5k_position_{suffix}_label_cache_meta.json"

        if rebuild_label_cache or not self.labels_cache.exists():
            self._build_label_cache()

        self.labels = np.load(self.labels_cache, mmap_mode="r")
        if self.labels.shape != (self.length,):
            raise ValueError(f"Label cache shape mismatch: expected {(self.length,)}, got {self.labels.shape}")
        if int(self.labels.min()) < 0 or int(self.labels.max()) >= self.num_views:
            raise ValueError(
                f"Label out of range in {self.labels_cache}: "
                f"min={int(self.labels.min())}, max={int(self.labels.max())}, views={self.num_views}"
            )
        self._grid_size = self._infer_grid_size()

    @staticmethod
    def _read_single_id(path: Path) -> int:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Empty id file: {path}")
        text = text.replace(",", " ").replace("\n", " ")
        values = [v for v in text.split() if v.strip()]
        if len(values) != 1:
            raise ValueError(f"{path} should contain exactly one integer id, got {values[:8]}")
        return int(values[0])

    def _label_path_for_record(self, record: dict, grid_path: Path) -> Path:
        label_path = record.get("label_path")
        if label_path:
            return Path(label_path)
        return grid_path.parent / "id.txt"

    def _build_label_cache(self) -> None:
        labels = np.zeros((self.length,), dtype=np.int64)
        missing: list[str] = []
        counter: Counter[int] = Counter()

        iterator = tqdm(self.records, desc=f"cache labels {self.split_dir.name}", leave=False)
        for i, record in enumerate(iterator):
            grid_path = Path(record.get("grid_npy_path", ""))
            if not grid_path.exists():
                missing.append(str(grid_path))
            label_path = self._label_path_for_record(record, grid_path)
            if not label_path.exists():
                missing.append(str(label_path))
            if missing:
                raise FileNotFoundError(f"Missing required sample file, first missing: {missing[0]}")

            label = self._read_single_id(label_path)
            if label < 0 or label >= self.num_views:
                raise ValueError(f"View id {label} out of range [0, {self.num_views - 1}] in {label_path}")
            labels[i] = label
            counter[label] += 1

        np.save(self.labels_cache, labels)
        meta = {
            "split_dir": str(self.split_dir),
            "index_path": str(self.index_path),
            "num_samples": self.length,
            "num_views": self.num_views,
            "labels_cache": str(self.labels_cache),
            "label_source": "single integer id.txt converted to int64 numpy labels",
            "top10_classes": counter.most_common(10),
        }
        self.cache_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @property
    def grid_size(self) -> int:
        return self._grid_size

    def _infer_grid_size(self) -> int:
        grid = np.load(self.records[0]["grid_npy_path"], mmap_mode="r")
        if grid.ndim == 3:
            return int(grid.shape[0])
        if grid.ndim == 4 and grid.shape[0] == 1:
            return int(grid.shape[1])
        if grid.ndim == 4 and grid.shape[-1] == 1:
            return int(grid.shape[0])
        if grid.ndim == 1:
            return int(round(grid.size ** (1.0 / 3.0)))
        raise ValueError(f"Unsupported first grid shape: {grid.shape}")

    @property
    def views(self) -> int:
        return self.num_views

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        record = self.records[index]
        grid_path = Path(record["grid_npy_path"])
        grid = np.load(grid_path).astype(np.float32, copy=False)
        label = int(self.labels[index])
        grid_size = self._grid_size

        if grid.ndim == 1:
            expected = grid_size**3
            if grid.size != expected:
                raise ValueError(f"Flat grid size={grid.size}, expected {expected}: {grid_path}")
            grid = grid.reshape(1, grid_size, grid_size, grid_size)
        elif grid.ndim == 3:
            grid = grid.reshape(1, grid_size, grid_size, grid_size)
        elif grid.ndim == 4:
            if grid.shape == (1, grid_size, grid_size, grid_size):
                pass
            elif grid.shape == (grid_size, grid_size, grid_size, 1):
                grid = np.transpose(grid, (3, 0, 1, 2))
            else:
                raise ValueError(f"Unsupported grid shape={grid.shape}: {grid_path}")
        else:
            raise ValueError(f"Unsupported grid ndim={grid.ndim}: {grid_path}")

        return {
            "grid": torch.from_numpy(grid.copy()),
            "nbv_class": torch.tensor(label, dtype=torch.long),
            "record_index": torch.tensor(int(record.get("index", index)), dtype=torch.long),
        }


class CandidatePositionContext(nn.Module):
    """Project fixed candidate xyz positions into a 3D context volume."""

    def __init__(
        self,
        view_positions: torch.Tensor,
        context_channels: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        if view_positions.ndim != 2 or view_positions.shape[1] != 3:
            raise ValueError(f"view_positions must have shape (V, 3), got {tuple(view_positions.shape)}")
        self.num_views = int(view_positions.shape[0])
        self.context_channels = int(context_channels)
        self.register_buffer("view_positions", view_positions.float())
        self.view_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_dim, self.context_channels),
        )

    def forward(self, batch_size: int, spatial_size: int) -> torch.Tensor:
        tokens = self.view_mlp(self.view_positions).transpose(0, 1).unsqueeze(0)
        num_voxels = int(spatial_size) ** 3
        if num_voxels % self.num_views == 0:
            context = tokens.repeat_interleave(num_voxels // self.num_views, dim=2)
        else:
            context = F.interpolate(tokens, size=num_voxels, mode="linear", align_corners=False)
        context = context.view(1, self.context_channels, spatial_size, spatial_size, spatial_size)
        return context.expand(batch_size, -1, -1, -1, -1)


class NBVNet5KPosition(nn.Module):
    """NBV-Net with xyz position context concatenated after the first conv/pool."""

    def __init__(
        self,
        view_positions: torch.Tensor,
        dropout_prob: float = 0.3,
        grid_size: int = 64,
        num_classes: int = NUM_VIEWS,
        context_channels: int = 4,
        position_hidden_dim: int = 64,
    ):
        super().__init__()
        self.grid_size = int(grid_size)
        self.num_classes = int(num_classes)
        self.context_channels = int(context_channels)

        self.conv1 = nn.Conv3d(1, 10, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.position_context = CandidatePositionContext(
            view_positions=view_positions,
            context_channels=context_channels,
            hidden_dim=position_hidden_dim,
            dropout=0.1,
        )

        self.conv2 = nn.Conv3d(10 + context_channels, 12, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv3d(12, 8, kernel_size=3, stride=1, padding=1)
        self.conv3_drop = nn.Dropout(dropout_prob)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        flat_dim = self._infer_flatten_dim(self.grid_size)
        self.fc1 = nn.Linear(flat_dim, self.num_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        context = self.position_context(batch_size=x.size(0), spatial_size=x.shape[-1])
        x = torch.cat([x, context], dim=1)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        return x

    def _infer_flatten_dim(self, grid_size: int) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, grid_size, grid_size, grid_size)
            x = self._forward_features(x)
            return int(x.reshape(1, -1).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = x.reshape(x.size(0), -1)
        return self.fc1(x)


def load_view_positions(path: str | Path, expected_views: int | None = None) -> torch.Tensor:
    positions = np.loadtxt(path, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"View positions must have shape (V, 3), got {positions.shape}")
    if expected_views is not None and positions.shape[0] != expected_views:
        raise ValueError(f"Expected {expected_views} view positions, got {positions.shape[0]}")
    return torch.from_numpy(positions)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def parse_gpu_ids(value: str) -> list[int]:
    if value is None or not str(value).strip():
        return []
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def resolve_device(device_name: str, gpu_ids: list[int]) -> torch.device:
    if gpu_ids:
        if not torch.cuda.is_available():
            print("CUDA is not available; falling back to CPU for this run.")
            return torch.device("cpu")
        count = torch.cuda.device_count()
        invalid = [gid for gid in gpu_ids if gid < 0 or gid >= count]
        if invalid:
            raise ValueError(f"Invalid GPU ids {invalid}; available ids are 0..{count - 1}")
        torch.cuda.set_device(gpu_ids[0])
        return torch.device(f"cuda:{gpu_ids[0]}")
    if device_name != "cpu" and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU for this run.")
        return torch.device("cpu")
    return torch.device(device_name)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def topk_correct(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    k = min(k, logits.size(1))
    pred = logits.topk(k, dim=1).indices
    return int(pred.eq(labels.view(-1, 1)).any(dim=1).sum().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> dict:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_top3 = 0
    total_top5 = 0
    total_samples = 0

    loop = tqdm(loader, desc="train" if train else "val", leave=False)
    with torch.set_grad_enabled(train):
        for batch_idx, sample in enumerate(loop):
            if max_batches is not None and batch_idx >= max_batches:
                break

            grids = sample["grid"].float().to(device, non_blocking=True)
            labels = sample["nbv_class"].to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(grids)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            batch_size = grids.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_top3 += topk_correct(logits, labels, 3)
            total_top5 += topk_correct(logits, labels, 5)
            total_samples += batch_size
            denom = max(total_samples, 1)
            loop.set_postfix(
                loss=f"{total_loss / denom:.4f}",
                acc=f"{total_correct / denom:.4f}",
                top3=f"{total_top3 / denom:.4f}",
                top5=f"{total_top5 / denom:.4f}",
            )

    denom = max(total_samples, 1)
    return {
        "loss": total_loss / denom,
        "acc": total_correct / denom,
        "top3_acc": total_top3 / denom,
        "top5_acc": total_top5 / denom,
        "samples": total_samples,
    }


def build_loader(
    split_dir: Path,
    cache_dir: Path,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
    shuffle: bool,
    rebuild_label_cache: bool,
):
    dataset = Public5KNBVPositionDataset(
        split_dir=split_dir,
        cache_dir=cache_dir,
        max_samples=max_samples,
        rebuild_label_cache=rebuild_label_cache,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return dataset, loader


def inspect_dataset(dataset: Public5KNBVPositionDataset, name: str, count: int = 3) -> None:
    labels = np.asarray(dataset.labels, dtype=np.int64)
    counter = Counter(labels.tolist())
    print(f"------ inspect {name} ------")
    print(
        f"samples={len(dataset)}, grid_size={dataset.grid_size}, views={dataset.views}, "
        f"labels=[{int(labels.min())}, {int(labels.max())}], top10={counter.most_common(10)}"
    )
    for i in range(min(count, len(dataset))):
        sample = dataset[i]
        grid = sample["grid"]
        label = int(sample["nbv_class"].item())
        record = dataset.records[i]
        label_path = record.get("label_path") or str(Path(record["grid_npy_path"]).parent / "id.txt")
        print(
            f"{name}[{i}] uid={record.get('uid')} view={record.get('view_id')} step={record.get('step_id')} "
            f"grid={tuple(grid.shape)} dtype={grid.dtype} min={float(grid.min()):.4f} "
            f"max={float(grid.max()):.4f} label={label} label_path={label_path}"
        )


def append_history_csv(csv_path: Path, row: dict) -> None:
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: dict,
    best: dict,
    args: argparse.Namespace,
    num_classes: int,
    grid_size: int,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "best": best,
            "num_classes": num_classes,
            "grid_size": grid_size,
            "model": "NBVNet5KPosition",
            "parameters": vars(args),
        },
        path,
    )


def selected_is_better(metric_name: str, value: float, best_value: float) -> bool:
    if metric_name == "val_loss":
        return value < best_value
    return value > best_value


def initial_best_value(metric_name: str) -> float:
    return float("inf") if metric_name == "val_loss" else -1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NBV-Net 5k position model on public split data.")
    parser.add_argument("--data-root", default="./network_training_data/main_public_split_data")
    parser.add_argument("--positions", default="./network_training_data/128_xyz.txt")
    parser.add_argument("--output-dir", default="./runs_nbvnet_5k_position/public_5k_position_gpus01")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--dropout-prob", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--context-channels", type=int, default=4)
    parser.add_argument("--position-hidden-dim", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--select-metric",
        choices=["val_acc", "val_loss", "val_top3_acc", "val_top5_acc"],
        default="val_acc",
    )
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--rebuild-label-cache", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run 1 epoch with one train and one val batch.")
    args = parser.parse_args()

    seed_everything(args.seed)
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    device = resolve_device(args.device, gpu_ids)

    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.epochs = 1
        args.max_train_batches = 1
        args.max_val_batches = 1

    train_set, train_loader = build_loader(
        split_dir=Path(args.data_root) / "train",
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_train_samples,
        shuffle=True,
        rebuild_label_cache=args.rebuild_label_cache,
    )
    val_batch_size = args.val_batch_size if args.val_batch_size is not None else args.batch_size
    val_set, val_loader = build_loader(
        split_dir=Path(args.data_root) / "val",
        cache_dir=cache_dir,
        batch_size=val_batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_val_samples,
        shuffle=False,
        rebuild_label_cache=args.rebuild_label_cache,
    )

    if train_set.grid_size != val_set.grid_size:
        raise ValueError(f"train/val grid size mismatch: {train_set.grid_size} vs {val_set.grid_size}")
    if train_set.views != val_set.views:
        raise ValueError(f"train/val views mismatch: {train_set.views} vs {val_set.views}")

    inspect_dataset(train_set, "train")
    inspect_dataset(val_set, "val")
    if args.inspect_only:
        return

    view_positions = load_view_positions(args.positions, expected_views=train_set.views)
    model = NBVNet5KPosition(
        view_positions=view_positions,
        dropout_prob=args.dropout_prob,
        grid_size=train_set.grid_size,
        num_classes=train_set.views,
        context_channels=args.context_channels,
        position_hidden_dim=args.position_hidden_dim,
    ).to(device)
    if len(gpu_ids) > 1 and device.type == "cuda":
        model = nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    config = {
        "args": vars(args),
        "device": str(device),
        "gpu_ids": gpu_ids if device.type == "cuda" else [],
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "grid_size": train_set.grid_size,
        "num_classes": train_set.views,
        "model": "NBVNet5KPosition",
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    best = {
        "selected": {
            "metric": args.select_metric,
            "value": initial_best_value(args.select_metric),
            "epoch": 0,
        },
        "val_loss": {"value": float("inf"), "epoch": 0},
        "val_acc": {"value": -1.0, "epoch": 0},
        "val_top3_acc": {"value": -1.0, "epoch": 0},
        "val_top5_acc": {"value": -1.0, "epoch": 0},
    }
    history = []
    history_csv = output_dir / "history.csv"

    print(f"data_root={args.data_root}")
    print(f"positions={args.positions}")
    print(f"train={len(train_set)}, val={len(val_set)}, device={device}, gpu_ids={gpu_ids}")
    print(f"batch_size={args.batch_size}, val_batch_size={val_batch_size}, lr={args.lr}, epochs={args.epochs}")
    print(f"output_dir={output_dir}")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            max_batches=args.max_val_batches,
        )
        elapsed = time.time() - start

        row = {
            "epoch": epoch,
            "seconds": elapsed,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_top3_acc": train_metrics["top3_acc"],
            "train_top5_acc": train_metrics["top5_acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_top3_acc": val_metrics["top3_acc"],
            "val_top5_acc": val_metrics["top5_acc"],
        }
        history.append(row)
        with (output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        append_history_csv(history_csv, row)

        metrics = {"train": train_metrics, "val": val_metrics, "row": row}
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, metrics, best, args, train_set.views, train_set.grid_size)

        selected_value = row[args.select_metric]
        if selected_is_better(args.select_metric, selected_value, best["selected"]["value"]):
            best["selected"] = {"metric": args.select_metric, "value": selected_value, "epoch": epoch, "row": row}
            save_checkpoint(
                output_dir / "best_selected.pt",
                model,
                optimizer,
                epoch,
                metrics,
                best,
                args,
                train_set.views,
                train_set.grid_size,
            )

        val_best_specs = {
            "val_loss": row["val_loss"],
            "val_acc": row["val_acc"],
            "val_top3_acc": row["val_top3_acc"],
            "val_top5_acc": row["val_top5_acc"],
        }
        for metric_name, metric_value in val_best_specs.items():
            if selected_is_better(metric_name, metric_value, best[metric_name]["value"]):
                best[metric_name] = {"value": metric_value, "epoch": epoch, "row": row}
                save_checkpoint(
                    output_dir / f"best_{metric_name}.pt",
                    model,
                    optimizer,
                    epoch,
                    metrics,
                    best,
                    args,
                    train_set.views,
                    train_set.grid_size,
                )

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                epoch,
                metrics,
                best,
                args,
                train_set.views,
                train_set.grid_size,
            )

        print(
            f"epoch {epoch:03d}/{args.epochs:03d} | "
            f"train loss {row['train_loss']:.6f} acc {row['train_acc']:.4f} "
            f"top3 {row['train_top3_acc']:.4f} top5 {row['train_top5_acc']:.4f} | "
            f"val loss {row['val_loss']:.6f} acc {row['val_acc']:.4f} "
            f"top3 {row['val_top3_acc']:.4f} top5 {row['val_top5_acc']:.4f} | "
            f"best_selected {best['selected']['metric']} e{best['selected']['epoch']}="
            f"{best['selected']['value']:.6f}"
        )

    with (output_dir / "best_summary.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    print("NBV-Net 5k position training complete.")


if __name__ == "__main__":
    main()
