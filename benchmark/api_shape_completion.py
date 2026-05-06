from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from PoinTr.infer_once import (
    IO,
    build_model_from_cfg,
    load_config,
    load_weights,
    maybe_denormalize_shapenet,
    maybe_normalize_shapenet,
    save_point_cloud,
    upsample_or_downsample_to_2048,
)


JSONRPC_VERSION = "2.0"
SHAPE_COMPLETION_METHODS = {"complete_shape"}


class ShapeCompletionError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class PoinTrCompletionConfig:
    config_path: str | Path
    ckpt_path: str | Path
    device: str = "cuda:0"
    model_input_points: int = 2048
    model_output_points: int = 8192
    rng_seed: int = 0


class PoinTrCompletionBackend:
    def __init__(self, cfg: PoinTrCompletionConfig) -> None:
        self.cfg = cfg
        self.config_path = Path(cfg.config_path).resolve()
        self.ckpt_path = Path(cfg.ckpt_path).resolve()
        self.device = torch.device(cfg.device.lower())
        if self.device.type != "cuda":
            raise RuntimeError("PoinTr completion backend currently expects a CUDA device.")

        self.runtime_config = load_config(self.config_path)
        self.rng = np.random.default_rng(cfg.rng_seed)

        self.model = build_model_from_cfg(self.runtime_config.model)
        load_weights(self.model, self.ckpt_path)
        self.model = self.model.to(self.device)
        self.model.eval()

    def capability_metadata(self) -> dict[str, Any]:
        return {
            "backend": "PoinTr-C",
            "config_path": str(self.config_path),
            "ckpt_path": str(self.ckpt_path),
            "device": str(self.device),
            "num_input_points_model": int(self.cfg.model_input_points),
            "num_output_points": int(self.cfg.model_output_points),
        }

    def warmup(self) -> dict[str, Any]:
        points = np.zeros((self.cfg.model_input_points, 3), dtype=np.float32)
        start = time.perf_counter()
        input_tensor = torch.from_numpy(points).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            _ = self.model(input_tensor)
        elapsed = time.perf_counter() - start
        return {
            "warmed_up": True,
            "runtime_sec": elapsed,
            **self.capability_metadata(),
        }

    def complete_point_cloud(self, input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()

        t0 = time.perf_counter()
        raw_points = IO.get(str(input_path)).astype(np.float32)
        sampled_points = upsample_or_downsample_to_2048(raw_points, self.rng)
        normalized_points, norm_meta = maybe_normalize_shapenet(sampled_points, self.runtime_config)
        preprocess_sec = time.perf_counter() - t0

        inference_start = time.perf_counter()
        input_tensor = torch.from_numpy(normalized_points.copy()).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            ret = self.model(input_tensor)
            dense_points = ret[-1].squeeze(0).detach().cpu().numpy()
        inference_sec = time.perf_counter() - inference_start

        post_start = time.perf_counter()
        dense_points = maybe_denormalize_shapenet(dense_points, norm_meta).astype(np.float32, copy=False)
        save_point_cloud(dense_points, output_path)
        postprocess_sec = time.perf_counter() - post_start

        return {
            "completed_pointcloud_path": str(output_path),
            "num_input_points_raw": int(raw_points.shape[0]),
            "num_input_points_model": int(sampled_points.shape[0]),
            "num_output_points": int(dense_points.shape[0]),
            "runtime": {
                "preprocess_sec": preprocess_sec,
                "inference_sec": inference_sec,
                "postprocess_sec": postprocess_sec,
                "total_sec": preprocess_sec + inference_sec + postprocess_sec,
            },
        }


class ShapeCompletionService:
    def __init__(
        self,
        service_root: str | Path,
        backend: PoinTrCompletionBackend,
        *,
        request_glob: str = "*.json.ready",
    ) -> None:
        self.service_root = Path(service_root)
        self.backend = backend
        self.request_glob = request_glob

        self.requests_dir = self.service_root / "requests"
        self.responses_dir = self.service_root / "responses"
        self.outputs_dir = self.service_root / "outputs"
        self.logs_dir = self.service_root / "logs"
        self.ready_path = self.service_root / "service_ready"
        self.info_path = self.service_root / "service_info.json"
        self.shutdown_path = self.service_root / "shutdown"

        for directory in (self.requests_dir, self.responses_dir, self.outputs_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def publish_ready(self) -> dict[str, Any]:
        info = {
            "protocol_name": "objview_shape_completion",
            "protocol_version": "v1",
            "methods": {
                "complete_shape": {
                    "description": "Run one completion inference from a partial point cloud file.",
                    "required_params": ["partial_pointcloud_path"],
                    "optional_params": ["output_pointcloud_path"],
                }
            },
            "capability": self.backend.capability_metadata(),
            "files": {
                "requests_root": self._rel(self.requests_dir),
                "responses_root": self._rel(self.responses_dir),
                "outputs_root": self._rel(self.outputs_dir),
                "ready_path": self._rel(self.ready_path),
                "request_ready_pattern": "requests/{request_name}.json.ready",
                "response_ready_pattern": "responses/{request_name}.json.ready",
            },
        }
        self._write_json(self.info_path, info)
        self._touch(self.ready_path)
        return info

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = request.get("id")
        try:
            self._validate_request(request)
            method = request["method"]
            params = request.get("params", {})
            if method == "complete_shape":
                result = self._handle_complete_shape(request_id=request_id, params=params)
            else:
                raise ShapeCompletionError("METHOD_NOT_FOUND", f"Unsupported method: {method}")
            response = {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}
        except ShapeCompletionError as exc:
            response = {
                "jsonrpc": JSONRPC_VERSION,
                "id": request_id,
                "error": {"code": exc.code, "message": exc.message},
            }
        except Exception as exc:  # pragma: no cover - defensive
            response = {
                "jsonrpc": JSONRPC_VERSION,
                "id": request_id,
                "error": {"code": "INTERNAL_ERROR", "message": str(exc)},
            }
        self._append_log({"time": time.time(), "request": request, "response": response})
        return response

    def process_pending_requests(self) -> list[dict[str, Any]]:
        processed = []
        for ready_path in sorted(self.requests_dir.glob(self.request_glob)):
            request_path = self._json_path_from_ready(ready_path)
            response_path = self.responses_dir / request_path.name
            response_ready_path = self._ready_path(response_path)

            if response_ready_path.exists():
                try:
                    ready_path.unlink()
                except FileNotFoundError:
                    pass
                continue

            try:
                request = self._read_json(request_path)
            except (FileNotFoundError, json.JSONDecodeError):
                continue

            response = self.handle_request(request)
            self._write_json(response_path, response)
            self._touch(response_ready_path)
            try:
                ready_path.unlink()
            except FileNotFoundError:
                pass

            processed.append(
                {
                    "request_path": self._rel(request_path),
                    "response_path": self._rel(response_path),
                    "request_ready_path": self._rel(ready_path),
                    "response_ready_path": self._rel(response_ready_path),
                    "request_id": response.get("id"),
                    "has_error": "error" in response,
                }
            )
        return processed

    def serve_forever(self, *, poll_interval_sec: float = 0.05) -> None:
        self.publish_ready()
        while not self.shutdown_path.exists():
            self.process_pending_requests()
            time.sleep(poll_interval_sec)

    def _handle_complete_shape(self, *, request_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        partial_path = params.get("partial_pointcloud_path")
        if not partial_path:
            raise ShapeCompletionError("INVALID_PARAMS", "Missing partial_pointcloud_path.")

        output_path = params.get("output_pointcloud_path")
        if output_path is None:
            stem = Path(str(partial_path)).stem
            suffix = Path(str(partial_path)).suffix or ".pcd"
            request_name = str(request_id) if request_id is not None else f"request_{int(time.time() * 1000)}"
            output_path = self.outputs_dir / f"{request_name}_{stem}_completed{suffix}"

        return self.backend.complete_point_cloud(partial_path, output_path)

    def _validate_request(self, request: dict[str, Any]) -> None:
        if not isinstance(request, dict):
            raise ShapeCompletionError("INVALID_REQUEST", "Request must be a JSON object.")
        if request.get("jsonrpc") != JSONRPC_VERSION:
            raise ShapeCompletionError("INVALID_REQUEST", f"Expected jsonrpc={JSONRPC_VERSION}.")
        if request.get("method") not in SHAPE_COMPLETION_METHODS:
            raise ShapeCompletionError("METHOD_NOT_FOUND", f"Unsupported method: {request.get('method')}")

    def _append_log(self, record: dict[str, Any]) -> None:
        log_path = self.logs_dir / "shape_completion_rpc_log.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2)
        try:
            os.replace(str(tmp_path), str(path))
        except FileNotFoundError:
            if path.exists():
                return
            with path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=True, indent=2)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _touch(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)

    @staticmethod
    def _ready_path(json_path: Path) -> Path:
        return json_path.with_suffix(json_path.suffix + ".ready")

    @staticmethod
    def _json_path_from_ready(ready_path: Path) -> Path:
        if ready_path.suffix != ".ready":
            raise ValueError(f"Expected .ready path, got {ready_path}")
        return ready_path.with_suffix("")

    def _rel(self, path: Path) -> str:
        return str(path.relative_to(self.service_root)).replace("\\", "/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ObjView shape completion service.")
    parser.add_argument(
        "--service-root",
        default=None,
        help="Service root containing requests/responses/outputs. If omitted, --session-dir/shape_completion is used.",
    )
    parser.add_argument(
        "--session-dir",
        default=None,
        help="Episode session dir. Used to derive a default service root at <session-dir>/shape_completion.",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "PoinTr" / "cfgs" / "ShapeNet55_models" / "PoinTr.yaml"),
        help="PoinTr config path.",
    )
    parser.add_argument(
        "--ckpt",
        default=str(Path(__file__).resolve().parent / "PoinTr" / "PoinTr-C" / "ckpt-best.pth"),
        help="PoinTr checkpoint path.",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device for the completion backend.")
    parser.add_argument("--poll-interval-sec", type=float, default=0.05, help="Polling interval for request processing.")
    parser.add_argument("--no-warmup", action="store_true", help="Skip one-time model warmup before publishing ready.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.service_root is None:
        if args.session_dir is None:
            raise RuntimeError("Either --service-root or --session-dir must be provided.")
        service_root = Path(args.session_dir).resolve() / "shape_completion"
    else:
        service_root = Path(args.service_root).resolve()
    backend = PoinTrCompletionBackend(
        PoinTrCompletionConfig(
            config_path=args.config,
            ckpt_path=args.ckpt,
            device=args.device,
        )
    )
    if not args.no_warmup:
        backend.warmup()
    service = ShapeCompletionService(service_root, backend)
    service.serve_forever(poll_interval_sec=args.poll_interval_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
