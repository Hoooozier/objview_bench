from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import torch


JSONRPC_VERSION = "2.0"
PLANNING_NETWORK_METHODS = {"infer"}


class PlanningNetworkError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _import_from_dir(module_dir: Path, module_name: str) -> Any:
    module_dir = module_dir.resolve()
    module_dir_str = str(module_dir)
    if module_dir_str not in sys.path:
        sys.path.insert(0, module_dir_str)
    return importlib.import_module(module_name)


def _torch_device(device_name: str) -> torch.device:
    requested = torch.device(device_name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print(f"[WARN] CUDA unavailable, falling back to CPU from {device_name}.", flush=True)
        return torch.device("cpu")
    return requested


@dataclass
class PlanningNetworkConfig:
    backend: str
    ckpt_path: Union[str, Path]
    device: str = "cuda:0"
    views_path: Optional[Union[str, Path]] = None
    view_position_radius: float = 1.0
    grid_size: int = 64
    mascvp_gamma: float = 0.5
    mascvp_decode_gammas: Optional[list[float]] = None
    pcnbv_num_points: int = 1024
    pcnbv_seed: int = 42


class PlanningNetworkBackend:
    def capability_metadata(self) -> dict[str, Any]:
        raise NotImplementedError

    def warmup(self) -> dict[str, Any]:
        raise NotImplementedError

    def infer_npz(self, input_npz: Union[str, Path], *, topk: int) -> dict[str, Any]:
        raise NotImplementedError


class BENBVBackend(PlanningNetworkBackend):
    def __init__(self, cfg: PlanningNetworkConfig) -> None:
        self.cfg = cfg
        self.backend = "benbv"
        self.ckpt_path = Path(cfg.ckpt_path).resolve()
        self.device = _torch_device(cfg.device)
        module_dir = Path(__file__).resolve().parent / "planning_network" / "BENBV"
        self.infer_once = _import_from_dir(module_dir, "infer_once")
        self.model = self.infer_once.load_model(self.ckpt_path, device=self.device)

    def capability_metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "ckpt_path": str(self.ckpt_path),
            "device": str(self.device),
            "output_semantics": "scores over 20 boundary candidates; best_index is the highest-scoring candidate.",
        }

    def warmup(self) -> dict[str, Any]:
        partial, candidates, context = self.infer_once.build_random_inputs(
            batch_size=1,
            num_partial_points=4096,
            num_candidates=20,
            device=self.device,
            zero_partial=True,
            zero_candidates=True,
            zero_context=True,
        )
        start = time.perf_counter()
        with torch.no_grad():
            _ = self.model(partial, candidates, context)
        return {"warmed_up": True, "runtime_sec": time.perf_counter() - start, **self.capability_metadata()}

    def infer_npz(self, input_npz: Union[str, Path], *, topk: int) -> dict[str, Any]:
        t0 = time.perf_counter()
        partial, candidates, context, case_meta = self.infer_once.build_inputs_from_npz(Path(input_npz), device=self.device)
        preprocess_sec = time.perf_counter() - t0

        inference_start = time.perf_counter()
        with torch.no_grad():
            scores = self.model(partial, candidates, context)
        inference_sec = time.perf_counter() - inference_start

        post_start = time.perf_counter()
        scores_2d = scores.squeeze(-1)
        num_candidates = int(scores_2d.shape[1])
        topk = min(max(1, int(topk)), num_candidates)
        topk_scores, topk_indices = torch.topk(scores_2d, k=topk, dim=1)
        best_scores, best_indices = torch.max(scores_2d, dim=1)
        postprocess_sec = time.perf_counter() - post_start

        return {
            "backend": self.backend,
            "input_npz": str(input_npz),
            "batch_size": int(scores_2d.shape[0]),
            "num_candidates": num_candidates,
            "scores": scores_2d.detach().cpu().tolist(),
            "best_index": best_indices.detach().cpu().tolist(),
            "best_score": best_scores.detach().cpu().tolist(),
            "topk_indices": topk_indices.detach().cpu().tolist(),
            "topk_scores": topk_scores.detach().cpu().tolist(),
            "case_meta": case_meta,
            "runtime": {
                "preprocess_sec": preprocess_sec,
                "inference_sec": inference_sec,
                "postprocess_sec": postprocess_sec,
                "total_sec": preprocess_sec + inference_sec + postprocess_sec,
            },
        }


class MASCVPBackend(PlanningNetworkBackend):
    def __init__(self, cfg: PlanningNetworkConfig) -> None:
        self.cfg = cfg
        self.backend = "mascvp"
        self.ckpt_path = Path(cfg.ckpt_path).resolve()
        if cfg.views_path is None:
            raise RuntimeError("MASCVP backend requires --views.")
        self.views_path = Path(cfg.views_path).resolve()
        self.device = _torch_device(cfg.device)
        self.grid_size = int(cfg.grid_size)
        self.gamma = float(cfg.mascvp_gamma)
        decode_gammas = cfg.mascvp_decode_gammas if cfg.mascvp_decode_gammas is not None else [0.3, 0.4, 0.5, 0.6, 0.7]
        self.decode_gammas = sorted({float(self.gamma), *(float(value) for value in decode_gammas)})
        module_dir = Path(__file__).resolve().parent / "planning_network" / "MASCVP"
        self.infer_once = _import_from_dir(module_dir, "infer_once")
        view_positions = self.infer_once.load_view_positions(self.views_path, radius=float(cfg.view_position_radius))
        self.num_views = int(view_positions.shape[0])
        self.model = self.infer_once.load_model(
            self.ckpt_path,
            view_positions,
            device=self.device,
            grid_size=self.grid_size,
            output_views=self.num_views,
        )

    def capability_metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "ckpt_path": str(self.ckpt_path),
            "views_path": str(self.views_path),
            "view_position_radius": float(self.cfg.view_position_radius),
            "device": str(self.device),
            "grid_size": self.grid_size,
            "output_views": self.num_views,
            "decode_gamma": self.gamma,
            "decode_gammas": self.decode_gammas,
            "output_semantics": "scores over view candidates; gamma_0.5 is a convenience decode, not the raw model output.",
        }

    def warmup(self) -> dict[str, Any]:
        grid, view_state = self.infer_once.build_random_inputs(
            batch_size=1,
            grid_size=self.grid_size,
            num_views=self.num_views,
            random_grid=False,
            random_view_state=False,
            device=self.device,
        )
        start = time.perf_counter()
        with torch.no_grad():
            _ = torch.sigmoid(self.model(grid, view_state))
        return {"warmed_up": True, "runtime_sec": time.perf_counter() - start, **self.capability_metadata()}

    def infer_npz(self, input_npz: Union[str, Path], *, topk: int) -> dict[str, Any]:
        t0 = time.perf_counter()
        grid, view_state, case_meta = self.infer_once.build_inputs_from_npz(Path(input_npz), device=self.device)
        preprocess_sec = time.perf_counter() - t0

        inference_start = time.perf_counter()
        with torch.no_grad():
            scores = torch.sigmoid(self.model(grid, view_state))
        inference_sec = time.perf_counter() - inference_start

        post_start = time.perf_counter()
        topk = min(max(1, int(topk)), self.num_views)
        topk_scores, topk_indices = torch.topk(scores, k=topk, dim=1)
        decodes = {}
        for gamma in self.decode_gammas:
            selected_mask = scores >= gamma
            gamma_indices = []
            gamma_scores = []
            for batch_idx in range(scores.shape[0]):
                batch_indices = torch.nonzero(selected_mask[batch_idx], as_tuple=False).flatten()
                gamma_indices.append(batch_indices.detach().cpu().tolist())
                gamma_scores.append(scores[batch_idx, batch_indices].detach().cpu().tolist())
            decodes[f"gamma_{gamma:g}"] = {
                "selected_indices": gamma_indices,
                "selected_scores": gamma_scores,
                "num_selected": [len(items) for items in gamma_indices],
            }
        postprocess_sec = time.perf_counter() - post_start

        return {
            "backend": self.backend,
            "input_npz": str(input_npz),
            "batch_size": int(scores.shape[0]),
            "num_candidates": self.num_views,
            "scores": scores.detach().cpu().tolist(),
            "topk_indices": topk_indices.detach().cpu().tolist(),
            "topk_scores": topk_scores.detach().cpu().tolist(),
            "decodes": decodes,
            "case_meta": case_meta,
            "runtime": {
                "preprocess_sec": preprocess_sec,
                "inference_sec": inference_sec,
                "postprocess_sec": postprocess_sec,
                "total_sec": preprocess_sec + inference_sec + postprocess_sec,
            },
        }


class NBVNETBackend(PlanningNetworkBackend):
    def __init__(self, cfg: PlanningNetworkConfig) -> None:
        self.cfg = cfg
        self.backend = "nbvnet"
        self.ckpt_path = Path(cfg.ckpt_path).resolve()
        if cfg.views_path is None:
            raise RuntimeError("NBVNET backend requires --views.")
        self.views_path = Path(cfg.views_path).resolve()
        self.device = _torch_device(cfg.device)
        self.grid_size = int(cfg.grid_size)
        module_dir = Path(__file__).resolve().parent / "planning_network" / "NBVNET"
        self.infer_once = _import_from_dir(module_dir, "infer_once")
        view_positions = self.infer_once.load_view_positions(
            self.views_path,
            radius=float(cfg.view_position_radius),
        )
        self.num_views = int(view_positions.shape[0])
        self.model = self.infer_once.load_model(
            self.ckpt_path,
            view_positions,
            device=self.device,
            grid_size=self.grid_size,
            num_classes=self.num_views,
        )

    def capability_metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "ckpt_path": str(self.ckpt_path),
            "views_path": str(self.views_path),
            "view_position_radius": float(self.cfg.view_position_radius),
            "device": str(self.device),
            "grid_size": self.grid_size,
            "output_views": self.num_views,
            "output_semantics": "single-class view classification; use best_index/ranking, not calibrated scores.",
        }

    def warmup(self) -> dict[str, Any]:
        grid, _view_state = self.infer_once.build_random_inputs(
            batch_size=1,
            grid_size=self.grid_size,
            num_views=self.num_views,
            random_grid=False,
            random_view_state=False,
            device=self.device,
        )
        start = time.perf_counter()
        with torch.no_grad():
            _ = self.model(grid)
        return {"warmed_up": True, "runtime_sec": time.perf_counter() - start, **self.capability_metadata()}

    def infer_npz(self, input_npz: Union[str, Path], *, topk: int) -> dict[str, Any]:
        t0 = time.perf_counter()
        grid, _view_state, case_meta = self.infer_once.build_inputs_from_npz(Path(input_npz), device=self.device)
        preprocess_sec = time.perf_counter() - t0

        inference_start = time.perf_counter()
        with torch.no_grad():
            logits = self.model(grid)
        inference_sec = time.perf_counter() - inference_start

        post_start = time.perf_counter()
        num_candidates = int(logits.shape[1])
        topk = min(max(1, int(topk)), num_candidates)
        _topk_values, topk_indices = torch.topk(logits, k=topk, dim=1)
        best_indices = torch.argmax(logits, dim=1)
        postprocess_sec = time.perf_counter() - post_start

        return {
            "backend": self.backend,
            "input_npz": str(input_npz),
            "batch_size": int(logits.shape[0]),
            "num_candidates": num_candidates,
            "best_index": best_indices.detach().cpu().tolist(),
            "topk_indices": topk_indices.detach().cpu().tolist(),
            "score_semantics": "single-class classification; logits are intentionally omitted and ranking is the usable output.",
            "case_meta": case_meta,
            "runtime": {
                "preprocess_sec": preprocess_sec,
                "inference_sec": inference_sec,
                "postprocess_sec": postprocess_sec,
                "total_sec": preprocess_sec + inference_sec + postprocess_sec,
            },
        }


class PCNBVBackend(PlanningNetworkBackend):
    def __init__(self, cfg: PlanningNetworkConfig) -> None:
        self.cfg = cfg
        self.backend = "pcnbv"
        self.ckpt_path = Path(cfg.ckpt_path).resolve()
        if cfg.views_path is None:
            raise RuntimeError("PCNBV backend requires --views.")
        self.views_path = Path(cfg.views_path).resolve()
        self.device = _torch_device(cfg.device)
        self.num_points = int(cfg.pcnbv_num_points)
        self.seed = int(cfg.pcnbv_seed)
        module_dir = Path(__file__).resolve().parent / "planning_network" / "PCNBV"
        self.infer_once = _import_from_dir(module_dir, "infer_once")
        view_positions = self.infer_once.load_view_positions(
            self.views_path,
            radius=float(cfg.view_position_radius),
            expected_views=128,
        )
        self.num_views = int(view_positions.shape[0])
        self.model = self.infer_once.load_model(
            self.ckpt_path,
            view_positions,
            device=self.device,
            views=self.num_views,
        )

    def capability_metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "ckpt_path": str(self.ckpt_path),
            "views_path": str(self.views_path),
            "view_position_radius": float(self.cfg.view_position_radius),
            "device": str(self.device),
            "num_points": self.num_points,
            "output_views": self.num_views,
            "output_semantics": "per-view regression scores; higher values are ranked as better candidates.",
        }

    def warmup(self) -> dict[str, Any]:
        points, view_state = self.infer_once.build_random_inputs(
            batch_size=1,
            num_points=self.num_points,
            num_views=self.num_views,
            random_view_state=False,
            device=self.device,
        )
        start = time.perf_counter()
        with torch.no_grad():
            _latent, _scores = self.model(points, view_state)
        return {"warmed_up": True, "runtime_sec": time.perf_counter() - start, **self.capability_metadata()}

    def infer_npz(self, input_npz: Union[str, Path], *, topk: int) -> dict[str, Any]:
        t0 = time.perf_counter()
        points, view_state, case_meta = self.infer_once.build_inputs_from_npz(
            Path(input_npz),
            device=self.device,
            num_points=self.num_points,
            num_views=self.num_views,
            seed=self.seed,
        )
        preprocess_sec = time.perf_counter() - t0

        inference_start = time.perf_counter()
        with torch.no_grad():
            _latent, scores = self.model(points, view_state)
        inference_sec = time.perf_counter() - inference_start

        post_start = time.perf_counter()
        num_candidates = int(scores.shape[1])
        topk = min(max(1, int(topk)), num_candidates)
        topk_scores, topk_indices = torch.topk(scores, k=topk, dim=1)
        best_scores, best_indices = torch.max(scores, dim=1)
        postprocess_sec = time.perf_counter() - post_start

        return {
            "backend": self.backend,
            "input_npz": str(input_npz),
            "batch_size": int(scores.shape[0]),
            "num_points": int(points.shape[2]),
            "num_candidates": num_candidates,
            "scores": scores.detach().cpu().tolist(),
            "best_index": best_indices.detach().cpu().tolist(),
            "best_score": best_scores.detach().cpu().tolist(),
            "topk_indices": topk_indices.detach().cpu().tolist(),
            "topk_scores": topk_scores.detach().cpu().tolist(),
            "score_semantics": "per-view regression scores; higher values are ranked as better candidates.",
            "case_meta": case_meta,
            "runtime": {
                "preprocess_sec": preprocess_sec,
                "inference_sec": inference_sec,
                "postprocess_sec": postprocess_sec,
                "total_sec": preprocess_sec + inference_sec + postprocess_sec,
            },
        }


class PlanningNetworkService:
    def __init__(
        self,
        service_root: Union[str, Path],
        backend: PlanningNetworkBackend,
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
            "protocol_name": "objview_planning_network",
            "protocol_version": "v1",
            "methods": {
                "infer": {
                    "description": "Run one planning network inference from an input npz file.",
                    "required_params": ["input_npz"],
                    "optional_params": ["topk"],
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
            params = request.get("params", {})
            result = self._handle_infer(params)
            response = {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}
        except PlanningNetworkError as exc:
            response = {
                "jsonrpc": JSONRPC_VERSION,
                "id": request_id,
                "error": {"code": exc.code, "message": exc.message},
            }
        except Exception as exc:
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

    def _handle_infer(self, params: dict[str, Any]) -> dict[str, Any]:
        input_npz = params.get("input_npz")
        if not input_npz:
            raise PlanningNetworkError("INVALID_PARAMS", "Missing input_npz.")
        topk = int(params.get("topk", 5))
        return self.backend.infer_npz(input_npz, topk=topk)

    def _validate_request(self, request: dict[str, Any]) -> None:
        if not isinstance(request, dict):
            raise PlanningNetworkError("INVALID_REQUEST", "Request must be a JSON object.")
        if request.get("jsonrpc") != JSONRPC_VERSION:
            raise PlanningNetworkError("INVALID_REQUEST", f"Expected jsonrpc={JSONRPC_VERSION}.")
        if request.get("method") not in PLANNING_NETWORK_METHODS:
            raise PlanningNetworkError("METHOD_NOT_FOUND", f"Unsupported method: {request.get('method')}")

    def _append_log(self, record: dict[str, Any]) -> None:
        log_path = self.logs_dir / "planning_network_rpc_log.jsonl"
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


def build_backend(args: argparse.Namespace) -> PlanningNetworkBackend:
    view_position_radius = args.view_position_radius
    if view_position_radius is None:
        view_position_radius = 3.0 if args.backend == "pcnbv" else 1.0
    cfg = PlanningNetworkConfig(
        backend=args.backend,
        ckpt_path=args.ckpt,
        device=args.device,
        views_path=args.views,
        view_position_radius=view_position_radius,
        grid_size=args.grid_size,
        mascvp_gamma=args.mascvp_gamma,
        mascvp_decode_gammas=args.mascvp_decode_gammas,
        pcnbv_num_points=args.pcnbv_num_points,
        pcnbv_seed=args.pcnbv_seed,
    )
    if args.backend == "benbv":
        return BENBVBackend(cfg)
    if args.backend == "mascvp":
        return MASCVPBackend(cfg)
    if args.backend == "nbvnet":
        return NBVNETBackend(cfg)
    if args.backend == "pcnbv":
        return PCNBVBackend(cfg)
    raise RuntimeError(f"Unsupported backend: {args.backend}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ObjView planning network service.")
    parser.add_argument(
        "--service-root",
        default=None,
        help="Service root containing requests/responses/outputs. If omitted, --session-dir/planning_network/<service-name> is used.",
    )
    parser.add_argument(
        "--session-dir",
        default=None,
        help="Episode session dir. Used to derive a default service root.",
    )
    parser.add_argument("--service-name", default="default", help="Named planning-network service under planning_network/.")
    parser.add_argument("--backend", choices=["benbv", "mascvp", "nbvnet", "pcnbv"], required=True)
    parser.add_argument("--ckpt", required=True, help="Backend checkpoint path.")
    parser.add_argument("--views", default=None, help="View xyz file for MASCVP/NBVNET/PCNBV.")
    parser.add_argument(
        "--view-position-radius",
        type=float,
        default=None,
        help="View-position radius. Defaults to 1.0, except PCNBV defaults to 3.0.",
    )
    parser.add_argument("--grid-size", type=int, default=64)
    parser.add_argument("--mascvp-gamma", type=float, default=0.5)
    parser.add_argument(
        "--mascvp-decode-gammas",
        type=float,
        nargs="+",
        default=None,
        help="MASCVP decode thresholds to include in the response. Defaults to 0.3 0.4 0.5 0.6 0.7.",
    )
    parser.add_argument("--pcnbv-num-points", type=int, default=1024)
    parser.add_argument("--pcnbv-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--poll-interval-sec", type=float, default=0.05)
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.service_root is None:
        if args.session_dir is None:
            raise RuntimeError("Either --service-root or --session-dir must be provided.")
        service_root = Path(args.session_dir).resolve() / "planning_network" / str(args.service_name)
    else:
        service_root = Path(args.service_root).resolve()

    backend = build_backend(args)
    if not args.no_warmup:
        backend.warmup()
    service = PlanningNetworkService(service_root, backend)
    service.serve_forever(poll_interval_sec=args.poll_interval_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
