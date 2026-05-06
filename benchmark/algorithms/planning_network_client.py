from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional


JSONRPC_VERSION = "2.0"


class PlanningNetworkClientError(RuntimeError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise PlanningNetworkClientError(f"Expected JSON object in {path}")
    return data


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    tmp_path.replace(path)


def wait_for_file(path: Path, timeout_sec: float, poll_interval_sec: float) -> None:
    start = time.perf_counter()
    while not path.exists():
        if time.perf_counter() - start > timeout_sec:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(poll_interval_sec)


def planning_network_paths(session_dir: Path, service_name: str) -> dict[str, Path]:
    if not service_name:
        raise ValueError("service_name must be non-empty")
    if "/" in service_name or "\\" in service_name:
        raise ValueError(f"service_name must not contain path separators: {service_name}")
    service_root = session_dir / "planning_network" / service_name
    return {
        "service_root": service_root,
        "requests_dir": service_root / "requests",
        "responses_dir": service_root / "responses",
        "outputs_dir": service_root / "outputs",
        "ready_path": service_root / "service_ready",
    }


def request_planning_network_infer(
    *,
    session_dir: Path,
    service_name: str,
    input_npz: Path,
    request_id: Optional[str] = None,
    topk: int = 5,
    wait_timeout_sec: float = 120.0,
    poll_interval_sec: float = 0.01,
) -> dict[str, Any]:
    paths = planning_network_paths(session_dir, service_name)
    wait_for_file(paths["ready_path"], wait_timeout_sec, poll_interval_sec)

    if request_id is None:
        request_id = f"{service_name}_infer_{int(time.time() * 1000)}"
    request_path = paths["requests_dir"] / f"{request_id}.json"
    request_ready_path = request_path.with_suffix(request_path.suffix + ".ready")
    response_path = paths["responses_dir"] / request_path.name
    response_ready_path = response_path.with_suffix(response_path.suffix + ".ready")

    request = {
        "jsonrpc": JSONRPC_VERSION,
        "id": request_id,
        "method": "infer",
        "params": {
            "input_npz": str(input_npz),
            "topk": int(topk),
        },
    }
    write_json_atomic(request_path, request)
    request_ready_path.touch()
    wait_for_file(response_ready_path, wait_timeout_sec, poll_interval_sec)

    response = read_json(response_path)
    try:
        response_ready_path.unlink()
    except FileNotFoundError:
        pass
    if "error" in response:
        raise PlanningNetworkClientError(f"Planning network RPC failed: {response['error']}")
    result = response.get("result")
    if not isinstance(result, dict):
        raise PlanningNetworkClientError(f"Planning network response missing result: {response}")
    return result
