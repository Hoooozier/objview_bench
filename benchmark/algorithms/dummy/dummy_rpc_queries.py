from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api_feasibility import FeasibilityAPI
from api_interaction import InteractionSession


def _request(method: str, params: dict, request_id: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone smoke test for InteractionSession RPC methods.")
    parser.add_argument("--session-dir", type=Path, default=Path("interaction/session_rpc"))
    parser.add_argument("--feasibility-json", type=Path, default=Path("configs/feasibility/quarter.json"))
    args = parser.parse_args()

    session = InteractionSession(
        session_dir=args.session_dir,
        feasibility_api=FeasibilityAPI(args.feasibility_json),
    )
    session.clear_submitted_action()
    session.publish_episode_config(
        {
            "episode_id": "dummy_rpc_queries",
            "uid": "dummy_uid",
            "task": {"type": "object_centric_active_3d_reconstruction", "evaluation_target": "geometry_only"},
        }
    )
    session.publish_current_step(
        step_index=0,
        current_step_record={
            "episode_id": "dummy_rpc_queries",
            "visited_view_num": 1,
            "current_pose": [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "observation_manifest_path": "observations/step_000.json",
        },
    )

    requests = [
        _request("describe_interface", {}, "describe"),
        _request("is_feasible", {"pose": [3.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]}, "feasible_single"),
        _request(
            "is_feasible",
            {
                "poses": [
                    [3.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [3.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                ]
            },
            "feasible_batch",
        ),
        _request("pose_to_matrix", {"pose": [3.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]}, "matrix_single"),
        _request(
            "pose_to_matrix",
            {
                "poses": [
                    [3.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                ]
            },
            "matrix_batch",
        ),
        _request(
            "submit_action",
            {
                "action": "stop",
                "step_index": 0,
                "stop_reason": "plan_end",
                "algorithm_runtime_sec": 0.01,
            },
            "submit_stop",
        ),
    ]
    responses = [session.handle_request(req) for req in requests]
    print(json.dumps(responses, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
