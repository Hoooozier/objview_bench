from __future__ import annotations

import argparse
import json
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from api_render import CameraPose
except ModuleNotFoundError:
    @dataclass(frozen=True)
    class CameraPose:
        camera_xyz: tuple[float, float, float]
        lookat_xyz: tuple[float, float, float]
        roll_rad: float = 0.0


POSE_FORMAT = "camera_lookat_roll"
POSE_FIELDS = (
    "camera_x",
    "camera_y",
    "camera_z",
    "lookat_x",
    "lookat_y",
    "lookat_z",
    "roll_rad",
    "camera_radius",
)


@dataclass(frozen=True)
class FeasibilityResult:
    feasible: bool
    reason: str = ""


class FeasibilityAPI:
    """
    Constraint-file based pose feasibility checker.

    Expected JSON shape:

    {
      "pose_format": "camera_lookat_roll",
      "atol": 0.0001,
      "structured_rule": {
        "logic": "and",
        "rules": [
          {"field": "camera_z", "op": ">=", "value": 0.0},
          {"field": "camera_y", "op": ">=", "value": 0.0}
        ]
      },
      "blocklist": {
        "poses": [
          [1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]
        ]
      }
    }

    Semantics:
    - structured_rule, when present, defines the allowed continuous pose space.
    - blocklist, when present, removes exact/tolerance-matched poses from it.
    - final feasible = passes structured_rule and is not in blocklist.
    """

    def __init__(self, constraint_json: str | Path) -> None:
        self.constraint_json = Path(constraint_json)
        with self.constraint_json.open("r", encoding="utf-8") as f:
            self.spec = json.load(f)

        pose_format = self.spec.get("pose_format", POSE_FORMAT)
        if pose_format != POSE_FORMAT:
            raise ValueError(f"Unsupported pose_format: {pose_format!r}; expected {POSE_FORMAT!r}")

        self.atol = float(self.spec.get("atol", 1e-4))
        if self.atol < 0:
            raise ValueError("atol must be non-negative")

        self.structured_rule = self.spec.get("structured_rule")
        self.blocklist = self._parse_blocklist(self.spec.get("blocklist"))

        if self.structured_rule is not None:
            self._validate_structured_rule(self.structured_rule)

    def is_feasible(self, pose: CameraPose | Sequence[float]) -> bool:
        return self.check(pose).feasible

    def check(self, pose: CameraPose | Sequence[float]) -> FeasibilityResult:
        camera_pose = self._coerce_pose(pose)

        if not self._passes_structured_rule(camera_pose):
            return FeasibilityResult(False, "structured_rule_not_satisfied")

        if self._is_blocked_pose(camera_pose):
            return FeasibilityResult(False, "blocked_by_pose_blocklist")

        return FeasibilityResult(True, "feasible")

    @staticmethod
    def pose_from_7d(values: Sequence[float]) -> CameraPose:
        if len(values) != 7:
            raise ValueError(f"Expected 7 pose values, got {len(values)}")
        vals = [float(v) for v in values]
        return CameraPose(
            camera_xyz=(vals[0], vals[1], vals[2]),
            lookat_xyz=(vals[3], vals[4], vals[5]),
            roll_rad=vals[6],
        )

    @staticmethod
    def pose_to_7d(pose: CameraPose) -> tuple[float, float, float, float, float, float, float]:
        return (
            float(pose.camera_xyz[0]),
            float(pose.camera_xyz[1]),
            float(pose.camera_xyz[2]),
            float(pose.lookat_xyz[0]),
            float(pose.lookat_xyz[1]),
            float(pose.lookat_xyz[2]),
            float(pose.roll_rad),
        )

    def _coerce_pose(self, pose: CameraPose | Sequence[float]) -> CameraPose:
        if isinstance(pose, CameraPose):
            return pose
        return self.pose_from_7d(pose)

    def _parse_blocklist(self, blocklist_spec: Any) -> list[CameraPose]:
        if blocklist_spec is None:
            return []
        if not isinstance(blocklist_spec, dict):
            raise ValueError("blocklist must be a JSON object")

        poses = blocklist_spec.get("poses", [])
        if not isinstance(poses, list):
            raise ValueError("blocklist.poses must be a list")
        return [self.pose_from_7d(pose) for pose in poses]

    def _validate_structured_rule(self, rule_spec: Any) -> None:
        if not isinstance(rule_spec, dict):
            raise ValueError("structured_rule must be a JSON object")

        logic = rule_spec.get("logic", "and")
        if logic not in {"and", "or"}:
            raise ValueError("structured_rule.logic must be 'and' or 'or'")

        rules = rule_spec.get("rules", [])
        if not isinstance(rules, list) or not rules:
            raise ValueError("structured_rule.rules must be a non-empty list")

        for rule in rules:
            if not isinstance(rule, dict):
                raise ValueError("Each structured_rule rule must be a JSON object")
            field = rule.get("field")
            op = rule.get("op")
            if field not in POSE_FIELDS:
                raise ValueError(f"Unsupported structured_rule field: {field!r}")
            if op not in _OPS:
                raise ValueError(f"Unsupported structured_rule op: {op!r}")
            if "value" not in rule:
                raise ValueError("Each structured_rule rule must contain a value")
            float(rule["value"])

    def _passes_structured_rule(self, pose: CameraPose) -> bool:
        if self.structured_rule is None:
            return True

        logic = self.structured_rule.get("logic", "and")
        results = [self._eval_rule(pose, rule) for rule in self.structured_rule["rules"]]
        if logic == "and":
            return all(results)
        return any(results)

    def _eval_rule(self, pose: CameraPose, rule: dict[str, Any]) -> bool:
        actual = self._pose_field_value(pose, rule["field"])
        expected = float(rule["value"])
        op = rule["op"]

        if op == "==":
            return abs(actual - expected) <= self.atol
        if op == "!=":
            return abs(actual - expected) > self.atol
        return bool(_OPS[op](actual, expected))

    def _is_blocked_pose(self, pose: CameraPose) -> bool:
        return any(self._pose_equal(pose, blocked_pose) for blocked_pose in self.blocklist)

    def _pose_equal(self, a: CameraPose, b: CameraPose) -> bool:
        return (
            np.allclose(np.asarray(a.camera_xyz), np.asarray(b.camera_xyz), atol=self.atol, rtol=0.0)
            and np.allclose(np.asarray(a.lookat_xyz), np.asarray(b.lookat_xyz), atol=self.atol, rtol=0.0)
            and abs(float(a.roll_rad) - float(b.roll_rad)) <= self.atol
        )

    @staticmethod
    def _pose_field_value(pose: CameraPose, field: str) -> float:
        values = {
            "camera_x": pose.camera_xyz[0],
            "camera_y": pose.camera_xyz[1],
            "camera_z": pose.camera_xyz[2],
            "lookat_x": pose.lookat_xyz[0],
            "lookat_y": pose.lookat_xyz[1],
            "lookat_z": pose.lookat_xyz[2],
            "roll_rad": pose.roll_rad,
            "camera_radius": float(np.linalg.norm(np.asarray(pose.camera_xyz, dtype=np.float32))),
        }
        return float(values[field])


_OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


def _default_constraint_path(name: str) -> Path:
    return Path(__file__).resolve().parent / "configs" / "feasibility" / f"{name}.json"


def _run_default_smoke_tests() -> None:
    test_poses = {
        "positive_yz": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "negative_z": [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        "negative_y": [0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    }

    for constraint_name in ("whole", "hemi", "quarter", "eighth"):
        api = FeasibilityAPI(_default_constraint_path(constraint_name))
        print(f"[{constraint_name}]")
        for pose_name, pose in test_poses.items():
            result = api.check(pose)
            print(f"  {pose_name}: {result.feasible} ({result.reason})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual tester for FeasibilityAPI.")
    parser.add_argument(
        "--constraint",
        choices=("whole", "hemi", "quarter", "eighth"),
        default="whole",
        help="Default feasibility config name.",
    )
    parser.add_argument(
        "--constraint-json",
        type=Path,
        default=None,
        help="Custom feasibility JSON path. Overrides --constraint.",
    )
    parser.add_argument(
        "--pose",
        type=float,
        nargs=7,
        metavar=("CX", "CY", "CZ", "LX", "LY", "LZ", "ROLL"),
        help="Pose in camera_lookat_roll format.",
    )
    parser.add_argument(
        "--run-default-tests",
        action="store_true",
        help="Run built-in checks against whole/hemi/quarter.",
    )
    args = parser.parse_args()

    if args.run_default_tests:
        _run_default_smoke_tests()
        return

    constraint_path = args.constraint_json or _default_constraint_path(args.constraint)
    api = FeasibilityAPI(constraint_path)

    pose = args.pose or [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    result = api.check(pose)

    print(f"constraint_json: {constraint_path}")
    print(f"pose: {pose}")
    print(f"feasible: {result.feasible}")
    print(f"reason: {result.reason}")


if __name__ == "__main__":
    main()

"""
python api_feasibility.py --run-default-tests
python api_feasibility.py --constraint whole --pose 0 1 1 0 0 0 0
python api_feasibility.py --constraint hemi --pose 0 1 -1 0 0 0 0
python api_feasibility.py --constraint quarter --pose 0 -1 1 0 0 0 0
python api_feasibility.py --constraint eighth --pose 1 1 1 0 0 0 0
python api_feasibility.py --constraint-json configs/feasibility/quarter.json --pose 0 -1 1 0 0 0 0
"""
