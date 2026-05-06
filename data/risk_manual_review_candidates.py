import json
from pathlib import Path
from typing import Any, Dict, List


BASE_DIR = Path("geometry_sampled")

INPUT_JSON_PATH = BASE_DIR / "manual_review_candidates.json"
OUTPUT_JSON_PATH = BASE_DIR / "manual_review_candidates_with_risk.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_float(x: Any) -> float:
    if x is None:
        return float("nan")
    return float(x)


def compute_bc_ratio(selected_view_count: float, observation_saturation_view_num: float) -> float:
    if observation_saturation_view_num <= 0:
        return -1.0
    return selected_view_count / observation_saturation_view_num


def build_risk_flags(row: Dict[str, Any]) -> List[str]:
    flags: List[str] = []

    B = safe_float(row.get("observation_saturation_view_num"))
    C = safe_float(row.get("selected_view_count"))
    V = safe_float(row.get("gt_surface_voxel_count"))
    pool_type = row.get("pool_type")

    # Truly higher-risk anomaly classes
    if V < 1000:
        flags.append("LOW_SURFACE_VOXEL")

    # C is already close to the fixed 128-candidate view-set ceiling
    if 100 <= C <= 128:
        flags.append("HIGH_C_NEAR_VIEWSET_LIMIT")

    # C significantly larger than B: semantically suspicious, worth priority review
    if (C - B) >= 5:
        flags.append("C_EXCEEDS_B_SIGNIFICANTLY")

    # Structural-indicator / representative flags
    if pool_type == "long_tail_pool":
        flags.append("LONG_TAIL_POOL")

    if B >= 80 and C <= 20:
        flags.append("HIGH_B_LOW_C")

    if C >= 40 and B <= 60:
        flags.append("HIGH_C_LOW_B")

    if B >= 80 and C >= 30:
        flags.append("HIGH_B_HIGH_C")

    return flags


def compute_manual_priority_score(row: Dict[str, Any], risk_flags: List[str]) -> int:
    score = 0
    V = safe_float(row.get("gt_surface_voxel_count"))

    # Prioritize true anomalies
    if "LOW_SURFACE_VOXEL" in risk_flags:
        score += 4
    if "C_EXCEEDS_B_SIGNIFICANTLY" in risk_flags:
        score += 5

    # Very high C should be reviewed early
    if "HIGH_C_NEAR_VIEWSET_LIMIT" in risk_flags:
        score += 2

    # Important/structural samples next
    if "LONG_TAIL_POOL" in risk_flags:
        score += 3
    if "HIGH_B_LOW_C" in risk_flags:
        score += 2
    if "HIGH_C_LOW_B" in risk_flags:
        score += 2
    if "HIGH_B_HIGH_C" in risk_flags:
        score += 2

    # Add a small bonus for very low voxel count
    if V < 1500:
        score += 1

    return score


def main() -> None:
    rows = load_json(INPUT_JSON_PATH)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list in {INPUT_JSON_PATH}, got {type(rows)}")

    output_rows: List[Dict[str, Any]] = []

    for row in rows:
        B = safe_float(row.get("observation_saturation_view_num"))
        C = safe_float(row.get("selected_view_count"))

        bc_ratio = compute_bc_ratio(C, B)
        risk_flags = build_risk_flags(row)
        manual_priority_score = compute_manual_priority_score(row, risk_flags)

        out_row = {
            "uid": row.get("uid"),
            "pool_type": row.get("pool_type"),
            "shape_type": row.get("shape_type"),
            "fill_bucket": row.get("fill_bucket"),
            "self_occlusion_attribute": row.get("self_occlusion_attribute"),
            "observation_saturation_view_num": row.get("observation_saturation_view_num"),
            "selected_view_count": row.get("selected_view_count"),
            "gt_surface_voxel_count": row.get("gt_surface_voxel_count"),
            "obj_path": row.get("obj_path"),
            "bc_ratio": round(bc_ratio, 6) if bc_ratio >= 0 else None,
            "risk_flags": risk_flags,
            "manual_priority_score": manual_priority_score,
        }
        output_rows.append(out_row)

    # Sorting:
    # 1) Higher priority score first
    # 2) long_tail first
    # 3) C_EXCEEDS_B_SIGNIFICANTLY / LOW_SURFACE_VOXEL first
    # 4) B descending
    # 5) C descending
    pool_order = {"long_tail_pool": 0, "main_pool": 1, "both": 2, "unknown": 3}

    def has_flag(row: Dict[str, Any], flag: str) -> int:
        return 1 if flag in row["risk_flags"] else 0

    output_rows.sort(
        key=lambda x: (
            -int(x["manual_priority_score"]),
            pool_order.get(x["pool_type"], 99),
            -has_flag(x, "C_EXCEEDS_B_SIGNIFICANTLY"),
            -has_flag(x, "LOW_SURFACE_VOXEL"),
            -safe_float(x["observation_saturation_view_num"]),
            -safe_float(x["selected_view_count"]),
            str(x["uid"]),
        )
    )

    save_json(OUTPUT_JSON_PATH, output_rows)

    # Print summary stats
    total = len(output_rows)
    print(f"[DONE] total rows: {total}")
    print(f"[DONE] saved to: {OUTPUT_JSON_PATH}")

    flag_counts: Dict[str, int] = {}
    for row in output_rows:
        for flag in row["risk_flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    print("[STATS] risk flag counts:")
    for k in sorted(flag_counts.keys()):
        print(f"  {k}: {flag_counts[k]}")

    priority_counts: Dict[int, int] = {}
    for row in output_rows:
        s = int(row["manual_priority_score"])
        priority_counts[s] = priority_counts.get(s, 0) + 1

    print("[STATS] manual_priority_score counts:")
    for k in sorted(priority_counts.keys(), reverse=True):
        print(f"  {k}: {priority_counts[k]}")

    print("[PREVIEW] top 15 rows:")
    for row in output_rows[:15]:
        print(
            f"  uid={row['uid']}, pool={row['pool_type']}, "
            f"B={row['observation_saturation_view_num']}, "
            f"C={row['selected_view_count']}, "
            f"vox={row['gt_surface_voxel_count']}, "
            f"score={row['manual_priority_score']}, "
            f"flags={row['risk_flags']}"
        )


if __name__ == "__main__":
    main()
