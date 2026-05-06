import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_REVIEW0 = Path("geometry_sampled/review_results_reviewer_0.json")
DEFAULT_REVIEW1 = Path("geometry_sampled/review_results_reviewer_1.json")
DEFAULT_OUTPUT = Path("geometry_sampled/review_results_reviewer_999.json")

REJECT_REASON_ORDER = {
    "pytorch3d_render_error": 0,
    "multi_object": 1,
    "scene": 2,
    "figure": 3,
    "transparent": 4,
    "single_color": 5,
    "single_layer_sheet_like_surface": 6,
    "malformed_or_weird_structure": 7,
    "structure_causes_unreliable_difficulty": 8,
    "other": 9,
}


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_review_dict(data: Any) -> Dict[str, Dict[str, Any]]:
    """
    Supported formats:
    1) {"uid1": {...}, "uid2": {...}}
    2) [{"uid": "...", ...}, ...]
    """
    if data is None:
        return {}

    if isinstance(data, dict):
        out = {}
        for uid, rec in data.items():
            if isinstance(rec, dict):
                rec2 = dict(rec)
                rec2.setdefault("uid", uid)
                out[str(uid)] = rec2
        return out

    if isinstance(data, list):
        out = {}
        for rec in data:
            if isinstance(rec, dict) and "uid" in rec:
                out[str(rec["uid"])] = dict(rec)
        return out

    raise ValueError(f"Unsupported review format: {type(data)}")


def choose_conservative_review(
    uid: str,
    r0: Optional[Dict[str, Any]],
    r1: Optional[Dict[str, Any]],
    rng: random.Random,
) -> Dict[str, Any]:
    """
    Conservative merge logic:
    - If only one reviewer has a result, use it.
    - If both sides are identical, keep it directly.
    - If there is a conflict:
      1) If either side is reject, prefer reject.
      2) If both are reject but reasons differ, randomly keep one reject.
      3) If there is no reject but at least one maybe, keep maybe.
      4) Otherwise keep yes.
    """
    if r0 is None and r1 is None:
        raise ValueError(f"Both reviews are None for uid={uid}")
    if r0 is None:
        out = dict(r1)
        out["reviewer"] = 999
        return out
    if r1 is None:
        out = dict(r0)
        out["reviewer"] = 999
        return out

    label0 = r0.get("manual_label")
    label1 = r1.get("manual_label")

    # Fully identical -> keep reviewer 0 record
    if (
        label0 == label1
        and r0.get("manual_reject_reason") == r1.get("manual_reject_reason")
        and (r0.get("manual_note", "") or "") == (r1.get("manual_note", "") or "")
    ):
        out = dict(r0)
        out["reviewer"] = 999
        return out

    # If any side is reject, prioritize reject
    rejects = []
    if label0 == "reject":
        rejects.append(r0)
    if label1 == "reject":
        rejects.append(r1)

    if len(rejects) == 1:
        out = dict(rejects[0])
        out["reviewer"] = 999
        return out

    if len(rejects) == 2:
        # Both are reject, randomly keep one
        chosen = rng.choice(rejects)
        out = dict(chosen)
        out["reviewer"] = 999
        return out

    # No reject: fallback to maybe if present
    if label0 == "maybe" or label1 == "maybe":
        chosen = r0 if label0 == "maybe" else r1
        out = dict(chosen)
        out["reviewer"] = 999
        return out

    # Otherwise yes/yes or other edge cases
    out = dict(r0)
    out["reviewer"] = 999
    return out


def sort_key(uid: str, rec: Dict[str, Any]):
    label = rec.get("manual_label")
    reason = rec.get("manual_reject_reason")

    # Sort order:
    # reject (by reason 0..9) -> maybe -> yes -> others
    if label == "reject":
        group_order = 0
        reason_order = REJECT_REASON_ORDER.get(reason, 999)
    elif label == "maybe":
        group_order = 1
        reason_order = 999
    elif label == "yes":
        group_order = 2
        reason_order = 999
    else:
        group_order = 3
        reason_order = 999

    return (group_order, reason_order, uid)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge reviewer_0 and reviewer_1 into reviewer_999 with conservative conflict resolution."
    )
    parser.add_argument("--review0", type=str, default=str(DEFAULT_REVIEW0))
    parser.add_argument("--review1", type=str, default=str(DEFAULT_REVIEW1))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when both sides are reject with different reasons.",
    )
    args = parser.parse_args()

    review0_path = Path(args.review0)
    review1_path = Path(args.review1)
    output_path = Path(args.output)

    review0 = normalize_review_dict(load_json(review0_path, default={}))
    review1 = normalize_review_dict(load_json(review1_path, default={}))

    all_uids = sorted(set(review0.keys()) | set(review1.keys()))
    rng = random.Random(args.seed)

    merged_dict: Dict[str, Dict[str, Any]] = {}
    merged_items: List[tuple] = []

    conflict_count = 0
    reject_conflict_count = 0
    maybe_conflict_count = 0

    for uid in all_uids:
        r0 = review0.get(uid)
        r1 = review1.get(uid)

        if r0 is not None and r1 is not None:
            if (
                r0.get("manual_label") != r1.get("manual_label")
                or r0.get("manual_reject_reason") != r1.get("manual_reject_reason")
                or (r0.get("manual_note", "") or "") != (r1.get("manual_note", "") or "")
            ):
                conflict_count += 1
                if r0.get("manual_label") == "reject" or r1.get("manual_label") == "reject":
                    reject_conflict_count += 1
                elif r0.get("manual_label") == "maybe" or r1.get("manual_label") == "maybe":
                    maybe_conflict_count += 1

        merged = choose_conservative_review(uid, r0, r1, rng)
        merged_dict[uid] = merged
        merged_items.append((uid, merged))

    merged_items.sort(key=lambda x: sort_key(x[0], x[1]))

    # Still output dict, but write with sorted order for easier inspection
    ordered_output = {uid: rec for uid, rec in merged_items}
    save_json(output_path, ordered_output)

    # Print summary stats
    label_counts: Dict[str, int] = {}
    reason_counts: Dict[str, int] = {}

    for _, rec in merged_items:
        label = rec.get("manual_label")
        label_counts[label] = label_counts.get(label, 0) + 1
        if label == "reject":
            reason = rec.get("manual_reject_reason") or "unknown"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    print("[DONE] merged to reviewer_999")
    print(f"[INFO] review0 count: {len(review0)}")
    print(f"[INFO] review1 count: {len(review1)}")
    print(f"[INFO] merged count: {len(ordered_output)}")
    print(f"[INFO] conflicts: {conflict_count}")
    print(f"[INFO] conflicts with reject involved: {reject_conflict_count}")
    print(f"[INFO] conflicts resolved to maybe-path: {maybe_conflict_count}")
    print()

    print("[STATS] manual_label counts:")
    for k in ["reject", "maybe", "yes", None]:
        if k in label_counts:
            print(f"  {k}: {label_counts[k]}")
    for k, v in label_counts.items():
        if k not in {"reject", "maybe", "yes", None}:
            print(f"  {k}: {v}")
    print()

    print("[STATS] reject reason counts:")
    for reason, idx in sorted(REJECT_REASON_ORDER.items(), key=lambda x: x[1]):
        if reason in reason_counts:
            print(f"  {idx}. {reason}: {reason_counts[reason]}")
    for reason, count in reason_counts.items():
        if reason not in REJECT_REASON_ORDER:
            print(f"  ?. {reason}: {count}")
    print()

    print(f"[FILE] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())