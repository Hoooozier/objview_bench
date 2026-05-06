import json
from pathlib import Path
from typing import Any, Dict, List, Set, Optional


BASE_DIR = Path("geometry_sampled")

OBJ_NORMALIZED_STATS_PATH = BASE_DIR / "obj_normalized_stats.json"

# A / B metadata source
AB_METADATA_PATH = BASE_DIR / "object_complexity_merged.json"
# C metadata source
C_METADATA_PATH = BASE_DIR / "planning_vs_object_complexity.json"

POOL_DIR = BASE_DIR / "object_complexity_split_by_saturation_partition"
EXTREME_POOL_PATH = POOL_DIR / "extreme_pool.json"
LONG_TAIL_POOL_PATH = POOL_DIR / "long_tail_pool.json"
MAIN_POOL_PATH = POOL_DIR / "main_pool.json"

OUTPUT_JSON_PATH = BASE_DIR / "manual_review_candidates.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



def normalize_uid_list(obj: Any, key_hint: Optional[str] = None) -> Set[str]:
    """
    Supports several common formats:
    1) ["uid1", "uid2", ...]
    2) {"uids": [...]}
    3) {"success_uids": [...]}
    4) [{"uid": "..."} , ...]
    """
    if isinstance(obj, list):
        if not obj:
            return set()
        if isinstance(obj[0], str):
            return set(obj)
        if isinstance(obj[0], dict):
            return {str(x["uid"]) for x in obj if "uid" in x}
        raise ValueError(f"Unsupported list format for uid list: {type(obj[0])}")

    if isinstance(obj, dict):
        candidate_keys = []
        if key_hint is not None:
            candidate_keys.append(key_hint)
        candidate_keys.extend(["uids", "success_uids", "uid_list"])

        for k in candidate_keys:
            if k in obj:
                value = obj[k]
                if isinstance(value, list):
                    if not value:
                        return set()
                    if isinstance(value[0], str):
                        return set(value)
                    if isinstance(value[0], dict):
                        return {str(x["uid"]) for x in value if "uid" in x}

        # If dict itself looks like uid -> something mapping, skip to avoid false positives
        raise ValueError(f"Could not find uid list in dict keys: {list(obj.keys())}")

    raise ValueError(f"Unsupported uid container type: {type(obj)}")


def build_uid_to_record(records: List[Dict[str, Any]], required_uid_key: str = "uid") -> Dict[str, Dict[str, Any]]:
    uid_to_record: Dict[str, Dict[str, Any]] = {}
    duplicate_uids = set()

    for rec in records:
        if required_uid_key not in rec:
            continue
        uid = str(rec[required_uid_key])
        if uid in uid_to_record:
            duplicate_uids.add(uid)
        uid_to_record[uid] = rec

    if duplicate_uids:
        print(f"[WARN] Duplicate uids found in records: {len(duplicate_uids)}")
        preview = list(sorted(duplicate_uids))[:10]
        print(f"       Examples: {preview}")

    return uid_to_record


def main() -> None:
    # 1) load success_uids
    obj_normalized_stats = load_json(OBJ_NORMALIZED_STATS_PATH)
    success_uids = normalize_uid_list(obj_normalized_stats, key_hint="success_uids")
    print(f"[INFO] success_uids: {len(success_uids)}")

    # 2) load pools
    extreme_uids = normalize_uid_list(load_json(EXTREME_POOL_PATH))
    long_tail_uids = normalize_uid_list(load_json(LONG_TAIL_POOL_PATH))
    main_uids = normalize_uid_list(load_json(MAIN_POOL_PATH))

    print(f"[INFO] extreme_pool: {len(extreme_uids)}")
    print(f"[INFO] long_tail_pool: {len(long_tail_uids)}")
    print(f"[INFO] main_pool: {len(main_uids)}")

    overlap_main_tail = main_uids & long_tail_uids
    if overlap_main_tail:
        print(f"[WARN] overlap between main_pool and long_tail_pool: {len(overlap_main_tail)}")
        preview = list(sorted(overlap_main_tail))[:20]
        print(f"       Examples: {preview}")

    # 3) load AB / C metadata
    ab_data = load_json(AB_METADATA_PATH)
    c_data = load_json(C_METADATA_PATH)

    if not isinstance(ab_data, list):
        raise ValueError(f"AB metadata must be a list of dicts: {AB_METADATA_PATH}")
    if not isinstance(c_data, list):
        raise ValueError(f"C metadata must be a list of dicts: {C_METADATA_PATH}")

    ab_by_uid = build_uid_to_record(ab_data)
    c_by_uid = build_uid_to_record(c_data)

    print(f"[INFO] AB records: {len(ab_by_uid)}")
    print(f"[INFO] C records: {len(c_by_uid)}")

    # 4) candidate uids = (main ∪ long_tail) - extreme, then intersect success_uids
    candidate_uids_all = (main_uids | long_tail_uids) - extreme_uids
    candidate_uids = candidate_uids_all & success_uids

    dropped_not_success = candidate_uids_all - success_uids
    print(f"[INFO] candidates before success filter: {len(candidate_uids_all)}")
    print(f"[INFO] candidates after success filter: {len(candidate_uids)}")
    print(f"[INFO] dropped because not in success_uids: {len(dropped_not_success)}")

    # 5) merge
    output_rows: List[Dict[str, Any]] = []
    missing_ab = []
    missing_c = []
    missing_fields = []

    for uid in sorted(candidate_uids):
        ab = ab_by_uid.get(uid)
        c = c_by_uid.get(uid)

        if ab is None:
            missing_ab.append(uid)
            continue
        if c is None:
            missing_c.append(uid)
            continue

        if uid in long_tail_uids and uid in main_uids:
            pool_type = "both"
        elif uid in long_tail_uids:
            pool_type = "long_tail_pool"
        elif uid in main_uids:
            pool_type = "main_pool"
        else:
            # Should not happen in normal flow
            pool_type = "unknown"

        row = {
            "uid": uid,
            "pool_type": pool_type,
            "shape_type": ab.get("shape_type"),
            "fill_bucket": ab.get("fill_bucket"),
            "self_occlusion_attribute": ab.get("self_occlusion_attribute"),
            "observation_saturation_view_num": ab.get("observation_saturation_view_num"),
            "selected_view_count": c.get("selected_view_count"),
            "gt_surface_voxel_count": ab.get("gt_surface_voxel_count"),
            "obj_path": str(BASE_DIR / "obj_normalized" / uid / f"{uid}.obj"),
        }

        required_fields = [
            "shape_type",
            "fill_bucket",
            "self_occlusion_attribute",
            "observation_saturation_view_num",
            "selected_view_count",
            "gt_surface_voxel_count",
        ]
        missing = [k for k in required_fields if row[k] is None]
        if missing:
            missing_fields.append({"uid": uid, "missing_fields": missing})
            continue

        output_rows.append(row)

    # 6) sort rows for easier review
    # Sort by pool_type first, then observation_saturation_view_num desc, then selected_view_count desc
    pool_order = {"long_tail_pool": 0, "main_pool": 1, "both": 2, "unknown": 3}
    output_rows.sort(
        key=lambda x: (
            pool_order.get(x["pool_type"], 99),
            -float(x["observation_saturation_view_num"]),
            -float(x["selected_view_count"]),
            x["uid"],
        )
    )

    # 7) save
    save_json(OUTPUT_JSON_PATH, output_rows)

    # 8) report
    print(f"[DONE] merged review candidates: {len(output_rows)}")
    print(f"[DONE] json saved to: {OUTPUT_JSON_PATH}")

    print(f"[CHECK] missing AB records: {len(missing_ab)}")
    if missing_ab:
        print(f"        Examples: {missing_ab[:10]}")

    print(f"[CHECK] missing C records: {len(missing_c)}")
    if missing_c:
        print(f"        Examples: {missing_c[:10]}")

    print(f"[CHECK] missing required fields: {len(missing_fields)}")
    if missing_fields:
        print(f"        Examples: {missing_fields[:5]}")

    count_main = sum(1 for x in output_rows if x["pool_type"] == "main_pool")
    count_tail = sum(1 for x in output_rows if x["pool_type"] == "long_tail_pool")
    count_both = sum(1 for x in output_rows if x["pool_type"] == "both")
    print(f"[STATS] main_pool rows:      {count_main}")
    print(f"[STATS] long_tail_pool rows: {count_tail}")
    print(f"[STATS] both rows:           {count_both}")


if __name__ == "__main__":
    main()
