import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


VALID_SHAPE_TYPES = [
    "bladed",
    "compact",
    "elongated",
    "flat_or_sheet_like",
]

VALID_FILL_BUCKETS = [
    "low",
    "mid",
    "high",
]

PLOT_TITLE_FONTSIZE = 22
PLOT_LABEL_FONTSIZE = 22
PLOT_TICK_FONTSIZE = 18


# =========================================================
# IO
# =========================================================

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_review_dict(data: Any) -> Dict[str, Dict[str, Any]]:
    """
    Supports two input formats:
    1) {"uid1": {...}, "uid2": {...}}
    2) [{"uid": "...", ...}, ...]
    """
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

    raise ValueError(f"Unsupported review json format: {type(data)}")


# =========================================================
# Merge review with candidate pool
# =========================================================

def build_final_clean_pool(
    candidates: List[Dict[str, Any]],
    reviews_by_uid: Dict[str, Dict[str, Any]],
    keep_label: str = "yes",
) -> List[Dict[str, Any]]:
    final_pool = []

    for item in candidates:
        uid = str(item["uid"])
        review = reviews_by_uid.get(uid)
        if review is None:
            continue

        if review.get("manual_label") != keep_label:
            continue

        merged = dict(item)
        merged["reviewer"] = review.get("reviewer")
        merged["manual_label"] = review.get("manual_label")
        merged["manual_reject_reason"] = review.get("manual_reject_reason")
        merged["manual_note"] = review.get("manual_note")
        merged["review_timestamp"] = review.get("timestamp")
        final_pool.append(merged)

    return final_pool


# =========================================================
# Stats helpers
# =========================================================

def safe_int(x):
    if x is None:
        return None
    return int(x)


def safe_float(x):
    if x is None:
        return None
    return float(x)


def counter_to_sorted_dict(counter: Counter) -> Dict[str, int]:
    return {str(k): v for k, v in sorted(counter.items(), key=lambda kv: kv[0])}


def compute_distribution_stats(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    A = self_occlusion_attribute
    B = observation_saturation_view_num
    C = selected_view_count
    """
    c_global = Counter()
    b_global = Counter()
    a_bin_global = Counter()

    bucket_c = defaultdict(Counter)
    bucket_b = defaultdict(Counter)
    bucket_a_bin = defaultdict(Counter)

    shape_c = defaultdict(Counter)
    shape_b = defaultdict(Counter)
    shape_a_bin = defaultdict(Counter)

    fill_c = defaultdict(Counter)
    fill_b = defaultdict(Counter)
    fill_a_bin = defaultdict(Counter)

    bucket_sizes = Counter()
    shape_sizes = Counter()
    fill_sizes = Counter()

    risk_flag_counter = Counter()

    a_values = []
    b_values = []
    c_values = []
    bc_ratio_values = []
    gt_voxel_values = []

    for item in data:
        shape_type = item.get("shape_type")
        fill_bucket = item.get("fill_bucket")

        A = safe_float(item.get("self_occlusion_attribute"))
        B = safe_int(item.get("observation_saturation_view_num"))
        C = safe_int(item.get("selected_view_count"))
        bc_ratio = safe_float(item.get("bc_ratio"))
        gt_vox = safe_int(item.get("gt_surface_voxel_count"))

        if shape_type not in VALID_SHAPE_TYPES:
            continue
        if fill_bucket not in VALID_FILL_BUCKETS:
            continue

        key = (shape_type, fill_bucket)

        if A is not None:
            a_values.append(A)
            a_bin = round((A // 0.05) * 0.05, 2)
            a_bin_global[a_bin] += 1
            shape_a_bin[shape_type][a_bin] += 1
            fill_a_bin[fill_bucket][a_bin] += 1
            bucket_a_bin[key][a_bin] += 1

        if B is not None:
            b_values.append(B)
            b_global[B] += 1
            shape_b[shape_type][B] += 1
            fill_b[fill_bucket][B] += 1
            bucket_b[key][B] += 1

        if C is not None:
            c_values.append(C)
            c_global[C] += 1
            shape_c[shape_type][C] += 1
            fill_c[fill_bucket][C] += 1
            bucket_c[key][C] += 1

        shape_sizes[shape_type] += 1
        fill_sizes[fill_bucket] += 1
        bucket_sizes[key] += 1

        if bc_ratio is not None:
            bc_ratio_values.append(bc_ratio)
        if gt_vox is not None:
            gt_voxel_values.append(gt_vox)

        for flag in item.get("risk_flags", []):
            risk_flag_counter[flag] += 1

    stats = {
        "num_objects": len(data),
        "global_A_distribution_binned": counter_to_sorted_dict(a_bin_global),
        "global_B_distribution": counter_to_sorted_dict(b_global),
        "global_C_distribution": counter_to_sorted_dict(c_global),
        "risk_flag_counts": dict(sorted(risk_flag_counter.items())),
        "summary": {
            "A_mean": (sum(a_values) / len(a_values)) if a_values else None,
            "B_mean": (sum(b_values) / len(b_values)) if b_values else None,
            "C_mean": (sum(c_values) / len(c_values)) if c_values else None,
            "bc_ratio_mean": (sum(bc_ratio_values) / len(bc_ratio_values)) if bc_ratio_values else None,
            "gt_surface_voxel_count_mean": (sum(gt_voxel_values) / len(gt_voxel_values)) if gt_voxel_values else None,
        },
        "shape_stats": {},
        "fill_stats": {},
        "bucket_stats": {},
    }

    for shape_type in VALID_SHAPE_TYPES:
        stats["shape_stats"][shape_type] = {
            "shape_type": shape_type,
            "num_objects": shape_sizes.get(shape_type, 0),
            "A_distribution_binned": counter_to_sorted_dict(shape_a_bin.get(shape_type, Counter())),
            "B_distribution": counter_to_sorted_dict(shape_b.get(shape_type, Counter())),
            "C_distribution": counter_to_sorted_dict(shape_c.get(shape_type, Counter())),
        }

    for fill_bucket in VALID_FILL_BUCKETS:
        stats["fill_stats"][fill_bucket] = {
            "fill_bucket": fill_bucket,
            "num_objects": fill_sizes.get(fill_bucket, 0),
            "A_distribution_binned": counter_to_sorted_dict(fill_a_bin.get(fill_bucket, Counter())),
            "B_distribution": counter_to_sorted_dict(fill_b.get(fill_bucket, Counter())),
            "C_distribution": counter_to_sorted_dict(fill_c.get(fill_bucket, Counter())),
        }

    for shape_type in VALID_SHAPE_TYPES:
        for fill_bucket in VALID_FILL_BUCKETS:
            key = (shape_type, fill_bucket)
            stats["bucket_stats"][f"{shape_type}__{fill_bucket}"] = {
                "shape_type": shape_type,
                "fill_bucket": fill_bucket,
                "num_objects": bucket_sizes.get(key, 0),
                "A_distribution_binned": counter_to_sorted_dict(bucket_a_bin.get(key, Counter())),
                "B_distribution": counter_to_sorted_dict(bucket_b.get(key, Counter())),
                "C_distribution": counter_to_sorted_dict(bucket_c.get(key, Counter())),
            }

    return stats


def compute_cross_analysis(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pairwise binned statistics for A/B/C relationships.
    """
    by_A_bin = defaultdict(list)   # A bin -> C list / B list
    by_B_bin = defaultdict(list)   # B bin -> A list / C list
    by_C_bin = defaultdict(list)   # C bin -> A list / B list

    for item in data:
        A = safe_float(item.get("self_occlusion_attribute"))
        B = safe_int(item.get("observation_saturation_view_num"))
        C = safe_int(item.get("selected_view_count"))

        if A is not None and B is not None:
            a_bin = round((A // 0.05) * 0.05, 2)
            by_A_bin[("B_given_A", a_bin)].append(B)
            b_bin = (B // 16) * 16
            by_B_bin[("A_given_B", b_bin)].append(A)

        if A is not None and C is not None:
            a_bin = round((A // 0.05) * 0.05, 2)
            by_A_bin[("C_given_A", a_bin)].append(C)
            c_bin = (C // 8) * 8
            by_C_bin[("A_given_C", c_bin)].append(A)

        if B is not None and C is not None:
            b_bin = (B // 16) * 16
            by_B_bin[("C_given_B", b_bin)].append(C)
            c_bin = (C // 8) * 8
            by_C_bin[("B_given_C", c_bin)].append(B)

    def summarize_group_dict(group_dict):
        out = {}
        for (name, k), vals in sorted(group_dict.items(), key=lambda x: (x[0][0], x[0][1])):
            if name not in out:
                out[name] = {}
            out[name][str(k)] = {
                "count": len(vals),
                "mean": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
            }
        return out

    return {
        "A_B_C_pairwise_binned_stats": {
            **summarize_group_dict(by_A_bin),
            **summarize_group_dict(by_B_bin),
            **summarize_group_dict(by_C_bin),
        }
    }


def run_sanity_checks(data: List[Dict[str, Any]], stats: Dict[str, Any], tag: str) -> None:
    n = len(data)
    assert stats["num_objects"] == n, f"[{tag}] num_objects mismatch"

    shape_total = sum(v["num_objects"] for v in stats["shape_stats"].values())
    fill_total = sum(v["num_objects"] for v in stats["fill_stats"].values())
    bucket_total = sum(v["num_objects"] for v in stats["bucket_stats"].values())

    assert shape_total == n, f"[{tag}] shape_total={shape_total}, n={n}"
    assert fill_total == n, f"[{tag}] fill_total={fill_total}, n={n}"
    assert bucket_total == n, f"[{tag}] bucket_total={bucket_total}, n={n}"

    for name, info in stats["shape_stats"].items():
        s = sum(info["C_distribution"].values())
        assert s == info["num_objects"], f"[{tag}] shape {name} C sum mismatch"

    for name, info in stats["fill_stats"].items():
        s = sum(info["C_distribution"].values())
        assert s == info["num_objects"], f"[{tag}] fill {name} C sum mismatch"

    for name, info in stats["bucket_stats"].items():
        s = sum(info["C_distribution"].values())
        assert s == info["num_objects"], f"[{tag}] bucket {name} C sum mismatch"


# =========================================================
# Split
# =========================================================

def split_main_long_tail(data: List[Dict[str, Any]]) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
    main_pool = []
    long_tail_pool = []

    for item in data:
        pool_type = item.get("pool_type")
        if pool_type == "main_pool":
            main_pool.append(item)
        elif pool_type == "long_tail_pool":
            long_tail_pool.append(item)

    return main_pool, long_tail_pool


# =========================================================
# Plot helpers
# =========================================================

def get_sorted_xs_from_counter_dict(distribution: Dict[str, int], cast_type=int) -> List:
    return sorted(cast_type(k) for k in distribution.keys())


def make_full_y_series(distribution: Dict[str, int], global_xs: List, cast_type=int) -> List[int]:
    counter = {x: 0 for x in global_xs}
    for k, v in distribution.items():
        counter[cast_type(k)] = v
    return [counter[x] for x in global_xs]


def choose_xticks(global_xs, max_ticks=16):
    if len(global_xs) <= max_ticks:
        return global_xs
    step = max(1, len(global_xs) // max_ticks)
    ticks = global_xs[::step]
    if ticks[-1] != global_xs[-1]:
        ticks.append(global_xs[-1])
    return ticks


def plot_distribution(global_xs, distribution, title, xlabel, save_path, cast_type=int):
    ys = make_full_y_series(distribution, global_xs, cast_type=cast_type)
    xticks = choose_xticks(global_xs, max_ticks=16)

    plt.figure(figsize=(10, 6))
    plt.bar(global_xs, ys, width=0.8 if len(global_xs) > 1 else 0.5)
    plt.xlim(min(global_xs) - 1, max(global_xs) + 1 if len(global_xs) > 0 else 1)
    plt.xticks(xticks, fontsize=PLOT_TICK_FONTSIZE)
    plt.yticks(fontsize=PLOT_TICK_FONTSIZE)
    plt.xlabel(xlabel, fontsize=PLOT_LABEL_FONTSIZE)
    plt.ylabel("count", fontsize=PLOT_LABEL_FONTSIZE)
    plt.title(title, fontsize=PLOT_TITLE_FONTSIZE)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_scatter(xs, ys, title, xlabel, ylabel, save_path, logx=False, logy=False):
    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, s=10, alpha=0.45)

    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")

    plt.xticks(fontsize=PLOT_TICK_FONTSIZE)
    plt.yticks(fontsize=PLOT_TICK_FONTSIZE)
    plt.xlabel(xlabel, fontsize=PLOT_LABEL_FONTSIZE)
    plt.ylabel(ylabel, fontsize=PLOT_LABEL_FONTSIZE)
    plt.title(title, fontsize=PLOT_TITLE_FONTSIZE)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def save_global_plots(stats: Dict[str, Any], output_dir: Path, prefix: str) -> None:
    global_dir = output_dir / prefix / "global"
    global_dir.mkdir(parents=True, exist_ok=True)

    # A
    dist_A = stats["global_A_distribution_binned"]
    if dist_A:
        xs_A = get_sorted_xs_from_counter_dict(dist_A, cast_type=float)
        plot_distribution(
            xs_A, dist_A,
            title=f"{prefix} | A(self_occlusion_attribute) | n={stats['num_objects']}",
            xlabel="A (binned self_occlusion_attribute)",
            save_path=global_dir / "A_distribution.png",
            cast_type=float,
        )

    # B
    dist_B = stats["global_B_distribution"]
    if dist_B:
        xs_B = get_sorted_xs_from_counter_dict(dist_B, cast_type=int)
        plot_distribution(
            xs_B, dist_B,
            title=f"{prefix} | B(observation_saturation_view_num) | n={stats['num_objects']}",
            xlabel="B",
            save_path=global_dir / "B_distribution.png",
            cast_type=int,
        )

    # C
    dist_C = stats["global_C_distribution"]
    if dist_C:
        xs_C = get_sorted_xs_from_counter_dict(dist_C, cast_type=int)
        plot_distribution(
            xs_C, dist_C,
            title=f"{prefix} | C(selected_view_count) | n={stats['num_objects']}",
            xlabel="C",
            save_path=global_dir / "C_distribution.png",
            cast_type=int,
        )


def save_shape_plots(stats: Dict[str, Any], output_dir: Path, prefix: str) -> None:
    shape_dir = output_dir / prefix / "4_shape_types"
    shape_dir.mkdir(parents=True, exist_ok=True)

    for shape_type in VALID_SHAPE_TYPES:
        info = stats["shape_stats"][shape_type]

        dist_B = info["B_distribution"]
        if dist_B:
            xs_B = get_sorted_xs_from_counter_dict(dist_B, cast_type=int)
            plot_distribution(
                xs_B, dist_B,
                title=f"{prefix} | {shape_type} | B | n={info['num_objects']}",
                xlabel="B",
                save_path=shape_dir / f"{shape_type}_B.png",
                cast_type=int,
            )

        dist_C = info["C_distribution"]
        if dist_C:
            xs_C = get_sorted_xs_from_counter_dict(dist_C, cast_type=int)
            plot_distribution(
                xs_C, dist_C,
                title=f"{prefix} | {shape_type} | C | n={info['num_objects']}",
                xlabel="C",
                save_path=shape_dir / f"{shape_type}_C.png",
                cast_type=int,
            )


def save_fill_plots(stats: Dict[str, Any], output_dir: Path, prefix: str) -> None:
    fill_dir = output_dir / prefix / "3_fill_buckets"
    fill_dir.mkdir(parents=True, exist_ok=True)

    for fill_bucket in VALID_FILL_BUCKETS:
        info = stats["fill_stats"][fill_bucket]

        dist_B = info["B_distribution"]
        if dist_B:
            xs_B = get_sorted_xs_from_counter_dict(dist_B, cast_type=int)
            plot_distribution(
                xs_B, dist_B,
                title=f"{prefix} | {fill_bucket} | B | n={info['num_objects']}",
                xlabel="B",
                save_path=fill_dir / f"{fill_bucket}_B.png",
                cast_type=int,
            )

        dist_C = info["C_distribution"]
        if dist_C:
            xs_C = get_sorted_xs_from_counter_dict(dist_C, cast_type=int)
            plot_distribution(
                xs_C, dist_C,
                title=f"{prefix} | {fill_bucket} | C | n={info['num_objects']}",
                xlabel="C",
                save_path=fill_dir / f"{fill_bucket}_C.png",
                cast_type=int,
            )


def save_bucket_plots(stats: Dict[str, Any], output_dir: Path, prefix: str) -> None:
    bucket_dir = output_dir / prefix / "12_buckets"
    bucket_dir.mkdir(parents=True, exist_ok=True)

    for shape_type in VALID_SHAPE_TYPES:
        for fill_bucket in VALID_FILL_BUCKETS:
            key = f"{shape_type}__{fill_bucket}"
            info = stats["bucket_stats"][key]

            dist_C = info["C_distribution"]
            if dist_C:
                xs_C = get_sorted_xs_from_counter_dict(dist_C, cast_type=int)
                plot_distribution(
                    xs_C, dist_C,
                    title=f"{prefix} | {shape_type} | {fill_bucket} | C | n={info['num_objects']}",
                    xlabel="C",
                    save_path=bucket_dir / f"{key}_C.png",
                    cast_type=int,
                )


def save_cross_plots(data: List[Dict[str, Any]], output_dir: Path, prefix: str) -> None:
    cross_dir = output_dir / prefix / "cross_analysis"
    cross_dir.mkdir(parents=True, exist_ok=True)

    A_vals, B_vals, C_vals = [], [], []

    for item in data:
        A = safe_float(item.get("self_occlusion_attribute"))
        B = safe_int(item.get("observation_saturation_view_num"))
        C = safe_int(item.get("selected_view_count"))

        if A is not None and B is not None:
            A_vals.append(("A_vs_B", A, B))
        if A is not None and C is not None:
            B_vals.append(("A_vs_C", A, C))
        if B is not None and C is not None:
            C_vals.append(("B_vs_C", B, C))

    if A_vals:
        xs = [x for _, x, _ in A_vals]
        ys = [y for _, _, y in A_vals]
        plot_scatter(
            xs, ys,
            title=f"{prefix} | A vs B",
            xlabel="A (self_occlusion_attribute)",
            ylabel="B (observation_saturation_view_num)",
            save_path=cross_dir / "A_vs_B.png",
        )

    if B_vals:
        xs = [x for _, x, _ in B_vals]
        ys = [y for _, _, y in B_vals]
        plot_scatter(
            xs, ys,
            title=f"{prefix} | A vs C",
            xlabel="A (self_occlusion_attribute)",
            ylabel="C (selected_view_count)",
            save_path=cross_dir / "A_vs_C.png",
        )

    if C_vals:
        xs = [x for _, x, _ in C_vals]
        ys = [y for _, _, y in C_vals]
        plot_scatter(
            xs, ys,
            title=f"{prefix} | B vs C",
            xlabel="B (observation_saturation_view_num)",
            ylabel="C (selected_view_count)",
            save_path=cross_dir / "B_vs_C.png",
        )

    # Extra plot: B vs gt voxel count
    xs, ys = [], []
    for item in data:
        gt = safe_int(item.get("gt_surface_voxel_count"))
        B = safe_int(item.get("observation_saturation_view_num"))
        if gt is not None and B is not None:
            xs.append(gt)
            ys.append(B)
    if xs:
        plot_scatter(
            xs, ys,
            title=f"{prefix} | gt_surface_voxel_count vs B",
            xlabel="gt_surface_voxel_count",
            ylabel="B",
            save_path=cross_dir / "gt_voxel_vs_B.png",
            logx=True,
        )

    # Extra plot: gt voxel count vs C
    xs, ys = [], []
    for item in data:
        gt = safe_int(item.get("gt_surface_voxel_count"))
        C = safe_int(item.get("selected_view_count"))
        if gt is not None and C is not None:
            xs.append(gt)
            ys.append(C)
    if xs:
        plot_scatter(
            xs, ys,
            title=f"{prefix} | gt_surface_voxel_count vs C",
            xlabel="gt_surface_voxel_count",
            ylabel="C",
            save_path=cross_dir / "gt_voxel_vs_C.png",
            logx=True,
        )


def save_all_plots(data: List[Dict[str, Any]], stats: Dict[str, Any], output_dir: Path, prefix: str) -> None:
    save_global_plots(stats, output_dir, prefix)
    save_shape_plots(stats, output_dir, prefix)
    save_fill_plots(stats, output_dir, prefix)
    save_bucket_plots(stats, output_dir, prefix)
    save_cross_plots(data, output_dir, prefix)


def generate_outputs_from_final_pool(final_pool: List[Dict[str, Any]], output_dir: Path) -> None:
    plot_output_dir = output_dir / "plots"

    main_pool, long_tail_pool = split_main_long_tail(final_pool)

    overall_stats = compute_distribution_stats(final_pool)
    main_stats = compute_distribution_stats(main_pool)
    long_tail_stats = compute_distribution_stats(long_tail_pool)

    overall_cross = compute_cross_analysis(final_pool)
    main_cross = compute_cross_analysis(main_pool)
    long_tail_cross = compute_cross_analysis(long_tail_pool)

    run_sanity_checks(final_pool, overall_stats, "overall")
    run_sanity_checks(main_pool, main_stats, "main")
    run_sanity_checks(long_tail_pool, long_tail_stats, "long_tail")

    save_json(final_pool, output_dir / "final_clean_pool.json")
    save_json(main_pool, output_dir / "final_clean_main_pool.json")
    save_json(long_tail_pool, output_dir / "final_clean_long_tail_pool.json")

    save_json(overall_stats, output_dir / "final_clean_pool_stats_overall.json")
    save_json(main_stats, output_dir / "final_clean_pool_stats_main.json")
    save_json(long_tail_stats, output_dir / "final_clean_pool_stats_long_tail.json")

    save_json(overall_cross, output_dir / "final_clean_pool_cross_overall.json")
    save_json(main_cross, output_dir / "final_clean_pool_cross_main.json")
    save_json(long_tail_cross, output_dir / "final_clean_pool_cross_long_tail.json")

    save_all_plots(final_pool, overall_stats, plot_output_dir, "overall")
    save_all_plots(main_pool, main_stats, plot_output_dir, "main")
    save_all_plots(long_tail_pool, long_tail_stats, plot_output_dir, "long_tail")


# =========================================================
# Summary print
# =========================================================

def print_summary(
    total_candidates: int,
    reviewed_count: int,
    final_count: int,
    main_count: int,
    long_tail_count: int,
    output_dir: Path,
) -> None:
    print("=" * 80)
    print(f"Total candidate objects:      {total_candidates}")
    print(f"Reviewed objects found:       {reviewed_count}")
    print(f"Final clean pool (yes only):  {final_count}")
    print(f"  Main pool:                  {main_count}")
    print(f"  Long-tail pool:             {long_tail_count}")
    print()
    print(f"Saved to directory: {output_dir}")
    print("Generated files:")
    print("  - final_clean_pool.json")
    print("  - final_clean_main_pool.json")
    print("  - final_clean_long_tail_pool.json")
    print("  - final_clean_pool_stats_overall.json")
    print("  - final_clean_pool_stats_main.json")
    print("  - final_clean_pool_stats_long_tail.json")
    print("  - final_clean_pool_cross_overall.json")
    print("  - final_clean_pool_cross_main.json")
    print("  - final_clean_pool_cross_long_tail.json")
    print("  - plots/overall/*")
    print("  - plots/main/*")
    print("  - plots/long_tail/*")
    print("=" * 80)


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Build final clean pool from reviewer_999 results.")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Load existing final_clean_pool.json from output-dir and regenerate stats/plots only.",
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default="geometry_sampled/manual_review_candidates_with_risk.json",
        help="Path to manual_review_candidates_with_risk.json",
    )
    parser.add_argument(
        "--reviews",
        type=str,
        default="geometry_sampled/review_results_reviewer_999.json",
        help="Path to review_results_reviewer_999.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="geometry_sampled/final_clean_pool",
        help="Output directory",
    )
    parser.add_argument(
        "--keep-label",
        type=str,
        default="yes",
        choices=["yes"],
        help="Currently only keep yes as final clean pool.",
    )
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    reviews_path = Path(args.reviews)
    output_dir = Path(args.output_dir)

    if args.plot_only:
        final_pool_path = output_dir / "final_clean_pool.json"
        final_pool = load_json(final_pool_path)
        if not isinstance(final_pool, list):
            raise ValueError(f"Expected list in final clean pool json: {final_pool_path}")

        generate_outputs_from_final_pool(final_pool, output_dir)

        main_pool, long_tail_pool = split_main_long_tail(final_pool)
        print_summary(
            total_candidates=len(final_pool),
            reviewed_count=len(final_pool),
            final_count=len(final_pool),
            main_count=len(main_pool),
            long_tail_count=len(long_tail_pool),
            output_dir=output_dir,
        )
        return

    candidates = load_json(candidates_path)
    if not isinstance(candidates, list):
        raise ValueError(f"Expected list in candidates json: {candidates_path}")

    reviews_raw = load_json(reviews_path)
    reviews_by_uid = normalize_review_dict(reviews_raw)

    final_pool = build_final_clean_pool(
        candidates=candidates,
        reviews_by_uid=reviews_by_uid,
        keep_label=args.keep_label,
    )
    generate_outputs_from_final_pool(final_pool, output_dir)

    main_pool, long_tail_pool = split_main_long_tail(final_pool)

    print_summary(
        total_candidates=len(candidates),
        reviewed_count=len(reviews_by_uid),
        final_count=len(final_pool),
        main_count=len(main_pool),
        long_tail_count=len(long_tail_pool),
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()

"""
python build_final_clean_pool.py \
  --candidates geometry_sampled/manual_review_candidates_with_risk.json \
  --reviews geometry_sampled/review_results_reviewer_999.json \
  --output-dir geometry_sampled/final_clean_pool

python build_final_clean_pool.py \
--plot-only \
--output-dir geometry_sampled/final_clean_pool
"""
