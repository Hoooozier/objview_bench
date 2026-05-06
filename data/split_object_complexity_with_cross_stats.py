import os
import json
from collections import Counter, defaultdict

import matplotlib.pyplot as plt


# =========================
# Config
# =========================

INPUT_JSON = "geometry_sampled/object_complexity_merged.json"
OUTPUT_DIR = "geometry_sampled/object_complexity_split_by_saturation_partition"

MAIN_THRESHOLD = 128
EXTREME_THRESHOLD = 300

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

FIG_DPI = 220


# =========================
# IO
# =========================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================
# Split
# =========================

def split_items(items, main_threshold, extreme_threshold):
    main_items = []
    long_tail_items = []
    extreme_items = []

    for item in items:
        b = int(item["observation_saturation_view_num"])

        if b <= main_threshold:
            main_items.append(item)
        elif b <= extreme_threshold:
            long_tail_items.append(item)
        else:
            extreme_items.append(item)

    return main_items, long_tail_items, extreme_items


# =========================
# Basic summaries
# =========================

def summarize_items(items):
    if not items:
        return {
            "num_objects": 0,
            "B_min": None,
            "B_max": None,
            "B_mean": None,
            "A_mean": None,
            "gt_surface_voxel_count_mean": None,
        }

    b_vals = [int(x["observation_saturation_view_num"]) for x in items]
    a_vals = [float(x["self_occlusion_attribute"]) for x in items]
    gt_vals = [int(x["gt_surface_voxel_count"]) for x in items if "gt_surface_voxel_count" in x]

    return {
        "num_objects": len(items),
        "B_min": min(b_vals),
        "B_max": max(b_vals),
        "B_mean": sum(b_vals) / len(b_vals),
        "A_mean": sum(a_vals) / len(a_vals),
        "gt_surface_voxel_count_mean": (sum(gt_vals) / len(gt_vals)) if gt_vals else None,
    }


def histogram_of_b(items):
    counter = Counter()
    for item in items:
        b = int(item["observation_saturation_view_num"])
        counter[b] += 1
    return dict(sorted(counter.items()))


def collect_examples(items, max_examples=50, sort_desc=False):
    if sort_desc:
        items = sorted(items, key=lambda x: (-int(x["observation_saturation_view_num"]), x["uid"]))
    else:
        items = sorted(items, key=lambda x: (int(x["observation_saturation_view_num"]), x["uid"]))

    out = []
    for x in items[:max_examples]:
        out.append({
            "uid": x["uid"],
            "observation_saturation_view_num": int(x["observation_saturation_view_num"]),
            "self_occlusion_attribute": float(x["self_occlusion_attribute"]),
            "gt_surface_voxel_count": int(x["gt_surface_voxel_count"]) if "gt_surface_voxel_count" in x else None,
            "shape_type": x.get("shape_type"),
            "fill_bucket": x.get("fill_bucket"),
        })
    return out


# =========================
# Cross stats
# =========================

def compute_cross_stats(items):
    shape_counter = Counter()
    fill_counter = Counter()
    bucket_counter = Counter()

    for item in items:
        shape_type = item.get("shape_type")
        fill_bucket = item.get("fill_bucket")

        if shape_type in VALID_SHAPE_TYPES:
            shape_counter[shape_type] += 1
        else:
            shape_counter["unknown"] += 1

        if fill_bucket in VALID_FILL_BUCKETS:
            fill_counter[fill_bucket] += 1
        else:
            fill_counter["unknown"] += 1

        shape_key = shape_type if shape_type in VALID_SHAPE_TYPES else "unknown"
        fill_key = fill_bucket if fill_bucket in VALID_FILL_BUCKETS else "unknown"
        bucket_counter[(shape_key, fill_key)] += 1

    shape_stats = {}
    for k in VALID_SHAPE_TYPES + ["unknown"]:
        shape_stats[k] = shape_counter.get(k, 0)

    fill_stats = {}
    for k in VALID_FILL_BUCKETS + ["unknown"]:
        fill_stats[k] = fill_counter.get(k, 0)

    bucket_stats = {}
    shape_keys = VALID_SHAPE_TYPES + ["unknown"]
    fill_keys = VALID_FILL_BUCKETS + ["unknown"]
    for s in shape_keys:
        for f in fill_keys:
            bucket_stats[f"{s}__{f}"] = bucket_counter.get((s, f), 0)

    return {
        "shape_type_distribution": shape_stats,
        "fill_bucket_distribution": fill_stats,
        "shape_fill_bucket_distribution": bucket_stats,
    }


# =========================
# Plotting
# =========================

def plot_histogram(items, title, save_path):
    hist = histogram_of_b(items)
    if not hist:
        return

    xs = list(hist.keys())
    ys = list(hist.values())

    plt.figure(figsize=(10, 5))
    plt.bar(xs, ys, width=1.0)
    plt.xlabel("observation_saturation_view_num")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI)
    plt.close()


def plot_split_overview(main_items, long_tail_items, extreme_items, save_path):
    labels = ["main (<=128)", "long_tail (129-300)", "extreme (>300)"]
    counts = [len(main_items), len(long_tail_items), len(extreme_items)]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, counts)
    plt.ylabel("count")
    plt.title("Object complexity split by saturation partition")
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI)
    plt.close()


def plot_scatter(items, x_key, y_key, title, xlabel, ylabel, save_path, logx=False):
    xs = []
    ys = []

    for item in items:
        if x_key not in item or y_key not in item:
            continue
        x = item[x_key]
        y = item[y_key]
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, s=18, alpha=0.65)
    if logx:
        plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI)
    plt.close()


def plot_categorical_bar(counter_dict, title, xlabel, save_path):
    labels = list(counter_dict.keys())
    values = list(counter_dict.values())

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI)
    plt.close()


# =========================
# Main
# =========================

def main():
    ensure_dir(OUTPUT_DIR)

    items = load_json(INPUT_JSON)

    main_items, long_tail_items, extreme_items = split_items(
        items,
        MAIN_THRESHOLD,
        EXTREME_THRESHOLD,
    )

    split_summary = {
        "config": {
            "input_json": INPUT_JSON,
            "main_threshold": MAIN_THRESHOLD,
            "extreme_threshold": EXTREME_THRESHOLD,
            "split_rule": {
                "main": f"observation_saturation_view_num <= {MAIN_THRESHOLD}",
                "long_tail": f"{MAIN_THRESHOLD} < observation_saturation_view_num <= {EXTREME_THRESHOLD}",
                "extreme": f"observation_saturation_view_num > {EXTREME_THRESHOLD}",
            },
        },
        "summary": {
            "all": summarize_items(items),
            "main": summarize_items(main_items),
            "long_tail": summarize_items(long_tail_items),
            "extreme": summarize_items(extreme_items),
        },
        "cross_stats": {
            "all": compute_cross_stats(items),
            "main": compute_cross_stats(main_items),
            "long_tail": compute_cross_stats(long_tail_items),
            "extreme": compute_cross_stats(extreme_items),
        },
        "examples": {
            "main_lowest_B_examples": collect_examples(main_items, max_examples=30, sort_desc=False),
            "main_highest_B_examples": collect_examples(main_items, max_examples=30, sort_desc=True),
            "long_tail_examples": collect_examples(long_tail_items, max_examples=50, sort_desc=False),
            "extreme_examples": collect_examples(extreme_items, max_examples=50, sort_desc=False),
        }
    }

    # Save split jsons
    save_json(main_items, os.path.join(OUTPUT_DIR, "main_pool.json"))
    save_json(long_tail_items, os.path.join(OUTPUT_DIR, "long_tail_pool.json"))
    save_json(extreme_items, os.path.join(OUTPUT_DIR, "extreme_pool.json"))
    save_json(split_summary, os.path.join(OUTPUT_DIR, "split_summary_with_cross_stats.json"))

    # Histograms
    plot_histogram(
        main_items,
        "Main pool (observation_saturation_view_num <= 128)",
        os.path.join(OUTPUT_DIR, "main_pool_histogram.png"),
    )
    plot_histogram(
        long_tail_items,
        "Long-tail pool (129 <= observation_saturation_view_num <= 300)",
        os.path.join(OUTPUT_DIR, "long_tail_pool_histogram.png"),
    )
    plot_histogram(
        extreme_items,
        "Extreme pool (observation_saturation_view_num > 300)",
        os.path.join(OUTPUT_DIR, "extreme_pool_histogram.png"),
    )
    plot_split_overview(
        main_items,
        long_tail_items,
        extreme_items,
        os.path.join(OUTPUT_DIR, "split_overview.png"),
    )

    # Cross-stat plots: shape_type
    for split_name, split_items_list in [
        ("all", items),
        ("main", main_items),
        ("long_tail", long_tail_items),
        ("extreme", extreme_items),
    ]:
        cross = compute_cross_stats(split_items_list)

        plot_categorical_bar(
            cross["shape_type_distribution"],
            f"{split_name} | shape_type distribution",
            "shape_type",
            os.path.join(OUTPUT_DIR, f"{split_name}_shape_type_distribution.png"),
        )

        plot_categorical_bar(
            cross["fill_bucket_distribution"],
            f"{split_name} | fill_bucket distribution",
            "fill_bucket",
            os.path.join(OUTPUT_DIR, f"{split_name}_fill_bucket_distribution.png"),
        )

    # Complexity-vs plots
    plot_scatter(
        items,
        x_key="observation_saturation_view_num",
        y_key="self_occlusion_attribute",
        title="all_objects | self_occlusion_attribute vs observation_saturation_view_num",
        xlabel="observation_saturation_view_num",
        ylabel="self_occlusion_attribute",
        save_path=os.path.join(OUTPUT_DIR, "all_objects_A_vs_saturation_view_num.png"),
        logx=False,
    )

    plot_scatter(
        items,
        x_key="gt_surface_voxel_count",
        y_key="observation_saturation_view_num",
        title="all_objects | gt_surface_voxel_count vs observation_saturation_view_num",
        xlabel="gt_surface_voxel_count",
        ylabel="observation_saturation_view_num",
        save_path=os.path.join(OUTPUT_DIR, "all_objects_gt_surface_voxel_count_vs_saturation_view_num.png"),
        logx=True,
    )

    print("=" * 80)
    print("Split + cross stats finished.")
    print(f"Input: {INPUT_JSON}")
    print(f"Output dir: {OUTPUT_DIR}")
    print("-" * 80)
    print(f"All objects:   {len(items)}")
    print(f"Main pool:     {len(main_items)}   (<= {MAIN_THRESHOLD})")
    print(f"Long-tail:     {len(long_tail_items)}   ({MAIN_THRESHOLD}, {EXTREME_THRESHOLD}]")
    print(f"Extreme pool:  {len(extreme_items)}   (> {EXTREME_THRESHOLD})")
    print("-" * 80)
    print("Saved:")
    print(f"  main json:      {os.path.join(OUTPUT_DIR, 'main_pool.json')}")
    print(f"  long-tail json: {os.path.join(OUTPUT_DIR, 'long_tail_pool.json')}")
    print(f"  extreme json:   {os.path.join(OUTPUT_DIR, 'extreme_pool.json')}")
    print(f"  summary json:   {os.path.join(OUTPUT_DIR, 'split_summary_with_cross_stats.json')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
