import os
import json
import glob
import math
from collections import Counter

import matplotlib.pyplot as plt


# =========================
# Config
# =========================

TXT_DIR = "geometry_sampled/pcd_saturation"
GEOMETRY_JSON_PATH = "geometry_sampled_12000.json"

OUTPUT_ROOT = "geometry_sampled/object_complexity_distribution"
MERGED_OUTPUT_PATH = "geometry_sampled/object_complexity_merged.json"
STATS_OUTPUT_PATH = "geometry_sampled/object_complexity_stats.json"

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

SELF_OCCLUSION_HIST_BINS = 20
SCATTER_MAX_POINTS = 5000  # for global scatter plot only


# =========================
# Parsing
# =========================

def parse_saturation_txt(txt_path):
    uid = os.path.splitext(os.path.basename(txt_path))[0]

    parsed = {
        "uid": uid,
        "A_fit": None,
        "A_discrete": None,
        "B_continuous": None,
        "B_discrete": None,
        "gt_surface_voxel_count": None,
        "fit_type": None,
        "fit_ymax": None,
        "fit_mu": None,
        "fit_sigma": None,
        "saturation_delta": None,
        "saturation_window": None,
        "saturation_epsilon_ratio": None,
    }

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if key == "A_fit":
                parsed["A_fit"] = float(value)
            elif key == "A_discrete":
                parsed["A_discrete"] = float(value)
            elif key == "B_continuous":
                parsed["B_continuous"] = float(value)
            elif key == "B_discrete":
                parsed["B_discrete"] = int(float(value))
            elif key == "gt_surface_voxel_count":
                parsed["gt_surface_voxel_count"] = int(float(value))
            elif key == "fit_type":
                parsed["fit_type"] = value
            elif key == "fit_ymax":
                parsed["fit_ymax"] = float(value)
            elif key == "fit_mu":
                parsed["fit_mu"] = float(value)
            elif key == "fit_sigma":
                parsed["fit_sigma"] = float(value)
            elif key == "saturation_delta":
                parsed["saturation_delta"] = float(value)
            elif key == "saturation_window":
                parsed["saturation_window"] = float(value)
            elif key == "saturation_epsilon_ratio":
                parsed["saturation_epsilon_ratio"] = float(value)

    required_keys = [
        "A_fit",
        "A_discrete",
        "B_continuous",
        "B_discrete",
        "gt_surface_voxel_count",
        "fit_type",
    ]
    missing = [k for k in required_keys if parsed[k] is None]
    if missing:
        raise ValueError(f"Missing fields {missing} in {txt_path}")

    return parsed


def load_all_txt_stats(txt_dir):
    txt_paths = sorted(glob.glob(os.path.join(txt_dir, "*.txt")))
    uid_to_stats = {}

    for txt_path in txt_paths:
        parsed = parse_saturation_txt(txt_path)

        uid_to_stats[parsed["uid"]] = {
            # main metrics
            "self_occlusion_attribute": parsed["A_fit"],
            "observation_saturation_view_num": int(math.ceil(parsed["B_continuous"])),

            # reference metrics
            "self_occlusion_attribute_discrete": parsed["A_discrete"],
            "observation_saturation_view_num_continuous": parsed["B_continuous"],
            "observation_saturation_view_num_discrete": parsed["B_discrete"],

            # metadata
            "gt_surface_voxel_count": parsed["gt_surface_voxel_count"],
            "fit_type": parsed["fit_type"],

            # fit parameters
            "fit_ymax": parsed["fit_ymax"],
            "fit_mu": parsed["fit_mu"],
            "fit_sigma": parsed["fit_sigma"],

            # threshold / protocol params
            "saturation_delta": parsed["saturation_delta"],
            "saturation_window": parsed["saturation_window"],
            "saturation_epsilon_ratio": parsed["saturation_epsilon_ratio"],
        }

    return uid_to_stats


def merge_with_geometry_json(geometry_json_path, uid_to_stats):
    with open(geometry_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    merged = []
    for item in data:
        uid = item.get("uid")
        if uid not in uid_to_stats:
            continue

        shape_type = item.get("shape_type")
        fill_bucket = item.get("fill_bucket")

        if shape_type not in VALID_SHAPE_TYPES:
            continue
        if fill_bucket not in VALID_FILL_BUCKETS:
            continue

        new_item = dict(item)
        new_item.update(uid_to_stats[uid])
        merged.append(new_item)

    return merged


# =========================
# Stats helpers
# =========================

def safe_mean(vals):
    return sum(vals) / len(vals) if vals else None


def safe_min(vals):
    return min(vals) if vals else None


def safe_max(vals):
    return max(vals) if vals else None


def percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]

    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    w = pos - lo
    return sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w


def summarize_continuous(vals):
    vals = list(vals)
    vals_sorted = sorted(vals)
    return {
        "count": len(vals),
        "mean": safe_mean(vals),
        "min": safe_min(vals),
        "p25": percentile(vals_sorted, 0.25),
        "median": percentile(vals_sorted, 0.5),
        "p75": percentile(vals_sorted, 0.75),
        "max": safe_max(vals),
    }


def counter_to_sorted_dict(counter):
    return {int(k): int(v) for k, v in sorted(counter.items(), key=lambda kv: int(kv[0]))}


def compute_group_stats(items):
    b_counter = Counter()
    a_vals = []
    b_cont_vals = []
    b_disc_vals = []
    a_disc_vals = []
    gt_voxel_vals = []
    fit_type_counter = Counter()

    for item in items:
        b_counter[item["observation_saturation_view_num"]] += 1
        a_vals.append(item["self_occlusion_attribute"])
        b_cont_vals.append(item["observation_saturation_view_num_continuous"])
        b_disc_vals.append(item["observation_saturation_view_num_discrete"])
        a_disc_vals.append(item["self_occlusion_attribute_discrete"])
        gt_voxel_vals.append(item["gt_surface_voxel_count"])
        fit_type_counter[item["fit_type"]] += 1

    return {
        "num_objects": len(items),
        "observation_saturation_view_num_distribution": counter_to_sorted_dict(b_counter),
        "self_occlusion_attribute_summary": summarize_continuous(a_vals),
        "observation_saturation_view_num_continuous_summary": summarize_continuous(b_cont_vals),
        "observation_saturation_view_num_discrete_summary": summarize_continuous(b_disc_vals),
        "self_occlusion_attribute_discrete_summary": summarize_continuous(a_disc_vals),
        "gt_surface_voxel_count_summary": summarize_continuous(gt_voxel_vals),
        "fit_type_distribution": dict(sorted(fit_type_counter.items())),
    }


def compute_all_stats(merged_data):
    stats = {
        "global_stats": compute_group_stats(merged_data),
        "shape_stats": {},
        "fill_stats": {},
        "bucket_stats": {},
    }

    for shape_type in VALID_SHAPE_TYPES:
        subset = [x for x in merged_data if x["shape_type"] == shape_type]
        info = compute_group_stats(subset)
        info["shape_type"] = shape_type
        stats["shape_stats"][shape_type] = info

    for fill_bucket in VALID_FILL_BUCKETS:
        subset = [x for x in merged_data if x["fill_bucket"] == fill_bucket]
        info = compute_group_stats(subset)
        info["fill_bucket"] = fill_bucket
        stats["fill_stats"][fill_bucket] = info

    for shape_type in VALID_SHAPE_TYPES:
        for fill_bucket in VALID_FILL_BUCKETS:
            subset = [
                x for x in merged_data
                if x["shape_type"] == shape_type and x["fill_bucket"] == fill_bucket
            ]
            info = compute_group_stats(subset)
            info["shape_type"] = shape_type
            info["fill_bucket"] = fill_bucket
            stats["bucket_stats"][f"{shape_type}__{fill_bucket}"] = info

    return stats


def run_sanity_checks(merged_data, stats):
    matched_count = len(merged_data)

    shape_total = sum(v["num_objects"] for v in stats["shape_stats"].values())
    fill_total = sum(v["num_objects"] for v in stats["fill_stats"].values())
    bucket_total = sum(v["num_objects"] for v in stats["bucket_stats"].values())
    global_total = stats["global_stats"]["num_objects"]

    assert shape_total == matched_count, f"shape_total={shape_total}, matched_count={matched_count}"
    assert fill_total == matched_count, f"fill_total={fill_total}, matched_count={matched_count}"
    assert bucket_total == matched_count, f"bucket_total={bucket_total}, matched_count={matched_count}"
    assert global_total == matched_count, f"global_total={global_total}, matched_count={matched_count}"

    for name, info in stats["shape_stats"].items():
        s = sum(info["observation_saturation_view_num_distribution"].values())
        assert s == info["num_objects"], f"{name}: B dist sum={s}, num_objects={info['num_objects']}"

    for name, info in stats["fill_stats"].items():
        s = sum(info["observation_saturation_view_num_distribution"].values())
        assert s == info["num_objects"], f"{name}: B dist sum={s}, num_objects={info['num_objects']}"

    for name, info in stats["bucket_stats"].items():
        s = sum(info["observation_saturation_view_num_distribution"].values())
        assert s == info["num_objects"], f"{name}: B dist sum={s}, num_objects={info['num_objects']}"

    global_b_sum = sum(stats["global_stats"]["observation_saturation_view_num_distribution"].values())
    assert global_b_sum == stats["global_stats"]["num_objects"], (
        f"global B dist sum={global_b_sum}, global_num_objects={stats['global_stats']['num_objects']}"
    )

    print("All sanity checks passed.")


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================
# Plot helpers
# =========================

def get_global_b_xs(stats):
    dist = stats["global_stats"]["observation_saturation_view_num_distribution"]
    xs = sorted(int(k) for k in dist.keys())
    return xs


def make_full_y_series(distribution, global_xs):
    counter = {x: 0 for x in global_xs}
    for k, v in distribution.items():
        counter[int(k)] = v
    return [counter[x] for x in global_xs]


def choose_xticks(global_xs, max_ticks=16):
    if len(global_xs) <= max_ticks:
        return global_xs

    step = max(1, len(global_xs) // max_ticks)
    ticks = global_xs[::step]
    if ticks[-1] != global_xs[-1]:
        ticks.append(global_xs[-1])
    return ticks


def plot_b_distribution(global_xs, distribution, title, save_path):
    ys = make_full_y_series(distribution, global_xs)
    xticks = choose_xticks(global_xs, max_ticks=16)

    plt.figure(figsize=(10, 6))
    plt.bar(global_xs, ys, width=0.8)
    plt.xlim(min(global_xs) - 1, max(global_xs) + 1)
    plt.xticks(xticks)
    plt.xlabel("observation_saturation_view_num")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_a_hist(values, title, save_path, bins=20):
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins, range=(0.0, 1.0))
    plt.xlim(0.0, 1.0)
    plt.xlabel("self_occlusion_attribute")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_global_scatter(merged_data, save_path, max_points=5000):
    xs = [x["observation_saturation_view_num"] for x in merged_data]
    ys = [x["self_occlusion_attribute"] for x in merged_data]

    if len(xs) > max_points:
        step = max(1, len(xs) // max_points)
        xs = xs[::step]
        ys = ys[::step]

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, s=8, alpha=0.6)
    plt.xlabel("observation_saturation_view_num")
    plt.ylabel("self_occlusion_attribute")
    plt.title("all_objects | self_occlusion_attribute vs observation_saturation_view_num")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_bucket_plots(stats, merged_data, output_dir, global_b_xs):
    bucket_dir = os.path.join(output_dir, "12_buckets")
    os.makedirs(bucket_dir, exist_ok=True)

    for shape_type in VALID_SHAPE_TYPES:
        for fill_bucket in VALID_FILL_BUCKETS:
            key = f"{shape_type}__{fill_bucket}"
            info = stats["bucket_stats"][key]
            subset = [
                x for x in merged_data
                if x["shape_type"] == shape_type and x["fill_bucket"] == fill_bucket
            ]

            title_b = f"{shape_type} | {fill_bucket} | B | n={info['num_objects']}"
            save_b = os.path.join(bucket_dir, f"{key}_B.png")
            plot_b_distribution(global_b_xs, info["observation_saturation_view_num_distribution"], title_b, save_b)

            title_a = f"{shape_type} | {fill_bucket} | A | n={info['num_objects']}"
            save_a = os.path.join(bucket_dir, f"{key}_A.png")
            plot_a_hist([x["self_occlusion_attribute"] for x in subset], title_a, save_a, bins=SELF_OCCLUSION_HIST_BINS)


def save_shape_plots(stats, merged_data, output_dir, global_b_xs):
    shape_dir = os.path.join(output_dir, "4_shape_types")
    os.makedirs(shape_dir, exist_ok=True)

    for shape_type in VALID_SHAPE_TYPES:
        info = stats["shape_stats"][shape_type]
        subset = [x for x in merged_data if x["shape_type"] == shape_type]

        title_b = f"{shape_type} | B | n={info['num_objects']}"
        save_b = os.path.join(shape_dir, f"{shape_type}_B.png")
        plot_b_distribution(global_b_xs, info["observation_saturation_view_num_distribution"], title_b, save_b)

        title_a = f"{shape_type} | A | n={info['num_objects']}"
        save_a = os.path.join(shape_dir, f"{shape_type}_A.png")
        plot_a_hist([x["self_occlusion_attribute"] for x in subset], title_a, save_a, bins=SELF_OCCLUSION_HIST_BINS)


def save_fill_plots(stats, merged_data, output_dir, global_b_xs):
    fill_dir = os.path.join(output_dir, "3_fill_buckets")
    os.makedirs(fill_dir, exist_ok=True)

    for fill_bucket in VALID_FILL_BUCKETS:
        info = stats["fill_stats"][fill_bucket]
        subset = [x for x in merged_data if x["fill_bucket"] == fill_bucket]

        title_b = f"{fill_bucket} | B | n={info['num_objects']}"
        save_b = os.path.join(fill_dir, f"{fill_bucket}_B.png")
        plot_b_distribution(global_b_xs, info["observation_saturation_view_num_distribution"], title_b, save_b)

        title_a = f"{fill_bucket} | A | n={info['num_objects']}"
        save_a = os.path.join(fill_dir, f"{fill_bucket}_A.png")
        plot_a_hist([x["self_occlusion_attribute"] for x in subset], title_a, save_a, bins=SELF_OCCLUSION_HIST_BINS)


def save_global_plots(stats, merged_data, output_dir, global_b_xs):
    global_dir = os.path.join(output_dir, "global")
    os.makedirs(global_dir, exist_ok=True)

    title_b = f"all_objects | B | n={stats['global_stats']['num_objects']}"
    save_b = os.path.join(global_dir, "all_objects_B.png")
    plot_b_distribution(
        global_b_xs,
        stats["global_stats"]["observation_saturation_view_num_distribution"],
        title_b,
        save_b,
    )

    title_a = f"all_objects | A | n={stats['global_stats']['num_objects']}"
    save_a = os.path.join(global_dir, "all_objects_A.png")
    plot_a_hist(
        [x["self_occlusion_attribute"] for x in merged_data],
        title_a,
        save_a,
        bins=SELF_OCCLUSION_HIST_BINS,
    )

    save_scatter = os.path.join(global_dir, "all_objects_A_vs_B.png")
    plot_global_scatter(merged_data, save_scatter, max_points=SCATTER_MAX_POINTS)


# =========================
# Summary
# =========================

def print_summary(matched_count, merged_output_path, stats_output_path, plot_output_dir):
    print("=" * 80)
    print(f"Matched objects: {matched_count}")
    print(f"Merged json: {merged_output_path}")
    print(f"Stats json:  {stats_output_path}")
    print(f"Plots dir:   {plot_output_dir}")
    print("Generated:")
    print("  - 12 bucket B plots")
    print("  - 12 bucket A plots")
    print("  - 4 shape_type B plots")
    print("  - 4 shape_type A plots")
    print("  - 3 fill_bucket B plots")
    print("  - 3 fill_bucket A plots")
    print("  - 1 global B plot")
    print("  - 1 global A plot")
    print("  - 1 global A_vs_B scatter")
    print("=" * 80)


# =========================
# Main
# =========================

def main():
    uid_to_stats = load_all_txt_stats(TXT_DIR)
    print(f"Loaded txt files: {len(uid_to_stats)}")

    merged_data = merge_with_geometry_json(GEOMETRY_JSON_PATH, uid_to_stats)
    matched_count = len(merged_data)
    print(f"Matched objects with geometry json: {matched_count}")

    save_json(merged_data, MERGED_OUTPUT_PATH)

    stats = compute_all_stats(merged_data)
    run_sanity_checks(merged_data, stats)
    save_json(stats, STATS_OUTPUT_PATH)

    global_b_xs = get_global_b_xs(stats)

    save_bucket_plots(stats, merged_data, OUTPUT_ROOT, global_b_xs)
    save_shape_plots(stats, merged_data, OUTPUT_ROOT, global_b_xs)
    save_fill_plots(stats, merged_data, OUTPUT_ROOT, global_b_xs)
    save_global_plots(stats, merged_data, OUTPUT_ROOT, global_b_xs)

    print_summary(
        matched_count=matched_count,
        merged_output_path=MERGED_OUTPUT_PATH,
        stats_output_path=STATS_OUTPUT_PATH,
        plot_output_dir=OUTPUT_ROOT,
    )


if __name__ == "__main__":
    main()
