import os
import json
import glob
from collections import Counter, defaultdict

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


# =========================================================
# Parsing set-cover txt
# =========================================================

def parse_set_cover_txt(txt_path):
    uid = os.path.splitext(os.path.basename(txt_path))[0]
    selected_view_count = None
    num_views = None

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if key == "selected_view_count":
                selected_view_count = int(value)
            elif key == "num_views":
                num_views = int(value)

    if selected_view_count is None or num_views is None:
        raise ValueError(f"Missing selected_view_count or num_views in {txt_path}")

    return {
        "uid": uid,
        "selected_view_count": selected_view_count,
        "num_views": num_views,
    }


def load_all_txt_stats(txt_dir):
    txt_paths = sorted(glob.glob(os.path.join(txt_dir, "*.txt")))
    uid_to_stats = {}

    for txt_path in txt_paths:
        parsed = parse_set_cover_txt(txt_path)
        uid_to_stats[parsed["uid"]] = {
            "selected_view_count": parsed["selected_view_count"],
            "num_views": parsed["num_views"],
        }

    return uid_to_stats


# =========================================================
# Merge with geometry + object complexity json
# =========================================================

def merge_all(geometry_json_path, object_complexity_json_path, uid_to_setcover):
    with open(geometry_json_path, "r", encoding="utf-8") as f:
        geometry_data = json.load(f)

    with open(object_complexity_json_path, "r", encoding="utf-8") as f:
        complexity_data = json.load(f)

    complexity_by_uid = {}
    for item in complexity_data:
        uid = item.get("uid")
        if uid is not None:
            complexity_by_uid[uid] = item

    merged = []
    for item in geometry_data:
        uid = item.get("uid")
        if uid not in uid_to_setcover:
            continue
        if uid not in complexity_by_uid:
            continue

        shape_type = item.get("shape_type")
        fill_bucket = item.get("fill_bucket")
        if shape_type not in VALID_SHAPE_TYPES:
            continue
        if fill_bucket not in VALID_FILL_BUCKETS:
            continue

        cpx = complexity_by_uid[uid]

        new_item = dict(item)
        new_item["selected_view_count"] = uid_to_setcover[uid]["selected_view_count"]
        new_item["num_views"] = uid_to_setcover[uid]["num_views"]

        # object complexity side
        new_item["self_occlusion_attribute"] = cpx.get("self_occlusion_attribute")
        new_item["observation_saturation_view_num"] = cpx.get("observation_saturation_view_num")
        new_item["gt_surface_voxel_count"] = cpx.get("gt_surface_voxel_count")
        new_item["fit_type"] = cpx.get("fit_type")

        merged.append(new_item)

    return merged


# =========================================================
# Stats
# =========================================================

def compute_distribution_stats(merged_data):
    bucket_counters = defaultdict(Counter)
    shape_counters = defaultdict(Counter)
    fill_counters = defaultdict(Counter)
    global_counter = Counter()

    bucket_sizes = Counter()
    shape_sizes = Counter()
    fill_sizes = Counter()

    for item in merged_data:
        shape_type = item["shape_type"]
        fill_bucket = item["fill_bucket"]
        diff = item["selected_view_count"]

        bucket_key = (shape_type, fill_bucket)

        bucket_counters[bucket_key][diff] += 1
        shape_counters[shape_type][diff] += 1
        fill_counters[fill_bucket][diff] += 1
        global_counter[diff] += 1

        bucket_sizes[bucket_key] += 1
        shape_sizes[shape_type] += 1
        fill_sizes[fill_bucket] += 1

    stats = {
        "global_planning_difficulty_distribution": dict(sorted(global_counter.items())),
        "global_num_objects": sum(global_counter.values()),
        "shape_stats": {},
        "fill_stats": {},
        "bucket_stats": {},
    }

    for shape_type in VALID_SHAPE_TYPES:
        stats["shape_stats"][shape_type] = {
            "shape_type": shape_type,
            "num_objects": shape_sizes.get(shape_type, 0),
            "planning_difficulty_distribution": dict(sorted(shape_counters.get(shape_type, Counter()).items())),
        }

    for fill_bucket in VALID_FILL_BUCKETS:
        stats["fill_stats"][fill_bucket] = {
            "fill_bucket": fill_bucket,
            "num_objects": fill_sizes.get(fill_bucket, 0),
            "planning_difficulty_distribution": dict(sorted(fill_counters.get(fill_bucket, Counter()).items())),
        }

    for shape_type in VALID_SHAPE_TYPES:
        for fill_bucket in VALID_FILL_BUCKETS:
            key = (shape_type, fill_bucket)
            stats["bucket_stats"][f"{shape_type}__{fill_bucket}"] = {
                "shape_type": shape_type,
                "fill_bucket": fill_bucket,
                "num_objects": bucket_sizes.get(key, 0),
                "planning_difficulty_distribution": dict(sorted(bucket_counters.get(key, Counter()).items())),
            }

    return stats


def compute_cross_analysis(merged_data):
    """
    Analyze relationship between:
      - planning difficulty: selected_view_count
      - self_occlusion_attribute
      - observation_saturation_view_num
      - gt_surface_voxel_count
    """
    by_saturation_bin = defaultdict(list)
    by_self_occ_bin = defaultdict(list)

    for item in merged_data:
        pdiff = item["selected_view_count"]
        sat = item.get("observation_saturation_view_num")
        occ = item.get("self_occlusion_attribute")

        if sat is not None:
            sat = int(sat)
            sat_bin = (sat // 16) * 16
            by_saturation_bin[sat_bin].append(pdiff)

        if occ is not None:
            occ = float(occ)
            occ_bin = round((occ // 0.05) * 0.05, 2)
            by_self_occ_bin[occ_bin].append(pdiff)

    saturation_bin_stats = {}
    for k in sorted(by_saturation_bin.keys()):
        vals = by_saturation_bin[k]
        saturation_bin_stats[str(k)] = {
            "count": len(vals),
            "selected_view_count_mean": sum(vals) / len(vals),
            "selected_view_count_min": min(vals),
            "selected_view_count_max": max(vals),
        }

    self_occ_bin_stats = {}
    for k in sorted(by_self_occ_bin.keys()):
        vals = by_self_occ_bin[k]
        self_occ_bin_stats[str(k)] = {
            "count": len(vals),
            "selected_view_count_mean": sum(vals) / len(vals),
            "selected_view_count_min": min(vals),
            "selected_view_count_max": max(vals),
        }

    return {
        "saturation_bin_stats": saturation_bin_stats,
        "self_occlusion_bin_stats": self_occ_bin_stats,
    }


# =========================================================
# Sanity
# =========================================================

def run_sanity_checks(merged_data, stats):
    matched_count = len(merged_data)

    bucket_total = sum(v["num_objects"] for v in stats["bucket_stats"].values())
    assert bucket_total == matched_count, f"bucket_total={bucket_total}, matched_count={matched_count}"

    shape_total = sum(v["num_objects"] for v in stats["shape_stats"].values())
    assert shape_total == matched_count, f"shape_total={shape_total}, matched_count={matched_count}"

    fill_total = sum(v["num_objects"] for v in stats["fill_stats"].values())
    assert fill_total == matched_count, f"fill_total={fill_total}, matched_count={matched_count}"

    global_total = stats["global_num_objects"]
    assert global_total == matched_count, f"global_total={global_total}, matched_count={matched_count}"

    for name, info in stats["bucket_stats"].items():
        s = sum(info["planning_difficulty_distribution"].values())
        assert s == info["num_objects"], f"{name}: dist_sum={s}, num_objects={info['num_objects']}"

    for name, info in stats["shape_stats"].items():
        s = sum(info["planning_difficulty_distribution"].values())
        assert s == info["num_objects"], f"{name}: dist_sum={s}, num_objects={info['num_objects']}"

    for name, info in stats["fill_stats"].items():
        s = sum(info["planning_difficulty_distribution"].values())
        assert s == info["num_objects"], f"{name}: dist_sum={s}, num_objects={info['num_objects']}"

    s = sum(stats["global_planning_difficulty_distribution"].values())
    assert s == stats["global_num_objects"], f"global dist sum={s}, global_num_objects={stats['global_num_objects']}"

    print("All sanity checks passed.")


# =========================================================
# Helpers for distributions
# =========================================================

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_global_xs(stats):
    global_dist = stats["global_planning_difficulty_distribution"]
    xs = sorted(int(k) for k in global_dist.keys())
    return xs


def make_full_y_series(distribution, global_xs):
    counter = {x: 0 for x in global_xs}
    for k, v in distribution.items():
        counter[int(k)] = v
    ys = [counter[x] for x in global_xs]
    return ys


def choose_xticks(global_xs, max_ticks=16):
    if len(global_xs) <= max_ticks:
        return global_xs

    step = max(1, len(global_xs) // max_ticks)
    ticks = global_xs[::step]
    if ticks[-1] != global_xs[-1]:
        ticks.append(global_xs[-1])
    return ticks


# =========================================================
# Plotting distribution bars
# =========================================================

def plot_distribution(global_xs, distribution, title, save_path):
    ys = make_full_y_series(distribution, global_xs)
    xticks = choose_xticks(global_xs, max_ticks=16)

    plt.figure(figsize=(10, 6))
    plt.bar(global_xs, ys, width=0.8)
    plt.xlim(min(global_xs) - 1, max(global_xs) + 1)
    plt.xticks(xticks)
    plt.xlabel("selected_view_count")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def save_bucket_plots(stats, output_dir, global_xs):
    bucket_dir = os.path.join(output_dir, "12_buckets")
    os.makedirs(bucket_dir, exist_ok=True)

    for shape_type in VALID_SHAPE_TYPES:
        for fill_bucket in VALID_FILL_BUCKETS:
            key = f"{shape_type}__{fill_bucket}"
            info = stats["bucket_stats"][key]
            title = f"{shape_type} | {fill_bucket} | n={info['num_objects']}"
            save_path = os.path.join(bucket_dir, f"{key}.png")
            plot_distribution(global_xs, info["planning_difficulty_distribution"], title, save_path)


def save_shape_plots(stats, output_dir, global_xs):
    shape_dir = os.path.join(output_dir, "4_shape_types")
    os.makedirs(shape_dir, exist_ok=True)

    for shape_type in VALID_SHAPE_TYPES:
        info = stats["shape_stats"][shape_type]
        title = f"{shape_type} | n={info['num_objects']}"
        save_path = os.path.join(shape_dir, f"{shape_type}.png")
        plot_distribution(global_xs, info["planning_difficulty_distribution"], title, save_path)


def save_fill_plots(stats, output_dir, global_xs):
    fill_dir = os.path.join(output_dir, "3_fill_buckets")
    os.makedirs(fill_dir, exist_ok=True)

    for fill_bucket in VALID_FILL_BUCKETS:
        info = stats["fill_stats"][fill_bucket]
        title = f"{fill_bucket} | n={info['num_objects']}"
        save_path = os.path.join(fill_dir, f"{fill_bucket}.png")
        plot_distribution(global_xs, info["planning_difficulty_distribution"], title, save_path)


def save_global_plot(stats, output_dir, global_xs):
    global_dir = os.path.join(output_dir, "global")
    os.makedirs(global_dir, exist_ok=True)

    title = f"all_objects | n={stats['global_num_objects']}"
    save_path = os.path.join(global_dir, "all_objects.png")
    plot_distribution(global_xs, stats["global_planning_difficulty_distribution"], title, save_path)


# =========================================================
# Plotting cross-analysis
# =========================================================

def plot_scatter(xs, ys, title, xlabel, ylabel, save_path, logx=False, logy=False):
    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, s=10, alpha=0.45)

    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def save_cross_plots(merged_data, output_dir):
    cross_dir = os.path.join(output_dir, "cross_analysis")
    os.makedirs(cross_dir, exist_ok=True)

    # planning difficulty vs saturation view num
    xs = []
    ys = []
    for item in merged_data:
        sat = item.get("observation_saturation_view_num")
        pdiff = item.get("selected_view_count")
        if sat is None or pdiff is None:
            continue
        xs.append(int(sat))
        ys.append(int(pdiff))
    plot_scatter(
        xs, ys,
        title="planning difficulty vs observation_saturation_view_num",
        xlabel="observation_saturation_view_num",
        ylabel="selected_view_count",
        save_path=os.path.join(cross_dir, "planning_vs_saturation_view_num.png"),
    )

    # planning difficulty vs self-occlusion attribute
    xs = []
    ys = []
    for item in merged_data:
        occ = item.get("self_occlusion_attribute")
        pdiff = item.get("selected_view_count")
        if occ is None or pdiff is None:
            continue
        xs.append(float(occ))
        ys.append(int(pdiff))
    plot_scatter(
        xs, ys,
        title="planning difficulty vs self_occlusion_attribute",
        xlabel="self_occlusion_attribute",
        ylabel="selected_view_count",
        save_path=os.path.join(cross_dir, "planning_vs_self_occlusion_attribute.png"),
    )

    # planning difficulty vs gt surface voxel count
    xs = []
    ys = []
    for item in merged_data:
        gt = item.get("gt_surface_voxel_count")
        pdiff = item.get("selected_view_count")
        if gt is None or pdiff is None:
            continue
        xs.append(int(gt))
        ys.append(int(pdiff))
    plot_scatter(
        xs, ys,
        title="planning difficulty vs gt_surface_voxel_count",
        xlabel="gt_surface_voxel_count",
        ylabel="selected_view_count",
        save_path=os.path.join(cross_dir, "planning_vs_gt_surface_voxel_count.png"),
        logx=True,
    )

    # saturation view num vs self-occlusion attribute
    xs = []
    ys = []
    for item in merged_data:
        sat = item.get("observation_saturation_view_num")
        occ = item.get("self_occlusion_attribute")
        if sat is None or occ is None:
            continue
        xs.append(int(sat))
        ys.append(float(occ))
    plot_scatter(
        xs, ys,
        title="self_occlusion_attribute vs observation_saturation_view_num",
        xlabel="observation_saturation_view_num",
        ylabel="self_occlusion_attribute",
        save_path=os.path.join(cross_dir, "self_occlusion_vs_saturation_view_num.png"),
    )


# =========================================================
# Summary print
# =========================================================

def print_summary(matched_count, merged_output_path, stats_output_path, cross_output_path, plot_output_dir):
    print("=" * 80)
    print(f"Matched objects: {matched_count}")
    print(f"Merged json:       {merged_output_path}")
    print(f"Stats json:        {stats_output_path}")
    print(f"Cross json:        {cross_output_path}")
    print(f"Plots dir:         {plot_output_dir}")
    print("Generated:")
    print("  - 12 bucket plots")
    print("  - 4 shape_type plots")
    print("  - 3 fill_bucket plots")
    print("  - 1 global plot")
    print("  - 4 cross-analysis scatter plots")
    print("=" * 80)


# =========================================================
# Main
# =========================================================

def main():
    txt_dir = "geometry_sampled/pcd_set_cover"
    geometry_json_path = "geometry_sampled_12000.json"
    object_complexity_json_path = "geometry_sampled/object_complexity_merged.json"
    plot_output_dir = "geometry_sampled/planning_difficulty_vs_object_complexity"

    uid_to_stats = load_all_txt_stats(txt_dir)
    print(f"Loaded txt files: {len(uid_to_stats)}")

    merged_data = merge_all(
        geometry_json_path=geometry_json_path,
        object_complexity_json_path=object_complexity_json_path,
        uid_to_setcover=uid_to_stats,
    )
    matched_count = len(merged_data)

    merged_output_path = f"geometry_sampled/planning_vs_object_complexity.json"
    stats_output_path = f"geometry_sampled/planning_vs_object_complexity_stats.json"
    cross_output_path = f"geometry_sampled/planning_vs_object_complexity_cross.json"

    save_json(merged_data, merged_output_path)

    stats = compute_distribution_stats(merged_data)
    run_sanity_checks(merged_data, stats)
    save_json(stats, stats_output_path)

    cross_stats = compute_cross_analysis(merged_data)
    save_json(cross_stats, cross_output_path)

    global_xs = get_global_xs(stats)

    save_bucket_plots(stats, plot_output_dir, global_xs)
    save_shape_plots(stats, plot_output_dir, global_xs)
    save_fill_plots(stats, plot_output_dir, global_xs)
    save_global_plot(stats, plot_output_dir, global_xs)
    save_cross_plots(merged_data, plot_output_dir)

    print_summary(
        matched_count=matched_count,
        merged_output_path=merged_output_path,
        stats_output_path=stats_output_path,
        cross_output_path=cross_output_path,
        plot_output_dir=plot_output_dir,
    )


if __name__ == "__main__":
    main()
