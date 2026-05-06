import os
import json
import glob
import statistics

import matplotlib.pyplot as plt


# =========================
# Config
# =========================

TXT_DIR = "geometry_sampled/pcd_saturation"
OUTPUT_DIR = "geometry_sampled/final_clean_pool_saturation_analysis"

FINAL_POOL_JSON = "geometry_sampled/final_clean_pool/final_clean_pool.json"
MAIN_POOL_JSON = "geometry_sampled/final_clean_pool/final_clean_main_pool.json"
LONG_TAIL_POOL_JSON = "geometry_sampled/final_clean_pool/final_clean_long_tail_pool.json"

# use this exact evaluated schedule
N_LIST = [6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64,
          80, 96, 112, 128, 144, 160, 180, 200, 270, 360, 432, 492]

# use which curve for analysis: "curve_raw" or "curve_monotonicized"
CURVE_NAME = "curve_raw"

FIG_DPI = 220


# =========================
# IO helpers
# =========================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def summarize(vals):
    vals = list(vals)
    if not vals:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "min": None,
            "max": None,
        }

    vals_sorted = sorted(vals)
    return {
        "count": len(vals),
        "mean": sum(vals) / len(vals),
        "median": statistics.median(vals),
        "p25": percentile(vals_sorted, 0.25),
        "p75": percentile(vals_sorted, 0.75),
        "min": vals_sorted[0],
        "max": vals_sorted[-1],
    }


# =========================
# Parsing saturation txt
# =========================

def parse_curve_block(lines, start_idx):
    curve = {}
    i = start_idx + 1
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            break
        if ":" not in line:
            break

        left, right = line.split(":", 1)
        left = left.strip()
        right = right.strip()

        try:
            n = int(left)
            y = float(right)
            curve[n] = y
        except ValueError:
            break

        i += 1

    return curve, i


def parse_saturation_txt(txt_path):
    uid = os.path.splitext(os.path.basename(txt_path))[0]

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    parsed = {
        "uid": uid,
        "observation_saturation_view_num": None,
        "self_occlusion_attribute": None,
        "curve_raw": {},
        "curve_monotonicized": {},
    }

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("B_discrete:"):
            parsed["observation_saturation_view_num"] = int(float(line.split(":", 1)[1].strip()))
        elif line.startswith("A_fit:"):
            parsed["self_occlusion_attribute"] = float(line.split(":", 1)[1].strip())
        elif line == "curve_raw:":
            curve, new_i = parse_curve_block(lines, i)
            parsed["curve_raw"] = curve
            i = new_i
            continue
        elif line == "curve_monotonicized:":
            curve, new_i = parse_curve_block(lines, i)
            parsed["curve_monotonicized"] = curve
            i = new_i
            continue

        i += 1

    if parsed["observation_saturation_view_num"] is None:
        raise ValueError(f"Missing observation_saturation_view_num (B_discrete) in {txt_path}")
    if parsed["self_occlusion_attribute"] is None:
        raise ValueError(f"Missing self_occlusion_attribute (A_fit) in {txt_path}")

    return parsed


def load_all_saturation_txts(txt_dir):
    txt_paths = sorted(glob.glob(os.path.join(txt_dir, "*.txt")))
    out = {}
    for txt_path in txt_paths:
        parsed = parse_saturation_txt(txt_path)
        out[parsed["uid"]] = parsed
    return out


# =========================
# Match final filtered pools with saturation txts
# =========================

def attach_saturation_info(pool_items, saturation_by_uid):
    matched = []
    missing = []

    for item in pool_items:
        uid = item["uid"]
        sat = saturation_by_uid.get(uid)
        if sat is None:
            missing.append(uid)
            continue

        merged = dict(item)
        merged["observation_saturation_view_num"] = sat["observation_saturation_view_num"]
        merged["self_occlusion_attribute"] = sat["self_occlusion_attribute"]
        merged["curve_raw"] = sat["curve_raw"]
        merged["curve_monotonicized"] = sat["curve_monotonicized"]
        matched.append(merged)

    return matched, missing


# =========================
# Saturation partition analysis
# =========================

def validate_curve_has_required_n(curve, n_list, uid):
    missing = [n for n in n_list if n not in curve]
    if missing:
        raise ValueError(f"{uid} missing N values in curve: {missing}")


def compute_normalized_curve(curve, n_list):
    n_max = n_list[-1]
    y_max = curve[n_max]
    if y_max <= 0:
        raise ValueError(f"Invalid y_max={y_max} at N_max={n_max}")
    return {n: curve[n] / y_max for n in n_list}


def analyze_pool(items, n_list, curve_name):
    per_object = []

    for item in items:
        curve = item[curve_name]
        validate_curve_has_required_n(curve, n_list, item["uid"])
        norm_curve = compute_normalized_curve(curve, n_list)

        per_object.append({
            "uid": item["uid"],
            "pool_type": item.get("pool_type"),
            "shape_type": item.get("shape_type"),
            "fill_bucket": item.get("fill_bucket"),
            "observation_saturation_view_num": item["observation_saturation_view_num"],
            "self_occlusion_attribute": item["self_occlusion_attribute"],
            "selected_view_count": item.get("selected_view_count"),
            "normalized_curve": norm_curve,
        })

    normalized_coverage_stats = {}
    for n in n_list:
        vals = [x["normalized_curve"][n] for x in per_object]
        normalized_coverage_stats[str(n)] = summarize(vals)

    marginal_gain_stats = {}
    for i in range(len(n_list) - 1):
        n1 = n_list[i]
        n2 = n_list[i + 1]
        vals = [
            x["normalized_curve"][n2] - x["normalized_curve"][n1]
            for x in per_object
        ]
        marginal_gain_stats[f"{n1}->{n2}"] = summarize(vals)

    B_vals = [x["observation_saturation_view_num"] for x in per_object]
    A_vals = [x["self_occlusion_attribute"] for x in per_object]
    C_vals = [x["selected_view_count"] for x in per_object if x["selected_view_count"] is not None]

    return {
        "num_objects": len(per_object),
        "curve_name": curve_name,
        "n_list": n_list,
        "B_summary": summarize(B_vals),
        "A_summary": summarize(A_vals),
        "C_summary": summarize(C_vals),
        "normalized_coverage_stats": normalized_coverage_stats,
        "marginal_gain_stats": marginal_gain_stats,
        "per_object_examples": per_object[:50],
    }


# =========================
# Plotting
# =========================

def plot_normalized_coverage_curve(stats, save_path, title):
    n_list = stats["n_list"]
    means = [stats["normalized_coverage_stats"][str(n)]["mean"] for n in n_list]
    medians = [stats["normalized_coverage_stats"][str(n)]["median"] for n in n_list]
    p25s = [stats["normalized_coverage_stats"][str(n)]["p25"] for n in n_list]
    p75s = [stats["normalized_coverage_stats"][str(n)]["p75"] for n in n_list]

    plt.figure(figsize=(8, 5))
    plt.plot(n_list, means, marker="o", label="mean")
    plt.plot(n_list, medians, marker="s", label="median")
    plt.fill_between(n_list, p25s, p75s, alpha=0.2, label="p25-p75")
    plt.xlabel("candidate view set size N")
    plt.ylabel("normalized observable surface ratio Y(N)/Y(Nmax)")
    plt.title(title)
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI)
    plt.close()


def plot_marginal_gain_curve(stats, save_path, title):
    keys = []
    means = []
    medians = []

    for i in range(len(stats["n_list"]) - 1):
        k = f"{stats['n_list'][i]}->{stats['n_list'][i+1]}"
        keys.append(k)
        means.append(stats["marginal_gain_stats"][k]["mean"])
        medians.append(stats["marginal_gain_stats"][k]["median"])

    xs = list(range(len(keys)))

    plt.figure(figsize=(12, 5))
    plt.plot(xs, means, marker="o", label="mean")
    plt.plot(xs, medians, marker="s", label="median")
    plt.xticks(xs, keys, rotation=45, ha="right")
    plt.xlabel("adjacent view-set interval")
    plt.ylabel("marginal normalized gain")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI)
    plt.close()


def plot_marginal_gain_curve_large_mean(stats, save_path):
    keys = []
    means = []

    for i in range(len(stats["n_list"]) - 1):
        k = f"{stats['n_list'][i]}->{stats['n_list'][i+1]}"
        keys.append(k)
        means.append(stats["marginal_gain_stats"][k]["mean"])

    xs = list(range(len(keys)))
    tick_step = 2
    tick_xs = xs[::tick_step]
    tick_labels = keys[::tick_step]

    plt.figure(figsize=(14, 7))
    plt.plot(xs, means, marker="o", linewidth=3, markersize=9)
    plt.xticks(tick_xs, tick_labels, rotation=45, ha="right", fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("Adjacent View-Set Interval", fontsize=22, labelpad=12)
    plt.ylabel("Marginal Normalized Coverage Gain", fontsize=22, labelpad=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI)
    plt.close()


def plot_saturation_view_num_hist(items, save_path, title):
    xs = sorted(int(x["observation_saturation_view_num"]) for x in items)
    counts = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1

    plt.figure(figsize=(10, 5))
    plt.bar(list(counts.keys()), list(counts.values()), width=1.0)
    plt.xlabel("observation_saturation_view_num (B)")
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

    final_pool = load_json(FINAL_POOL_JSON)
    main_pool = load_json(MAIN_POOL_JSON)
    long_tail_pool = load_json(LONG_TAIL_POOL_JSON)

    saturation_by_uid = load_all_saturation_txts(TXT_DIR)

    final_items, final_missing = attach_saturation_info(final_pool, saturation_by_uid)
    main_items, main_missing = attach_saturation_info(main_pool, saturation_by_uid)
    long_tail_items, long_tail_missing = attach_saturation_info(long_tail_pool, saturation_by_uid)

    final_analysis = analyze_pool(final_items, N_LIST, CURVE_NAME)
    main_analysis = analyze_pool(main_items, N_LIST, CURVE_NAME)
    long_tail_analysis = analyze_pool(long_tail_items, N_LIST, CURVE_NAME)

    report = {
        "config": {
            "txt_dir": TXT_DIR,
            "output_dir": OUTPUT_DIR,
            "final_pool_json": FINAL_POOL_JSON,
            "main_pool_json": MAIN_POOL_JSON,
            "long_tail_pool_json": LONG_TAIL_POOL_JSON,
            "n_list": N_LIST,
            "curve_name": CURVE_NAME,
            "normalization_reference": f"Y({N_LIST[-1]})",
        },
        "summary": {
            "final_pool_num_objects": len(final_pool),
            "main_pool_num_objects": len(main_pool),
            "long_tail_pool_num_objects": len(long_tail_pool),
            "final_pool_missing_txt": final_missing,
            "main_pool_missing_txt": main_missing,
            "long_tail_pool_missing_txt": long_tail_missing,
        },
        "final_pool_saturation_partition": final_analysis,
        "main_pool_saturation_partition": main_analysis,
        "long_tail_pool_saturation_partition": long_tail_analysis,
    }

    save_json(report, os.path.join(OUTPUT_DIR, "filtered_pool_saturation_partition_report.json"))

    # overall
    plot_saturation_view_num_hist(
        final_items,
        os.path.join(OUTPUT_DIR, "final_pool_B_histogram.png"),
        f"Final clean pool B histogram | n={len(final_items)}",
    )
    plot_normalized_coverage_curve(
        final_analysis,
        os.path.join(OUTPUT_DIR, "final_pool_normalized_coverage_curve.png"),
        "Final clean pool: normalized observable surface gain",
    )
    plot_marginal_gain_curve(
        final_analysis,
        os.path.join(OUTPUT_DIR, "final_pool_marginal_gain_curve.png"),
        "Final clean pool: marginal normalized gain",
    )
    plot_marginal_gain_curve_large_mean(
        final_analysis,
        os.path.join(OUTPUT_DIR, "final_pool_marginal_gain_curve_large_mean.png"),
    )

    # main
    plot_saturation_view_num_hist(
        main_items,
        os.path.join(OUTPUT_DIR, "main_pool_B_histogram.png"),
        f"Main pool B histogram | n={len(main_items)}",
    )
    plot_normalized_coverage_curve(
        main_analysis,
        os.path.join(OUTPUT_DIR, "main_pool_normalized_coverage_curve.png"),
        "Main pool: normalized observable surface gain",
    )
    plot_marginal_gain_curve(
        main_analysis,
        os.path.join(OUTPUT_DIR, "main_pool_marginal_gain_curve.png"),
        "Main pool: marginal normalized gain",
    )

    # long-tail
    plot_saturation_view_num_hist(
        long_tail_items,
        os.path.join(OUTPUT_DIR, "long_tail_pool_B_histogram.png"),
        f"Long-tail pool B histogram | n={len(long_tail_items)}",
    )
    plot_normalized_coverage_curve(
        long_tail_analysis,
        os.path.join(OUTPUT_DIR, "long_tail_pool_normalized_coverage_curve.png"),
        "Long-tail pool: normalized observable surface gain",
    )
    plot_marginal_gain_curve(
        long_tail_analysis,
        os.path.join(OUTPUT_DIR, "long_tail_pool_marginal_gain_curve.png"),
        "Long-tail pool: marginal normalized gain",
    )

    print("=" * 80)
    print("Filtered-pool saturation partition analysis finished.")
    print(f"Final clean pool:   {len(final_items)}")
    print(f"Main pool:          {len(main_items)}")
    print(f"Long-tail pool:     {len(long_tail_items)}")
    print(f"Curve used:         {CURVE_NAME}")
    print(f"Normalization:      Y(N) / Y({N_LIST[-1]})")
    print(f"Saved to:           {OUTPUT_DIR}")
    if final_missing or main_missing or long_tail_missing:
        print("[WARN] Missing saturation txt for some uids.")
    print("=" * 80)


if __name__ == "__main__":
    main()
