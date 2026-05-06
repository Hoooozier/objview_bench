import json
import random
from collections import defaultdict, Counter
from pathlib import Path

import matplotlib.pyplot as plt


# =========================
# Config
# =========================

MAIN_POOL_JSON = Path("geometry_sampled/final_clean_pool/final_clean_main_pool.json")
SHOWCASE_UIDS_JSON = Path("geometry_sampled/paper_showcase_uids.json")
OUTPUT_DIR = Path("geometry_sampled/analysis_splits")

RANDOM_SEED = 42

ANALYSIS_TEST_ROUNDS = 3
BALANCED_TRAIN_ROUNDS = 21

ALLOW_TRAIN_OVERLAP = True

FIG_DPI = 220


# =========================
# IO helpers
# =========================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_showcase_uids(path: Path):
    data = load_json(path)
    if isinstance(data, dict):
        if "uids" in data and isinstance(data["uids"], list):
            return set(data["uids"])
        raise ValueError(f"Expected dict with key 'uids' in {path}")
    elif isinstance(data, list):
        return set(data)
    else:
        raise ValueError(f"Unsupported showcase uid format in {path}")


# =========================
# Sampling helpers
# =========================

def group_by_c(items):
    buckets = defaultdict(list)
    for item in items:
        c = item.get("selected_view_count")
        if c is None:
            continue
        buckets[int(c)].append(item)
    return buckets


def shuffle_buckets_inplace(buckets, rng):
    for c in buckets:
        rng.shuffle(buckets[c])


def round_robin_sample(items, rounds, rng):
    """
    Round-robin over non-empty C buckets.
    Each round: visit C buckets in ascending order, pop one if available.
    """
    buckets = group_by_c(items)
    shuffle_buckets_inplace(buckets, rng)

    cs_sorted = sorted(buckets.keys())
    out = []

    for _ in range(rounds):
        took_any = False
        for c in cs_sorted:
            if buckets[c]:
                out.append(buckets[c].pop())
                took_any = True
        if not took_any:
            break

    return out


def random_sample(items, k, rng):
    if k > len(items):
        raise ValueError(f"Cannot sample {k} items from only {len(items)} available items.")
    return rng.sample(items, k)


def filter_items(items, forbidden_uids):
    forbidden_uids = set(forbidden_uids)
    return [x for x in items if x["uid"] not in forbidden_uids]


def count_c(items):
    counter = Counter()
    for item in items:
        c = item.get("selected_view_count")
        if c is not None:
            counter[int(c)] += 1
    return counter


def compute_proportional_quotas_exact(items, target_size):
    """
    Allocate exact integer quotas per C bucket such that:
      - total quota == target_size
      - quotas approximate the original C distribution as closely as possible

    Uses largest remainder (Hamilton apportionment):
      1. compute exact fractional quotas
      2. take floor
      3. distribute remaining seats to largest fractional parts
    """
    counter = count_c(items)
    total_n = len(items)

    if target_size > total_n:
        raise ValueError(f"target_size={target_size} exceeds available items={total_n}")

    cs = sorted(counter.keys())

    exact = {c: counter[c] / total_n * target_size for c in cs}
    base = {c: int(exact[c]) for c in cs}  # floor
    used = sum(base.values())
    remain = target_size - used

    remainders = sorted(
        [(exact[c] - base[c], c) for c in cs],
        reverse=True
    )

    quotas = dict(base)
    for _, c in remainders[:remain]:
        quotas[c] += 1

    return quotas


def sample_by_bucket_quotas(items, quotas, rng):
    buckets = group_by_c(items)
    sampled = []

    for c in sorted(quotas.keys()):
        bucket = list(buckets[c])
        rng.shuffle(bucket)

        k = quotas[c]
        if k > len(bucket):
            raise ValueError(
                f"Bucket C={c} has only {len(bucket)} items but quota asks for {k}."
            )

        sampled.extend(bucket[:k])

    return sampled


# =========================
# Plot helpers
# =========================

def summarize_numeric_list(vals):
    vals = list(vals)
    if not vals:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p25": None,
            "p75": None,
        }

    vals_sorted = sorted(vals)
    n = len(vals_sorted)

    def percentile(q):
        if q <= 0:
            return vals_sorted[0]
        if q >= 1:
            return vals_sorted[-1]
        pos = (n - 1) * q
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        w = pos - lo
        return vals_sorted[lo] * (1.0 - w) + vals_sorted[hi] * w

    return {
        "count": n,
        "mean": sum(vals_sorted) / n,
        "median": vals_sorted[n // 2] if n % 2 == 1 else (vals_sorted[n // 2 - 1] + vals_sorted[n // 2]) / 2,
        "min": vals_sorted[0],
        "max": vals_sorted[-1],
        "p25": percentile(0.25),
        "p75": percentile(0.75),
    }


def build_c_summary(items):
    c_vals = [int(x["selected_view_count"]) for x in items if x.get("selected_view_count") is not None]
    counter = Counter(c_vals)
    return {
        "summary": summarize_numeric_list(c_vals),
        "histogram": {str(k): counter[k] for k in sorted(counter.keys())},
    }


def split_analysis_test_by_quantile(items):
    """
    Step 1:
      sort by C and split by rank into 3 tertiles

    Step 2:
      convert them into mutually exclusive contiguous C intervals
      by assigning boundary-tied C values to the earlier group
    """
    valid_items = [x for x in items if x.get("selected_view_count") is not None]
    sorted_items = sorted(valid_items, key=lambda x: (int(x["selected_view_count"]), x["uid"]))

    n = len(sorted_items)
    i1 = n // 3
    i2 = (2 * n) // 3

    # initial rank-based split
    low_rank = sorted_items[:i1]
    mid_rank = sorted_items[i1:i2]
    high_rank = sorted_items[i2:]

    if not low_rank or not mid_rank or not high_rank:
        raise ValueError("Analysis test is too small to split into low/mid/high.")

    low_max_c = max(int(x["selected_view_count"]) for x in low_rank)
    mid_max_c = max(int(x["selected_view_count"]) for x in mid_rank)

    # enforce strict intervals:
    # low:  C <= low_max_c
    # mid:  low_max_c < C <= mid_max_c
    # high: C > mid_max_c
    low_items = []
    mid_items = []
    high_items = []

    for x in sorted_items:
        c = int(x["selected_view_count"])
        if c <= low_max_c:
            low_items.append(x)
        elif c <= mid_max_c:
            mid_items.append(x)
        else:
            high_items.append(x)

    def c_range(xs):
        if not xs:
            return {"min_c": None, "max_c": None}
        c_vals = [int(x["selected_view_count"]) for x in xs]
        return {"min_c": min(c_vals), "max_c": max(c_vals)}

    return {
        "sizes": {
            "total": n,
            "low": len(low_items),
            "mid": len(mid_items),
            "high": len(high_items),
        },
        "ranges": {
            "low": c_range(low_items),
            "mid": c_range(mid_items),
            "high": c_range(high_items),
        },
        "thresholds": {
            "low_max_c": low_max_c,
            "mid_max_c": mid_max_c,
            "high_min_c": mid_max_c + 1,
        },
        "uids": {
            "low": [x["uid"] for x in low_items],
            "mid": [x["uid"] for x in mid_items],
            "high": [x["uid"] for x in high_items],
        },
        "full_records": {
            "low": low_items,
            "mid": mid_items,
            "high": high_items,
        },
    }


def plot_c_histograms(main_items, raw_train, balanced_train, analysis_test, save_path: Path):
    main_counter = count_c(main_items)
    raw_counter = count_c(raw_train)
    bal_counter = count_c(balanced_train)
    test_counter = count_c(analysis_test)

    all_cs = sorted(set(main_counter.keys()) | set(raw_counter.keys()) | set(bal_counter.keys()) | set(test_counter.keys()))
    xs = list(range(len(all_cs)))
    labels = [str(c) for c in all_cs]

    main_vals = [main_counter.get(c, 0) for c in all_cs]
    raw_vals = [raw_counter.get(c, 0) for c in all_cs]
    bal_vals = [bal_counter.get(c, 0) for c in all_cs]
    test_vals = [test_counter.get(c, 0) for c in all_cs]

    plt.figure(figsize=(16, 8))
    plt.plot(xs, main_vals, marker="o", label=f"main_pool | n={len(main_items)}")
    plt.plot(xs, raw_vals, marker="s", label=f"Main-Raw-Train | n={len(raw_train)}")
    plt.plot(xs, bal_vals, marker="^", label=f"Main-Balanced-Train | n={len(balanced_train)}")
    plt.plot(xs, test_vals, marker="d", label=f"Main-Analysis-Test | n={len(analysis_test)}")

    tick_step = max(1, len(xs) // 25)
    plt.xticks(xs[::tick_step], labels[::tick_step], rotation=45, ha="right")
    plt.xlabel("C (selected_view_count)")
    plt.ylabel("count")
    plt.title("C difficulty distribution: main pool vs analysis splits")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI)
    plt.close()


# =========================
# Main
# =========================

def main():
    rng = random.Random(RANDOM_SEED)

    main_items = load_json(MAIN_POOL_JSON)
    showcase_uids = load_showcase_uids(SHOWCASE_UIDS_JSON)

    if not isinstance(main_items, list):
        raise ValueError(f"Expected list in {MAIN_POOL_JSON}")

    main_by_uid = {x["uid"]: x for x in main_items}
    missing_showcase = sorted(uid for uid in showcase_uids if uid not in main_by_uid)
    if missing_showcase:
        print(f"[WARN] {len(missing_showcase)} showcase uids not found in main pool.")

    ensure_dir(OUTPUT_DIR)

    # -------------------------------------------------
    # Step 1: Main-Analysis-Test
    # forbid showcase uids
    # -------------------------------------------------
    analysis_test_candidates = filter_items(main_items, showcase_uids)
    main_analysis_test = round_robin_sample(
        analysis_test_candidates,
        rounds=ANALYSIS_TEST_ROUNDS,
        rng=rng,
    )
    analysis_test_uids = {x["uid"] for x in main_analysis_test}

    # -------------------------------------------------
    # Step 2: Main-Balanced-Train
    # forbid test, but do NOT forbid showcase uids
    # -------------------------------------------------
    balanced_train_candidates = filter_items(main_items, analysis_test_uids)
    main_balanced_train = round_robin_sample(
        balanced_train_candidates,
        rounds=BALANCED_TRAIN_ROUNDS,
        rng=rng,
    )
    balanced_train_uids = {x["uid"] for x in main_balanced_train}

    # -------------------------------------------------
    # Step 3: Main-Raw-Train
    # same size as balanced train
    # forbid test
    # overlap with balanced train configurable
    # -------------------------------------------------
    raw_train_candidates = filter_items(main_items, analysis_test_uids)

    if not ALLOW_TRAIN_OVERLAP:
        raw_train_candidates = filter_items(raw_train_candidates, balanced_train_uids)

    raw_train_quotas = compute_proportional_quotas_exact(
        raw_train_candidates,
        target_size=len(main_balanced_train),
    )

    main_raw_train = sample_by_bucket_quotas(
        raw_train_candidates,
        quotas=raw_train_quotas,
        rng=rng,
    )
    raw_train_uids = {x["uid"] for x in main_raw_train}

    # -------------------------------------------------
    # Extra summaries
    # -------------------------------------------------
    main_pool_c_summary = build_c_summary(main_items)
    analysis_test_c_summary = build_c_summary(main_analysis_test)
    balanced_train_c_summary = build_c_summary(main_balanced_train)
    raw_train_c_summary = build_c_summary(main_raw_train)

    analysis_test_quantile_split = split_analysis_test_by_quantile(main_analysis_test)

    # -------------------------------------------------
    # Save splits
    # -------------------------------------------------
    save_json(main_analysis_test, OUTPUT_DIR / "main_analysis_test.json")
    save_json(main_balanced_train, OUTPUT_DIR / "main_balanced_train.json")
    save_json(main_raw_train, OUTPUT_DIR / "main_raw_train.json")

    meta = {
        "config": {
            "main_pool_json": str(MAIN_POOL_JSON),
            "showcase_uids_json": str(SHOWCASE_UIDS_JSON),
            "random_seed": RANDOM_SEED,
            "analysis_test_rounds": ANALYSIS_TEST_ROUNDS,
            "balanced_train_rounds": BALANCED_TRAIN_ROUNDS,
            "allow_train_overlap": ALLOW_TRAIN_OVERLAP,
            "raw_train_sampling_rule": (
                "sample by exact proportional C-bucket quotas "
                "(largest remainder), matching the natural C distribution "
                "of the raw-train candidate pool"
            ),
        },
        "summary": {
            "main_pool_size": len(main_items),
            "showcase_uid_count": len(showcase_uids),
            "main_analysis_test_size": len(main_analysis_test),
            "main_balanced_train_size": len(main_balanced_train),
            "main_raw_train_size": len(main_raw_train),
            "raw_balanced_overlap_size": len(raw_train_uids & balanced_train_uids),
        },
        "forbidden": {
            "analysis_test_forbid_showcase_uids": True,
            "balanced_train_forbid_analysis_test": True,
            "raw_train_forbid_analysis_test": True,
            "raw_train_forbid_balanced_train": not ALLOW_TRAIN_OVERLAP,
        },
    }
    save_json(meta, OUTPUT_DIR / "analysis_split_meta.json")

    save_json(main_pool_c_summary, OUTPUT_DIR / "main_pool_c_summary.json")
    save_json(raw_train_c_summary, OUTPUT_DIR / "main_raw_train_c_summary.json")
    save_json(balanced_train_c_summary, OUTPUT_DIR / "main_balanced_train_c_summary.json")
    save_json(analysis_test_c_summary, OUTPUT_DIR / "main_analysis_test_c_summary.json")

    save_json(
        {str(c): raw_train_quotas[c] for c in sorted(raw_train_quotas.keys())},
        OUTPUT_DIR / "main_raw_train_c_quotas.json"
    )

    save_json(
        {
            "sizes": analysis_test_quantile_split["sizes"],
            "ranges": analysis_test_quantile_split["ranges"],
            "thresholds": analysis_test_quantile_split["thresholds"],
            "uids": analysis_test_quantile_split["uids"],
        },
        OUTPUT_DIR / "main_analysis_test_low_mid_high_split.json"
    )

    save_json(
        analysis_test_quantile_split["full_records"]["low"],
        OUTPUT_DIR / "main_analysis_test_low.json"
    )
    save_json(
        analysis_test_quantile_split["full_records"]["mid"],
        OUTPUT_DIR / "main_analysis_test_mid.json"
    )
    save_json(
        analysis_test_quantile_split["full_records"]["high"],
        OUTPUT_DIR / "main_analysis_test_high.json"
    )

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    plot_c_histograms(
        main_items=main_items,
        raw_train=main_raw_train,
        balanced_train=main_balanced_train,
        analysis_test=main_analysis_test,
        save_path=OUTPUT_DIR / "c_distribution_main_vs_analysis_splits.png",
    )

    print("=" * 80)
    print("Analysis splits generated.")
    print(f"Main pool size:            {len(main_items)}")
    print(f"Showcase uid count:        {len(showcase_uids)}")
    print(f"Main-Analysis-Test size:   {len(main_analysis_test)}  (rounds={ANALYSIS_TEST_ROUNDS})")
    print(f"Main-Balanced-Train size:  {len(main_balanced_train)} (rounds={BALANCED_TRAIN_ROUNDS})")
    print(f"Main-Raw-Train size:       {len(main_raw_train)}")
    print(f"Raw/Balanced overlap size: {len(raw_train_uids & balanced_train_uids)}")
    print(f"Saved to:                  {OUTPUT_DIR}")
    print("=" * 80)

    print("Analysis-Test low/mid/high sizes:")
    print(
        f"  low={analysis_test_quantile_split['sizes']['low']}, "
        f"mid={analysis_test_quantile_split['sizes']['mid']}, "
        f"high={analysis_test_quantile_split['sizes']['high']}"
    )
    print("Analysis-Test low/mid/high C ranges:")
    print(
        f"  low=[{analysis_test_quantile_split['ranges']['low']['min_c']}, "
        f"{analysis_test_quantile_split['ranges']['low']['max_c']}], "
        f"mid=[{analysis_test_quantile_split['ranges']['mid']['min_c']}, "
        f"{analysis_test_quantile_split['ranges']['mid']['max_c']}], "
        f"high=[{analysis_test_quantile_split['ranges']['high']['min_c']}, "
        f"{analysis_test_quantile_split['ranges']['high']['max_c']}]"
    )
    print("Analysis-Test strict thresholds:")
    print(
        f"  low:  C <= {analysis_test_quantile_split['thresholds']['low_max_c']}, "
        f"mid: {analysis_test_quantile_split['thresholds']['low_max_c'] + 1} <= C <= {analysis_test_quantile_split['thresholds']['mid_max_c']}, "
        f"high: C >= {analysis_test_quantile_split['thresholds']['high_min_c']}"
    )


if __name__ == "__main__":
    main()