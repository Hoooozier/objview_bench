import json
import math
import random
from collections import defaultdict, Counter
from pathlib import Path

import matplotlib.pyplot as plt


# =========================
# Config
# =========================

MAIN_POOL_JSON = Path("geometry_sampled/final_clean_pool/final_clean_main_pool.json")
SHOWCASE_UIDS_JSON = Path("geometry_sampled/paper_showcase_uids.json")
OUTPUT_DIR = Path("geometry_sampled/released_benchmark_splits")

RANDOM_SEED = 42
TARGET_HIDDEN_TEST_SIZE = 500

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
# Helpers
# =========================

def filter_items(items, forbidden_uids):
    forbidden_uids = set(forbidden_uids)
    return [x for x in items if x["uid"] not in forbidden_uids]


def group_by_c(items):
    buckets = defaultdict(list)
    for item in items:
        c = item.get("selected_view_count")
        if c is None:
            continue
        buckets[int(c)].append(item)
    return buckets


def count_c(items):
    counter = Counter()
    for item in items:
        c = item.get("selected_view_count")
        if c is not None:
            counter[int(c)] += 1
    return counter


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


def compute_bucket_quotas(candidate_items, target_size):
    """
    For each non-empty C bucket:
        quota_c = max(1, round(n_c / N * target_size))
    """
    counter = count_c(candidate_items)
    total_n = len(candidate_items)

    quotas = {}
    for c in sorted(counter.keys()):
        q = counter[c] / total_n * target_size
        quotas[c] = max(1, round(q))
    return quotas


def sample_by_bucket_quotas(candidate_items, quotas, rng):
    buckets = group_by_c(candidate_items)
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
# Plot
# =========================

def plot_c_histograms(main_items, hidden_test, train_public, save_path: Path):
    main_counter = count_c(main_items)
    hidden_counter = count_c(hidden_test)
    train_counter = count_c(train_public)

    all_cs = sorted(set(main_counter.keys()) | set(hidden_counter.keys()) | set(train_counter.keys()))
    xs = list(range(len(all_cs)))
    labels = [str(c) for c in all_cs]

    main_vals = [main_counter.get(c, 0) for c in all_cs]
    hidden_vals = [hidden_counter.get(c, 0) for c in all_cs]
    train_vals = [train_counter.get(c, 0) for c in all_cs]

    plt.figure(figsize=(16, 8))
    plt.plot(xs, main_vals, marker="o", label=f"main_pool | n={len(main_items)}")
    plt.plot(xs, hidden_vals, marker="d", label=f"Main-Hidden-Test | n={len(hidden_test)}")
    plt.plot(xs, train_vals, marker="s", label=f"Main-Train-Public | n={len(train_public)}")

    tick_step = max(1, len(xs) // 25)
    plt.xticks(xs[::tick_step], labels[::tick_step], rotation=45, ha="right")
    plt.xlabel("C (selected_view_count)")
    plt.ylabel("count")
    plt.title("C difficulty distribution: main pool vs released benchmark splits")
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
    # Step 1: forbid showcase uids from hidden test
    # -------------------------------------------------
    hidden_test_candidates = filter_items(main_items, showcase_uids)

    # -------------------------------------------------
    # Step 2: compute per-bucket quotas
    # -------------------------------------------------
    quotas = compute_bucket_quotas(
        candidate_items=hidden_test_candidates,
        target_size=TARGET_HIDDEN_TEST_SIZE,
    )

    # -------------------------------------------------
    # Step 3: sample hidden test by quotas
    # -------------------------------------------------
    main_hidden_test = sample_by_bucket_quotas(
        candidate_items=hidden_test_candidates,
        quotas=quotas,
        rng=rng,
    )
    hidden_test_uids = {x["uid"] for x in main_hidden_test}

    # -------------------------------------------------
    # Step 4: public train = all remaining main pool items
    # -------------------------------------------------
    main_train_public = filter_items(main_items, hidden_test_uids)
    train_public_uids = {x["uid"] for x in main_train_public}

    # sanity check: showcase uids should remain in public train if they exist in main pool
    showcase_in_main = {uid for uid in showcase_uids if uid in main_by_uid}
    leaked_showcase = sorted(uid for uid in showcase_in_main if uid not in train_public_uids)
    if leaked_showcase:
        raise ValueError(f"Some showcase uids are missing from public train: {leaked_showcase}")

    # -------------------------------------------------
    # Save splits
    # -------------------------------------------------
    save_json(main_hidden_test, OUTPUT_DIR / "main_hidden_test.json")
    save_json(main_train_public, OUTPUT_DIR / "main_train_public.json")

    main_pool_c_summary = build_c_summary(main_items)
    hidden_test_c_summary = build_c_summary(main_hidden_test)
    train_public_c_summary = build_c_summary(main_train_public)

    save_json(main_pool_c_summary, OUTPUT_DIR / "main_pool_c_summary.json")
    save_json(hidden_test_c_summary, OUTPUT_DIR / "main_hidden_test_c_summary.json")
    save_json(train_public_c_summary, OUTPUT_DIR / "main_train_public_c_summary.json")

    save_json(
        {str(c): quotas[c] for c in sorted(quotas.keys())},
        OUTPUT_DIR / "main_hidden_test_c_quotas.json"
    )

    meta = {
        "config": {
            "main_pool_json": str(MAIN_POOL_JSON),
            "showcase_uids_json": str(SHOWCASE_UIDS_JSON),
            "random_seed": RANDOM_SEED,
            "target_hidden_test_size": TARGET_HIDDEN_TEST_SIZE,
            "hidden_test_sampling_rule": (
                "For each non-empty C bucket, sample "
                "max(1, round(n_c / N * target_hidden_test_size)) objects, "
                "where N is the hidden-test candidate pool size after excluding showcase uids."
            ),
        },
        "summary": {
            "main_pool_size": len(main_items),
            "showcase_uid_count": len(showcase_uids),
            "hidden_test_candidate_size": len(hidden_test_candidates),
            "actual_hidden_test_size": len(main_hidden_test),
            "main_train_public_size": len(main_train_public),
        },
        "forbidden": {
            "hidden_test_forbid_showcase_uids": True,
        },
        "showcase_check": {
            "showcase_in_main_pool_count": len(showcase_in_main),
            "all_showcase_uids_remain_in_public_train": len(leaked_showcase) == 0,
        },
    }
    save_json(meta, OUTPUT_DIR / "released_benchmark_meta.json")

    # -------------------------------------------------
    # Plot
    # -------------------------------------------------
    plot_c_histograms(
        main_items=main_items,
        hidden_test=main_hidden_test,
        train_public=main_train_public,
        save_path=OUTPUT_DIR / "c_distribution_main_vs_released_benchmark.png",
    )

    print("=" * 80)
    print("Released benchmark splits generated.")
    print(f"Main pool size:             {len(main_items)}")
    print(f"Showcase uid count:         {len(showcase_uids)}")
    print(f"Hidden-test candidate size: {len(hidden_test_candidates)}")
    print(f"Target hidden-test size:    {TARGET_HIDDEN_TEST_SIZE}")
    print(f"Actual hidden-test size:    {len(main_hidden_test)}")
    print(f"Main-Train-Public size:     {len(main_train_public)}")
    print(f"Saved to:                   {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()