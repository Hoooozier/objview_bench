#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Three-stage geometry filtering and stratified sampling for mesh annotation jsonl.

Updated features:
1. Read multiple part files such as:
   geometry_parts/geometry_annotations_part_000000_000100.jsonl
   geometry_parts/geometry_annotations_part_000100_000200.jsonl
   ...
2. Merge all records first, then run three-stage filtering.

Stage 1: hard filtering
Stage 2: statistical filtering (lightweight anomaly screening)
         - area-related anomaly signals
         - signed Euler anomaly signal
Stage 3: build two stratification fields
         - shape_type
         - fill_bucket

3. Output an extra json for final stratified dataset:
   only UID + shape_type + fill_bucket
4. Output an extra json for sampled dataset as well.
5. summary keeps unknown buckets, but sampling excludes them and only uses valid 4x3 strata.
"""

import os
import json
import glob
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd


EPS = 1e-12


VALID_SHAPE_TYPES = {
    "bladed",
    "elongated",
    "flat_or_sheet_like",
    "compact",
}

VALID_FILL_BUCKETS = {
    "low",
    "mid",
    "high",
}


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def safe_bool(x, default=False):
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    return bool(x)


def sorted_bbox_extents(extent) -> Tuple[float, float, float]:
    if not isinstance(extent, (list, tuple)) or len(extent) != 3:
        return (np.nan, np.nan, np.nan)
    vals = [safe_float(v, np.nan) for v in extent]
    vals = sorted(vals, reverse=True)
    return vals[0], vals[1], vals[2]


def robust_zscore(series: pd.Series) -> pd.Series:
    """
    Robust z-score using median and MAD:
        z = (x - median) / (1.4826 * MAD)
    Keeps sign.
    """
    s = pd.to_numeric(series, errors="coerce")
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad < EPS:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - med) / (1.4826 * mad + EPS)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Skip bad json at line {line_idx} in {path}: {e}")
    return rows


def read_jsonl_parts(input_pattern: str) -> List[Dict[str, Any]]:
    paths = sorted(glob.glob(input_pattern))
    if len(paths) == 0:
        raise FileNotFoundError(f"No input files matched pattern: {input_pattern}")

    all_rows = []
    for p in paths:
        rows = read_jsonl(p)
        print(f"[INFO] Loaded {len(rows)} rows from: {p}")
        all_rows.extend(rows)

    print(f"[INFO] Total merged rows: {len(all_rows)} from {len(paths)} files")
    return all_rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def qbucket(series: pd.Series, labels=("low", "mid", "high")) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()

    if len(valid) < 3:
        return pd.Series(["unknown"] * len(s), index=s.index)

    try:
        q1, q2 = valid.quantile([1 / 3, 2 / 3]).values
    except Exception:
        return pd.Series(["unknown"] * len(s), index=s.index)

    out = []
    for v in s:
        if pd.isna(v):
            out.append("unknown")
        elif v <= q1:
            out.append(labels[0])
        elif v <= q2:
            out.append(labels[1])
        else:
            out.append(labels[2])
    return pd.Series(out, index=s.index)


def build_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)

    required_cols = [
        "UID",
        "load_ok",
        "error",
        "vertex_count",
        "face_count",
        "non_empty",
        "finite_vertices",
        "valid_face_indices",
        "is_winding_consistent",
        "is_watertight",
        "bbox_extent",
        "euler_number",
        "degenerate_face_ratio",
        "area",
        "volume",
        "convex_hull_area",
        "convex_hull_volume",
    ]
    for c in required_cols:
        if c not in df.columns:
            df[c] = np.nan

    bool_cols = [
        "load_ok",
        "non_empty",
        "finite_vertices",
        "valid_face_indices",
        "is_winding_consistent",
        "is_watertight",
    ]
    for c in bool_cols:
        df[c] = df[c].apply(lambda x: safe_bool(x, default=False))

    num_cols = [
        "vertex_count",
        "face_count",
        "euler_number",
        "degenerate_face_ratio",
        "area",
        "volume",
        "convex_hull_area",
        "convex_hull_volume",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    bbox_sorted = df["bbox_extent"].apply(sorted_bbox_extents)
    df["bbox_e1"] = bbox_sorted.apply(lambda x: x[0])
    df["bbox_e2"] = bbox_sorted.apply(lambda x: x[1])
    df["bbox_e3"] = bbox_sorted.apply(lambda x: x[2])

    df["elongation"] = df["bbox_e1"] / (df["bbox_e2"] + EPS)
    df["planarity"] = df["bbox_e2"] / (df["bbox_e3"] + EPS)

    df["abs_volume"] = df["volume"].abs()

    df.loc[df["area"] < 0, "area"] = np.nan
    df.loc[df["convex_hull_area"] < 0, "convex_hull_area"] = np.nan
    df.loc[df["convex_hull_volume"] < 0, "convex_hull_volume"] = np.nan

    valid_hull_vol = df["convex_hull_volume"] > EPS
    valid_hull_area = df["convex_hull_area"] > EPS

    df["fill_ratio"] = np.where(
        valid_hull_vol,
        df["abs_volume"] / (df["convex_hull_volume"] + EPS),
        np.nan
    )
    df["hull_area_ratio"] = np.where(
        valid_hull_area,
        df["area"] / (df["convex_hull_area"] + EPS),
        np.nan
    )

    df.loc[df["fill_ratio"] < 0, "fill_ratio"] = np.nan
    df.loc[df["hull_area_ratio"] < 0, "hull_area_ratio"] = np.nan

    bbox_area = 2.0 * (
        df["bbox_e1"] * df["bbox_e2"] +
        df["bbox_e1"] * df["bbox_e3"] +
        df["bbox_e2"] * df["bbox_e3"]
    )
    bbox_volume = df["bbox_e1"] * df["bbox_e2"] * df["bbox_e3"]

    valid_bbox_area = bbox_area > EPS
    valid_bbox_volume = bbox_volume > EPS

    df["bbox_area_ratio"] = np.where(
        valid_bbox_area,
        df["area"] / (bbox_area + EPS),
        np.nan
    )
    df["bbox_volume_ratio"] = np.where(
        valid_bbox_volume,
        df["abs_volume"] / (bbox_volume + EPS),
        np.nan
    )

    df.loc[df["bbox_area_ratio"] < 0, "bbox_area_ratio"] = np.nan
    df.loc[df["bbox_volume_ratio"] < 0, "bbox_volume_ratio"] = np.nan

    df["log_bbox_area_ratio"] = np.log1p(df["bbox_area_ratio"].clip(lower=0))
    df["log_hull_area_ratio"] = np.log1p(df["hull_area_ratio"].clip(lower=0))

    # signed topology cue
    df["euler_signed_norm"] = df["euler_number"] / (np.log1p(df["face_count"].clip(lower=0)) + EPS)

    return df


def hard_filter(df: pd.DataFrame, args) -> pd.DataFrame:
    fail_reasons = []

    for _, row in df.iterrows():
        reasons = []

        if not row["load_ok"]:
            reasons.append("load_failed")

        err = row["error"]
        if pd.notna(err):
            err_str = str(err).strip().lower()
            if err_str not in ("", "none", "null", "nan"):
                reasons.append("non_null_error")

        if not row["non_empty"]:
            reasons.append("empty_mesh")
        if not row["finite_vertices"]:
            reasons.append("non_finite_vertices")
        if not row["valid_face_indices"]:
            reasons.append("invalid_face_indices")

        if not row["is_winding_consistent"]:
            reasons.append("winding_inconsistent")

        vc = row["vertex_count"]
        fc = row["face_count"]
        deg = row["degenerate_face_ratio"]

        if pd.isna(vc) or vc < args.min_vertex_count:
            reasons.append("too_few_vertices")
        if pd.isna(fc) or fc < args.min_face_count:
            reasons.append("too_few_faces")

        if pd.notna(vc) and vc > args.max_vertex_count:
            reasons.append("too_many_vertices")
        if pd.notna(fc) and fc > args.max_face_count:
            reasons.append("too_many_faces")

        if pd.isna(deg) or deg > args.max_degenerate_face_ratio:
            reasons.append("high_degenerate_face_ratio")

        fail_reasons.append(reasons)

    df = df.copy()
    df["hard_fail_reasons"] = fail_reasons
    df["hard_pass"] = df["hard_fail_reasons"].apply(lambda x: len(x) == 0)
    return df


def statistical_filter(df: pd.DataFrame, args) -> pd.DataFrame:
    """
    Statistical filtering after hard filtering.

    Signals:
        - log_bbox_area_ratio
        - log_hull_area_ratio
        - euler_signed_norm  (signed robust z-score)

    Filtering rule:
        reject if any enabled signal is abnormal.
    """
    df = df.copy()

    df["rz_log_bbox_area_ratio"] = robust_zscore(df["log_bbox_area_ratio"])
    df["rz_log_hull_area_ratio"] = robust_zscore(df["log_hull_area_ratio"])
    df["rz_euler_signed_norm"] = robust_zscore(df["euler_signed_norm"])

    df["flag_bbox_area_ratio"] = df["rz_log_bbox_area_ratio"].abs().fillna(np.inf) > args.tau_area
    df["flag_hull_area_ratio"] = df["rz_log_hull_area_ratio"].abs().fillna(np.inf) > args.tau_hull_area
    df["flag_euler_signed_norm"] = df["rz_euler_signed_norm"].abs().fillna(np.inf) > args.tau_euler

    flag_cols = [
        "flag_bbox_area_ratio",
        "flag_hull_area_ratio",
        "flag_euler_signed_norm",
    ]
    df["stat_anomaly_count"] = df[flag_cols].sum(axis=1)
    df["stat_pass"] = ~df[flag_cols].any(axis=1)

    return df


def stratification_buckets(df: pd.DataFrame, args) -> pd.DataFrame:
    """
    Build two stratification fields:
        1. shape_type (4-way)
        2. fill_bucket
    """
    df = df.copy()

    def shape_type(row):
        el = row["elongation"]
        pl = row["planarity"]

        if pd.isna(el) or pd.isna(pl):
            return "unknown"

        el_thr = args.shape_elongation_thr
        pl_thr = args.shape_planarity_thr

        if el >= el_thr and pl >= pl_thr:
            return "bladed"
        elif el >= el_thr and pl < pl_thr:
            return "elongated"
        elif el < el_thr and pl >= pl_thr:
            return "flat_or_sheet_like"
        else:
            return "compact"

    df["shape_type"] = df.apply(shape_type, axis=1)
    df["fill_bucket"] = qbucket(df["fill_ratio"], labels=("low", "mid", "high"))

    return df


def filter_valid_12_strata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only valid 4x3 strata:
        shape_type in {bladed, elongated, flat_or_sheet_like, compact}
        fill_bucket in {low, mid, high}
    """
    df = df.copy()
    keep_mask = (
        df["shape_type"].isin(VALID_SHAPE_TYPES) &
        df["fill_bucket"].isin(VALID_FILL_BUCKETS)
    )
    return df[keep_mask].copy()


def balanced_sample_12_buckets(
    df: pd.DataFrame,
    target_n: int = 10000,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Balanced sampling over valid 12 strata:
        shape_type x fill_bucket

    Assumes input df already excludes unknown buckets.
    """
    rng = np.random.RandomState(random_seed)
    df = df.copy()

    # extra safety: force 12-strata only
    df = filter_valid_12_strata(df)

    bucket_cols = ["shape_type", "fill_bucket"]
    df["stratum_key"] = df[bucket_cols].astype(str).agg("|".join, axis=1)

    groups = {k: g.copy() for k, g in df.groupby("stratum_key")}
    nonempty_keys = [k for k, g in groups.items() if len(g) > 0]

    if len(nonempty_keys) == 0:
        return df.iloc[0:0].copy()

    remaining_keys = set(nonempty_keys)
    selected_parts = []
    remaining_quota = min(target_n, len(df))

    while remaining_quota > 0 and len(remaining_keys) > 0:
        per_group_quota = max(1, remaining_quota // len(remaining_keys))
        newly_completed = set()

        for key in list(remaining_keys):
            g = groups[key]
            n_avail = len(g)

            if n_avail <= per_group_quota:
                selected_parts.append(g)
                remaining_quota -= n_avail
                newly_completed.add(key)
            else:
                sampled_idx = rng.choice(g.index.to_numpy(), size=per_group_quota, replace=False)
                sampled = g.loc[sampled_idx]
                leftover = g.drop(sampled_idx)

                selected_parts.append(sampled)
                groups[key] = leftover
                remaining_quota -= per_group_quota

                if len(leftover) == 0:
                    newly_completed.add(key)

            if remaining_quota <= 0:
                break

        remaining_keys -= newly_completed

    sampled_df = pd.concat(selected_parts, axis=0) if len(selected_parts) > 0 else df.iloc[0:0].copy()

    if len(sampled_df) > target_n:
        sampled_df = sampled_df.sample(n=target_n, random_state=random_seed)

    sampled_df = sampled_df.drop(columns=["stratum_key"], errors="ignore")
    return sampled_df


def summarize(
    df_all: pd.DataFrame,
    df_hard: pd.DataFrame,
    df_kept_all: pd.DataFrame,
    df_kept_12: pd.DataFrame
) -> Dict[str, Any]:
    """
    Summary keeps unknown buckets in the all-stat-passed view,
    while also reporting the valid 12-strata view used for sampling.
    """
    summary = {
        "num_total": int(len(df_all)),
        "num_after_hard_filter": int(len(df_hard)),
        "num_after_stat_filter": int(len(df_kept_all)),
        "num_after_valid_12_strata": int(len(df_kept_12)),
        "num_unknown_strata_removed_before_sampling": int(len(df_kept_all) - len(df_kept_12)),
        "hard_filter_keep_ratio": float(len(df_hard) / max(len(df_all), 1)),
        "stat_filter_keep_ratio": float(len(df_kept_all) / max(len(df_hard), 1)),
        "valid_12_strata_keep_ratio_from_stat": float(len(df_kept_12) / max(len(df_kept_all), 1)),
    }

    if len(df_kept_all) > 0:
        summary["shape_type_distribution"] = df_kept_all["shape_type"].value_counts(dropna=False).to_dict()
        summary["fill_bucket_distribution"] = df_kept_all["fill_bucket"].value_counts(dropna=False).to_dict()

        joint_counts_all = df_kept_all.groupby(["shape_type", "fill_bucket"]).size().to_dict()
        summary["shape_fill_joint_distribution"] = {
            f"{k[0]}|{k[1]}": int(v)
            for k, v in joint_counts_all.items()
        }

    if len(df_kept_12) > 0:
        joint_counts_valid = df_kept_12.groupby(["shape_type", "fill_bucket"]).size().to_dict()
        summary["shape_fill_joint_distribution_valid_12"] = {
            f"{k[0]}|{k[1]}": int(v)
            for k, v in joint_counts_valid.items()
        }

    return summary


def main():
    parser = argparse.ArgumentParser()

    # input: pattern for multiple part files
    parser.add_argument(
        "--input_pattern",
        type=str,
        default="geometry_parts/geometry_annotations_part_*.jsonl",
        help="Glob pattern for input jsonl part files"
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")

    # stage 1 hard filter thresholds
    parser.add_argument("--min_vertex_count", type=int, default=200)
    parser.add_argument("--max_vertex_count", type=int, default=200000)
    parser.add_argument("--min_face_count", type=int, default=200)
    parser.add_argument("--max_face_count", type=int, default=200000)
    parser.add_argument("--max_degenerate_face_ratio", type=float, default=0.02)

    # stage 2 stat filter thresholds
    parser.add_argument("--tau_area", type=float, default=2.0)
    parser.add_argument("--tau_hull_area", type=float, default=2.0)
    parser.add_argument("--tau_euler", type=float, default=5.0)

    # stage 3 shape thresholds
    parser.add_argument("--shape_elongation_thr", type=float, default=1.8)
    parser.add_argument("--shape_planarity_thr", type=float, default=1.8)

    # optional balanced sampling
    parser.add_argument(
        "--sample_target",
        type=int,
        default=0,
        help="If > 0, perform balanced sampling over 12 strata to this target size."
    )
    parser.add_argument("--sample_seed", type=int, default=42)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) read and merge all parts
    records = read_jsonl_parts(args.input_pattern)
    if len(records) == 0:
        raise RuntimeError("No valid records loaded from input jsonl parts.")

    # optional: save merged raw input
    write_jsonl(os.path.join(args.output_dir, "merged_all.jsonl"), records)

    df_all = build_dataframe(records)

    # stage 1
    df_stage1 = hard_filter(df_all, args)
    df_hard = df_stage1[df_stage1["hard_pass"]].copy()

    # stage 2
    df_stage2 = statistical_filter(df_hard, args)

    # stage 3
    df_stage2 = stratification_buckets(df_stage2, args)

    # all stat-passed records, summary should keep unknown here
    df_kept_all = df_stage2[df_stage2["stat_pass"]].copy()

    # valid 12 strata only, used for final bucket dataset and sampling
    df_kept = filter_valid_12_strata(df_kept_all)

    hard_jsonl_path = os.path.join(args.output_dir, "filtered_hard.jsonl")
    kept_jsonl_path = os.path.join(args.output_dir, "filtered_stat.jsonl")

    keep_cols_for_jsonl = [
        "UID",
        "load_ok",
        "error",
        "vertex_count",
        "face_count",
        "non_empty",
        "finite_vertices",
        "valid_face_indices",
        "is_winding_consistent",
        "is_watertight",
        "bbox_extent",
        "euler_number",
        "degenerate_face_ratio",
        "area",
        "volume",
        "convex_hull_area",
        "convex_hull_volume",
        "bbox_e1",
        "bbox_e2",
        "bbox_e3",
        "elongation",
        "planarity",
        "fill_ratio",
        "hull_area_ratio",
        "bbox_area_ratio",
        "bbox_volume_ratio",
        "log_bbox_area_ratio",
        "log_hull_area_ratio",
        "euler_signed_norm",
        "rz_log_bbox_area_ratio",
        "rz_log_hull_area_ratio",
        "rz_euler_signed_norm",
        "flag_bbox_area_ratio",
        "flag_hull_area_ratio",
        "flag_euler_signed_norm",
        "stat_anomaly_count",
        "stat_pass",
        "shape_type",
        "fill_bucket",
    ]

    write_jsonl(
        hard_jsonl_path,
        df_hard[[c for c in keep_cols_for_jsonl if c in df_hard.columns]].to_dict(orient="records")
    )

    # keep filtered_stat as all stat-passed records so it can still include unknown
    write_jsonl(
        kept_jsonl_path,
        df_kept_all[[c for c in keep_cols_for_jsonl if c in df_kept_all.columns]].to_dict(orient="records")
    )

    df_stage2.to_csv(os.path.join(args.output_dir, "scored_all.csv"), index=False)
    df_kept_all.to_csv(os.path.join(args.output_dir, "scored_kept.csv"), index=False)
    df_kept.to_csv(os.path.join(args.output_dir, "scored_kept_valid_12.csv"), index=False)

    df_stage1[~df_stage1["hard_pass"]][["UID", "hard_fail_reasons"]].to_csv(
        os.path.join(args.output_dir, "hard_filter_failures.csv"), index=False
    )

    df_stage2[~df_stage2["stat_pass"]][[
        "UID",
        "flag_bbox_area_ratio",
        "flag_hull_area_ratio",
        "flag_euler_signed_norm",
        "stat_anomaly_count",
        "rz_euler_signed_norm",
        "euler_signed_norm",
    ]].to_csv(
        os.path.join(args.output_dir, "stat_filter_failures.csv"),
        index=False
    )

    # final stratified dataset only keeps uid + valid 12-strata bucket labels
    final_strata_records = df_kept[["UID", "shape_type", "fill_bucket"]].rename(
        columns={"UID": "uid"}
    ).to_dict(orient="records")
    write_json(
        os.path.join(args.output_dir, "final_strata_uid_buckets.json"),
        final_strata_records
    )

    if args.sample_target > 0:
        df_sampled = balanced_sample_12_buckets(
            df_kept,
            target_n=args.sample_target,
            random_seed=args.sample_seed
        )

        df_sampled.to_csv(os.path.join(args.output_dir, f"sampled_{args.sample_target}.csv"), index=False)
        write_jsonl(
            os.path.join(args.output_dir, f"sampled_{args.sample_target}.jsonl"),
            df_sampled.to_dict(orient="records")
        )

        # sampled uid + bucket labels, guaranteed no unknown
        sampled_uid_buckets = df_sampled[["UID", "shape_type", "fill_bucket"]].rename(
            columns={"UID": "uid"}
        ).to_dict(orient="records")
        write_json(
            os.path.join(args.output_dir, f"sampled_{args.sample_target}_uid_buckets.json"),
            sampled_uid_buckets
        )

        sampled_joint_counts = df_sampled.groupby(
            ["shape_type", "fill_bucket"]
        ).size().to_dict()

        sampled_summary = {
            "sample_target": int(args.sample_target),
            "num_sampled": int(len(df_sampled)),
            "shape_type_distribution": df_sampled["shape_type"].value_counts(dropna=False).to_dict(),
            "fill_bucket_distribution": df_sampled["fill_bucket"].value_counts(dropna=False).to_dict(),
            "shape_fill_joint_distribution": {
                f"{k[0]}|{k[1]}": int(v)
                for k, v in sampled_joint_counts.items()
            },
        }
        write_json(
            os.path.join(args.output_dir, f"sampled_{args.sample_target}_summary.json"),
            sampled_summary
        )

    summary = summarize(df_all, df_hard, df_kept_all, df_kept)
    write_json(os.path.join(args.output_dir, "summary.json"), summary)

    print("\n=== Done ===")
    print(f"Input total:                        {len(df_all)}")
    print(f"After hard filter:                  {len(df_hard)}")
    print(f"After stat filter (all kept):       {len(df_kept_all)}")
    print(f"After valid 12-strata filtering:    {len(df_kept)}")
    if args.sample_target > 0:
        print(f"After balanced sample:              {min(args.sample_target, len(df_kept))}")
    print(f"Outputs saved to:                   {args.output_dir}")


if __name__ == "__main__":
    main()

"""
python geometry_filter_sample.py \
  --input_pattern "geometry_parts/geometry_annotations_part_*.jsonl" \
  --output_dir out_geom \
  --sample_target 12000
"""
