#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from launcher_algorithm import METHODS


DEFAULT_CHECKPOINTS = ("terminal",)
DEFAULT_THRESHOLDS = ("0.01", "0.02", "0.03")


def _load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _std(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return 0.0 if len(values) == 1 else None
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def _median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _fmt(value, digits=4):
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _checkpoint(summary, name):
    checkpoints = summary.get("checkpoints", {})
    if name == "terminal":
        return summary.get("terminal_checkpoint") or checkpoints.get("terminal")
    return checkpoints.get(name)


def _checkpoint_accepted_nbv_action_num(summary, checkpoint_name, checkpoint):
    value = checkpoint.get("accepted_nbv_action_num")
    if value is not None:
        return value

    visited_view_num = checkpoint.get("visited_view_num")
    try:
        visited = int(visited_view_num)
    except (TypeError, ValueError):
        visited = None
    if visited is not None and visited > 0:
        return visited - 1

    if checkpoint_name == "terminal":
        return summary.get("accepted_nbv_action_num")
    return None


def _extract_row(summary, summary_path, checkpoint_name, thresholds):
    checkpoint = _checkpoint(summary, checkpoint_name)
    if checkpoint is None:
        return None
    method_name = summary.get("method_name")
    method_spec = METHODS.get(method_name) if isinstance(method_name, str) else None

    ms = summary.get("evaluator_summary", {}).get("map_stabilization", {})
    if not ms:
        ms = summary.get("map_stabilization", {})

    row = {
        "summary_path": str(summary_path),
        "episode_id": summary.get("episode_id"),
        "uid": summary.get("uid"),
        "method_name": method_name,
        "method_display_name": summary.get("method_display_name")
        or (method_spec.display_name if method_spec is not None else None)
        or method_name,
        "family": summary.get("family") or (method_spec.family if method_spec is not None else None),
        "execution_mode_compatibility": summary.get("execution_mode_compatibility")
        or summary.get("execution_compatibility")
        or (method_spec.execution_mode_compatibility if method_spec is not None else None),
        "execution_mode": summary.get("execution_mode") or (method_spec.execution_mode if method_spec is not None else None),
        "constraint": summary.get("viewspace_constraint_name"),
        "start_view_id": summary.get("start_view_id"),
        "start_view_set": summary.get("start_view_set"),
        "checkpoint": checkpoint_name,
        "checkpoint_reason": checkpoint.get("reason"),
        "terminal_source": summary.get("terminal_source"),
        "terminal_reason": summary.get("terminal_reason"),
        "algorithm_stop_reason": summary.get("algorithm_stop_reason"),
        "accepted_nbv_action_num": _checkpoint_accepted_nbv_action_num(summary, checkpoint_name, checkpoint),
        "visited_view_num": checkpoint.get("visited_view_num"),
        "effective_stop_view_num": summary.get("effective_stop_view_num"),
        "chamfer_distance": checkpoint.get("chamfer_distance"),
        "total_path_cost": checkpoint.get("total_path_cost"),
        "total_algorithm_runtime_sec": checkpoint.get("total_algorithm_runtime_sec"),
        "total_benchmark_wall_time_sec": checkpoint.get("total_benchmark_wall_time_sec"),
        "num_fused_points": checkpoint.get("num_fused_points"),
    }

    sc = checkpoint.get("surface_coverage", {})
    nsc = checkpoint.get("normalized_surface_coverage", {})
    ms_views = ms.get("trigger_visited_view_nums", {})
    ms_steps = ms.get("trigger_step_indices", {})
    for threshold in thresholds:
        row[f"SC@{threshold}"] = sc.get(threshold)
        row[f"NSC@{threshold}"] = nsc.get(threshold)
        row[f"MS@{threshold}_visited_view_num"] = ms_views.get(threshold)
        row[f"MS@{threshold}_step_index"] = ms_steps.get(threshold)
    return row


def _collect_rows(summaries_dirs, checkpoints, thresholds, method_filter=None):
    rows = []
    for summaries_dir in summaries_dirs:
        for path in sorted(Path(summaries_dir).glob("*.json")):
            summary = _load_json(path)
            if method_filter and summary.get("method_name") not in method_filter:
                continue
            for checkpoint_name in checkpoints:
                row = _extract_row(summary, path, checkpoint_name, thresholds)
                if row is not None:
                    rows.append(row)
    return rows


def _group_key(row, include_start_view):
    key = [
        row.get("method_name"),
        row.get("method_display_name"),
        row.get("family"),
        row.get("execution_mode_compatibility"),
        row.get("execution_mode"),
        row.get("constraint"),
        row.get("checkpoint"),
    ]
    if include_start_view:
        key.append(row.get("start_view_id"))
    return tuple(key)


def _aggregate_rows(rows, thresholds, include_start_view=False):
    groups = defaultdict(list)
    for row in rows:
        groups[_group_key(row, include_start_view)].append(row)

    out = []
    for key in sorted(groups):
        items = groups[key]
        method_name, method_display_name, family, execution_mode_compatibility, execution_mode, constraint, checkpoint = key[:7]
        agg = {
            "method_name": method_name,
            "method_display_name": method_display_name,
            "family": family,
            "execution_mode_compatibility": execution_mode_compatibility,
            "execution_mode": execution_mode,
            "constraint": constraint,
            "checkpoint": checkpoint,
        }
        if include_start_view:
            agg["start_view_id"] = key[7]

        agg["num_episodes"] = len(items)
        agg["num_objects"] = len({x.get("uid") for x in items})
        terminal_reason_counter = Counter(x.get("terminal_reason") for x in items)
        algorithm_stop_reason_counter = Counter(x.get("algorithm_stop_reason") for x in items)
        agg["terminal_reason_counts"] = ";".join(
            f"{name}:{count}"
            for name, count in sorted(
                terminal_reason_counter.items(),
                key=lambda kv: (kv[0] is None, "" if kv[0] is None else str(kv[0])),
            )
        )
        agg["algorithm_stop_reason_counts"] = ";".join(
            f"{name}:{count}"
            for name, count in sorted(
                algorithm_stop_reason_counter.items(),
                key=lambda kv: (kv[0] is None, "" if kv[0] is None else str(kv[0])),
            )
        )

        for field in (
            "visited_view_num",
            "accepted_nbv_action_num",
            "chamfer_distance",
            "total_path_cost",
            "total_algorithm_runtime_sec",
            "total_benchmark_wall_time_sec",
        ):
            values = [_as_float(x.get(field)) for x in items]
            agg[f"{field}_mean"] = _mean(values)
            agg[f"{field}_std"] = _std(values)
            agg[f"{field}_median"] = _median(values)
            agg[f"{field}_min"] = min((v for v in values if v is not None), default=None)
            agg[f"{field}_max"] = max((v for v in values if v is not None), default=None)

        for threshold in thresholds:
            for prefix in ("SC", "NSC"):
                field = f"{prefix}@{threshold}"
                values = [_as_float(x.get(field)) for x in items]
                agg[f"{field}_mean"] = _mean(values)
                agg[f"{field}_std"] = _std(values)
                agg[f"{field}_median"] = _median(values)
                agg[f"{field}_min"] = min((v for v in values if v is not None), default=None)
                agg[f"{field}_max"] = max((v for v in values if v is not None), default=None)

            ms_field = f"MS@{threshold}_visited_view_num"
            ms_values = [_as_float(x.get(ms_field)) for x in items]
            triggered = [v for v in ms_values if v is not None]
            agg[f"MS@{threshold}_trigger_rate"] = len(triggered) / len(items) if items else None
            agg[f"MS@{threshold}_visited_view_num_mean"] = _mean(triggered)
            agg[f"MS@{threshold}_visited_view_num_median"] = _median(triggered)

        out.append(agg)
    return out


def _write_csv(path, rows):
    if not rows:
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _print_compact_table(rows, thresholds):
    if not rows:
        print("No rows.")
        return

    cols = [
        "method_name",
        "method_display_name",
        "execution_mode_compatibility",
        "execution_mode",
        "constraint",
        "checkpoint",
        "num_episodes",
        "visited_view_num_mean",
        "accepted_nbv_action_num_mean",
        "chamfer_distance_mean",
        "NSC@0.02_mean",
        "NSC@0.02_min",
        "total_path_cost_mean",
        "total_algorithm_runtime_sec_mean",
        "total_benchmark_wall_time_sec_mean",
        "MS@0.02_trigger_rate",
        "MS@0.02_visited_view_num_mean",
        "terminal_reason_counts",
    ]
    if "0.02" not in thresholds:
        cols[6] = f"NSC@{thresholds[0]}_mean"
        cols[7] = f"NSC@{thresholds[0]}_min"
        cols[9] = f"MS@{thresholds[0]}_trigger_rate"
        cols[10] = f"MS@{thresholds[0]}_visited_view_num_mean"

    print("\t".join(cols))
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col)
            vals.append(_fmt(value))
        print("\t".join(vals))


def main():
    parser = argparse.ArgumentParser(description="Summarize ObjView-Bench episode summaries into method-level tables.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory containing summaries/.")
    parser.add_argument("--run-dirs", type=Path, nargs="+", default=None, help="One or more run directories containing summaries/.")
    parser.add_argument(
        "--sharded-run-dir-template",
        type=str,
        default=None,
        help="Run directory template with {shard_id}, e.g. runs/method_shard{shard_id}of4.",
    )
    parser.add_argument("--num-shards", type=int, default=None, help="Number of shards for --sharded-run-dir-template.")
    parser.add_argument("--shard-ids", type=int, nargs="+", default=None, help="Shard ids for --sharded-run-dir-template.")
    parser.add_argument("--summaries-dir", type=Path, default=None, help="Directory containing summary JSON files.")
    parser.add_argument("--summaries-dirs", type=Path, nargs="+", default=None, help="One or more directories containing summary JSON files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for CSV tables.")
    parser.add_argument("--method", action="append", default=None, help="Filter by method_name. Can be repeated.")
    parser.add_argument("--checkpoints", nargs="+", default=list(DEFAULT_CHECKPOINTS))
    parser.add_argument("--thresholds", nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--include-start-view", action="store_true", help="Keep start_view_id as a grouping column.")
    parser.add_argument("--no-print", action="store_true")
    args = parser.parse_args()

    summaries_dirs = []
    if args.summaries_dir is not None:
        summaries_dirs.append(args.summaries_dir)
    if args.summaries_dirs is not None:
        summaries_dirs.extend(args.summaries_dirs)
    if args.run_dir is not None:
        summaries_dirs.append(args.run_dir / "summaries")
    if args.run_dirs is not None:
        summaries_dirs.extend(run_dir / "summaries" for run_dir in args.run_dirs)
    if args.sharded_run_dir_template is not None:
        shard_ids = args.shard_ids
        if shard_ids is None:
            if args.num_shards is None:
                parser.error("Provide --num-shards or --shard-ids with --sharded-run-dir-template.")
            shard_ids = list(range(args.num_shards))
        for shard_id in shard_ids:
            summaries_dirs.append(Path(args.sharded_run_dir_template.format(shard_id=shard_id)) / "summaries")
    if not summaries_dirs:
        parser.error("Provide --run-dir, --run-dirs, --summaries-dir, --summaries-dirs, or --sharded-run-dir-template.")

    if args.output_dir is None:
        if args.run_dir is not None:
            args.output_dir = args.run_dir / "tables"
        else:
            args.output_dir = summaries_dirs[0].parent / "tables"

    rows = _collect_rows(summaries_dirs, args.checkpoints, args.thresholds, set(args.method or []))
    aggregate = _aggregate_rows(rows, args.thresholds, include_start_view=args.include_start_view)

    _write_csv(args.output_dir / "episodes.csv", rows)
    _write_csv(args.output_dir / "method_summary.csv", aggregate)

    meta = {
        "summaries_dirs": [str(path.resolve()) for path in summaries_dirs],
        "output_dir": str(args.output_dir.resolve()),
        "num_episode_rows": len(rows),
        "num_aggregate_rows": len(aggregate),
        "checkpoints": args.checkpoints,
        "thresholds": args.thresholds,
        "method_filter": args.method,
        "include_start_view": args.include_start_view,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if not args.no_print:
        print(json.dumps(meta, indent=2))
        _print_compact_table(aggregate, args.thresholds)


if __name__ == "__main__":
    raise SystemExit(main())

"""
python summarize_method_table.py \
  --sharded-run-dir-template 'runs/final_clean_long_tail_oracle_rollout_hybrid_sorted_shard{shard_id}of3' \
  --num-shards 3 \
  --output-dir runs/tables/final_clean_long_tail_oracle_rollout_hybrid_sorted/ \
  --checkpoints MS@0.01 MS@0.02 MS@0.03 K=1 K=2 K=3 K=4 K=5 K=10 K=15 K=20 K=30 K=40 K=50 K=60 K=70 K=80 K=90 K=100 K=110 K=120 terminal 

python summarize_method_table.py \
  --sharded-run-dir-template 'runs/main_hidden_oracle_rollout_hybrid_sorted_shard{shard_id}of3' \
  --num-shards 3 \
  --output-dir runs/tables/main_hidden_oracle_rollout_hybrid_sorted/ \
  --checkpoints MS@0.01 MS@0.02 MS@0.03 K=1 K=2 K=3 K=4 K=5 K=10 K=15 K=20 K=30 K=40 K=50 K=60 K=70 K=80 K=90 K=100 K=110 K=120 terminal 

python summarize_method_table.py \
  --sharded-run-dir-template 'runs/main_simple_random_tsporder_5_10_30_shard{shard_id}of3' \
  --num-shards 3 \
  --output-dir runs/tables/main_simple_random_tsporder_5_10_30/ \
  --checkpoints terminal

python summarize_method_table.py \
  --sharded-run-dir-template 'runs/main_classical_voxel_ig_rse_shard{shard_id}of2' \
  --num-shards 2 \
  --output-dir runs/tables/main_classical_voxel_ig_rse/ \
  --checkpoints K=5 K=10 K=30 MS@0.01 MS@0.02 MS@0.03

python summarize_method_table.py \
  --sharded-run-dir-template 'runs/main_classical_voxel_ig_rse_mov_shard{shard_id}of2' \
  --num-shards 2 \
  --output-dir runs/tables/main_classical_voxel_ig_rse_mov/ \
  --checkpoints K=5 K=10 K=30 MS@0.01 MS@0.02 MS@0.03

python summarize_method_table.py \
  --sharded-run-dir-template 'runs/main_completion_planning_pointr_c_nbv_shard{shard_id}of2' \
  --num-shards 2 \
  --output-dir runs/tables/main_completion_planning_pointr_c_nbv/ \
  --checkpoints K=5 K=10 K=30 MS@0.01 MS@0.02 MS@0.03

python summarize_method_table.py \
  --sharded-run-dir-template 'runs/main_completion_planning_pointr_c_scp_shard{shard_id}of2' \
  --num-shards 2 \
  --output-dir runs/tables/main_completion_planning_pointr_c_scp/ \
  --checkpoints terminal

python summarize_method_table.py \
  --sharded-run-dir-template 'runs/main_completion_planning_pointr_c_mcp_shard{shard_id}of2' \
  --num-shards 2 \
  --output-dir runs/tables/main_completion_planning_pointr_c_mcp/ \
  --checkpoints terminal

python summarize_method_table.py \
  --sharded-run-dir-template 'runs/main_learned_set_cover_mascvp_shard{shard_id}of4' \
  --num-shards 4 \
  --output-dir runs/tables/main_learned_set_cover_mascvp/ \
  --checkpoints terminal

python summarize_method_table.py \
  --sharded-run-dir-template 'runs/main_learned_nbv_benbv_shard{shard_id}of2' \
  --num-shards 2 \
  --output-dir runs/tables/main_learned_nbv_benbv/ \
  --checkpoints K=5 K=10 K=30 MS@0.01 MS@0.02 MS@0.03 terminal

python summarize_method_table.py \
  --sharded-run-dir-template 'runs/main_learned_nbv_nbvnet_shard{shard_id}of4' \
  --num-shards 4 \
  --output-dir runs/tables/main_learned_nbv_nbvnet/ \
  --checkpoints K=5 K=10 K=30 MS@0.01 MS@0.02 MS@0.03 terminal
"""
