# ObjView Data Tools

This folder contains data-processing and annotation utilities for ObjView/ObjView-Bench.

## Environment

Use Python 3.9+ and install required packages used by scripts in this folder:

```bash
pip install numpy pandas matplotlib open3d trimesh objaverse torch torchvision pytorch3d
```

Some scripts additionally depend on GUI/runtime tools:
- `review_app_qt_open3d.py`: `PyQt6` + Open3D display support
- `glb_blender_obj.py`: Blender Python runtime

## Unified Usage Rule

Use this pattern for every Python script:

```bash
python <script>.py --help
```

## Main Script Groups

- **Pool/split building**
  - `build_final_clean_pool.py`
  - `build_analysis_splits.py`
  - `build_released_benchmark_splits.py`
  - `split_object_complexity_with_cross_stats.py`

- **Saturation / complexity analysis**
  - `analyze_filtered_pools_saturation.py`
  - `analyze_saturation_partition.py`
  - `collect_object_complexity_data.py`
  - `collect_planning_difficulty_data.py`

- **Geometry download / conversion / normalization**
  - `download_geometry_sampled_Objaverse.py`
  - `glb_blender_obj.py`
  - `normalize_obj.py`
  - `glb_poisson_disk_normalized_pcd_worker.py`
  - `glb_poisson_disk_normalized_pcd_master.py`

- **Geometry annotation pipeline**
  - `geometry_annotate_from_json.py`
  - `run_geometry_annotation_in_batches.py`
  - `repair_geometry_annotations.py`

- **Manual review candidate workflow**
  - `get_manual_review_candidates.py`
  - `risk_manual_review_candidates.py`
  - `review_app_qt_open3d.py`
  - `merge_review_results.py`

- **Utilities**
  - `parallel_copy.py`
  - `render_previews_pytorch3d.py`
  - `filter_ObjaversePlusPlus.py`
  - `geometry_filter_sample.py`

