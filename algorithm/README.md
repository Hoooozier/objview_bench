# ObjView Algorithm Module

This folder contains the C++/CUDA implementation of planning algorithms, cache/rollout tools, evaluation workers, and test binaries used by ObjView.

## Environment Setup

You can use the reference setup script in `env.txt` (now anonymized, no user-specific home paths).

Minimum dependencies from `CMakeLists.txt`:
- CMake >= 3.24
- C++17 and CUDA 11.8 toolchain
- Boost
- PCL
- OpenCV
- jsoncpp
- ZLIB
- OctoMap
- Gurobi (for optimization-based planners)
- cnpy

## Build

Run from this directory:

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

## Quick Usage

Most executables support `--help`:

```bash
./<BinaryName> --help
```

Representative run entrypoints:
- `ObjViewAlgorithmRunner`: online benchmark interaction entrypoint.
- `PcdSetCover`, `PcdSaturation`, `PcdCache`: single-object offline tools.
- `Batch*` executables: batch wrappers for the corresponding single-object tools.

## File-by-File Guide

### Build and environment files
- `CMakeLists.txt`: defines all executable targets and linked dependencies.
- `env.txt`: container/toolchain setup notes and dependency installation script.
- `README.md`: this file.

### Core algorithm interfaces and runtime
- `objview_algorithm.h`: base algorithm interface and decision payload types.
- `objview_algorithm_runner.cpp`: main runtime that talks to benchmark session directories and dispatches selected algorithm.
- `objview_benchmark_submitter.h`: session I/O protocol helper for benchmark interaction.
- `objview_interaction_rpc_client.h`: RPC client for remote benchmark/interaction services.

### Planning-network-backed algorithms
- `mascvp_planning_network_algorithm.h`: MascVP planning-network algorithm implementation.
- `benbv_planning_network_algorithm.h`: BenBV planning-network algorithm implementation.
- `nbvnet_planning_network_algorithm.h`: NBVNet planning-network algorithm implementation.
- `objview_planning_network_client.h`: generic planning-network client abstraction.

### PointR-C family algorithms
- `pointr_c_nbv_algorithm.h`: PointR-C NBV variant.
- `pointr_c_scp_algorithm.h`: PointR-C set-cover-style planner.
- `pointr_c_mcp_algorithm.h`: PointR-C movement-cost-aware planner.
- `objview_shape_completion_client.h`: shape completion service client used by PointR-C-based methods.
- `test_shape_completion_client.cpp`: unit/integration-style test for shape completion client and capability parsing.

### Voxel information-gain algorithms
- `voxel_ig_algorithm.h`: CPU/CUDA voxel information gain planner logic.
- `voxel_ig_cuda.h`: CUDA interface declarations for voxel IG kernels.
- `voxel_ig_cuda.cu`: CUDA implementation for voxel IG acceleration.
- `test_voxel_ig_algorithm.cpp`: tests for voxel IG algorithm behavior.

### Geometry, view, and observation utilities
- `objview_geometry.h`: geometry primitives and helper math utilities.
- `objview_view_io.h`: view set file parsing/loading helpers.
- `objview_observation_io.h`: observation serialization/deserialization helpers.
- `objview_pointcloud_io.h`: point cloud file I/O helpers.
- `test_observation_io.cpp`: tests for observation I/O.

### Global path planning
- `global_path_planner.h`: path planning and runtime accounting utilities (Gurobi-backed).
- `test_global_path_planner.cpp`: executable test for global planner API and runtime accounting.
- `random_tsp_order_algorithm.h`: random/TSP-order baseline built on global path planning.
- `test_random_tsp_order_algorithm.cpp`: executable test for random/TSP-order baseline.

### CUDA raycaster
- `cuda_raycaster.h`: CUDA raycaster API.
- `cuda_raycaster.cu`: CUDA raycaster implementation.
- `test_raycaster.cu`: CUDA raycaster test executable.
- `test_cuda.cu`: CUDA sanity test source (used by `TestEnv` target).
- `test_env.cpp`: full environment sanity test source (used by `TestEnv` target).

### Offline single-object tools
- `set_cover_from_pcd.cpp` -> binary `PcdSetCover`: compute set-cover-based view plans from one `.pcd`.
- `saturation_from_pcd.cpp` -> binary `PcdSaturation`: compute saturation/coverage metrics from one `.pcd`.
- `cache_from_pcd.cpp` -> binary `PcdCache`: precompute/render per-view cache from one `.pcd`.
- `rollout_from_cache.cpp` -> binary `RolloutCache`: run rollout/evaluation using precomputed cache.
- `export_method_case.cpp` -> binary `ExportMethodCase`: export detailed method-level evaluation case artifacts.
- `export_mascvp_offline_case.cpp` -> binary `ExportMascvpOfflineCase`: export MascVP offline case artifacts.
- `eval_mascvp_object_worker.cpp` -> binary `EvalMascvpObjectWorker`: evaluate one object for MascVP workflow.
- `benbv_single_worker.cpp` -> binary `BenbvSingleWorker`: evaluate/process one object for BenBV workflow.

### Offline batch wrappers
- `batch_set_cover_from_pcd.cpp` -> binary `BatchPcdSetCover`: batch run `PcdSetCover`.
- `batch_saturation_from_pcd.cpp` -> binary `BatchPcdSaturation`: batch run `PcdSaturation`.
- `batch_cache_from_pcd.cpp` -> binary `BatchPcdCache`: batch run `PcdCache`.
- `batch_rollout_from_cache.cpp` -> binary `BatchRolloutCache`: batch run `RolloutCache`.
- `batch_export_method_case.cpp` -> binary `BatchExportMethodCase`: batch run `ExportMethodCase`.
- `batch_export_mascvp_offline_case.cpp` -> binary `BatchExportMascvpOfflineCase`: batch run `ExportMascvpOfflineCase`.
- `batch_eval_mascvp_object_worker.cpp` -> binary `BatchEvalMascvpObjectWorker`: batch run `EvalMascvpObjectWorker`.
- `batch_benbv_single_worker.cpp` -> binary `BatchBenbvSingleWorker`: batch run `BenbvSingleWorker`.

## Run Examples

From `build/`, use this unified pattern for every executable:

```bash
./<target> --help
```

