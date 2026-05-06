# ObjView Benchmark Module

This directory contains the benchmark runtime, APIs, launcher scripts, baseline/dummy algorithms, and planning-network inference/training utilities.

## Environment Setup

Use `requirements.txt` as the main pip dependency list, and `pyenv.txt` as the full environment recipe (including conda/PyTorch/PyTorch3D steps):
- Python 3.9
- PyTorch 2.0 + CUDA 11.8
- PyTorch3D
- Open3D, OpenCV, matplotlib, scikit-image, imageio, plotly
- PoinTr-related packages (`easydict`, `transforms3d`, `h5py`, `timm`)

Example:

```bash
conda create -n pytorch3d python=3.9
conda activate pytorch3d
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html
# install Python packages in this folder
pip install -r requirements.txt
# then follow pyenv.txt for optional extras
```

For C++ algorithm execution via Enroot, use:
- `algorithms/cpp_enroot/run_cpp_enroot_algorithm.sh`
- Environment variables:
  - `OBJVIEW_ENROOT_NAME`
  - `OBJVIEW_ALGO_ROOT`
  - `OBJVIEW_RUNNER`
  - `OBJVIEW_ALGO_WORKDIR`
  - `OBJVIEW_GUROBI_LICENSE_CONTAINER`

## Unified Usage Rule

For all Python entrypoints in this folder, use:

```bash
python <script>.py --help
```

For shell wrappers, use:

```bash
bash <script>.sh --help
```

## Minimal Demo (Oracle, 5 Moves)

Run one minimal benchmark episode with `oracle_greedy_nsc01` and cap it to 5 move actions.
Execute from `objview_code/benchmark`:

```bash
python launcher_episode.py \
  --episode-id demo_oracle_greedy_5step \
  --uid ff825afe79e8447cbe12d9ea36998d90 \
  --obj-path obj_normalized/ff825afe79e8447cbe12d9ea36998d90/ff825afe79e8447cbe12d9ea36998d90.obj \
  --gt-pointcloud /ABS/PATH/TO/ff825afe79e8447cbe12d9ea36998d90.pcd \
  --cache-index-json render_cache/cache_index.json \
  --feasibility-json configs/feasibility/whole.json \
  --session-dir interaction/session_demo_oracle_greedy_5step \
  --summary-json interaction/summaries/demo_oracle_greedy_5step.json \
  --method-name oracle_greedy_nsc01 \
  --family oracle \
  --max-visited-view-num 6 \
  --algorithm-command "python launcher_algorithm.py --method oracle_greedy_nsc01 --session-dir {session_dir} --cache-index-json render_cache/cache_index.json"
```

Notes:
- `--max-visited-view-num 6` means 1 initial observation + up to 5 move actions.
- Replace `--gt-pointcloud` with your real normalized `.pcd` path.
- Result summary is written to `interaction/summaries/demo_oracle_greedy_5step.json`.

## Key Entry Points

- `launcher_episode.py`: launch one benchmark episode and one algorithm command together.
- `runner_episode.py`: run one episode directly (without launcher wrapper).
- `launcher_algorithm.py`: resolve method name/family to concrete algorithm command.
- `suite_runner.py`: run multi-episode benchmark suites from configs.
- `summarize_method_table.py`: aggregate per-episode summaries into method-level tables.

## Core APIs

- `api_render.py`: rendering API (online/cache/auto mode).
- `api_feasibility.py`: geometric feasibility checks.
- `api_evaluation.py`: coverage/path/evaluation metrics.
- `api_interaction.py`: filesystem interaction protocol between benchmark and algorithm.
- `api_planning_network.py`: planning-network service API.
- `api_shape_completion.py`: shape completion service API.

## Workers and Caches

- `render_cache_worker.py`: precompute render cache and build/merge cache index.
- `eval_cache_worker.py`: batch evaluation from cache index.

## Algorithms

- `algorithms/oracle/*`: oracle baselines.
- `algorithms/dummy/*`: protocol validation and failure-mode dummy methods.
- `algorithms/planning_network_client.py`: client for remote planning-network inference.
- `algorithms/cpp_enroot/run_cpp_enroot_algorithm.sh`: wrapper for running C++ planner in Enroot.

## Planning Networks and Shape Completion

- `planning_network/NBVNET/*`: NBVNET inference and training.
- `planning_network/MASCVP/*`: MASCVP inference/model code.
- `planning_network/BENBV/*`: BENBV inference/model/data utilities.
- `PoinTr/*`: Point cloud completion model and dependencies.

## Data and Config Assets

- `Tammes_sphere/*.txt`: view set definitions.
- `configs/feasibility/*.json`: benchmark feasibility constraints.
- `showcases.json`: sample benchmark method configurations.

