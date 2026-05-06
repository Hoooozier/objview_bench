#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONTAINER="${OBJVIEW_ENROOT_NAME:-ObjViewBench}"
ALGO_ROOT="${OBJVIEW_ALGO_ROOT:-$HOME/ObjViewBench}"
RUNNER="${OBJVIEW_RUNNER:-$ALGO_ROOT/build/ObjViewAlgorithmRunner}"
WORKDIR="${OBJVIEW_ALGO_WORKDIR:-$ALGO_ROOT/build}"
GUROBI_LICENSE_CONTAINER="${OBJVIEW_GUROBI_LICENSE_CONTAINER:-/gurobi.lic}"

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --session-dir)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --session-dir" >&2
        exit 2
      fi
      ARGS+=("$1" "$(readlink -f "$2")")
      shift 2
      ;;
    --cache-index-json)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --cache-index-json" >&2
        exit 2
      fi
      ARGS+=("$1" "$(readlink -f "$2")")
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

exec enroot start \
  -w -r \
  -e DISPLAY="${DISPLAY:-}" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e GRB_LICENSE_FILE="$GUROBI_LICENSE_CONTAINER" \
  -m ~:/home/${USER} \
  -m /mnt:/mnt \
  -m /dev:/dev \
  -m /tmp/.X11-unix:/tmp/.X11-unix \
  "$CONTAINER" \
  bash -lc 'cd "$1"; shift; exec "$@"' \
  bash "$WORKDIR" "$RUNNER" "${ARGS[@]}"
