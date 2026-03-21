#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONDA_ENV=${CONDA_ENV:-carl}

# Checkpoints and route
export CHECKPOINT_DIR=outputs/checkpoints/CaRL
export ROUTES=data/benchmark_routes/bench2drive/24240.xml

# Environment
export BENCHMARK_ROUTE_ID="$(basename "${ROUTES}" .xml)"
export EVALUATION_OUTPUT_DIR="${SCRIPT_DIR}/outputs/local_evaluation/${BENCHMARK_ROUTE_ID}"
export CARLA_ROOT=3rd_party/CARLA_0915
export SCENARIO_RUNNER_ROOT="${SCRIPT_DIR}/3rd_party/scenario_runner"
export LEADERBOARD_ROOT="${SCRIPT_DIR}/3rd_party/leaderboard"
export PYTHONPATH=3rd_party/leaderboard:$PYTHONPATH
export PYTHONPATH=3rd_party/scenario_runner:$PYTHONPATH
export PYTHONPATH=${SCRIPT_DIR}:${CARLA_ROOT}/PythonAPI/carla:${SCENARIO_RUNNER_ROOT}:${LEADERBOARD_ROOT}

# Agent runtime env vars
export SAVE_PATH="${EVALUATION_OUTPUT_DIR}/"
export DEBUG_ENV_AGENT=1
export RECORD=0
export SAVE_PNG=0
export UPSCALE_FACTOR=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HIGH_FREQ_INFERENCE=0
export SAMPLE_TYPE=mean
export CPP=0
export CPP_PORT=5555
export CARLA_RPC_PORT=${CARLA_RPC_PORT:-2000}
export CARLA_TM_PORT=${CARLA_TM_PORT:-8000}

# This machine's GPU arch is unsupported by the pinned torch build.
# Keep evaluation on CPU by default.
export CUDA_VISIBLE_DEVICES=""

set -x
set -e

# Recreate output folder
rm -rf "${EVALUATION_OUTPUT_DIR}"
mkdir -p "${EVALUATION_OUTPUT_DIR}" "${SAVE_PATH}"

python "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py" \
  --routes="${ROUTES}" \
  --track=MAP \
  --checkpoint="${EVALUATION_OUTPUT_DIR}/checkpoint_endpoint.json" \
  --agent="${SCRIPT_DIR}/lead/carl_agent/carl_agent.py" \
  --agent-config="${CHECKPOINT_DIR}" \
  --debug=0 \
  --record= \
  --resume=1 \
  --port="${CARLA_RPC_PORT}" \
  --traffic-manager-port="${CARLA_TM_PORT}" \
  --timeout=900 \
  --traffic-manager-seed=0 \
  --repetitions=1
