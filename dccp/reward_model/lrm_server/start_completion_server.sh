#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

LRM_PYTHON_BIN="${LRM_PYTHON_BIN:-python}"
LRM_COMPLETION_MODEL_PATH="${LRM_COMPLETION_MODEL_PATH:-${DCCP_ASSET_ROOT:-${PROJECT_ROOT}/../assets}/lrm/progress}"
LRM_BASE_MODEL_PATH="${LRM_BASE_MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
LRM_OFFICIAL_SERVER_PY="${LRM_OFFICIAL_SERVER_PY:-${PROJECT_ROOT}/../Large-Reward-Models/vlm_reward/vlm_reward_server.py}"

LRM_HOST="${LRM_HOST:-127.0.0.1}"
LRM_COMPLETION_PORT="${LRM_COMPLETION_PORT:-8001}"
LRM_COMPLETION_GPU_ID="${LRM_COMPLETION_GPU_ID:-0}"
LRM_COMPLETION_USE_REFERENCES="${LRM_COMPLETION_USE_REFERENCES:-true}"
LRM_COMPLETION_BACKEND="${LRM_COMPLETION_BACKEND:-progress}"
LRM_COMPLETION_THRESHOLD="${LRM_COMPLETION_THRESHOLD:-0.9}"
LRM_COMPLETION_INITIAL_IMAGE="${LRM_COMPLETION_INITIAL_IMAGE:-}"
LRM_COMPLETION_GOAL_IMAGE="${LRM_COMPLETION_GOAL_IMAGE:-${PROJECT_ROOT}/../Large-Reward-Models/frame_000229.jpg}"
export LRM_COMPLETION_SCORE_THRESHOLD="${LRM_COMPLETION_THRESHOLD}"

USE_REFERENCES=false
if [[ "${LRM_COMPLETION_USE_REFERENCES,,}" == "true" || "${LRM_COMPLETION_USE_REFERENCES}" == "1" ]]; then
  USE_REFERENCES=true
fi
if [[ "${LRM_COMPLETION_BACKEND}" != "yesno" && "${LRM_COMPLETION_BACKEND}" != "progress" ]]; then
  echo "LRM_COMPLETION_BACKEND must be yesno or progress" >&2
  exit 2
fi
if [[ "${USE_REFERENCES}" == "true" && "${LRM_COMPLETION_BACKEND}" != "progress" ]]; then
  echo "reference images require LRM_COMPLETION_BACKEND=progress" >&2
  exit 2
fi
if [[ "${USE_REFERENCES}" == "true" && ! -f "${LRM_COMPLETION_GOAL_IMAGE}" ]]; then
  echo "completion goal image does not exist: ${LRM_COMPLETION_GOAL_IMAGE}" >&2
  exit 2
fi
if [[ "${LRM_COMPLETION_BACKEND}" == "yesno" && "${LRM_COMPLETION_MODEL_PATH,,}" != *completion* ]]; then
  echo "yesno backend requires a completion checkpoint: ${LRM_COMPLETION_MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -d "${LRM_COMPLETION_MODEL_PATH}" ]]; then
  echo "completion checkpoint directory does not exist: ${LRM_COMPLETION_MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -f "${LRM_OFFICIAL_SERVER_PY}" ]]; then
  echo "LRM backend does not exist: ${LRM_OFFICIAL_SERVER_PY}" >&2
  exit 2
fi
if ! command -v "${LRM_PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${LRM_PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${LRM_PYTHON_BIN}" >&2
  exit 2
fi

REFERENCE_ARGS=()
if [[ "${USE_REFERENCES}" == "true" ]]; then
  [[ -n "${LRM_COMPLETION_INITIAL_IMAGE}" ]] && REFERENCE_ARGS+=(--completion_initial_image "${LRM_COMPLETION_INITIAL_IMAGE}")
  REFERENCE_ARGS+=(--completion_goal_image "${LRM_COMPLETION_GOAL_IMAGE}")
fi

printf '[completion-server] model_path=%s backend=%s references=%s official_server_py=%s\n' \
  "${LRM_COMPLETION_MODEL_PATH}" \
  "${LRM_COMPLETION_BACKEND}" \
  "${USE_REFERENCES}" \
  "${LRM_OFFICIAL_SERVER_PY}"

exec "${LRM_PYTHON_BIN}" reward_model/lrm_server/dccp_lrm_server.py \
  --mode completion \
  --model_path "${LRM_COMPLETION_MODEL_PATH}" \
  --base_model_path "${LRM_BASE_MODEL_PATH}" \
  --official_server_py "${LRM_OFFICIAL_SERVER_PY}" \
  --gpu_id "${LRM_COMPLETION_GPU_ID}" \
  --host "${LRM_HOST}" \
  --port "${LRM_COMPLETION_PORT}" \
  --completion_backend "${LRM_COMPLETION_BACKEND}" \
  --completion_threshold "${LRM_COMPLETION_THRESHOLD}" \
  "${REFERENCE_ARGS[@]}"
