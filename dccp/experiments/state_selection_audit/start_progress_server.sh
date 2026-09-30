#!/usr/bin/env bash
# 建议在 neu_lab4 或另一张空闲 GPU 上启动；选择阶段结束后即可停止。
set -euo pipefail

DCCP_PACKAGE_ROOT="${DCCP_PACKAGE_ROOT:-/path/to/DCCP/dccp}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LRM_PROGRESS_GPU_ID="${LRM_PROGRESS_GPU_ID:-0}"
export LRM_PROGRESS_MODEL_PATH="${LRM_PROGRESS_MODEL_PATH:-/path/to/assets/lrm/model/progress}"
export LRM_OFFICIAL_SERVER_PY="${LRM_OFFICIAL_SERVER_PY:-/path/to/DCCP/Large-Reward-Models/vlm_reward/vlm_reward_server.py}"
cd "${DCCP_PACKAGE_ROOT}"
exec bash reward_model/lrm_server/start_progress_server.sh
