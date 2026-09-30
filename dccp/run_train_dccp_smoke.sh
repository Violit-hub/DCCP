#!/usr/bin/env bash
set -euo pipefail
# Reduced run of the same real pipeline. Requires models, data, CUDA, and both LRMs.
# Its one-state/two-candidate settings are for interface checks, not paper results.
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-1}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1}
export SAVE_FREQ=${SAVE_FREQ:--1}
export DCCP_STATE_BUDGET_PER_TRAJ=${DCCP_STATE_BUDGET_PER_TRAJ:-1}
export DCCP_SELECTED_STATES=${DCCP_SELECTED_STATES:-1}
export DCCP_NUM_CANDIDATES=${DCCP_NUM_CANDIDATES:-2}
export DCCP_HORIZON_H=${DCCP_HORIZON_H:-1}
export DCCP_BRANCH_HORIZON=${DCCP_BRANCH_HORIZON:-1}
export DCCP_MARGIN_POS=${DCCP_MARGIN_POS:-0.02}
export DCCP_MARGIN_NEG=${DCCP_MARGIN_NEG:-0.02}
exec bash "$SCRIPT_ROOT/run_train_dccp_full.sh" "$@"
