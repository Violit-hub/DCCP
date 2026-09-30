#!/usr/bin/env bash
# 分阶段入口。为避免误触发数百次 rollout，不提供无参数的一键全跑。
set -euo pipefail
# neu_lab2 与 neu_lab4 共享 neu_lab2 组；只给该组续跑写权限。
umask 002

AUDIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${DCCP_PYTHON:-python}"
CONFIG="${AUDIT_CONFIG:-${AUDIT_ROOT}/configs/coffee_smoke.yaml}"
STAGE="${1:-}"
if [[ $# -gt 0 ]]; then
  shift
fi
OUTPUT_ROOT="/path/to/DCCP/dccp/outputs/state_selection_audit"

# setgid 让 neu_lab2/neu_lab4 新建的 run 都继承共同组。
mkdir -p "${OUTPUT_ROOT}"
chgrp neu_lab2 "${OUTPUT_ROOT}"
chmod 2775 "${OUTPUT_ROOT}"

export PYTHONPATH="/path/to/DCCP/dccp:${AUDIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_PY_MUJOCO_PATH="/path/to/mujoco210"
export LD_LIBRARY_PATH="/path/to/mujoco210/bin:/usr/lib/nvidia${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"

case "${STAGE}" in
  setup)     exec "${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/00_validate_setup.py" --config "${CONFIG}" "$@" ;;
  restore)   exec "${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/01_validate_state_restore.py" --config "${CONFIG}" "$@" ;;
  collect)   exec "${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/02_collect_nominal.py" --config "${CONFIG}" "$@" ;;
  select)    exec "${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/03_select_states.py" --config "${CONFIG}" "$@" ;;
  evaluate)  exec "${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/04_evaluate_all_states.py" --config "${CONFIG}" "$@" ;;
  summarize) exec "${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/05_summarize.py" --config "${CONFIG}" "$@" ;;
  inspect)   exec "${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/06_visualize.py" --config "${CONFIG}" "$@" ;;
  *)
    echo "用法: $0 {setup|restore|collect|select|evaluate|summarize|inspect} [阶段参数]" >&2
    exit 2
    ;;
esac
