#!/usr/bin/env bash
# Deliberately stage-by-stage: no default "run everything" path.
set -euo pipefail
umask 002

AUDIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${DCCP_PYTHON:-python}"
CONFIG="${AUDIT_CONFIG:-${AUDIT_ROOT}/configs/coffee_smoke.yaml}"
STAGE="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi

OUTPUT_ROOT="/path/to/DCCP/dccp/outputs/branch_validity_audit"
mkdir -p "${OUTPUT_ROOT}"
chgrp neu_lab2 "${OUTPUT_ROOT}"
chmod 2775 "${OUTPUT_ROOT}"

export PYTHONPATH="/path/to/DCCP/dccp:${AUDIT_ROOT}:${AUDIT_ROOT}/../state_selection_audit${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_PY_MUJOCO_PATH="/path/to/mujoco210"
export LD_LIBRARY_PATH="/path/to/mujoco210/bin:/usr/lib/nvidia${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"

case "${STAGE}" in
  setup)       exec "${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/00_validate_setup.py" --config "${CONFIG}" "$@" ;;
  states)      exec "${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/01_prepare_states.py" --config "${CONFIG}" "$@" ;;
  candidates)  exec "${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/02_generate_candidates.py" --config "${CONFIG}" "$@" ;;
  wm)          exec "${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/03_predict_world_model.py" --config "${CONFIG}" "$@" ;;
  freeze)      exec "${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/04_score_and_freeze.py" --config "${CONFIG}" "$@" ;;
  simulator)   exec "${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/05_evaluate_simulator.py" --config "${CONFIG}" "$@" ;;
  metrics)     exec "${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/06_compute_metrics.py" --config "${CONFIG}" "$@" ;;
  inspect)     exec "${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/07_visualize.py" --config "${CONFIG}" "$@" ;;
  *) echo "Usage: $0 {setup|states|candidates|wm|freeze|simulator|metrics|inspect} [options]" >&2; exit 2 ;;
esac
