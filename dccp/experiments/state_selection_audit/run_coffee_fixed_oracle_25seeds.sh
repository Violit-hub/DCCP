#!/usr/bin/env bash
set -euo pipefail
umask 002

AUDIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${DCCP_PYTHON:-python}"
CONFIG="${AUDIT_CONFIG:-${AUDIT_ROOT}/configs/coffee_fixed_oracle_s3_s6_25seeds_v1.yaml}"
SOURCE_RUN="${STABILITY_SOURCE_RUN:-/path/to/DCCP/dccp/outputs/state_selection_audit/coffee_one_traj_v2}"
PREPARER="${AUDIT_ROOT}/scripts/08_prepare_fixed_seed_stability.py"
ORCHESTRATOR="${AUDIT_ROOT}/scripts/07_run_to_completion.py"
RUN_DIR="$("${PYTHON_BIN}" "${ORCHESTRATOR}" --config "${CONFIG}" --print-run-dir)"
LOG_DIR="${RUN_DIR}/automation"
LOG_FILE="${LOG_DIR}/run.log"
PID_FILE="${LOG_DIR}/run.pid"
STATUS_FILE="${RUN_DIR}/automation_status.json"
LOCK_FILE="${RUN_DIR}/automation.lock"
STOP_FILE="${LOG_DIR}/STOP_REQUESTED"
ACTION="${1:-start}"

prepare_run() {
  "${PYTHON_BIN}" "${PREPARER}" \
    --config "${CONFIG}" \
    --source-run-dir "${SOURCE_RUN}" \
    --trajectory-id traj_0000 \
    --decision-index 3 \
    --decision-index 6
  mkdir -p "${LOG_DIR}"
  chmod 2775 "${LOG_DIR}"
}

is_running() {
  if flock -n "${LOCK_FILE}" -c true; then
    return 1
  fi
  return 0
}

case "${ACTION}" in
  prepare)
    prepare_run
    ;;
  run)
    prepare_run
    exec "${PYTHON_BIN}" "${ORCHESTRATOR}" \
      --config "${CONFIG}" --poll-seconds 10 --evaluate-chunk-size 8
    ;;
  start)
    prepare_run
    if is_running; then
      echo "25-seed 自动化总控已运行"
      exit 0
    fi
    rm -f "${STOP_FILE}"
    touch "${LOG_FILE}"
    printf '\n[%s] launcher=start\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"${LOG_FILE}"
    nohup setsid env PYTHONUNBUFFERED=1 \
      "${PYTHON_BIN}" "${ORCHESTRATOR}" \
      --config "${CONFIG}" --poll-seconds 10 --evaluate-chunk-size 8 \
      >>"${LOG_FILE}" 2>&1 </dev/null &
    pid=$!
    echo "${pid}" >"${PID_FILE}"
    chmod 660 "${LOG_FILE}" "${PID_FILE}"
    echo "已启动: PID ${pid}"
    echo "独立实验目录: ${RUN_DIR}"
    echo "固定状态: 3, 6；固定候选: 每状态 8 个；新配对 seeds: 25 个"
    echo "计划分支数: 400"
    echo "日志: ${LOG_FILE}"
    echo "状态: $0 status"
    echo "本实验已冻结 selection，不需要启动 LRM"
    ;;
  status)
    if [[ ! -f "${RUN_DIR}/fixed_seed_stability.json" ]]; then
      echo "process=NOT_PREPARED"
      echo "先运行: $0 prepare"
      exit 0
    fi
    if is_running; then
      echo "process=RUNNING"
    else
      echo "process=STOPPED"
    fi
    "${PYTHON_BIN}" "${ORCHESTRATOR}" --config "${CONFIG}" --status-only
    if [[ -f "${STATUS_FILE}" ]]; then
      echo "automation_status=${STATUS_FILE}"
    fi
    ;;
  log)
    prepare_run
    touch "${LOG_FILE}"
    exec tail -n 50 -f "${LOG_FILE}"
    ;;
  stop)
    if [[ ! -e "${LOCK_FILE}" ]] || ! is_running; then
      echo "25-seed 自动化总控未运行"
      exit 0
    fi
    touch "${STOP_FILE}"
    chmod 660 "${STOP_FILE}"
    echo "已请求停止；当前 evaluate 小块结束后生效"
    ;;
  *)
    echo "用法: $0 {prepare|start|status|log|stop|run}" >&2
    exit 2
    ;;
esac
