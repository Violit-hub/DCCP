#!/usr/bin/env bash
set -euo pipefail
umask 002

AUDIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${DCCP_PYTHON:-python}"
CONFIG="${AUDIT_CONFIG:-${AUDIT_ROOT}/configs/coffee_one_traj_v2.yaml}"
RUN_DIR="$("${PYTHON_BIN}" "${AUDIT_ROOT}/scripts/07_run_to_completion.py" \
  --config "${CONFIG}" --print-run-dir)"
LOG_DIR="${RUN_DIR}/automation"
LOG_FILE="${LOG_DIR}/run.log"
PID_FILE="${LOG_DIR}/run.pid"
STATUS_FILE="${RUN_DIR}/automation_status.json"
LOCK_FILE="${RUN_DIR}/automation.lock"
STOP_FILE="${LOG_DIR}/STOP_REQUESTED"
ORCHESTRATOR="${AUDIT_ROOT}/scripts/07_run_to_completion.py"
ACTION="${1:-start}"

mkdir -p "${LOG_DIR}"
chmod 2775 "${LOG_DIR}"

is_running() {
  if flock -n "${LOCK_FILE}" -c true; then
    return 1
  fi
  return 0
}

case "${ACTION}" in
  run)
    exec "${PYTHON_BIN}" "${ORCHESTRATOR}" --config "${CONFIG}"
    ;;
  start)
    if is_running; then
      echo "自动化总控已运行"
      exit 0
    fi
    rm -f "${STOP_FILE}"
    : >"${LOG_FILE}"
    nohup setsid env PYTHONUNBUFFERED=1 \
      "${PYTHON_BIN}" "${ORCHESTRATOR}" --config "${CONFIG}" \
      >>"${LOG_FILE}" 2>&1 </dev/null &
    pid=$!
    echo "${pid}" >"${PID_FILE}"
    chmod 660 "${LOG_FILE}" "${PID_FILE}"
    echo "已启动: PID ${pid}"
    echo "实验目录: ${RUN_DIR}"
    echo "日志: ${LOG_FILE}"
    echo "状态: $0 status"
    echo "到 select 阶段若 LRM 未启动，会显示 WAITING_FOR_LRM 并自动等待"
    ;;
  status)
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
    touch "${LOG_FILE}"
    exec tail -n 50 -f "${LOG_FILE}"
    ;;
  stop)
    if ! is_running; then
      echo "自动化总控未运行"
      exit 0
    fi
    touch "${STOP_FILE}"
    chmod 660 "${STOP_FILE}"
    echo "已请求停止；当前 evaluate 小块结束后生效"
    ;;
  *)
    echo "用法: $0 {start|status|log|stop|run}" >&2
    exit 2
    ;;
esac
