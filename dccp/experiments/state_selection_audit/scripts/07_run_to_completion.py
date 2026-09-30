#!/usr/bin/env python3
"""自动完成一次可断点续跑的状态选择实验。"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_ROOT = Path(__file__).resolve().parents[1]
if str(AUDIT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIT_ROOT))

from state_audit.artifact_store import ArtifactStore  # noqa: E402
from state_audit.config import load_config  # noqa: E402
from state_audit.progress import check_progress_service  # noqa: E402
from state_audit.stage_gates import stage_status  # noqa: E402


EVALUATE_RE = re.compile(
    r"OK=(?P<ok>\d+)/(?P<expected>\d+), unresolved FAILED=(?P<failed>\d+)"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)
    path.chmod(0o660)


def _snapshot(cfg) -> dict[str, Any]:
    rows = stage_status(cfg)
    stages = {str(row["stage"]): row for row in rows}
    match = EVALUATE_RE.fullmatch(str(stages["evaluate"]["detail"]))
    if match is None:
        raise RuntimeError(f"无法解析 evaluate 状态: {stages['evaluate']['detail']}")
    return {
        "updated_at_utc": _utc_now(),
        "run_name": cfg.run_name,
        "run_dir": str(cfg.run_dir),
        "pid": os.getpid(),
        "restore": stages["restore"]["state"],
        "collect": stages["collect"]["state"],
        "select": stages["select"]["state"],
        "evaluate": stages["evaluate"]["state"],
        "summarize": stages["summarize"]["state"],
        "ok_branches": int(match.group("ok")),
        "expected_branches": int(match.group("expected")),
        "unresolved_failed": int(match.group("failed")),
    }


def _phase(snapshot: dict[str, Any]) -> str:
    if snapshot["restore"] != "PASSED":
        return "restore"
    if snapshot["collect"] != "DONE":
        return "collect"
    if snapshot["select"] != "FROZEN":
        return "select"
    if snapshot["evaluate"] != "DONE":
        return "evaluate"
    if snapshot["summarize"] != "DONE":
        return "summarize"
    return "complete"


def _publish(status_path: Path, snapshot: dict[str, Any], **extra: Any) -> None:
    payload = dict(snapshot)
    payload.update(extra)
    _write_json_atomic(status_path, payload)


def _progress_health(cfg) -> tuple[bool, dict[str, Any] | None, str | None]:
    """探测 Progress LRM；服务未启动是正常等待状态，不算阶段错误。"""
    try:
        health = check_progress_service(
            cfg.paths.progress_lrm_config,
            cfg.runtime.progress_health_timeout_sec,
        )
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"
    return True, health, None


def _merge_controller_status(
    status_path: Path, snapshot: dict[str, Any]
) -> dict[str, Any]:
    """让 status-only 同时显示产物阶段和后台总控的等待状态。"""
    payload = dict(snapshot)
    payload["phase"] = _phase(snapshot)
    if not status_path.exists():
        payload["controller_status"] = "NOT_STARTED"
        return payload
    try:
        controller = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        payload["controller_status"] = "UNKNOWN"
        payload["controller_status_error"] = f"{type(exc).__name__}: {exc}"
        return payload
    payload["controller_status"] = controller.get("status", "UNKNOWN")
    for key in (
        "lrm_required",
        "lrm_error",
        "lrm_health",
        "resume_automatic",
        "consecutive_errors",
        "error",
    ):
        if key in controller:
            payload[key] = controller[key]
    return payload


def _run_stage(config_path: Path, stage: str, *, evaluate_chunk_size: int) -> int:
    runner = AUDIT_ROOT / "run_coffee_smoke.sh"
    env = os.environ.copy()
    env["AUDIT_CONFIG"] = str(config_path)
    env["PYTHONUNBUFFERED"] = "1"
    print(f"[{_utc_now()}] stage={stage}: start", flush=True)
    command = [str(runner), stage]
    if stage == "evaluate":
        command.extend(["--max-branches", str(evaluate_chunk_size)])
    result = subprocess.run(command, cwd=AUDIT_ROOT, env=env, check=False)
    print(f"[{_utc_now()}] stage={stage}: exit={result.returncode}", flush=True)
    return int(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--lrm-poll-seconds", type=int, default=30)
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--print-run-dir", action="store_true")
    parser.add_argument("--max-consecutive-errors", type=int, default=10)
    parser.add_argument("--evaluate-chunk-size", type=int, default=8)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path, require_inputs=True)
    if args.print_run_dir:
        print(cfg.run_dir)
        return 0
    store = ArtifactStore(cfg.run_dir)
    store.ensure_layout()
    status_path = cfg.run_dir / "automation_status.json"
    stop_path = cfg.run_dir / "automation" / "STOP_REQUESTED"
    snapshot = _snapshot(cfg)
    if args.status_only:
        print(
            json.dumps(
                _merge_controller_status(status_path, snapshot),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    lock_path = cfg.run_dir / "automation.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"已有自动化总控持有锁: {lock_path}", file=sys.stderr)
        return 2

    poll_seconds = max(int(args.poll_seconds), 10)
    lrm_poll_seconds = max(int(args.lrm_poll_seconds), 10)
    max_errors = max(int(args.max_consecutive_errors), 1)
    evaluate_chunk_size = max(int(args.evaluate_chunk_size), 1)
    consecutive_errors = 0
    previous_ok = int(snapshot["ok_branches"])
    waiting_for_lrm = False
    _publish(status_path, snapshot, status="RUNNING", phase=_phase(snapshot))

    while True:
        snapshot = _snapshot(cfg)
        phase = _phase(snapshot)
        if stop_path.exists():
            _publish(status_path, snapshot, status="STOPPED", phase=phase)
            print(f"检测到停止请求: {stop_path}", flush=True)
            return 130
        if phase == "complete":
            _publish(status_path, snapshot, status="COMPLETE", phase=phase)
            print(json.dumps(snapshot, ensure_ascii=False, indent=2), flush=True)
            return 0

        lrm_health = None
        if phase == "select":
            lrm_ready, lrm_health, lrm_error = _progress_health(cfg)
            if not lrm_ready:
                _publish(
                    status_path,
                    snapshot,
                    status="WAITING_FOR_LRM",
                    phase=phase,
                    lrm_required=True,
                    lrm_error=lrm_error,
                    resume_automatic=True,
                    consecutive_errors=consecutive_errors,
                )
                if not waiting_for_lrm:
                    print(
                        f"[{_utc_now()}] 已完成 LRM 之前的阶段；"
                        "正在等待 Progress LRM。请启动服务，健康检查通过后会自动继续。",
                        flush=True,
                    )
                    print(f"LRM health error: {lrm_error}", flush=True)
                waiting_for_lrm = True
                time.sleep(lrm_poll_seconds)
                continue
            if waiting_for_lrm:
                print(
                    f"[{_utc_now()}] Progress LRM 已就绪，自动继续 select。",
                    flush=True,
                )
            waiting_for_lrm = False

        _publish(
            status_path,
            snapshot,
            status="RUNNING",
            phase=phase,
            consecutive_errors=consecutive_errors,
            lrm_required=False,
            lrm_health=lrm_health,
        )
        return_code = _run_stage(
            config_path, phase, evaluate_chunk_size=evaluate_chunk_size
        )
        latest = _snapshot(cfg)
        current_ok = int(latest["ok_branches"])
        progressed = current_ok > previous_ok or _phase(latest) != phase
        if progressed:
            consecutive_errors = 0
        elif return_code != 0:
            consecutive_errors += 1
        previous_ok = current_ok

        if consecutive_errors >= max_errors:
            _publish(
                status_path,
                latest,
                status="FAILED",
                phase=_phase(latest),
                consecutive_errors=consecutive_errors,
                error=f"连续 {consecutive_errors} 次非零退出且没有进度",
            )
            return 1

        if _phase(latest) == "complete":
            continue
        print(
            f"[{_utc_now()}] phase={_phase(latest)} "
            f"progress={latest['ok_branches']}/{latest['expected_branches']}; "
            f"{poll_seconds}s 后重试",
            flush=True,
        )
        time.sleep(poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
