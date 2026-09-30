"""后台总控的纯逻辑回归测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "07_run_to_completion.py"
SPEC = importlib.util.spec_from_file_location("state_audit_automation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUTOMATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUTOMATION)


def _snapshot(**updates):
    payload = {
        "restore": "PASSED",
        "collect": "DONE",
        "select": "FROZEN",
        "evaluate": "DONE",
        "summarize": "DONE",
    }
    payload.update(updates)
    return payload


def test_phase_routes_first_incomplete_stage():
    assert AUTOMATION._phase(_snapshot(restore="PENDING")) == "restore"
    assert AUTOMATION._phase(_snapshot(collect="PENDING")) == "collect"
    assert AUTOMATION._phase(_snapshot(select="PENDING")) == "select"
    assert AUTOMATION._phase(_snapshot(evaluate="STARTED")) == "evaluate"
    assert AUTOMATION._phase(_snapshot(summarize="PENDING")) == "summarize"
    assert AUTOMATION._phase(_snapshot()) == "complete"


def test_atomic_status_roundtrip(tmp_path):
    path = tmp_path / "automation_status.json"
    AUTOMATION._write_json_atomic(path, {"status": "RUNNING", "ok": 7})
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert '"ok": 7' in path.read_text(encoding="utf-8")



def test_progress_health_treats_offline_service_as_wait(monkeypatch):
    class Config:
        class Paths:
            progress_lrm_config = "/tmp/progress.yaml"

        class Runtime:
            progress_health_timeout_sec = 1.0

        paths = Paths()
        runtime = Runtime()

    def unavailable(*_args, **_kwargs):
        raise ConnectionRefusedError("offline")

    monkeypatch.setattr(AUTOMATION, "check_progress_service", unavailable)
    ready, health, error = AUTOMATION._progress_health(Config())
    assert ready is False
    assert health is None
    assert "ConnectionRefusedError" in error


def test_status_reports_waiting_for_lrm(tmp_path):
    path = tmp_path / "automation_status.json"
    AUTOMATION._write_json_atomic(
        path,
        {
            "status": "WAITING_FOR_LRM",
            "lrm_required": True,
            "lrm_error": "offline",
            "resume_automatic": True,
        },
    )
    payload = AUTOMATION._merge_controller_status(
        path, _snapshot(select="PENDING", evaluate="PENDING", summarize="PENDING")
    )
    assert payload["phase"] == "select"
    assert payload["controller_status"] == "WAITING_FOR_LRM"
    assert payload["lrm_required"] is True
    assert payload["resume_automatic"] is True
