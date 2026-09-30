"""实验阶段闸门和可视化状态摘要。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifact_store import ArtifactStore
from .config import AuditConfig


def require_restore_gate(cfg: AuditConfig) -> dict[str, Any]:
    """只有状态恢复报告明确通过后，才允许采集 nominal。"""
    report_path = cfg.run_dir / "reports" / "state_restore_gate.json"
    if not report_path.exists():
        raise RuntimeError(
            "状态恢复闸门尚未执行；请先运行 ./run_coffee_smoke.sh restore，"
            "并人工检查 reports/state_restore_gate.json"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise RuntimeError(f"状态恢复闸门未通过，禁止采集 nominal: {report_path}")
    return report


def stage_status(cfg: AuditConfig) -> list[dict[str, Any]]:
    """根据稳定产物推断阶段状态，供 HTML 检查页和人工验收使用。"""
    store = ArtifactStore(cfg.run_dir)
    restore_path = store.report_dir / "state_restore_gate.json"
    trajectory_path = store.nominal_dir / "trajectories.json"
    frozen_path = store.selection_dir / "SELECTIONS_FROZEN.json"
    branch_path = store.branch_dir / "branch_results.jsonl"
    summary_path = store.report_dir / "summary.json"

    restore_passed = False
    if restore_path.exists():
        restore_passed = json.loads(restore_path.read_text(encoding="utf-8")).get("passed") is True
    trajectory_count = 0
    if trajectory_path.exists():
        trajectory_count = len(json.loads(trajectory_path.read_text(encoding="utf-8")))
    branch_rows = ArtifactStore.read_jsonl(branch_path)
    ok_keys = {
        (
            str(row.get("trajectory_id")),
            int(row.get("decision_index", -1)),
            int(row.get("candidate_index", -1)),
            int(row.get("evaluation_seed", -1)),
        )
        for row in branch_rows
        if row.get("status", "OK") == "OK"
    }
    failed_keys = {
        (
            str(row.get("trajectory_id")),
            int(row.get("decision_index", -1)),
            int(row.get("candidate_index", -1)),
            int(row.get("evaluation_seed", -1)),
        )
        for row in branch_rows
        if row.get("status", "OK") != "OK"
    }
    selection_rows = ArtifactStore.read_jsonl(store.selection_dir / "selected_states.jsonl")
    eligible_by_traj: dict[str, set[int]] = {}
    for row in selection_rows:
        eligible_by_traj.setdefault(str(row["trajectory_id"]), set()).update(
            int(index) for index in row.get("eligible_indices", [])
        )
    expected_branches = (
        sum(len(indices) for indices in eligible_by_traj.values())
        * cfg.pilot.total_candidates
        * len(cfg.pilot.evaluation_seeds)
    )
    unresolved_failed = failed_keys - ok_keys
    if expected_branches > 0 and len(ok_keys) >= expected_branches:
        evaluation_state = "DONE"
    elif branch_rows:
        evaluation_state = "STARTED"
    else:
        evaluation_state = "PENDING"

    return [
        {"stage": "setup", "state": "READY", "detail": "配置已成功加载"},
        {
            "stage": "restore",
            "state": "PASSED" if restore_passed else "PENDING",
            "detail": str(restore_path),
        },
        {
            "stage": "collect",
            "state": "DONE" if trajectory_count else "PENDING",
            "detail": f"{trajectory_count}/{cfg.pilot.num_trajectories} trajectories",
        },
        {
            "stage": "select",
            "state": "FROZEN" if frozen_path.exists() else "PENDING",
            "detail": str(frozen_path),
        },
        {
            "stage": "evaluate",
            "state": evaluation_state,
            "detail": (
                f"OK={len(ok_keys)}/{expected_branches}, "
                f"unresolved FAILED={len(unresolved_failed)}"
            ),
        },
        {
            "stage": "summarize",
            "state": "DONE" if summary_path.exists() else "PENDING",
            "detail": str(summary_path),
        },
    ]
