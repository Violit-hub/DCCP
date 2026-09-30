"""完整性审计、指标汇总与 CSV/JSON 报告。"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from typing import Any

import numpy as np

from .artifact_store import ArtifactStore
from .config import AuditConfig
from .metrics import compute_state_metrics, summarize_methods
from .seeding import derive_seed


def _validate_complete(
    cfg: AuditConfig,
    selections: list[dict[str, Any]],
    branches: list[dict[str, Any]],
) -> None:
    eligible: dict[str, set[int]] = defaultdict(set)
    for row in selections:
        eligible[str(row["trajectory_id"])].update(int(x) for x in row["eligible_indices"])
    expected = {
        (trajectory_id, decision_index, candidate_index, seed)
        for trajectory_id, indices in eligible.items()
        for decision_index in indices
        for candidate_index in range(cfg.pilot.total_candidates)
        for seed in cfg.pilot.evaluation_seeds
    }
    ok = {
        (
            str(row["trajectory_id"]),
            int(row["decision_index"]),
            int(row["candidate_index"]),
            int(row["evaluation_seed"]),
        )
        for row in branches
        if row.get("status", "OK") == "OK"
    }
    missing = sorted(expected - ok)
    failed_keys = {
        (
            str(row["trajectory_id"]),
            int(row["decision_index"]),
            int(row["candidate_index"]),
            int(row["evaluation_seed"]),
        )
        for row in branches
        if row.get("status", "OK") != "OK"
    }
    unresolved_failed = failed_keys - ok
    if missing or unresolved_failed:
        preview = missing[:5]
        raise RuntimeError(
            f"branch 结果不完整：missing={len(missing)}, failed={len(unresolved_failed)}, "
            f"missing_preview={preview}；请重跑 04_evaluate_all_states.py"
        )


def _with_bootstrap_ci(
    summary: list[dict[str, Any]],
    per_trajectory: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    for row in summary:
        method_rows = [item for item in per_trajectory if item["method"] == row["method"]]
        enriched = dict(row)
        if not method_rows or samples <= 0:
            output.append(enriched)
            continue
        rng = np.random.default_rng(derive_seed(seed, row["method"], "bootstrap"))
        indices = rng.integers(0, len(method_rows), size=(samples, len(method_rows)))
        for field in ("mean_improvement", "opportunity_rate", "paired_fail_to_success_rate"):
            values = np.asarray([float(item[field]) for item in method_rows])
            estimates = values[indices].mean(axis=1)
            enriched[f"{field}_ci95_low"] = float(np.quantile(estimates, 0.025))
            enriched[f"{field}_ci95_high"] = float(np.quantile(estimates, 0.975))
        output.append(enriched)
    return output


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def build_reports(cfg: AuditConfig) -> dict[str, Any]:
    store = ArtifactStore(cfg.run_dir)
    selections = ArtifactStore.read_jsonl(store.selection_dir / "selected_states.jsonl")
    branches = ArtifactStore.read_jsonl(store.branch_dir / "branch_results.jsonl")
    if not selections:
        raise RuntimeError("没有冻结的选择结果")
    _validate_complete(cfg, selections, branches)
    state_metrics = compute_state_metrics(branches)
    per_trajectory, summary = summarize_methods(
        state_metrics,
        selections,
        state_budget=cfg.selection.state_budget,
        nms_gap=cfg.selection.nms_gap,
        random_draws=cfg.selection.random_draws,
        random_seed=cfg.pilot.candidate_seed,
    )
    summary = _with_bootstrap_ci(
        summary,
        per_trajectory,
        samples=cfg.pilot.bootstrap_samples,
        seed=cfg.pilot.candidate_seed,
    )
    ArtifactStore.write_jsonl_atomic(
        store.report_dir / "state_metrics.jsonl", [item.to_dict() for item in state_metrics]
    )
    ArtifactStore.write_json_atomic(store.report_dir / "per_trajectory.json", per_trajectory)
    ArtifactStore.write_json_atomic(store.report_dir / "summary.json", summary)
    ArtifactStore.write_text_atomic(
        store.report_dir / "per_trajectory.csv", _csv_text(per_trajectory)
    )
    ArtifactStore.write_text_atomic(store.report_dir / "summary.csv", _csv_text(summary))
    return {
        "num_states": len(state_metrics),
        "num_branch_rows": len(branches),
        "summary": summary,
    }
