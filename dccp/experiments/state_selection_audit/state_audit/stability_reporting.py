"""固定状态、固定候选、全新配对 seed 的确认性稳定性统计。"""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .artifact_store import ArtifactStore
from .config import AuditConfig
from .seeding import derive_seed


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _exact_mcnemar_p(rescued: int, harmed: int) -> float:
    discordant = rescued + harmed
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(rescued, harmed) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def compute_confirmatory_state(
    *,
    trajectory_id: str,
    decision_index: int,
    confirmatory_candidate_index: int,
    nominal_outcomes: dict[int, bool],
    candidate_outcomes: dict[int, bool],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """比较预先固定的 candidate 和 nominal；不在新 seed 上重新挑 winner。"""
    nominal_seeds = set(nominal_outcomes)
    candidate_seeds = set(candidate_outcomes)
    if nominal_seeds != candidate_seeds or not nominal_seeds:
        raise ValueError(
            f"确认性比较需要完全相同且非空的 paired seeds: "
            f"nominal={sorted(nominal_seeds)}, candidate={sorted(candidate_seeds)}"
        )
    seeds = sorted(nominal_seeds)
    nominal = np.asarray([bool(nominal_outcomes[seed]) for seed in seeds], dtype=np.int8)
    candidate = np.asarray([bool(candidate_outcomes[seed]) for seed in seeds], dtype=np.int8)
    differences = candidate - nominal
    rescued = int(np.sum((nominal == 0) & (candidate == 1)))
    harmed = int(np.sum((nominal == 1) & (candidate == 0)))
    both_success = int(np.sum((nominal == 1) & (candidate == 1)))
    both_fail = int(np.sum((nominal == 0) & (candidate == 0)))
    total = len(seeds)
    nominal_successes = int(nominal.sum())
    candidate_successes = int(candidate.sum())
    nominal_ci = _wilson_interval(nominal_successes, total)
    candidate_ci = _wilson_interval(candidate_successes, total)

    rng = np.random.default_rng(bootstrap_seed)
    sample_count = max(int(bootstrap_samples), 1)
    sampled = rng.integers(0, total, size=(sample_count, total))
    bootstrapped_differences = differences[sampled].mean(axis=1)
    delta_ci = (
        float(np.quantile(bootstrapped_differences, 0.025)),
        float(np.quantile(bootstrapped_differences, 0.975)),
    )
    mcnemar_p = _exact_mcnemar_p(rescued, harmed)
    return {
        "trajectory_id": trajectory_id,
        "decision_index": int(decision_index),
        "nominal_candidate_index": 0,
        "confirmatory_candidate_index": int(confirmatory_candidate_index),
        "paired_seed_count": total,
        "nominal_successes": nominal_successes,
        "candidate_successes": candidate_successes,
        "nominal_success_rate": float(nominal_successes / total),
        "nominal_success_rate_ci95_low": nominal_ci[0],
        "nominal_success_rate_ci95_high": nominal_ci[1],
        "candidate_success_rate": float(candidate_successes / total),
        "candidate_success_rate_ci95_low": candidate_ci[0],
        "candidate_success_rate_ci95_high": candidate_ci[1],
        "paired_improvement": float(differences.mean()),
        "paired_improvement_ci95_low": delta_ci[0],
        "paired_improvement_ci95_high": delta_ci[1],
        "rescued_nominal_failures": rescued,
        "harmed_nominal_successes": harmed,
        "both_success": both_success,
        "both_fail": both_fail,
        "exact_mcnemar_p": mcnemar_p,
    }


def _latest_ok_rows(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    latest: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("status", "OK") != "OK":
            continue
        key = (
            str(row["trajectory_id"]),
            int(row["decision_index"]),
            int(row["candidate_index"]),
            int(row["evaluation_seed"]),
        )
        latest[key] = row
    return latest


def _source_confirmatory_candidates(
    source_run: Path,
    trajectory_id: str,
    decision_indices: list[int],
) -> dict[int, int]:
    rows = ArtifactStore.read_jsonl(source_run / "reports" / "state_metrics.jsonl")
    output = {
        int(row["decision_index"]): int(row["best_candidate_index"])
        for row in rows
        if str(row["trajectory_id"]) == trajectory_id
        and int(row["decision_index"]) in decision_indices
    }
    missing = sorted(set(decision_indices) - set(output))
    if missing:
        raise RuntimeError(f"source state_metrics 缺少预注册候选: {missing}")
    return output


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def build_fixed_seed_stability_report(cfg: AuditConfig) -> dict[str, Any] | None:
    store = ArtifactStore(cfg.run_dir)
    provenance_path = store.run_dir / "fixed_seed_stability.json"
    if not provenance_path.exists():
        return None
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    trajectory_id = str(provenance["trajectory_id"])
    decision_indices = [int(value) for value in provenance["decision_indices"]]
    source_run = Path(str(provenance["source_run_dir"])).resolve()
    confirmatory = _source_confirmatory_candidates(
        source_run, trajectory_id, decision_indices
    )
    latest = _latest_ok_rows(
        ArtifactStore.read_jsonl(store.branch_dir / "branch_results.jsonl")
    )
    expected_seeds = [int(seed) for seed in cfg.pilot.evaluation_seeds]

    state_rows: list[dict[str, Any]] = []
    exploratory: list[dict[str, Any]] = []
    for decision_index in decision_indices:
        candidate_rates: dict[int, float] = {}
        outcomes_by_candidate: dict[int, dict[int, bool]] = {}
        for candidate_index in range(cfg.pilot.total_candidates):
            outcomes: dict[int, bool] = {}
            for seed in expected_seeds:
                key = (trajectory_id, decision_index, candidate_index, seed)
                if key not in latest:
                    raise RuntimeError(f"稳定性报告缺少已完成分支: {key}")
                outcomes[seed] = bool(latest[key]["success"])
            outcomes_by_candidate[candidate_index] = outcomes
            candidate_rates[candidate_index] = float(np.mean(list(outcomes.values())))
            exploratory.append(
                {
                    "trajectory_id": trajectory_id,
                    "decision_index": decision_index,
                    "candidate_index": candidate_index,
                    "is_nominal": candidate_index == 0,
                    "is_confirmatory": candidate_index == confirmatory[decision_index],
                    "successes": int(sum(outcomes.values())),
                    "paired_seed_count": len(outcomes),
                    "success_rate": candidate_rates[candidate_index],
                }
            )
        row = compute_confirmatory_state(
            trajectory_id=trajectory_id,
            decision_index=decision_index,
            confirmatory_candidate_index=confirmatory[decision_index],
            nominal_outcomes=outcomes_by_candidate[0],
            candidate_outcomes=outcomes_by_candidate[confirmatory[decision_index]],
            bootstrap_samples=max(cfg.pilot.bootstrap_samples, 10000),
            bootstrap_seed=derive_seed(
                cfg.run_name, trajectory_id, decision_index, "confirmatory-bootstrap"
            ),
        )
        exploratory_best = max(
            sorted(candidate_rates), key=lambda index: (candidate_rates[index], -index)
        )
        row["exploratory_best_candidate_index"] = int(exploratory_best)
        row["exploratory_best_success_rate"] = candidate_rates[exploratory_best]
        state_rows.append(row)

    correction_factor = max(len(state_rows), 1)
    for row in state_rows:
        row["bonferroni_mcnemar_p"] = min(
            1.0, float(row["exact_mcnemar_p"]) * correction_factor
        )
        row["confirmatory_sensitive"] = bool(
            row["paired_improvement"] > 0
            and row["paired_improvement_ci95_low"] > 0
            and row["bonferroni_mcnemar_p"] < 0.05
        )

    report = {
        "experiment": "fixed-state/fixed-candidate fresh-seed stability",
        "interpretation": (
            "Confirmatory comparisons use the candidate selected by the source run; "
            "the best candidate on the new seeds is exploratory only."
        ),
        "source_run_dir": str(source_run),
        "new_seed_count": len(expected_seeds),
        "num_confirmatory_tests": len(state_rows),
        "multiple_testing_correction": "Bonferroni over fixed states",
        "states": state_rows,
        "all_candidate_rates_exploratory": exploratory,
    }
    ArtifactStore.write_json_atomic(
        store.report_dir / "fixed_seed_stability.json", report
    )
    ArtifactStore.write_text_atomic(
        store.report_dir / "fixed_seed_stability.csv", _csv_text(state_rows)
    )
    ArtifactStore.write_text_atomic(
        store.report_dir / "fixed_seed_candidate_rates_exploratory.csv",
        _csv_text(exploratory),
    )
    return report
