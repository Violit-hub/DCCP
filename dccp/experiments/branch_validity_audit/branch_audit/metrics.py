"""Independent simulator-grounded validity metrics for frozen DCCP labels."""

from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from typing import Any

import numpy as np

from .artifacts import AuditStore
from .candidates import load_candidates
from .config import AuditConfig
from .lrm_runner import verify_frozen_labels
from .source_states import load_states


def _rankdata(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    rx, ry = _rankdata(x), _rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall_tau_b(x: list[float], y: list[float]) -> float | None:
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = np.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return float((concordant - discordant) / denominator) if denominator else None


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _latest_success(rows: list[dict]) -> dict[tuple[str, int, int], dict]:
    output = {}
    for row in rows:
        if row.get("status") == "OK":
            output[(str(row["state_id"]), int(row["candidate_index"]), int(row["evaluation_seed"]))] = row
    return output


def _bootstrap_state_mean(rows: list[dict], field: str, samples: int, seed: int) -> dict[str, float | None]:
    by_state: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None:
            by_state[str(row["state_id"])].append(float(value))
    state_means = [float(np.mean(values)) for values in by_state.values() if values]
    if not state_means:
        return {"mean": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(seed)
    draws = [float(np.mean(rng.choice(state_means, size=len(state_means), replace=True))) for _ in range(samples)]
    return {
        "mean": float(np.mean(state_means)),
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
    }


def compute_metrics(cfg: AuditConfig) -> dict[str, Any]:
    verify_frozen_labels(cfg)
    store = AuditStore(cfg.run_dir)
    labels = {str(row["state_id"]): row for row in AuditStore.read_jsonl(store.labels / "labels.jsonl")}
    outcomes = _latest_success(AuditStore.read_jsonl(store.simulator / "outcomes.jsonl"))
    expected = {
        (str(state["state_id"]), int(candidate["candidate_index"]), int(seed))
        for state in load_states(cfg)
        for candidate in load_candidates(cfg)
        if str(candidate["state_id"]) == str(state["state_id"])
        for seed in cfg.simulator.evaluation_seeds
    }
    missing = sorted(expected - set(outcomes))
    if missing:
        raise RuntimeError(f"Simulator evaluation incomplete: {len(missing)} outcomes missing; first={missing[:3]}")

    candidate_rows = []
    by_state_candidate: dict[tuple[str, int], dict[str, float]] = {}
    for state in load_states(cfg):
        state_id = str(state["state_id"])
        candidates = sorted(
            int(row["candidate_index"]) for row in load_candidates(cfg) if row["state_id"] == state_id
        )
        for candidate_index in candidates:
            rows = [outcomes[(state_id, candidate_index, int(seed))] for seed in cfg.simulator.evaluation_seeds]
            summary = {
                "success_rate": float(np.mean([bool(row["success"]) for row in rows])),
                "mean_short_progress": float(np.mean([float(row["short_progress"]) for row in rows])),
                "mean_final_progress": float(np.mean([float(row["final_progress"]) for row in rows])),
                "mean_executed_steps": float(np.mean([int(row["executed_low_level_steps"]) for row in rows])),
            }
            by_state_candidate[(state_id, candidate_index)] = summary
            candidate_rows.append({"state_id": state_id, "candidate_index": candidate_index, **summary})

    pair_rows = []
    state_rows = []
    for state_id, label in labels.items():
        predicted_scores = {
            int(key): float(value)
            for key, value in label.get("label_scores", label["candidate_mean_scores"]).items()
        }
        indices = sorted(predicted_scores)
        actual_success = [by_state_candidate[(state_id, index)]["success_rate"] for index in indices]
        actual_short = [by_state_candidate[(state_id, index)]["mean_short_progress"] for index in indices]
        actual_final = [by_state_candidate[(state_id, index)]["mean_final_progress"] for index in indices]
        best_tuple = max(zip(actual_success, actual_short, actual_final))
        true_best = [
            index for index in indices
            if (
                by_state_candidate[(state_id, index)]["success_rate"],
                by_state_candidate[(state_id, index)]["mean_short_progress"],
                by_state_candidate[(state_id, index)]["mean_final_progress"],
            ) == best_tuple
        ]
        predicted_top = [int(value) for value in label["predicted_top_candidates"]]
        predicted_top_success = max(by_state_candidate[(state_id, index)]["success_rate"] for index in predicted_top)
        state_pair_rows = []
        for pair_index, pair in enumerate(label["dccp_emitted_pairs"]):
            winner, loser = int(pair["winner_candidate_index"]), int(pair["loser_candidate_index"])
            winner_actual, loser_actual = by_state_candidate[(state_id, winner)], by_state_candidate[(state_id, loser)]
            success_gap = winner_actual["success_rate"] - loser_actual["success_rate"]
            short_gap = winner_actual["mean_short_progress"] - loser_actual["mean_short_progress"]
            final_gap = winner_actual["mean_final_progress"] - loser_actual["mean_final_progress"]
            row = {
                "state_id": state_id,
                "pair_index": pair_index,
                "winner_candidate_index": winner,
                "loser_candidate_index": loser,
                "predicted_abs_margin": float(pair["weight_abs_margin"]),
                "predicted_signed_margin_alt_nominal": float(pair["margin_alt_minus_nominal"]),
                "actual_success_gap": success_gap,
                "actual_short_progress_gap": short_gap,
                "actual_final_progress_gap": final_gap,
                "strict_success_correct": float(success_gap > cfg.metrics.actual_gap_epsilon),
                "success_nonworse": float(success_gap >= -cfg.metrics.actual_gap_epsilon),
                "short_progress_correct": float(short_gap > cfg.metrics.actual_gap_epsilon),
                "fail_to_success": float(loser_actual["success_rate"] == 0.0 and winner_actual["success_rate"] > 0.0),
            }
            pair_rows.append(row)
            state_pair_rows.append(row)
        state_rows.append({
            "state_id": state_id,
            "num_emitted_pairs": len(state_pair_rows),
            "pair_strict_accuracy": _mean([row["strict_success_correct"] for row in state_pair_rows]),
            "pair_nonworse_rate": _mean([row["success_nonworse"] for row in state_pair_rows]),
            "winner_loser_success_lift": _mean([row["actual_success_gap"] for row in state_pair_rows]),
            "fail_to_success_rate": _mean([row["fail_to_success"] for row in state_pair_rows]),
            "top1_hit": float(bool(set(predicted_top) & set(true_best))),
            "top1_success_regret": float(best_tuple[0] - predicted_top_success),
            "spearman_predicted_vs_success": spearman([predicted_scores[i] for i in indices], actual_success),
            "spearman_predicted_vs_short_progress": spearman([predicted_scores[i] for i in indices], actual_short),
            "kendall_predicted_vs_success": kendall_tau_b([predicted_scores[i] for i in indices], actual_success),
            "predicted_top_candidates": predicted_top,
            "simulator_best_candidates": true_best,
        })

    decisive = [row for row in pair_rows if abs(row["actual_success_gap"]) > cfg.metrics.actual_gap_epsilon]
    margin_actual_corr = spearman(
        [row["predicted_abs_margin"] for row in pair_rows],
        [row["actual_success_gap"] for row in pair_rows],
    )
    bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.01)]
    calibration = []
    for low, high in bins:
        bucket = [row for row in pair_rows if low <= row["predicted_abs_margin"] < high]
        calibration.append({
            "margin_low": low, "margin_high": high, "count": len(bucket),
            "strict_success_accuracy": _mean([row["strict_success_correct"] for row in bucket]),
            "mean_actual_success_gap": _mean([row["actual_success_gap"] for row in bucket]),
            "mean_actual_short_progress_gap": _mean([row["actual_short_progress_gap"] for row in bucket]),
        })
    bootstrap = {
        field: _bootstrap_state_mean(pair_rows, field, cfg.metrics.bootstrap_samples, cfg.metrics.bootstrap_seed + offset)
        for offset, field in enumerate(("strict_success_correct", "success_nonworse", "actual_success_gap", "fail_to_success"))
    }
    summary = {
        "num_states": len(state_rows),
        "num_candidates": len(candidate_rows),
        "num_emitted_pairs": len(pair_rows),
        "num_success_decisive_pairs": len(decisive),
        "strict_pair_accuracy_all_pairs": _mean([row["strict_success_correct"] for row in pair_rows]),
        "decisive_pair_accuracy": _mean([row["strict_success_correct"] for row in decisive]),
        "pair_nonworse_rate": _mean([row["success_nonworse"] for row in pair_rows]),
        "winner_loser_success_lift": _mean([row["actual_success_gap"] for row in pair_rows]),
        "fail_to_success_rate": _mean([row["fail_to_success"] for row in pair_rows]),
        "top1_accuracy_tie_aware": _mean([row["top1_hit"] for row in state_rows]),
        "mean_top1_success_regret": _mean([row["top1_success_regret"] for row in state_rows]),
        "mean_state_spearman_vs_success": _mean([row["spearman_predicted_vs_success"] for row in state_rows if row["spearman_predicted_vs_success"] is not None]),
        "predicted_margin_vs_actual_success_gap_spearman": margin_actual_corr,
        "bootstrap_by_state": bootstrap,
        "interpretation": {
            "primary_ground_truth": "simulator final success rate",
            "secondary_ground_truth": "Coffee privileged short-horizon milestones",
            "tie_policy": "top1 is correct if any predicted-top candidate is simulator-best",
        },
    }
    report = {
        "summary": summary,
        "state_metrics": state_rows,
        "pair_metrics": pair_rows,
        "candidate_metrics": candidate_rows,
        "margin_calibration": calibration,
    }
    AuditStore.write_json(store.reports / "validity_report.json", report)
    _write_csv(store.reports / "candidate_metrics.csv", candidate_rows)
    _write_csv(store.reports / "pair_metrics.csv", pair_rows)
    _write_csv(store.reports / "state_metrics.csv", state_rows)
    return report


def _write_csv(path, rows: list[dict]) -> None:
    if not rows:
        AuditStore.write_text(path, "")
        return
    output = io.StringIO()
    fields = sorted({key for row in rows for key in row})
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    AuditStore.write_text(path, output.getvalue())
