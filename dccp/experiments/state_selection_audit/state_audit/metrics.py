"""逐状态改善量、失败转成功和完整上限统计。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .selectors import draw_random_selections, select_oracle_indices


@dataclass(frozen=True)
class StateAuditMetric:
    trajectory_id: str
    decision_index: int
    nominal_success_rate: float
    best_success_rate: float
    best_candidate_index: int
    improvement: float
    paired_fail_to_success_rate: float
    has_fail_to_success_opportunity: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def compute_state_metrics(branch_rows: Iterable[dict[str, Any]]) -> list[StateAuditMetric]:
    """从逐 seed 分支结果计算每个状态的经验 ground truth。"""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in branch_rows:
        if row.get("status", "OK") != "OK":
            continue
        grouped[(str(row["trajectory_id"]), int(row["decision_index"]))].append(row)

    metrics: list[StateAuditMetric] = []
    for (trajectory_id, decision_index), rows in sorted(grouped.items()):
        by_candidate: dict[int, dict[int, bool]] = defaultdict(dict)
        nominal_index = None
        for row in rows:
            candidate = int(row["candidate_index"])
            seed = int(row["evaluation_seed"])
            by_candidate[candidate][seed] = bool(row["success"])
            if bool(row.get("is_nominal", candidate == 0)):
                nominal_index = candidate
        if nominal_index is None:
            raise ValueError(f"状态缺少 nominal branch: {trajectory_id}/{decision_index}")
        seed_sets = [set(values) for values in by_candidate.values()]
        common_seeds = set.intersection(*seed_sets) if seed_sets else set()
        if not common_seeds:
            raise ValueError(f"候选动作没有共同 evaluation seed: {trajectory_id}/{decision_index}")
        ordered_seeds = sorted(common_seeds)
        rates = {
            candidate: float(np.mean([outcomes[seed] for seed in ordered_seeds]))
            for candidate, outcomes in by_candidate.items()
        }
        best_candidate = max(sorted(rates), key=lambda candidate: (rates[candidate], -candidate))
        nominal_rate = rates[nominal_index]
        best_rate = rates[best_candidate]
        # “失败可转成功”按同 seed 下任一候选能否救回计算；改善量仍使用聚合成功率最佳候选。
        paired = [
            (not by_candidate[nominal_index][seed])
            and any(outcomes[seed] for outcomes in by_candidate.values())
            for seed in ordered_seeds
        ]
        metrics.append(
            StateAuditMetric(
                trajectory_id=trajectory_id,
                decision_index=decision_index,
                nominal_success_rate=nominal_rate,
                best_success_rate=best_rate,
                best_candidate_index=int(best_candidate),
                improvement=max(0.0, best_rate - nominal_rate),
                paired_fail_to_success_rate=float(np.mean(paired)),
                has_fail_to_success_opportunity=bool(any(paired)),
            )
        )
    return metrics


def _selected_metric(values: dict[int, StateAuditMetric], selected: Iterable[int]) -> dict[str, float]:
    chosen = [values[int(index)] for index in selected if int(index) in values]
    if not chosen:
        return {
            "selected_count": 0.0,
            "mean_improvement": 0.0,
            "total_improvement": 0.0,
            "opportunity_rate": 0.0,
            "paired_fail_to_success_rate": 0.0,
        }
    return {
        "selected_count": float(len(chosen)),
        "mean_improvement": float(np.mean([item.improvement for item in chosen])),
        "total_improvement": float(np.sum([item.improvement for item in chosen])),
        "opportunity_rate": float(np.mean([item.has_fail_to_success_opportunity for item in chosen])),
        "paired_fail_to_success_rate": float(np.mean([item.paired_fail_to_success_rate for item in chosen])),
    }


def summarize_methods(
    state_metrics: Iterable[StateAuditMetric],
    selection_rows: Iterable[dict[str, Any]],
    *,
    state_budget: int,
    nms_gap: int,
    random_draws: int,
    random_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """返回逐轨迹统计和跨轨迹方法汇总。"""
    by_traj_metric: dict[str, dict[int, StateAuditMetric]] = defaultdict(dict)
    for item in state_metrics:
        by_traj_metric[item.trajectory_id][item.decision_index] = item
    by_traj_selection: dict[str, dict[str, list[int]]] = defaultdict(dict)
    eligible_by_traj: dict[str, set[int]] = defaultdict(set)
    for row in selection_rows:
        trajectory_id = str(row["trajectory_id"])
        eligible_by_traj[trajectory_id].update(int(x) for x in row.get("eligible_indices", []))
        method = str(row["method"])
        if method == "random":
            continue
        by_traj_selection[trajectory_id][method] = [int(x) for x in row["selected_indices"]]

    per_trajectory: list[dict[str, Any]] = []
    for trajectory_id, indexed in sorted(by_traj_metric.items()):
        eligible = eligible_by_traj.get(trajectory_id, set(indexed))
        max_index = max(set(indexed) | set(eligible)) if (indexed or eligible) else -1
        valid_mask = np.zeros(max_index + 1, dtype=bool)
        improvement = np.zeros(max_index + 1, dtype=np.float64)
        for index, item in indexed.items():
            improvement[index] = item.improvement
        for index in eligible:
            # 只有完成 ground-truth 分支评估的 eligible 状态才能进入 oracle/random。
            valid_mask[index] = index in indexed
        oracle = select_oracle_indices(
            improvement, valid_mask, budget=state_budget, nms_gap=nms_gap
        )
        oracle_stats = _selected_metric(indexed, oracle)
        methods = dict(by_traj_selection.get(trajectory_id, {}))
        methods["oracle"] = list(oracle)
        random_sets = draw_random_selections(
            valid_mask,
            budget=state_budget,
            nms_gap=nms_gap,
            draws=random_draws,
            seed=random_seed + sum(ord(char) for char in trajectory_id),
        )
        random_stats = [_selected_metric(indexed, selection) for selection in random_sets]
        for method, selected in sorted(methods.items()):
            stats = _selected_metric(indexed, selected)
            denominator = oracle_stats["total_improvement"]
            per_trajectory.append(
                {
                    "trajectory_id": trajectory_id,
                    "method": method,
                    "selected_indices": selected,
                    **stats,
                    "oracle_total_improvement": denominator,
                    "capture_ratio": stats["total_improvement"] / denominator if denominator > 0 else None,
                }
            )
        if random_stats:
            mean_stats = {
                key: float(np.mean([item[key] for item in random_stats]))
                for key in random_stats[0]
            }
            denominator = oracle_stats["total_improvement"]
            per_trajectory.append(
                {
                    "trajectory_id": trajectory_id,
                    "method": "random",
                    "selected_indices": None,
                    **mean_stats,
                    "oracle_total_improvement": denominator,
                    "capture_ratio": mean_stats["total_improvement"] / denominator if denominator > 0 else None,
                }
            )

    method_rows: list[dict[str, Any]] = []
    methods = sorted({str(row["method"]) for row in per_trajectory})
    for method in methods:
        rows = [row for row in per_trajectory if row["method"] == method]
        valid_ratios = [float(row["capture_ratio"]) for row in rows if row["capture_ratio"] is not None]
        total_method = float(sum(float(row["total_improvement"]) for row in rows))
        total_oracle = float(sum(float(row["oracle_total_improvement"]) for row in rows))
        method_rows.append(
            {
                "method": method,
                "num_trajectories": len(rows),
                "mean_improvement": float(np.mean([row["mean_improvement"] for row in rows])) if rows else 0.0,
                "opportunity_rate": float(np.mean([row["opportunity_rate"] for row in rows])) if rows else 0.0,
                "paired_fail_to_success_rate": float(
                    np.mean([row["paired_fail_to_success_rate"] for row in rows])
                ) if rows else 0.0,
                "macro_capture_ratio": float(np.mean(valid_ratios)) if valid_ratios else None,
                "micro_capture_ratio": total_method / total_oracle if total_oracle > 0 else None,
            }
        )
    return per_trajectory, method_rows
