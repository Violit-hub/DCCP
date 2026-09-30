"""四种状态选择方法和同约束 simulator oracle。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MethodSelection:
    method: str
    local_indices: list[int]
    decision_indices: list[int]
    scores: list[float]


def _mining_imports():
    # 延迟导入，保证纯统计单元测试不要求完整训练依赖在 import 时加载。
    from verl.utils.dccp_mining import DCCPMiningConfig, select_decision_sensitive_states

    return DCCPMiningConfig, select_decision_sensitive_states


def select_scored_methods(
    progress_scores: np.ndarray,
    entropy_scores: np.ndarray,
    valid_start_indices: np.ndarray,
    *,
    state_budget: int,
    nms_gap: int,
    lambda_curvature: float,
    lambda_entropy: float,
) -> dict[str, MethodSelection]:
    """计算 curvature、entropy 和 joint 三个确定性选择器。"""
    DCCPMiningConfig, select_states = _mining_imports()
    progress = np.asarray(progress_scores, dtype=np.float32).reshape(-1)
    entropy = np.asarray(entropy_scores, dtype=np.float32).reshape(-1)
    starts = np.asarray(valid_start_indices, dtype=np.int64).reshape(-1)
    if not (len(progress) == len(entropy) == len(starts)):
        raise ValueError("progress、entropy 和 valid_start_indices 长度必须一致")

    method_weights = {
        "curvature": (1.0, 0.0),
        "entropy": (0.0, 1.0),
        "joint": (float(lambda_curvature), float(lambda_entropy)),
    }
    output: dict[str, MethodSelection] = {}
    for method, (weight_c, weight_h) in method_weights.items():
        cfg = DCCPMiningConfig(
            state_budget_per_traj=int(state_budget),
            nms_gap=int(nms_gap),
            lambda_curvature=weight_c,
            lambda_entropy=weight_h,
            require_entropy=weight_h != 0.0,
        )
        result = select_states(progress, entropy, cfg)
        local = [int(state.index) for state in result.selected_states]
        output[method] = MethodSelection(
            method=method,
            local_indices=local,
            decision_indices=[int(starts[index]) for index in local],
            scores=[float(result.decision_scores[index]) for index in local],
        )
    return output


def legal_index_sets(valid_mask: Iterable[bool], budget: int, nms_gap: int) -> list[tuple[int, ...]]:
    """枚举满足预算和时间间隔约束的索引组合。"""
    mask = np.asarray(list(valid_mask), dtype=bool).reshape(-1)
    candidates = np.flatnonzero(mask).tolist()
    budget = min(int(budget), len(candidates))
    if budget <= 0:
        return []
    legal = []
    for combo in combinations(candidates, budget):
        if all(abs(left - right) > int(nms_gap) for left, right in combinations(combo, 2)):
            legal.append(tuple(int(index) for index in combo))
    return legal


def draw_random_selections(
    valid_mask: Iterable[bool],
    *,
    budget: int,
    nms_gap: int,
    draws: int,
    seed: int,
) -> list[tuple[int, ...]]:
    """从全部合法组合中均匀抽样；允许重复以估计随机基线期望。"""
    legal = legal_index_sets(valid_mask, budget, nms_gap)
    if not legal:
        return []
    generator = np.random.default_rng(int(seed))
    sampled = generator.integers(0, len(legal), size=int(draws))
    return [legal[int(index)] for index in sampled]


def select_oracle_indices(
    improvement: Iterable[float],
    valid_mask: Iterable[bool],
    *,
    budget: int,
    nms_gap: int,
) -> tuple[int, ...]:
    """在相同预算/NMS 约束下最大化真实改善量，仅用于最终评估。"""
    values = np.asarray(list(improvement), dtype=np.float64).reshape(-1)
    legal = legal_index_sets(valid_mask, budget, nms_gap)
    if not legal:
        return tuple()
    return max(legal, key=lambda combo: (sum(float(values[index]) for index in combo), tuple(-x for x in combo)))
