"""Simulator-privileged Coffee progress and failure taxonomy."""

from __future__ import annotations

from typing import Any


MILESTONE_WEIGHTS = {"grasp": 0.4, "rim": 0.6, "insertion": 0.8, "task": 1.0}


def unwrap_coffee_env(env):
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if callable(getattr(current, "_get_partial_task_metrics", None)):
            return current
        current = getattr(current, "env", None)
    raise AttributeError("Unable to find Coffee environment with _get_partial_task_metrics")


def partial_metrics(env) -> dict[str, bool]:
    raw = unwrap_coffee_env(env)._get_partial_task_metrics()
    return {name: bool(raw.get(name, False)) for name in MILESTONE_WEIGHTS}


def progress_score(metrics: dict[str, Any]) -> float:
    reached = [weight for name, weight in MILESTONE_WEIGHTS.items() if bool(metrics.get(name, False))]
    return float(max(reached, default=0.0))


def failure_reason(metrics: dict[str, Any], trace: list[dict[str, Any]], success: bool) -> str:
    if success or bool(metrics.get("task")):
        return "success"
    if bool(metrics.get("insertion")):
        return "inserted_but_task_incomplete_or_lid_open"
    if bool(metrics.get("rim")):
        return "pod_at_rim_not_inserted"
    if bool(metrics.get("grasp")):
        return "pod_still_grasped_at_timeout"
    if any(bool(row.get("grasp")) for row in trace):
        return "pod_dropped_after_grasp"
    return "never_grasped_or_no_progress"
