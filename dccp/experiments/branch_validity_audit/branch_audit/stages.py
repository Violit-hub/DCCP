"""Stage completion evidence used by CLI gates and the inspection page."""

from __future__ import annotations

import json

from .artifacts import AuditStore
from .config import AuditConfig


def stage_status(cfg: AuditConfig) -> list[dict]:
    store = AuditStore(cfg.run_dir)
    state_manifest = store.states / "manifest.json"
    candidate_manifest = store.candidates / "manifest.json"
    states = json.loads(state_manifest.read_text(encoding="utf-8"))["states"] if state_manifest.exists() else []
    candidates = json.loads(candidate_manifest.read_text(encoding="utf-8"))["candidates"] if candidate_manifest.exists() else []
    expected_wm = len(candidates) * len(cfg.world_model.seeds)
    wm_rows = AuditStore.read_jsonl(store.world_model / "predictions.jsonl")
    wm_ok = {
        (row.get("state_id"), row.get("candidate_index"), row.get("wm_seed"))
        for row in wm_rows if row.get("status") == "OK"
    }
    score_rows = AuditStore.read_jsonl(store.labels / "prediction_scores.jsonl")
    score_ok = {
        (row.get("state_id"), row.get("candidate_index"), row.get("wm_seed"))
        for row in score_rows if row.get("status") == "OK"
    }
    expected_sim = len(candidates) * len(cfg.simulator.evaluation_seeds)
    sim_rows = AuditStore.read_jsonl(store.simulator / "outcomes.jsonl")
    sim_ok = {
        (row.get("state_id"), row.get("candidate_index"), row.get("evaluation_seed"))
        for row in sim_rows if row.get("status") == "OK"
    }
    return [
        {"stage": "setup", "state": "READY", "detail": "configuration loaded"},
        {"stage": "states", "state": "FROZEN" if states else "PENDING", "detail": f"{len(states)}/{cfg.states.max_states}"},
        {"stage": "candidates", "state": "FROZEN" if candidates else "PENDING", "detail": f"{len(candidates)} artifacts"},
        {"stage": "world_model", "state": "DONE" if expected_wm and len(wm_ok) == expected_wm else ("STARTED" if wm_rows else "PENDING"), "detail": f"OK={len(wm_ok)}/{expected_wm}"},
        {"stage": "lrm_scores", "state": "DONE" if expected_wm and len(score_ok) == expected_wm else ("STARTED" if score_rows else "PENDING"), "detail": f"OK={len(score_ok)}/{expected_wm}"},
        {"stage": "labels", "state": "FROZEN" if (store.labels / "LABELS_FROZEN.json").exists() else "PENDING", "detail": str(store.labels / "LABELS_FROZEN.json")},
        {"stage": "simulator", "state": "DONE" if expected_sim and len(sim_ok) == expected_sim else ("STARTED" if sim_rows else "PENDING"), "detail": f"OK={len(sim_ok)}/{expected_sim}"},
        {"stage": "metrics", "state": "DONE" if (store.reports / "validity_report.json").exists() else "PENDING", "detail": str(store.reports / "validity_report.json")},
    ]
