"""Generate and freeze the exact candidate actions used by both WM and simulator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .artifacts import AuditStore, relative, sha256_file, stable_hash
from .config import AuditConfig
from .schemas import CandidateRecord
from .seeding import derive_seed
from .source_states import load_states


def candidate_fingerprint(cfg: AuditConfig) -> str:
    return stable_hash({
        "total": cfg.candidates.total_candidates,
        "seed": cfg.candidates.seed,
        "deduplicate": cfg.candidates.deduplicate,
        "temperature": cfg.policy.candidate_temperature,
        "checkpoint": str(Path(cfg.paths.policy_checkpoint).resolve()),
        "unnorm_key": cfg.policy.unnorm_key,
    })


def _load_nominal(store: AuditStore, state: dict[str, Any]):
    from state_audit.policy_adapter import PolicyAction

    with np.load(store.run_dir / state["nominal_action_path"], allow_pickle=False) as data:
        actions = np.asarray(data["actions"])
        normalized = np.asarray(data["normalized_actions"])
        seed = int(data["seed"])
    with np.load(store.run_dir / state["nominal_tokens_path"], allow_pickle=False) as data:
        tokens = np.asarray(data["response_tokens"])
    return PolicyAction(actions, normalized, tokens, float("nan"), seed)


def generate_candidates(cfg: AuditConfig, policy=None) -> list[dict[str, Any]]:
    from state_audit.policy_adapter import FrozenVLAAdapter

    store = AuditStore(cfg.run_dir)
    store.ensure_layout()
    states = load_states(cfg)
    manifest_path = store.candidates / "manifest.json"
    fingerprint = candidate_fingerprint(cfg)
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("config_fingerprint") != fingerprint:
            raise RuntimeError("Candidate manifest conflicts with config; use a new run_name")
        return list(payload["candidates"])

    owns_policy = policy is None
    if policy is None:
        policy = FrozenVLAAdapter(cfg.paths.policy_checkpoint, cfg.policy)
        policy.load()
    records: list[CandidateRecord] = []
    for state in states:
        state_id = str(state["state_id"])
        nominal = _load_nominal(store, state)
        frame = np.asarray(Image.open(store.run_dir / state["frame_path"]).convert("RGB"))
        generated = [nominal]
        seen = {tuple(np.asarray(nominal.response_tokens).reshape(-1).tolist())}
        attempts = 0
        while len(generated) < cfg.candidates.total_candidates:
            if attempts >= cfg.policy.candidate_max_attempts:
                raise RuntimeError(f"Only generated {len(generated)} candidates for {state_id}")
            seed = derive_seed(cfg.candidates.seed, state_id, attempts)
            action = policy.generate_action(
                frame, cfg.policy.instruction, seed=seed, do_sample=True,
                temperature=cfg.policy.candidate_temperature, compute_entropy=False,
            )
            attempts += 1
            key = tuple(np.asarray(action.response_tokens).reshape(-1).tolist())
            if cfg.candidates.deduplicate and key in seen:
                continue
            seen.add(key)
            generated.append(action)
        for candidate_index, action in enumerate(generated):
            target = store.candidates / state_id / f"candidate_{candidate_index:02d}.npz"
            AuditStore.write_npz(
                target,
                actions=np.asarray(action.actions, dtype=np.float32),
                normalized_actions=np.asarray(action.normalized_actions, dtype=np.float32),
                response_tokens=np.asarray(action.response_tokens, dtype=np.int64),
                seed=np.asarray(int(action.seed), dtype=np.int64),
            )
            records.append(CandidateRecord(
                state_id=state_id,
                candidate_index=candidate_index,
                is_nominal=candidate_index == 0,
                action_path=relative(target, store.run_dir),
                seed=int(action.seed),
                token_hash=stable_hash(np.asarray(action.response_tokens).reshape(-1).tolist()),
            ))
    payload = {
        "schema_version": 1,
        "config_fingerprint": fingerprint,
        "candidate_count_per_state": cfg.candidates.total_candidates,
        "candidate_sampling_semantics": "deduplicated" if cfg.candidates.deduplicate else "faithful_allow_duplicates",
        "candidates": [row.to_dict() for row in records],
    }
    AuditStore.write_json(manifest_path, payload)
    AuditStore.write_json(store.candidates / "CANDIDATES_FROZEN.json", {
        "count": len(records), "manifest_sha256": sha256_file(manifest_path)
    })
    if owns_policy:
        del policy
    return payload["candidates"]


def load_candidates(cfg: AuditConfig) -> list[dict[str, Any]]:
    path = AuditStore(cfg.run_dir).candidates / "manifest.json"
    if not path.exists():
        raise RuntimeError("Candidates are not frozen; run generate_candidates first")
    return list(json.loads(path.read_text(encoding="utf-8"))["candidates"])


def candidates_by_state(cfg: AuditConfig) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in load_candidates(cfg):
        grouped.setdefault(str(row["state_id"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item["candidate_index"]))
    return grouped
