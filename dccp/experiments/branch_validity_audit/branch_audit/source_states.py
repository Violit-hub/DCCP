"""Import a frozen set of real, restorable states from state_selection_audit."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .artifacts import AuditStore, relative, sha256_file, stable_hash
from .config import AuditConfig
from .schemas import StateRecord


def _selected_keys(cfg: AuditConfig, source_run: Path) -> list[tuple[str, int]]:
    if cfg.states.targets:
        return [
            (str(item["trajectory_id"]), int(item["decision_index"]))
            for item in cfg.states.targets
        ][: cfg.states.max_states]
    selection_file = source_run / "selections" / "selected_states.jsonl"
    if not selection_file.exists():
        raise RuntimeError(
            "No explicit states.targets and source selections are missing. Run the source "
            "state_selection_audit through select, or list targets in this config."
        )
    rows = AuditStore.read_jsonl(selection_file)
    keys: list[tuple[str, int]] = []
    for row in rows:
        if str(row.get("method")) != cfg.states.selection_method:
            continue
        for index in row.get("selected_indices", []):
            key = str(row["trajectory_id"]), int(index)
            if key not in keys:
                keys.append(key)
    if not keys:
        methods = sorted({str(row.get("method")) for row in rows})
        raise ValueError(
            f"No source selections for method={cfg.states.selection_method!r}; available={methods}"
        )
    return keys[: cfg.states.max_states]


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(target)
    target.chmod(0o660)


def prepare_states(cfg: AuditConfig) -> list[dict[str, Any]]:
    store = AuditStore(cfg.run_dir)
    store.ensure_layout()
    manifest_path = store.states / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("config_fingerprint") != state_config_fingerprint(cfg):
            raise RuntimeError("Frozen state manifest conflicts with config; use a new run_name")
        return list(existing["states"])

    source_run = Path(cfg.paths.source_state_run).expanduser().resolve()
    trajectory_path = source_run / "nominal" / "trajectories.json"
    trajectories = json.loads(trajectory_path.read_text(encoding="utf-8"))
    trajectory_by_id = {str(row["trajectory_id"]): row for row in trajectories}
    requested = _selected_keys(cfg, source_run)
    records: list[StateRecord] = []
    for trajectory_id, decision_index in requested:
        if trajectory_id not in trajectory_by_id:
            raise KeyError(f"Unknown source trajectory: {trajectory_id}")
        trajectory = trajectory_by_id[trajectory_id]
        step = next(
            (item for item in trajectory.get("steps", []) if int(item["decision_index"]) == decision_index),
            None,
        )
        if step is None:
            raise KeyError(f"Unknown source state: {trajectory_id}@{decision_index}")
        state_id = f"{trajectory_id}__step_{decision_index:04d}"
        target_dir = store.states / state_id
        sources = {
            "frame": source_run / step["frame_path"],
            "simulator_state": source_run / step["simulator_state_path"],
            "model_xml": source_run / trajectory["model_xml_path"],
            "nominal_action": source_run / step["nominal_action_path"],
            "nominal_tokens": source_run / step["nominal_tokens_path"],
        }
        suffixes = {
            "frame": ".png", "simulator_state": ".npz", "model_xml": ".xml",
            "nominal_action": ".npz", "nominal_tokens": ".npz",
        }
        targets = {name: target_dir / f"{name}{suffixes[name]}" for name in sources}
        for name, source in sources.items():
            if not source.exists():
                raise FileNotFoundError(f"Missing source state artifact: {source}")
            _copy(source, targets[name])
        records.append(
            StateRecord(
                state_id=state_id,
                trajectory_id=trajectory_id,
                decision_index=decision_index,
                low_level_step=int(step["low_level_step"]),
                frame_path=relative(targets["frame"], store.run_dir),
                simulator_state_path=relative(targets["simulator_state"], store.run_dir),
                model_xml_path=relative(targets["model_xml"], store.run_dir),
                nominal_action_path=relative(targets["nominal_action"], store.run_dir),
                nominal_tokens_path=relative(targets["nominal_tokens"], store.run_dir),
                source_paths={name: str(path) for name, path in sources.items()},
                hashes={name: sha256_file(path) for name, path in targets.items()},
            )
        )
    payload = {
        "schema_version": 1,
        "task": cfg.task,
        "source_state_run": str(source_run),
        "source_state_config": str(Path(cfg.paths.source_state_config).resolve()),
        "config_fingerprint": state_config_fingerprint(cfg),
        "states": [row.to_dict() for row in records],
    }
    AuditStore.write_json(manifest_path, payload)
    AuditStore.write_json(store.states / "STATES_FROZEN.json", {
        "count": len(records), "manifest_sha256": sha256_file(manifest_path)
    })
    return payload["states"]


def state_config_fingerprint(cfg: AuditConfig) -> str:
    return stable_hash({
        "source_state_run": str(Path(cfg.paths.source_state_run).expanduser().resolve()),
        "selection_method": cfg.states.selection_method,
        "max_states": cfg.states.max_states,
        "targets": cfg.states.targets,
    })


def load_states(cfg: AuditConfig) -> list[dict[str, Any]]:
    path = AuditStore(cfg.run_dir).states / "manifest.json"
    if not path.exists():
        raise RuntimeError("States are not prepared; run stage prepare_states first")
    return list(json.loads(path.read_text(encoding="utf-8"))["states"])
