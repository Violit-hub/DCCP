"""Atomic, group-friendly artifacts and integrity helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


class AuditStore:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.states = self.run_dir / "states"
        self.candidates = self.run_dir / "candidates"
        self.world_model = self.run_dir / "world_model"
        self.labels = self.run_dir / "labels"
        self.simulator = self.run_dir / "simulator"
        self.reports = self.run_dir / "reports"
        self.inspection = self.run_dir / "inspection"

    def ensure_layout(self) -> None:
        for path in (
            self.run_dir, self.states, self.candidates, self.world_model,
            self.labels, self.simulator, self.reports, self.inspection,
        ):
            path.mkdir(parents=True, exist_ok=True)
            path.chmod(0o2775)

    @staticmethod
    def write_text(path: str | Path, content: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
            handle.write(content)
            temporary = handle.name
        os.replace(temporary, target)
        target.chmod(0o660)

    @staticmethod
    def write_json(path: str | Path, payload: Any) -> None:
        AuditStore.write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n")

    @staticmethod
    def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
        AuditStore.write_text(path, "".join(json.dumps(row, ensure_ascii=False, default=json_default) + "\n" for row in rows))

    @staticmethod
    def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        target.chmod(0o660)

    @staticmethod
    def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
        target = Path(path)
        if not target.exists():
            return []
        rows = []
        for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Corrupt JSONL {target}:{number}: {exc}") from exc
        return rows

    @staticmethod
    def write_npz(path: str | Path, **arrays: Any) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(suffix=".npz", dir=target.parent)
        os.close(descriptor)
        try:
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, target)
            target.chmod(0o660)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def relative(path: str | Path, root: str | Path) -> str:
    return str(Path(path).resolve().relative_to(Path(root).resolve()))
