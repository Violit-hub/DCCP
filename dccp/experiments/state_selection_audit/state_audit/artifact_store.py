"""可恢复、原子化的实验产物存储。"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"不能序列化对象: {type(value)!r}")


class ArtifactStore:
    """统一管理一次 run 的目录和断点续跑记录。"""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.nominal_dir = self.run_dir / "nominal"
        self.selection_dir = self.run_dir / "selections"
        self.branch_dir = self.run_dir / "branches"
        self.report_dir = self.run_dir / "reports"

    def ensure_layout(self) -> None:
        for path in (
            self.run_dir,
            self.nominal_dir / "model_xml",
            self.nominal_dir / "simulator_states",
            self.nominal_dir / "frames",
            self.nominal_dir / "actions",
            self.nominal_dir / "tokens",
            self.selection_dir,
            self.branch_dir / "actions",
            self.branch_dir / "tokens",
            self.report_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
            # setgid 保证 neu_lab2/neu_lab4 后续创建的子目录继续继承公共组。
            path.chmod(0o2775)

    @staticmethod
    def write_json_atomic(path: str | Path, payload: Any) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, delete=False
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
            handle.write("\n")
            temp_name = handle.name
        os.replace(temp_name, target)
        target.chmod(0o660)

    @staticmethod
    def write_text_atomic(path: str | Path, text: str) -> None:
        """原子写入文本，避免中断后留下半个文件。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, delete=False
        ) as handle:
            handle.write(text)
            temp_name = handle.name
        os.replace(temp_name, target)
        target.chmod(0o660)

    @staticmethod
    def write_jsonl_atomic(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
        """一次性冻结 JSONL；用于禁止选择结果被分支回报污染。"""
        lines = "".join(
            json.dumps(row, ensure_ascii=False, default=_json_default) + "\n" for row in rows
        )
        ArtifactStore.write_text_atomic(path, lines)

    @staticmethod
    def write_npz_atomic(path: str | Path, **arrays: Any) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temp_name = tempfile.mkstemp(suffix=".npz", dir=target.parent)
        os.close(file_descriptor)
        try:
            np.savez_compressed(temp_name, **arrays)
            os.replace(temp_name, target)
            target.chmod(0o660)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    @staticmethod
    def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                for row in rows:
                    handle.write(
                        json.dumps(row, ensure_ascii=False, default=_json_default) + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        target.chmod(0o660)

    @staticmethod
    def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
        target = Path(path)
        if not target.exists():
            return []
        rows: list[dict[str, Any]] = []
        with target.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL 损坏: {target}:{line_number}: {exc}") from exc
        return rows

    @staticmethod
    def existing_keys(path: str | Path, fields: tuple[str, ...]) -> set[tuple[Any, ...]]:
        return {tuple(row[field] for field in fields) for row in ArtifactStore.read_jsonl(path)}
