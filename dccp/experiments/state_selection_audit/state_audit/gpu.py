"""GPU 占用检查；测试脚本在繁忙时可安全跳过。"""

from __future__ import annotations

import csv
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GPUStatus:
    index: int
    name: str
    memory_used_mib: int
    memory_total_mib: int
    utilization_percent: int

    @property
    def free_memory_mib(self) -> int:
        return self.memory_total_mib - self.memory_used_mib


def query_gpu_status() -> list[GPUStatus]:
    if shutil.which("nvidia-smi") is None:
        return []
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = csv.reader(completed.stdout.splitlines(), skipinitialspace=True)
    return [
        GPUStatus(
            index=int(row[0]),
            name=row[1].strip(),
            memory_used_mib=int(row[2]),
            memory_total_mib=int(row[3]),
            utilization_percent=int(row[4]),
        )
        for row in rows
        if row
    ]


def gpu_is_busy(status: GPUStatus, memory_threshold_mib: int) -> bool:
    return (
        status.memory_used_mib >= int(memory_threshold_mib)
        or status.utilization_percent >= 20
    )


def query_compute_processes() -> list[dict[str, str]]:
    """返回 GPU 计算进程；无进程时 nvidia-smi 可能输出空行。"""
    if shutil.which("nvidia-smi") is None:
        return []
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    rows = csv.reader(completed.stdout.splitlines(), skipinitialspace=True)
    return [
        {"pid": row[0].strip(), "process_name": row[1].strip(), "used_memory_mib": row[2].strip()}
        for row in rows
        if len(row) >= 3 and row[0].strip()
    ]
