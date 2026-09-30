"""可重复随机数与 common-random-numbers 支持。"""

from __future__ import annotations

import hashlib
import random
from contextlib import contextmanager

import numpy as np


def derive_seed(*parts: object, modulo: int = 2**31 - 1) -> int:
    """由稳定标识派生 seed，不能使用进程随机化的内置 hash。"""
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % int(modulo)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        pass


@contextmanager
def temporary_seed(seed: int):
    """临时设置随机状态，退出后恢复 Python/NumPy/Torch 状态。"""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = None
    cuda_states = None
    try:
        import torch

        torch_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            cuda_states = torch.cuda.get_rng_state_all()
    except ImportError:
        torch = None
    seed_everything(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if torch_state is not None:
            torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
