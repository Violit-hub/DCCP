"""Deterministic seed derivation shared across stages."""

from __future__ import annotations

import hashlib
import random
from contextlib import contextmanager

import numpy as np


def derive_seed(*parts) -> int:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "little") & 0x7FFFFFFF


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
    py_state, np_state = random.getstate(), np.random.get_state()
    seed_everything(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
