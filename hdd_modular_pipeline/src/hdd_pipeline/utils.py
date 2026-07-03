from __future__ import annotations

import gc
import os
import random
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_memory() -> None:
    """Collect Python garbage and clear CuPy memory pool when available."""
    gc.collect()
    try:
        import cupy as cp  # type: ignore

        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def to_numpy_safe(x):
    """Convert pandas/cupy/sklearn outputs to numpy without hard dependency on cupy."""
    try:
        import cupy as cp  # type: ignore

        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    return x.to_numpy() if hasattr(x, "to_numpy") else np.asarray(x)


def memory_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except Exception:
        return float("nan")


def iter_slices(n: int, batch_size: int) -> Iterator[slice]:
    for start in range(0, n, batch_size):
        yield slice(start, min(start + batch_size, n))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
