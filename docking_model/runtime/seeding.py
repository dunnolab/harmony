from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: Optional[int] = None, workers: bool = False, verbose: bool = True) -> int:
    """Set random seeds for Python, NumPy, and PyTorch."""

    seed = normalize_seed(seed)

    if verbose:
        print(f"Seed set to {seed}")

    os.environ["GLOBAL_SEED"] = str(seed)
    os.environ["SEED_WORKERS"] = str(int(workers))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy RNGs inside a PyTorch dataloader worker."""

    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(seed: Optional[int]) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(normalize_seed(seed))
    return generator


def normalize_seed(seed: Optional[int]) -> int:
    if seed is None:
        seed = os.environ.get("GLOBAL_SEED", 0)

    try:
        seed = int(seed)
    except (TypeError, ValueError):
        print(f"Invalid seed {seed!r}; using seed 0.")
        seed = 0

    if seed < 0 or seed > 2**32 - 1:
        print(f"Seed {seed} is out of bounds; using seed 0.")
        seed = 0

    return seed
