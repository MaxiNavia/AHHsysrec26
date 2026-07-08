from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, TypeVar
import numpy as np


T = TypeVar("T")


def load_or_build(cache_path: Path, builder: Callable[[], T], force: bool = False) -> T:
    """Load an object from pickle cache or build and persist it."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        with cache_path.open("rb") as file:
            return pickle.load(file)

    value = builder()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)
    return value


def save_pickle(value: T, cache_path: Path) -> None:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(cache_path: Path) -> T:
    with Path(cache_path).open("rb") as file:
        return pickle.load(file)


def split_val_test(eval_users: list[int], seed: int) -> tuple[list[int], list[int]]:
    """Split the sampled evaluation population into a tuning (val) set and a
    held-out reporting (test) set. Hyperparameters are selected only on
    val_users; final metrics are reported only on final_test_users.
    """
    rng = np.random.default_rng(seed + 1_000_003)
    shuffled = rng.permutation(np.array(eval_users, dtype=int))
    half = len(shuffled) // 2
    val_users = [int(u) for u in shuffled[:half]]
    final_test_users = [int(u) for u in shuffled[half:]]
    return val_users, final_test_users