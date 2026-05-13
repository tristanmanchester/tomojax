from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True, slots=True)
class FoldArrays:
    train_idx: jnp.ndarray
    train_mask: jnp.ndarray
    val_idx: jnp.ndarray
    val_mask: jnp.ndarray

    @property
    def n_folds(self) -> int:
        return int(self.train_idx.shape[0])

    def to_metadata(self) -> dict[str, object]:
        return {
            "n_folds": int(self.train_idx.shape[0]),
            "max_train": int(self.train_idx.shape[1]),
            "max_val": int(self.val_idx.shape[1]),
            "train_counts": [int(v) for v in np.asarray(jnp.sum(self.train_mask, axis=1))],
            "val_counts": [int(v) for v in np.asarray(jnp.sum(self.val_mask, axis=1))],
        }


@dataclass(frozen=True, slots=True)
class FoldSpec:
    n_folds: int = 4
    mode: Literal["interleaved"] = "interleaved"

    def build(self, n_views: int) -> FoldArrays:
        n = int(n_views)
        k = int(self.n_folds)
        if k < 2:
            raise ValueError("bilevel CV requires at least two folds")
        if n < k:
            raise ValueError(f"bilevel CV requires at least {k} views for {k} folds; got {n}")
        indices = np.arange(n, dtype=np.int32)
        val_parts = [indices[(indices % k) == fold] for fold in range(k)]
        if any(part.size == 0 for part in val_parts):
            raise ValueError("bilevel CV split produced an empty validation fold")
        train_parts = [np.setdiff1d(indices, val, assume_unique=True) for val in val_parts]
        max_train = max(int(part.size) for part in train_parts)
        max_val = max(int(part.size) for part in val_parts)

        def pad(parts: list[np.ndarray], width: int) -> tuple[jnp.ndarray, jnp.ndarray]:
            idx = np.zeros((k, width), dtype=np.int32)
            mask = np.zeros((k, width), dtype=np.float32)
            for row, part in enumerate(parts):
                idx[row, : part.size] = part
                mask[row, : part.size] = 1.0
            return jnp.asarray(idx, dtype=jnp.int32), jnp.asarray(mask, dtype=jnp.float32)

        train_idx, train_mask = pad(train_parts, max_train)
        val_idx, val_mask = pad(val_parts, max_val)
        return FoldArrays(
            train_idx=train_idx,
            train_mask=train_mask,
            val_idx=val_idx,
            val_mask=val_mask,
        )
