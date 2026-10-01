"""Balanced batch sampler: every batch contains exactly batch_size/4 items of each input condition."""
from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler


class BalancedConditionBatchSampler(Sampler):
    """Yields lists of ``(image_index, condition_index)``.

    Each epoch: images are shuffled, chunked into batches, and every batch gets an exactly balanced,
    shuffled list of condition labels. Requires batch_size % n_conditions == 0 (rounded down otherwise).
    """

    def __init__(self, n_items: int, batch_size: int, n_conditions: int = 4, seed: int = 42,
                 drop_last: bool = True, condition_ids: list[int] | None = None):
        self.condition_ids = condition_ids if condition_ids is not None else list(range(n_conditions))
        k = len(self.condition_ids)
        self.batch_size = max(k, (batch_size // k) * k)
        self.n_items, self.seed, self.drop_last = n_items, seed, drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        perm = rng.permutation(self.n_items)
        k = len(self.condition_ids)
        for start in range(0, self.n_items, self.batch_size):
            idx = perm[start:start + self.batch_size]
            if len(idx) < self.batch_size and self.drop_last:
                break
            conds = np.resize(np.repeat(self.condition_ids, self.batch_size // k), len(idx))
            rng.shuffle(conds)
            yield [(int(i), int(c)) for i, c in zip(idx, conds)]

    def __len__(self) -> int:
        n = self.n_items // self.batch_size
        return n if self.drop_last else n + int(self.n_items % self.batch_size > 0)
