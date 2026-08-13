"""Asset-disjoint splitting.

Per the project rules, we NEVER split at the record (or window-instance)
level — an asset (device/IP) must land entirely in one of train/val/test,
otherwise the model could see one time-slice of a device at train time and
a neighboring time-slice of the *same* device at test time, which leaks
identity/behavioral signal and inflates every downstream metric.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def asset_disjoint_split(
    asset_ids: List[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Dict[str, str]:
    """Return {asset_id: 'train'|'val'|'test'}.

    Splitting is done once over the *set* of unique assets (order-independent,
    seeded), so it is reproducible and identical regardless of how records
    happen to be ordered in the source file.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "fractions must sum to 1"
    unique_assets = sorted(set(asset_ids))  # sort first so shuffle is deterministic across runs/platforms
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_assets))
    shuffled = [unique_assets[i] for i in perm]

    n = len(shuffled)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    # remainder goes to test to make sure all assets are assigned
    train_assets = shuffled[:n_train]
    val_assets = shuffled[n_train : n_train + n_val]
    test_assets = shuffled[n_train + n_val :]

    split_map = {}
    for a in train_assets:
        split_map[a] = "train"
    for a in val_assets:
        split_map[a] = "val"
    for a in test_assets:
        split_map[a] = "test"
    return split_map
