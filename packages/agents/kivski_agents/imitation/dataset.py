"""ImitationDataset — wraps in-memory demos as a torch Dataset."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

_REQUIRED_KEYS = ("obs", "joint_obs", "received_comm", "move", "discrete", "hidden_state")


class ImitationDataset(Dataset):
    """In-memory (obs, action) demos for behavior cloning.

    All tensors are pre-stacked so __getitem__ is just a per-index slice.
    """

    def __init__(self, demos: dict[str, torch.Tensor]) -> None:
        for k in _REQUIRED_KEYS:
            if k not in demos:
                raise KeyError(f"demos missing required key: {k!r}")
        self.demos = demos
        self._n = int(demos["obs"].shape[0])

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {k: self.demos[k][idx] for k in _REQUIRED_KEYS}
