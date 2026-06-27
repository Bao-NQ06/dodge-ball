"""Opponent snapshot pool for self-play (Bansal-style).

We keep a list of frozen PPO models. The learner samples from this pool:
~80% latest policy, ~20% older snapshots.
"""
from __future__ import annotations

import os
import random
import shutil
from collections import deque
from dataclasses import dataclass
from typing import Optional

from stable_baselines3 import PPO


@dataclass
class OpponentSnapshot:
    path: str
    step: int


class OpponentPool:
    def __init__(self,
                 root: str = "models/snapshots",
                 max_size: int = 12,
                 latest_prob: float = 0.8):
        os.makedirs(root, exist_ok=True)
        self.root = root
        self.max_size = max_size
        self.latest_prob = latest_prob
        self._pool: deque[OpponentSnapshot] = deque(maxlen=max_size)
        self._always_latest = True  # first iteration uses current self as opp

    def __len__(self) -> int:
        return len(self._pool)

    def add(self, model: PPO, step: int) -> OpponentSnapshot:
        path = os.path.join(self.root, f"step_{step:08d}.zip")
        # Save to temp then move so we never half-write
        tmp = path + ".tmp"
        model.save(tmp)
        if os.path.exists(path):
            os.remove(path)
        os.rename(tmp, path)
        snap = OpponentSnapshot(path=path, step=step)
        self._pool.append(snap)
        return snap

    def sample(self, current_step: int) -> Optional[str]:
        """Return path to a snapshot, or None for 'use current learner as opp'."""
        if not self._pool:
            return None
        if random.random() < self.latest_prob or len(self._pool) == 1:
            return self._pool[-1].path
        # sample from older pool excluding the latest
        older = list(self._pool)[:-1]
        return random.choice(older).path

    def latest(self) -> Optional[OpponentSnapshot]:
        return self._pool[-1] if self._pool else None