"""ELO rating for tracking self-play progress."""
from __future__ import annotations

K = 32.0
INITIAL = 1200.0


class ELORater:
    def __init__(self, k: float = K):
        self.k = k
        self.ratings: dict[str, float] = {}

    def get(self, name: str) -> float:
        return self.ratings.setdefault(name, INITIAL)

    def update(self, a: str, b: str, a_score: float) -> None:
        """a_score in {0.0, 0.5, 1.0}."""
        ra = self.get(a)
        rb = self.get(b)
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
        ra_new = ra + self.k * (a_score - ea)
        self.ratings[a] = ra_new
        self.ratings[b] = rb - self.k * (a_score - ea)

    def snapshot(self) -> dict[str, float]:
        return dict(self.ratings)