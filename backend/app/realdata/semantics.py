"""Semantic space over the 200 real-world objects.

Exact-match accuracy is the wrong single number for a reconstruction of real
life. If someone looked at a *monkey* and the system answers *cow*, that is a
far better reconstruction than *scissors* — same animacy, same category, wrong
individual — and a metric that scores both as zero throws away the part of the
signal EEG actually carries.

So reconstructions are scored on a graded scale over the label hierarchy that
ships with the dataset: animate/inanimate, ten categories, fifty concepts,
and the individual photograph.

The tier weights below are a choice, and an arbitrary one in the sense that no
experiment fixes them. What is *not* a choice is the chance level: it is
computed from the actual catalogue by averaging the metric over every
(truth, guess) pair, so whatever the weights are, the baseline they imply is
measured rather than assumed. A result is only meaningful against that number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .images import Stimulus

#: Graded similarity, most to least specific. A pair scores at the first tier
#: whose condition it satisfies.
TIERS: Tuple[Tuple[str, float], ...] = (
    ("exact", 1.00),      # the same photograph
    ("concept", 0.80),    # a different photograph of the same thing
    ("category", 0.50),   # same category: another mammal, another tool
    ("animacy", 0.20),    # only animate/inanimate agrees
    ("unrelated", 0.00),
)

_TIER_SCORE: Dict[str, float] = dict(TIERS)


def relation(a: Stimulus, b: Stimulus) -> str:
    """The most specific tier at which two stimuli agree."""
    if a.number == b.number:
        return "exact"
    if a.level_c == b.level_c:
        return "concept"
    if a.level_b == b.level_b:
        return "category"
    if a.level_a == b.level_a:
        return "animacy"
    return "unrelated"


def similarity(a: Stimulus, b: Stimulus) -> float:
    return _TIER_SCORE[relation(a, b)]


@dataclass
class SemanticSpace:
    """Pairwise semantic similarity over the whole candidate set."""

    numbers: np.ndarray
    #: (n, n) similarity, indexed by position in ``numbers``.
    matrix: np.ndarray
    labels: List[str]
    level_a: np.ndarray
    level_b: np.ndarray
    level_c: np.ndarray
    _index: Dict[int, int]

    @classmethod
    def from_catalogue(cls, catalogue: Dict[int, Stimulus]) -> "SemanticSpace":
        numbers = np.array(sorted(catalogue), dtype=np.int32)
        items = [catalogue[int(n)] for n in numbers]
        size = len(items)

        matrix = np.zeros((size, size), dtype=np.float32)
        for i, first in enumerate(items):
            for j, second in enumerate(items):
                matrix[i, j] = similarity(first, second)

        return cls(
            numbers=numbers,
            matrix=matrix,
            labels=[item.label for item in items],
            level_a=np.array([item.level_a for item in items]),
            level_b=np.array([item.level_b for item in items]),
            level_c=np.array([item.level_c for item in items]),
            _index={int(n): i for i, n in enumerate(numbers)},
        )

    @property
    def size(self) -> int:
        return int(self.numbers.size)

    def position(self, number: int) -> Optional[int]:
        return self._index.get(int(number))

    def score(self, truth: int, guess: int) -> float:
        """Graded similarity between a true and a guessed stimulus number."""
        i, j = self.position(truth), self.position(guess)
        if i is None or j is None:
            return 0.0
        return float(self.matrix[i, j])

    def tier(self, truth: int, guess: int) -> str:
        i, j = self.position(truth), self.position(guess)
        if i is None or j is None:
            return "unrelated"
        value = float(self.matrix[i, j])
        # Recover the tier name from the score; the weights are distinct.
        for name, weight in TIERS:
            if abs(value - weight) < 1e-6:
                return name
        return "unrelated"

    # ----------------------------------------------------------------- chance

    def chance_similarity(self) -> float:
        """Expected score of a uniformly random guess.

        Averaged over every (truth, guess) pair including the diagonal, since
        a random guess does occasionally land on the right photograph. This is
        the number a reconstruction has to beat to mean anything.
        """
        return float(self.matrix.mean())

    def chance_by_tier(self) -> Dict[str, float]:
        """Probability a random guess lands in each tier."""
        counts: Dict[str, float] = {name: 0.0 for name, _ in TIERS}
        total = float(self.matrix.size)
        for name, weight in TIERS:
            counts[name] = float(np.isclose(self.matrix, weight).sum()) / total
        return counts

    def best_possible(self, truth: Sequence[int]) -> float:
        """Ceiling: 1.0. Present so reports never imply a lower ceiling."""
        return 1.0

    # ------------------------------------------------------------------ score

    def score_many(
        self, truth: Sequence[int], guess: Sequence[int]
    ) -> np.ndarray:
        """Per-item graded similarity for two aligned sequences."""
        out = np.zeros(len(truth), dtype=np.float32)
        for k, (t, g) in enumerate(zip(truth, guess)):
            out[k] = self.score(int(t), int(g))
        return out

    def tier_counts(
        self, truth: Sequence[int], guess: Sequence[int]
    ) -> Dict[str, int]:
        counts: Dict[str, int] = {name: 0 for name, _ in TIERS}
        for t, g in zip(truth, guess):
            counts[self.tier(int(t), int(g))] += 1
        return counts
