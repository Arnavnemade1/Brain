"""Reconstruction by retrieval.

Given EEG from a single viewing, rank all 200 candidate objects and return the
best matches. The reconstruction *is* the ranked candidate set, and it is
recognizable for a straightforward reason: every candidate is a real
photograph, so whenever the ranking is right the output is exactly what the
person saw.

This is the honest way to get a recognizable image out of EEG. Generating
pixels instead would produce something that looks like a reconstruction while
being mostly the generative prior's invention — scalp EEG does not carry
per-pixel information. Retrieval makes the contribution auditable: the rank of
the true stimulus says precisely how much the brain signal narrowed the field,
and the top-1 image is either right or it is not.

What retrieval cannot do is generalise beyond its candidate set. That is a
real limitation and it is reported rather than hidden.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .images import Stimulus, StimulusEmbeddings, thumbnail

log = logging.getLogger("mindscape.reconstruct")


@dataclass
class Candidate:
    """One ranked hypothesis about what was being viewed."""

    rank: int
    stimulus_number: int
    filename: str
    label: str
    level_a: str
    level_b: str
    score: float
    #: Normalised 0-1 confidence across the candidate set.
    probability: float


@dataclass
class Reconstruction:
    """The output of one reconstruction attempt."""

    candidates: List[Candidate]
    #: True stimulus, when known (always, in this validation setting).
    truth_number: Optional[int]
    truth_filename: Optional[str]
    #: Rank of the true stimulus, 0-based. None when truth is unknown.
    truth_rank: Optional[int]
    #: Semantic agreement of the top candidate with the truth.
    correct_animacy: Optional[bool]
    correct_category: Optional[bool]
    trials_used: int

    @property
    def top(self) -> Candidate:
        return self.candidates[0]

    @property
    def is_exact(self) -> bool:
        return self.truth_rank == 0

    @property
    def in_top5(self) -> bool:
        return self.truth_rank is not None and self.truth_rank < 5


class RetrievalReconstructor:
    """Matches EEG against per-stimulus templates learned from training data."""

    def __init__(
        self,
        embeddings: StimulusEmbeddings,
        stimuli_dir: Path,
    ) -> None:
        self.embeddings = embeddings
        self.stimuli_dir = stimuli_dir
        self.candidates: np.ndarray = np.array([], dtype=np.int32)
        self.templates: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._scale: Optional[np.ndarray] = None

    # ------------------------------------------------------------------- fit

    def fit(self, x: np.ndarray, stimulus: np.ndarray) -> "RetrievalReconstructor":
        """Learn one EEG template per stimulus from training trials."""
        self._mean = x.mean(axis=0, keepdims=True)
        scale = x.std(axis=0, keepdims=True)
        scale[scale < 1e-9] = 1.0
        self._scale = scale

        standardised = (x - self._mean) / self._scale

        self.candidates = np.unique(stimulus)
        self.templates = np.zeros((self.candidates.size, x.shape[1]))
        for i, value in enumerate(self.candidates):
            rows = standardised[stimulus == value]
            if rows.size:
                self.templates[i] = rows.mean(axis=0)

        # Pre-normalise so scoring is a single matrix product.
        self._templates_unit = self._unit(self.templates)
        return self

    @staticmethod
    def _unit(a: np.ndarray) -> np.ndarray:
        centred = a - a.mean(axis=-1, keepdims=True)
        norm = np.linalg.norm(centred, axis=-1, keepdims=True)
        norm[norm < 1e-12] = 1.0
        return centred / norm

    # --------------------------------------------------------------- predict

    def reconstruct(
        self,
        x: np.ndarray,
        truth_number: Optional[int] = None,
        top_k: int = 8,
        temperature: float = 0.02,
    ) -> Reconstruction:
        """Rank all candidates for one or more trials of the same viewing.

        ``x`` is ``(trials, features)``; trials are averaged first, which is
        what repeated exposure to the same content buys.
        """
        if self.templates is None or self._mean is None:
            raise RuntimeError("reconstructor not fitted")

        if x.ndim == 1:
            x = x[None, :]
        trials_used = int(x.shape[0])

        standardised = (x - self._mean) / self._scale
        pattern = self._unit(standardised.mean(axis=0))

        scores = self._templates_unit @ pattern

        # Softmax over correlations gives a comparable confidence across the
        # candidate set. The temperature is small because correlations here
        # are small in absolute terms even when the ranking is informative.
        shifted = (scores - scores.max()) / max(temperature, 1e-6)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()

        order = np.argsort(-scores)
        catalogue = self.embeddings.catalogue

        candidates: List[Candidate] = []
        for rank, index in enumerate(order[:top_k]):
            number = int(self.candidates[index])
            stimulus = catalogue.get(number)
            candidates.append(
                Candidate(
                    rank=rank,
                    stimulus_number=number,
                    filename=stimulus.filename if stimulus else f"stim{number:03d}",
                    label=stimulus.label if stimulus else str(number),
                    level_a=stimulus.level_a if stimulus else "",
                    level_b=stimulus.level_b if stimulus else "",
                    score=float(scores[index]),
                    probability=float(probabilities[index]),
                )
            )

        truth_rank = None
        truth_filename = None
        correct_animacy = None
        correct_category = None

        if truth_number is not None:
            matches = np.nonzero(self.candidates == truth_number)[0]
            if matches.size:
                truth_rank = int(np.nonzero(order == matches[0])[0][0])
            truth_stimulus = catalogue.get(int(truth_number))
            if truth_stimulus is not None:
                truth_filename = truth_stimulus.filename
                if candidates:
                    correct_animacy = candidates[0].level_a == truth_stimulus.level_a
                    correct_category = candidates[0].level_b == truth_stimulus.level_b

        return Reconstruction(
            candidates=candidates,
            truth_number=int(truth_number) if truth_number is not None else None,
            truth_filename=truth_filename,
            truth_rank=truth_rank,
            correct_animacy=correct_animacy,
            correct_category=correct_category,
            trials_used=trials_used,
        )

    # ------------------------------------------------------------- rendering

    def composite(self, reconstruction: Reconstruction, top_k: int = 4, size: int = 64) -> np.ndarray:
        """Probability-weighted blend of the top candidates.

        A useful visualisation of *uncertainty*: when the EEG is decisive the
        composite is nearly the top image, and when it is not the blend is
        visibly smeared across several objects. It is a summary of the
        ranking, not a claim about what was seen — the top-1 image is the
        actual reconstruction.
        """
        selected = reconstruction.candidates[:top_k]
        if not selected:
            return np.zeros((size, size, 3), dtype=np.float32)

        weights = np.array([c.probability for c in selected], dtype=np.float32)
        total = weights.sum()
        weights = weights / total if total > 1e-9 else np.ones_like(weights) / len(weights)

        blend = np.zeros((size, size, 3), dtype=np.float32)
        for weight, candidate in zip(weights, selected):
            path = self.stimuli_dir / candidate.filename
            if not path.exists():
                continue
            blend += weight * thumbnail(path, size=size)

        return np.clip(blend, 0.0, 1.0)


def encode_png(array: np.ndarray) -> str:
    """RGB float array to a base64 PNG, for transport to the client."""
    from io import BytesIO

    from PIL import Image

    image = Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8), mode="RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_stimulus(stimuli_dir: Path, filename: str, size: int = 96) -> str:
    """A stimulus image as a base64 PNG."""
    path = stimuli_dir / filename
    if not path.exists():
        return ""
    return encode_png(thumbnail(path, size=size))
