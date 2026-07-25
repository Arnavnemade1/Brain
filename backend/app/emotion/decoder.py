"""Continuous emotion decoding.

Maps the affective feature sequence onto valence and arousal, producing an
emotion value for every moment of a clip rather than one label for the whole
thing.

Two honesty constraints shape this module, and both cost accuracy:

**Supervision is trial-level.** DENS records one valence and one arousal
rating per clip, so every window inside a trial inherits the same target. The
decoder therefore learns what a *sustained* affective state looks like, and
the within-trial variation it produces is an extrapolation from that, not
something directly supervised. The click timestamps are the only per-moment
ground truth available, and they are held back for exactly that reason — see
`validation.click_alignment`.

**Validation is leave-one-subject-out.** EEG differs enormously between
people; a model tested on the same subject it was fitted on mostly measures
that subject's idiosyncratic scalp. Splitting by subject is the only split
that answers "would this work on someone new".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("mindscape.emotion.decoder")

#: Ratings are 1-9; the decoder works in a centred -1..+1 space.
RATING_MIN = 1.0
RATING_MAX = 9.0


def to_unit(rating: float) -> float:
    """Map a 1-9 self-report rating onto -1..+1."""
    return float(
        np.clip((rating - RATING_MIN) / (RATING_MAX - RATING_MIN) * 2.0 - 1.0, -1.0, 1.0)
    )


def to_rating(unit: float) -> float:
    """Inverse of :func:`to_unit`."""
    return float(np.clip((unit + 1.0) / 2.0 * (RATING_MAX - RATING_MIN) + RATING_MIN, 1.0, 9.0))


@dataclass
class TrialFeatures:
    """Feature sequence for one trial, with its labels."""

    subject: str
    trial_index: int
    stimulus: str
    times: np.ndarray
    features: np.ndarray
    valence: float
    arousal: float
    quadrant: str
    emotion: str
    click_times: List[float] = field(default_factory=list)
    duration: float = 0.0


def normalise_per_subject(trials: Sequence[TrialFeatures]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Per-subject feature mean and scale.

    Absolute EEG feature values differ by more between two people than
    between two emotions in the same person — skull thickness, electrode
    impedance and resting rhythms all shift them. Without per-subject
    standardisation a cross-subject decoder mostly learns who is wearing the
    cap.
    """
    stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    subjects = sorted({t.subject for t in trials})

    for subject in subjects:
        stacked = np.concatenate(
            [t.features for t in trials if t.subject == subject and t.features.size]
        )
        mean = stacked.mean(axis=0)
        scale = stacked.std(axis=0)
        scale[scale < 1e-9] = 1.0
        stats[subject] = (mean, scale)

    return stats


def build_matrix(
    trials: Sequence[TrialFeatures],
    stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stack every window into ``(x, valence, arousal, trial_id)``."""
    rows: List[np.ndarray] = []
    valence: List[float] = []
    arousal: List[float] = []
    trial_id: List[int] = []

    for index, trial in enumerate(trials):
        if not trial.features.size:
            continue
        mean, scale = stats.get(
            trial.subject, (np.zeros(trial.features.shape[1]), np.ones(trial.features.shape[1]))
        )
        standardised = (trial.features - mean) / scale
        rows.append(standardised)
        valence.extend([to_unit(trial.valence)] * standardised.shape[0])
        arousal.extend([to_unit(trial.arousal)] * standardised.shape[0])
        trial_id.extend([index] * standardised.shape[0])

    if not rows:
        empty = np.zeros((0, 0))
        return empty, np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)

    return (
        np.concatenate(rows),
        np.array(valence),
        np.array(arousal),
        np.array(trial_id, dtype=int),
    )


@dataclass
class EmotionModel:
    """A fitted valence/arousal decoder."""

    valence_weights: np.ndarray
    valence_intercept: float
    arousal_weights: np.ndarray
    arousal_intercept: float
    feature_names: List[str]
    #: Held-out performance recorded at fit time, reported never used.
    scores: Dict[str, float] = field(default_factory=dict)

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Per-window valence and arousal in -1..+1."""
        if x.ndim == 1:
            x = x[None, :]
        valence = np.clip(x @ self.valence_weights + self.valence_intercept, -1.0, 1.0)
        arousal = np.clip(x @ self.arousal_weights + self.arousal_intercept, -1.0, 1.0)
        return valence, arousal

    def to_dict(self) -> dict:
        return {
            "valenceWeights": self.valence_weights.tolist(),
            "valenceIntercept": self.valence_intercept,
            "arousalWeights": self.arousal_weights.tolist(),
            "arousalIntercept": self.arousal_intercept,
            "featureNames": self.feature_names,
            "scores": self.scores,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "EmotionModel":
        return cls(
            valence_weights=np.array(payload["valenceWeights"], dtype=np.float64),
            valence_intercept=float(payload["valenceIntercept"]),
            arousal_weights=np.array(payload["arousalWeights"], dtype=np.float64),
            arousal_intercept=float(payload["arousalIntercept"]),
            feature_names=list(payload["featureNames"]),
            scores=dict(payload.get("scores", {})),
        )


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[np.ndarray, float]:
    """Ridge regression returning ``(weights, intercept)``."""
    mean = x.mean(axis=0)
    centred = x - mean
    target_mean = y.mean()

    gram = centred.T @ centred + alpha * np.eye(centred.shape[1])
    weights = np.linalg.solve(gram, centred.T @ (y - target_mean))
    intercept = float(target_mean - mean @ weights)
    return weights, intercept


def fit(
    trials: Sequence[TrialFeatures],
    feature_names: List[str],
    alpha: float = 50.0,
) -> Tuple[EmotionModel, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Fit a decoder on every supplied trial."""
    stats = normalise_per_subject(trials)
    x, valence, arousal, _ = build_matrix(trials, stats)

    if x.size == 0:
        raise RuntimeError("no windows to fit on")

    valence_weights, valence_intercept = _ridge(x, valence, alpha)
    arousal_weights, arousal_intercept = _ridge(x, arousal, alpha)

    return (
        EmotionModel(
            valence_weights=valence_weights,
            valence_intercept=valence_intercept,
            arousal_weights=arousal_weights,
            arousal_intercept=arousal_intercept,
            feature_names=list(feature_names),
        ),
        stats,
    )


def decode_trial(
    model: EmotionModel,
    trial: TrialFeatures,
    stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
    smooth: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-window valence and arousal for one trial.

    Lightly smoothed in time. Emotional state does not change meaningfully
    between adjacent half-second windows, so unsmoothed output is dominated by
    measurement noise; the width is kept short enough to preserve a genuine
    turn within a couple of seconds.
    """
    if not trial.features.size:
        return np.zeros(0), np.zeros(0)

    mean, scale = stats.get(
        trial.subject,
        (np.zeros(trial.features.shape[1]), np.ones(trial.features.shape[1])),
    )
    x = (trial.features - mean) / scale
    valence, arousal = model.predict(x)

    if smooth > 0 and valence.size > 2:
        # Hop is 0.5 s, so a `smooth` of 1.5 s spans three windows.
        width = max(1, int(round(smooth / 0.5)))
        kernel = np.hanning(width + 2)[1:-1]
        kernel = kernel / kernel.sum() if kernel.sum() > 0 else np.ones(1)
        valence = np.convolve(valence, kernel, mode="same")
        arousal = np.convolve(arousal, kernel, mode="same")

    return valence, arousal


def quadrant_of(valence: float, arousal: float) -> str:
    return ("HV" if valence >= 0 else "LV") + ("HA" if arousal >= 0 else "LA")


#: Named emotions positioned on the valence-arousal circumplex, used to give
#: the decoded trajectory a readable label at each moment.
CIRCUMPLEX: List[Tuple[str, float, float]] = [
    ("excited", 0.6, 0.8),
    ("joyous", 0.8, 0.5),
    ("content", 0.7, -0.3),
    ("calm", 0.5, -0.7),
    ("bored", -0.4, -0.7),
    ("sad", -0.7, -0.4),
    ("distressed", -0.7, 0.4),
    ("afraid", -0.6, 0.7),
    ("angry", -0.8, 0.7),
    ("alert", 0.1, 0.8),
    ("neutral", 0.0, 0.0),
]


def label_for(valence: float, arousal: float) -> str:
    """Nearest circumplex emotion to a decoded point."""
    best = min(
        CIRCUMPLEX,
        key=lambda entry: (valence - entry[1]) ** 2 + (arousal - entry[2]) ** 2,
    )
    return best[0]
