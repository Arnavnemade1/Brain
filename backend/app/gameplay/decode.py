"""Reconstructing gameplay from EEG.

Given only the brain activity recorded while someone played, recover what was
happening: when they fired, when they were rewarded, when they crashed.

Validation is leave-one-subject-out and every accuracy is **balanced**, which
matters here more than usual — rewards outnumber crashes five to one, so raw
accuracy would reward a decoder that simply never predicts a crash.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("mindscape.gameplay.decode")

#: Post-event window used for decoding. Covers the feedback complex and the
#: motor response; earlier and later samples add noise.
DECODE_WINDOW = (-0.10, 0.70)
DECODE_TIMEPOINTS = 22


@dataclass
class Features:
    x: np.ndarray
    categories: np.ndarray
    onsets: np.ndarray
    subject: str
    duration: float


def build(epochs, window: Tuple[float, float] = DECODE_WINDOW) -> Features:
    mask = (epochs.times >= window[0]) & (epochs.times <= window[1])
    selected = np.nonzero(mask)[0]
    if selected.size > DECODE_TIMEPOINTS:
        picks = np.linspace(selected[0], selected[-1], DECODE_TIMEPOINTS).astype(int)
    else:
        picks = selected

    data = epochs.data[:, :, picks]
    n = data.shape[0]
    return Features(
        x=data.reshape(n, -1).astype(np.float64),
        categories=np.array(epochs.categories),
        onsets=np.asarray(epochs.onsets),
        subject=epochs.subject,
        duration=epochs.duration,
    )


def balanced_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Mean per-class accuracy. 50% for any constant prediction."""
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)
    if predicted.size == 0:
        return 0.0
    rates = []
    for value in np.unique(actual):
        mask = actual == value
        if mask.any():
            rates.append(float(np.mean(predicted[mask] == value)))
    return float(np.mean(rates)) if rates else 0.0


def _standardise(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-9] = 1.0
    return (train - mean) / scale, (test - mean) / scale


def _classifier():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    # Shrinkage is required: features (channels x timepoints) are comparable
    # in number to the trials, so the covariance estimate is singular without.
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def classify_pair(
    train: List[Features],
    test: Features,
    positive: str,
    negative: str,
    shuffle: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """Binary decoding of one event category against another."""
    rng = rng or np.random.default_rng(0)

    def subset(features: Features):
        mask = np.isin(features.categories, [positive, negative])
        return features.x[mask], features.categories[mask]

    x_train = np.concatenate([subset(f)[0] for f in train])
    y_train = np.concatenate([subset(f)[1] for f in train])
    x_test, y_test = subset(test)

    if x_train.size == 0 or x_test.size == 0:
        return {"accuracy": 0.0, "n": 0}

    if shuffle:
        y_train = rng.permutation(y_train)

    x_train, x_test = _standardise(x_train, x_test)

    model = _classifier()
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)

    return {
        "accuracy": balanced_accuracy(predicted, y_test),
        "n": int(y_test.size),
    }


@dataclass
class Reconstruction:
    """The recovered sequence of events for one session."""

    onsets: np.ndarray
    actual: np.ndarray
    predicted: np.ndarray
    #: Decoder confidence per event, 0-1.
    confidence: np.ndarray
    subject: str
    duration: float

    @property
    def accuracy(self) -> float:
        return balanced_accuracy(self.predicted, self.actual)

    @property
    def raw_accuracy(self) -> float:
        return float(np.mean(self.predicted == self.actual))


def reconstruct(
    train: List[Features], test: Features, categories: Sequence[str] = ("reward", "crash", "shoot")
) -> Reconstruction:
    """Decode the whole session: what was happening at each event."""
    keep = np.isin(test.categories, list(categories))

    x_train = np.concatenate(
        [f.x[np.isin(f.categories, list(categories))] for f in train]
    )
    y_train = np.concatenate(
        [f.categories[np.isin(f.categories, list(categories))] for f in train]
    )

    x_test = test.x[keep]
    y_test = test.categories[keep]

    x_train, x_test = _standardise(x_train, x_test)

    model = _classifier()
    model.fit(x_train, y_train)

    predicted = model.predict(x_test)
    try:
        probabilities = model.predict_proba(x_test)
        confidence = probabilities.max(axis=1)
    except Exception:  # noqa: BLE001
        confidence = np.full(predicted.shape, 0.5)

    return Reconstruction(
        onsets=test.onsets[keep],
        actual=y_test,
        predicted=predicted,
        confidence=confidence,
        subject=test.subject,
        duration=test.duration,
    )
