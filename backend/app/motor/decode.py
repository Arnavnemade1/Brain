"""Decoding what the body was doing, from sensorimotor rhythms.

Every accuracy here is **balanced** — the mean of the per-class rates. It
matters more in this dataset than usual: rest occupies over half of every
recording, so a decoder that predicted "rest" and nothing else would score
54% raw and look like it worked. Balanced accuracy puts any constant
prediction at 1/5.

Two validation regimes, and they answer different questions:

* **leave-one-run-out** — train on a person's other runs, decode the held-out
  one. This is the honest model of a headset that has been calibrated on its
  wearer, which is how such a device would actually be used.
* **leave-one-subject-out** — train on other people entirely. Harder, and the
  question of whether sensorimotor rhythms transfer across skulls.

Windows overlap (one second wide, quarter-second hop), so a random split would
put neighbouring windows sharing three quarters of their samples on both sides
of the fold boundary. Splits are therefore always by *run* or by *subject*,
never by window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .windows import CLASSES, WindowSet

log = logging.getLogger("mindscape.motor.decode")


@dataclass
class Prediction:
    """Held-out predictions for every window of a session."""

    predicted: np.ndarray
    actual: np.ndarray
    #: (windows, classes) posterior.
    probability: np.ndarray
    centre: np.ndarray
    run: np.ndarray
    subject: str


def balanced_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Mean per-class accuracy. 1/n_classes for any constant prediction."""
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)
    rates = []
    for value in np.unique(actual):
        mask = actual == value
        if mask.any():
            rates.append(float(np.mean(predicted[mask] == value)))
    return float(np.mean(rates)) if rates else 0.0


def confusion(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Row-normalised confusion, rows are truth."""
    size = len(CLASSES)
    matrix = np.zeros((size, size))
    for truth in range(size):
        mask = actual == truth
        if not mask.any():
            continue
        for guess in range(size):
            matrix[truth, guess] = float(np.mean(predicted[mask] == guess))
    return matrix


def _model():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    # Shrinkage LDA: 128 correlated band-power features against a few thousand
    # windows is exactly the regime where an unregularised covariance estimate
    # falls apart. Linear, so the weights stay attributable to a rhythm over a
    # scalp location.
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def _fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit on one fold, predict another. Log band power only.

    Common spatial patterns were tried here and removed. Fitted per fold on
    the training data only, one-versus-rest over these classes, CSP came out
    +0.6 points on moving-versus-rest, +1.5 on left-versus-right fist and
    **-3.9** on fists-versus-feet, and 2.5 points *worse* on the five-class
    problem. It did not earn the extra machinery, so the features are the
    plain per-channel log band powers, which are also individually nameable.
    """
    mean = train_x.mean(axis=0, keepdims=True)
    scale = train_x.std(axis=0, keepdims=True)
    scale[scale < 1e-9] = 1.0

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        model = _model().fit((train_x - mean) / scale, train_y)
        standardised = (test_x - mean) / scale
        probability = model.predict_proba(standardised)
    # Map back to the full class list: a training fold may not contain every
    # class, and `predict_proba` only spans the ones it saw.
    full = np.zeros((standardised.shape[0], len(CLASSES)))
    for column, label in enumerate(model.classes_):
        full[:, int(label)] = probability[:, column]

    return full.argmax(axis=1), full


def leave_one_run_out(windows: WindowSet, shuffle: bool = False, seed: int = 0) -> Prediction:
    """Decode every window of a subject from a model fitted on their other runs."""
    y = windows.y.copy()
    if shuffle:
        # Permute labels within run, so the class balance and the run structure
        # both survive and only the association with the EEG is destroyed.
        rng = np.random.default_rng(seed)
        for run in np.unique(windows.run):
            mask = windows.run == run
            y[mask] = rng.permutation(y[mask])

    predicted = np.zeros(y.size, dtype=np.int16)
    probability = np.zeros((y.size, len(CLASSES)))

    for run in np.unique(windows.run):
        test = windows.run == run
        train = ~test
        if not train.any() or not test.any():
            continue
        predicted[test], probability[test] = _fit_predict(
            windows.x[train], y[train], windows.x[test]
        )

    return Prediction(
        predicted=predicted,
        actual=y,
        probability=probability,
        centre=windows.centre,
        run=windows.run,
        subject=windows.subject,
    )


def leave_one_subject_out(
    subjects: Sequence[WindowSet], held_out: int, shuffle: bool = False, seed: int = 0
) -> Prediction:
    """Decode one person's session from a model fitted on everyone else."""
    test = subjects[held_out]
    train_sets = [s for i, s in enumerate(subjects) if i != held_out]

    train_x = np.concatenate([s.x for s in train_sets])
    train_y = np.concatenate([s.y for s in train_sets])

    y = test.y.copy()
    if shuffle:
        rng = np.random.default_rng(seed)
        train_y = rng.permutation(train_y)

    predicted, probability = _fit_predict(train_x, train_y, test.x)

    return Prediction(
        predicted=predicted,
        actual=y,
        probability=probability,
        centre=test.centre,
        run=test.run,
        subject=test.subject,
    )


@dataclass
class Report:
    balanced: float
    raw: float
    per_class: Dict[str, float]
    confusion: np.ndarray
    n_windows: int

    def summary(self) -> str:
        return (
            f"balanced {self.balanced * 100:.1f}%  raw {self.raw * 100:.1f}%  "
            f"({self.n_windows} windows, chance {100 / len(CLASSES):.0f}%)"
        )


def score(prediction: Prediction) -> Report:
    per_class = {}
    for index, name in enumerate(CLASSES):
        mask = prediction.actual == index
        if mask.any():
            per_class[name] = float(np.mean(prediction.predicted[mask] == index))

    return Report(
        balanced=balanced_accuracy(prediction.predicted, prediction.actual),
        raw=float(np.mean(prediction.predicted == prediction.actual)),
        per_class=per_class,
        confusion=confusion(prediction.predicted, prediction.actual),
        n_windows=int(prediction.actual.size),
    )


#: The questions worth asking separately. The five-way problem forces a model
#: to tell "both fists" from "left fist" — actions that never occur in the
#: same run — and the resulting number hides a real structure: how well a
#: distinction decodes tracks how far apart the body parts sit on the cortex.
BREAKDOWN: Tuple[Tuple[str, Tuple[int, ...], int], ...] = (
    ("moving vs rest", (0, 1, 2, 3, 4), 2),
    ("fists vs feet", (3, 4), 2),
    ("left vs right fist", (1, 2), 2),
    ("which of four limbs", (1, 2, 3, 4), 4),
)


def decompose(windows: WindowSet, shuffle: bool = False, seed: int = 0) -> Dict[str, float]:
    """Balanced accuracy for each sub-question, leave-one-run-out."""
    out: Dict[str, float] = {}

    for name, classes, _ in BREAKDOWN:
        if name == "moving vs rest":
            keep = np.ones(windows.y.size, dtype=bool)
            labels = (windows.y != 0).astype(np.int16)
        else:
            keep = np.isin(windows.y, classes)
            labels = windows.y

        y = labels[keep]
        run = windows.run[keep]
        x = windows.x[keep]

        if shuffle:
            rng = np.random.default_rng(seed)
            y = rng.permutation(y)

        predicted = np.zeros(y.size, dtype=np.int16)
        for value in np.unique(run):
            test = run == value
            train = ~test
            if not train.any() or np.unique(y[train]).size < 2:
                continue
            predicted[test], _ = _fit_predict(x[train], y[train], x[test])

        out[name] = balanced_accuracy(predicted, y)

    return out


#: Which effector each action uses. Getting "both fists" when the truth was
#: "right fist" is a different quality of error from answering "both feet" —
#: the decoder saw hand movement and could not lateralise it. Scoring both as
#: a plain miss hides the shape of what the signal carries, the same way exact
#: match hid it for the object reconstruction.
EFFECTOR: Tuple[int, ...] = (0, 1, 1, 1, 2)  # rest, hands, hands, hands, feet

TIERS: Tuple[str, ...] = ("exact", "effector", "moving", "wrong")


def tier_of(truth: int, guess: int) -> str:
    """Most specific level at which a guess agrees with the truth."""
    if truth == guess:
        return "exact"
    if EFFECTOR[truth] == EFFECTOR[guess]:
        return "effector"
    if (truth == 0) == (guess == 0):
        return "moving"
    return "wrong"


def graded(prediction: Prediction) -> Dict[str, float]:
    """Tier rates, **balanced** across the true action.

    Computed per true class and then averaged, for the same reason every
    accuracy here is balanced: rest is 54% of the windows, so a raw tier rate
    of 53% would be *below* the score of always answering "rest" while looking
    like a result.
    """
    per_class: Dict[str, List[float]] = {name: [] for name in TIERS}

    for truth in np.unique(prediction.actual):
        mask = prediction.actual == truth
        guesses = prediction.predicted[mask]
        for name in TIERS:
            share = np.mean(
                [tier_of(int(truth), int(g)) == name for g in guesses]
            )
            per_class[name].append(float(share))

    out = {name: float(np.mean(values)) for name, values in per_class.items()}
    out["exact or effector"] = out["exact"] + out["effector"]
    out["right about moving"] = out["exact"] + out["effector"] + out["moving"]
    return out
