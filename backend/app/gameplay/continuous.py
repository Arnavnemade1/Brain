"""Continuous scene reconstruction — no event log at inference time.

The previous reconstruction had a real limitation: it was told *when* things
happened and only recovered *what* they were. This module removes that. A
window slides across the entire recording and, at every step, the EEG alone
decides what is happening right now — firing, being rewarded, crashing, or
nothing at all.

That "nothing at all" class is what makes it honest. Most of a session is
uneventful, so a decoder that never predicts the absence of an event would
report constant action and score well on the moments that happen to be events.
The idle class is sampled from the gaps between real events, and every metric
below includes it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("mindscape.gameplay.continuous")

#: Sliding window step. The decoded scene updates at this rate.
HOP_SECONDS = 0.25

#: Classes recovered at every step. "idle" is the absence of an event.
CLASSES = ["idle", "shoot", "reward", "crash"]

#: A window is labelled with an event if one falls within this distance of
#: its centre. Wider than the epoch so labels are not razor-thin.
MATCH_TOLERANCE = 0.35

#: Minimum distance from any event for a window to count as idle.
IDLE_MARGIN = 1.2


@dataclass
class ContinuousScene:
    """What the EEG says was happening, moment by moment."""

    times: np.ndarray
    predicted: np.ndarray
    actual: np.ndarray
    #: (steps, classes) posterior over CLASSES.
    probabilities: np.ndarray
    subject: str
    duration: float

    def accuracy(self) -> float:
        from .decode import balanced_accuracy

        return balanced_accuracy(self.predicted, self.actual)

    def event_accuracy(self) -> float:
        """Balanced accuracy restricted to moments where something happened."""
        from .decode import balanced_accuracy

        mask = self.actual != "idle"
        if not mask.any():
            return 0.0
        return balanced_accuracy(self.predicted[mask], self.actual[mask])

    def detection_rate(self) -> Tuple[float, float]:
        """(recall, precision) for "something is happening" against idle."""
        actual_event = self.actual != "idle"
        predicted_event = self.predicted != "idle"
        recall = (
            float(np.mean(predicted_event[actual_event])) if actual_event.any() else 0.0
        )
        precision = (
            float(np.mean(actual_event[predicted_event])) if predicted_event.any() else 0.0
        )
        return recall, precision


def label_windows(
    centres: np.ndarray, onsets: np.ndarray, categories: Sequence[str]
) -> np.ndarray:
    """Ground-truth class for each window centre."""
    labels = np.full(centres.shape, "idle", dtype=object)
    categories = np.asarray(categories)

    for i, centre in enumerate(centres):
        distances = np.abs(onsets - centre)
        nearest = int(np.argmin(distances)) if distances.size else -1
        if nearest < 0:
            continue
        if distances[nearest] <= MATCH_TOLERANCE:
            labels[i] = categories[nearest]
        elif distances[nearest] < IDLE_MARGIN:
            # Ambiguous: close to an event but not on it. Excluded rather
            # than forced into a class it may not belong to.
            labels[i] = "ambiguous"

    return labels


def build_windows(
    set_path,
    events_path,
    subject: str,
    hop: float = HOP_SECONDS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, List[str]]:
    """Slide a window across the whole recording.

    Returns ``(centres, features, labels, duration, channel_names)``.
    """
    from scipy import signal as sp_signal

    from ..emotion.eeglab import read
    from .events import BAND, EPOCH_END, EPOCH_START, TARGET_RATE, read_events

    events = read_events(events_path)
    onsets = np.array([e.onset for e in events])
    categories = [e.category for e in events]

    with read(set_path) as recording:
        rate = recording.sampling_rate
        channels = recording.eeg_channels
        names = [recording.channel_names[i] for i in channels]
        continuous = recording.window(0, recording.n_samples)[channels]
        duration = recording.duration

    nyquist = rate / 2.0
    sos = sp_signal.butter(
        4, [BAND[0] / nyquist, min(BAND[1] / nyquist, 0.99)], btype="band", output="sos"
    )
    continuous = sp_signal.sosfiltfilt(sos, continuous, axis=-1).astype(np.float32)
    continuous -= continuous.mean(axis=0, keepdims=True)

    start_offset = int(round(EPOCH_START * rate))
    length = int(round((EPOCH_END - EPOCH_START) * rate))
    decimation = max(1, int(round(rate / TARGET_RATE)))
    baseline_hi = int(round(-EPOCH_START * rate))

    centres = np.arange(1.0, duration - 1.0, hop)
    labels = label_windows(centres, onsets, categories)

    keep_index = [i for i, l in enumerate(labels) if l != "ambiguous"]
    centres = centres[keep_index]
    labels = labels[keep_index]

    rows: List[np.ndarray] = []
    valid: List[int] = []

    for i, centre in enumerate(centres):
        begin = int(round(centre * rate)) + start_offset
        if begin < 0 or begin + length > continuous.shape[1]:
            continue
        epoch = continuous[:, begin : begin + length]
        epoch = epoch - epoch[:, :baseline_hi].mean(axis=1, keepdims=True)
        rows.append(epoch[:, ::decimation].astype(np.float32))
        valid.append(i)

    if not rows:
        raise RuntimeError(f"{subject}: no windows")

    data = np.stack(rows)
    n = data.shape[0]

    log.info(
        "%s: %d windows over %.0f s — %s",
        subject,
        n,
        duration,
        ", ".join(
            f"{c} {int(np.sum(labels[valid] == c))}" for c in CLASSES if np.any(labels[valid] == c)
        ),
    )

    return centres[valid], data.reshape(n, -1).astype(np.float64), labels[valid], duration, names


def _balance(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, cap: int = 900):
    """Subsample each class to a common size.

    Idle windows outnumber events by an order of magnitude. Left unbalanced,
    the classifier maximises accuracy by predicting idle forever, which is
    both useless and would look respectable in raw accuracy.
    """
    keep: List[int] = []
    smallest = min(int(np.sum(y == c)) for c in np.unique(y))
    target = min(cap, max(smallest, 1))

    for value in np.unique(y):
        index = np.nonzero(y == value)[0]
        if index.size > target:
            index = rng.choice(index, size=target, replace=False)
        keep.extend(index.tolist())

    keep = np.array(sorted(keep))
    return x[keep], y[keep]


def reconstruct_session(
    train: List[Tuple[np.ndarray, np.ndarray]],
    test_x: np.ndarray,
    test_y: np.ndarray,
    test_times: np.ndarray,
    subject: str,
    duration: float,
    seed: int = 0,
) -> ContinuousScene:
    """Decode every moment of one session from a model fitted on others."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    rng = np.random.default_rng(seed)

    x_train = np.concatenate([x for x, _ in train])
    y_train = np.concatenate([y for _, y in train])
    x_train, y_train = _balance(x_train, y_train, rng)

    mean = x_train.mean(axis=0, keepdims=True)
    scale = x_train.std(axis=0, keepdims=True)
    scale[scale < 1e-9] = 1.0

    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    model.fit((x_train - mean) / scale, y_train)

    standardised = (test_x - mean) / scale
    predicted = model.predict(standardised)

    try:
        raw = model.predict_proba(standardised)
        order = list(model.classes_)
        probabilities = np.zeros((standardised.shape[0], len(CLASSES)))
        for i, name in enumerate(CLASSES):
            if name in order:
                probabilities[:, i] = raw[:, order.index(name)]
    except Exception:  # noqa: BLE001
        probabilities = np.full((standardised.shape[0], len(CLASSES)), 1.0 / len(CLASSES))

    return ContinuousScene(
        times=test_times,
        predicted=np.asarray(predicted, dtype=object),
        actual=np.asarray(test_y, dtype=object),
        probabilities=probabilities,
        subject=subject,
        duration=duration,
    )
