"""Decoding stimulus identity and semantics from real EEG.

The central question this module answers is *identification accuracy*: given
the EEG recorded while someone looked at one of 200 objects, can the system
pick the right object out of the 200? That is the meaningful definition of a
"recognizable" reconstruction — far more informative than a pixel metric,
and it is what the fMRI and MEG reconstruction literature reports.

Two things are guarded carefully here, because both are easy ways to get an
impressive number that is not real:

* **Temporal leakage.** Stimuli appear every 200 ms while epochs are 700 ms
  long, so neighbouring trials overlap in time. Splitting trials at random
  puts a test trial's own recording window inside training trials. Alongside
  the conventional repetition-fold split, a strict *time split* trains on the
  first part of the session and tests on the last, with a gap between them.
* **Chance level.** Every accuracy is reported next to its actual chance
  level, and against a label-shuffled control run through the same pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("mindscape.decoding")

#: Post-stimulus window used for decoding. Visual object information peaks
#: between roughly 80 and 300 ms; the later window adds little and dilutes.
DECODE_WINDOW = (0.05, 0.40)

#: Timepoints retained after decimation within that window.
DECODE_TIMEPOINTS = 18

#: Gap left between train and test in the time split, in trials. Comfortably
#: longer than one epoch, so no test epoch overlaps a training epoch.
TIME_SPLIT_GAP = 25


@dataclass
class DecodingFeatures:
    """Flattened spatiotemporal patterns ready for a linear model."""

    #: (trials, channels x timepoints)
    x: np.ndarray
    stimulus: np.ndarray
    level_a: np.ndarray
    level_b: np.ndarray
    #: Trial order in the recording, for the time split.
    order: np.ndarray
    n_channels: int
    n_times: int


def build_features(epochs, window: Tuple[float, float] = DECODE_WINDOW) -> DecodingFeatures:
    """Extract the decoding matrix from an :class:`EpochSet`."""
    times = epochs.times
    mask = (times >= window[0]) & (times <= window[1])
    selected = np.nonzero(mask)[0]

    if selected.size > DECODE_TIMEPOINTS:
        # Even sub-selection rather than a moving average: adjacent samples at
        # 250 Hz are already smooth, and averaging would blur the sharp
        # component timing that carries much of the identity information.
        picks = np.linspace(selected[0], selected[-1], DECODE_TIMEPOINTS).astype(int)
    else:
        picks = selected

    data = epochs.data[:, :, picks]
    n_trials, n_channels, n_times = data.shape

    return DecodingFeatures(
        x=data.reshape(n_trials, n_channels * n_times).astype(np.float64),
        stimulus=np.asarray(epochs.stimulus, dtype=np.int32),
        level_a=np.asarray(epochs.level_a),
        level_b=np.asarray(epochs.level_b),
        order=np.arange(n_trials),
        n_channels=n_channels,
        n_times=n_times,
    )


def _standardise(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Z-score using training statistics only."""
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-9] = 1.0
    return (train - mean) / scale, (test - mean) / scale


# --------------------------------------------------------------------------- #
# Splits                                                                      #
# --------------------------------------------------------------------------- #


def repetition_folds(stimulus: np.ndarray, n_folds: int = 5) -> List[np.ndarray]:
    """Fold assignment that keeps each stimulus balanced across folds.

    Each stimulus is seen ~40 times; folds are assigned by repetition index so
    every fold contains every stimulus roughly equally.
    """
    folds = np.zeros(stimulus.size, dtype=np.int32)
    for value in np.unique(stimulus):
        indices = np.nonzero(stimulus == value)[0]
        folds[indices] = np.arange(indices.size) % n_folds
    return [np.nonzero(folds == k)[0] for k in range(n_folds)]


def time_split(
    n_trials: int, train_fraction: float = 0.6, gap: int = TIME_SPLIT_GAP
) -> Tuple[np.ndarray, np.ndarray]:
    """Train on the earlier part of the session, test on the later part.

    The strict control against temporal leakage: with a gap between the two,
    no test epoch shares any samples with a training epoch.
    """
    cut = int(n_trials * train_fraction)
    train = np.arange(0, cut)
    test = np.arange(min(cut + gap, n_trials), n_trials)
    return train, test


# --------------------------------------------------------------------------- #
# Identification                                                              #
# --------------------------------------------------------------------------- #


@dataclass
class IdentificationResult:
    """How often the right stimulus is recovered, and how close it comes."""

    top1: float
    top5: float
    top10: float
    #: Mean normalised rank of the correct stimulus, 0 = always first.
    mean_rank: float
    #: Pairwise identification: correct beaten against one random other.
    pairwise: float
    n_candidates: int
    n_test: int
    #: Trials averaged per test instance.
    trials_averaged: int
    label: str = ""

    @property
    def chance_top1(self) -> float:
        return 1.0 / self.n_candidates

    def summary(self) -> str:
        return (
            f"{self.label:<26} top-1 {self.top1 * 100:5.1f}% "
            f"(chance {self.chance_top1 * 100:.1f}%)  "
            f"top-5 {self.top5 * 100:5.1f}%  "
            f"pairwise {self.pairwise * 100:5.1f}%  "
            f"rank {self.mean_rank:.3f}"
        )


def _rank_metrics(
    scores: np.ndarray, truth_index: np.ndarray, label: str, trials_averaged: int
) -> IdentificationResult:
    """Turn a (test, candidates) score matrix into identification metrics."""
    n_test, n_candidates = scores.shape

    order = np.argsort(-scores, axis=1)
    ranks = np.empty(n_test, dtype=np.int32)
    for i in range(n_test):
        ranks[i] = int(np.nonzero(order[i] == truth_index[i])[0][0])

    truth_score = scores[np.arange(n_test), truth_index]
    # Pairwise identification: fraction of other candidates the correct one
    # outscores. Equivalent to averaging over every 2-way test.
    better = (scores < truth_score[:, None]).sum(axis=1)
    pairwise = float(np.mean(better / max(n_candidates - 1, 1)))

    return IdentificationResult(
        top1=float(np.mean(ranks == 0)),
        top5=float(np.mean(ranks < 5)),
        top10=float(np.mean(ranks < 10)),
        mean_rank=float(np.mean(ranks / max(n_candidates - 1, 1))),
        pairwise=pairwise,
        n_candidates=n_candidates,
        n_test=n_test,
        trials_averaged=trials_averaged,
        label=label,
    )


def identify_by_template(
    features: DecodingFeatures,
    train_index: np.ndarray,
    test_index: np.ndarray,
    trials_averaged: int = 1,
    label: str = "template",
    shuffle: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> IdentificationResult:
    """Identification by matching against per-stimulus EEG templates.

    Builds an average response per stimulus from the training trials, then
    scores each test instance against all 200 templates by correlation. No
    image features are involved — this measures how much stimulus-specific
    structure the EEG itself carries.
    """
    rng = rng or np.random.default_rng(0)

    x_train, x_test = _standardise(features.x[train_index], features.x[test_index])
    y_train = features.stimulus[train_index]
    y_test = features.stimulus[test_index].copy()

    if shuffle:
        # Control: destroy the trial/label correspondence in the *test* set
        # only, leaving everything else identical. Accuracy must collapse to
        # chance; if it does not, the pipeline is leaking.
        y_test = rng.permutation(y_test)

    candidates = np.unique(features.stimulus)
    templates = np.zeros((candidates.size, x_train.shape[1]))
    for i, value in enumerate(candidates):
        rows = x_train[y_train == value]
        if rows.size:
            templates[i] = rows.mean(axis=0)

    # Group test trials into averaged instances, which is how a real session
    # would use repeated exposures to the same content.
    #
    # The requested averaging is capped at what the test set actually holds
    # per stimulus. A fold with eight test trials per stimulus cannot supply
    # ten, and silently producing no instances at all would report a crash
    # where the honest answer is "averaged as far as the data allows".
    per_stimulus = min(
        int(np.sum(y_test == value)) for value in candidates if np.any(y_test == value)
    )
    effective = max(1, min(trials_averaged, per_stimulus))

    instances: List[np.ndarray] = []
    truth: List[int] = []
    for value in candidates:
        rows = np.nonzero(y_test == value)[0]
        if rows.size == 0:
            continue
        rng.shuffle(rows)
        for start in range(0, rows.size - effective + 1, effective):
            chunk = rows[start : start + effective]
            instances.append(x_test[chunk].mean(axis=0))
            truth.append(int(value))

    if not instances:
        raise RuntimeError("no test instances built")

    matrix = np.stack(instances)

    # Correlation between each instance and each template.
    def centre(a: np.ndarray) -> np.ndarray:
        a = a - a.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(a, axis=1, keepdims=True)
        norm[norm < 1e-12] = 1.0
        return a / norm

    scores = centre(matrix) @ centre(templates).T

    lookup = {int(v): i for i, v in enumerate(candidates)}
    truth_index = np.array([lookup[t] for t in truth], dtype=np.int32)

    return _rank_metrics(scores, truth_index, label, effective)


# --------------------------------------------------------------------------- #
# Semantic decoding                                                           #
# --------------------------------------------------------------------------- #


@dataclass
class ClassificationResult:
    accuracy: float
    chance: float
    n_classes: int
    n_test: int
    label: str
    per_class: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"{self.label:<26} {self.accuracy * 100:5.1f}%  "
            f"(chance {self.chance * 100:.1f}%, {self.n_classes}-way, n={self.n_test})"
        )


def classify(
    features: DecodingFeatures,
    labels: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    label: str,
    shuffle: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> ClassificationResult:
    """Linear discriminant classification of a semantic label."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    rng = rng or np.random.default_rng(0)

    x_train, x_test = _standardise(features.x[train_index], features.x[test_index])
    y_train = labels[train_index]
    y_test = labels[test_index]

    if shuffle:
        y_train = rng.permutation(y_train)

    # Shrinkage is essential: the feature count (channels x timepoints) is
    # comparable to the trial count, and an unregularised covariance estimate
    # is singular.
    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)

    classes = np.unique(labels)
    per_class = {
        str(c): float(np.mean(predicted[y_test == c] == c))
        for c in classes
        if np.any(y_test == c)
    }

    return ClassificationResult(
        accuracy=float(np.mean(predicted == y_test)),
        chance=1.0 / classes.size,
        n_classes=int(classes.size),
        n_test=int(test_index.size),
        label=label,
        per_class=per_class,
    )


# --------------------------------------------------------------------------- #
# Time-resolved decoding                                                      #
# --------------------------------------------------------------------------- #


def time_resolved_accuracy(
    epochs,
    labels: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    step: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Classification accuracy as a function of time after stimulus onset.

    The standard sanity check for any EEG decoding result: accuracy must sit
    at chance before onset and rise afterwards. A curve that is elevated
    *before* the stimulus appeared is proof of leakage, not of decoding.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    times = epochs.times[::step]
    accuracies = np.zeros(times.size)

    for i, index in enumerate(range(0, epochs.times.size, step)):
        if i >= times.size:
            break
        x = epochs.data[:, :, index].astype(np.float64)
        x_train, x_test = _standardise(x[train_index], x[test_index])

        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        model.fit(x_train, labels[train_index])
        accuracies[i] = float(np.mean(model.predict(x_test) == labels[test_index]))

    return times, accuracies
