"""Continuous windows of "what the body was doing", from motor cortex.

The signal being read here is **event-related desynchronization**: when a limb
moves, the mu (8-13 Hz) and beta (13-30 Hz) rhythms over the contralateral
sensorimotor strip drop in power. Moving the left hand suppresses the rhythm
over the right hemisphere and vice versa; feet, whose cortical representation
sits on the midline between the hemispheres, suppress it over the vertex.

That is why this works when reading someone's *visual* experience barely does.
The effect is large, it is spatially organised in a way scalp electrodes can
actually see, and it does not depend on recovering fine detail.

Features are log band power per channel per band — nameable quantities, so a
decoding result stays attributable to a rhythm over a location rather than to
an opaque learned filter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal as sp_signal

from .edf import EdfRecording, read_edf

log = logging.getLogger("mindscape.motor")

#: What the body was doing. "rest" is a real class, not an absence — half the
#: recording is deliberate rest, and a decoder that cannot recognise stillness
#: would smear movement across the whole session.
CLASSES: Tuple[str, ...] = (
    "rest",
    "left fist",
    "right fist",
    "both fists",
    "both feet",
)

#: Which physical action the T1/T2 annotations mean, per run. The database
#: reuses the same two annotation codes for different limbs depending on the
#: run, so reading them without this table silently mislabels half the data.
RUN_TASKS: Dict[int, Tuple[str, str]] = {
    3: ("left fist", "right fist"),
    7: ("left fist", "right fist"),
    11: ("left fist", "right fist"),
    5: ("both fists", "both feet"),
    9: ("both fists", "both feet"),
    13: ("both fists", "both feet"),
}

#: Analysis window and hop, in seconds. One second is long enough to estimate
#: band power stably and short enough to follow a four-second action.
WINDOW = 1.0
STEP = 0.25

#: Rhythms read, in Hz. Split rather than pooled so the decoder can weight mu
#: and beta separately — they desynchronise with different time courses. Two
#: bands rather than three keeps the per-window covariances (64x64 each) to a
#: size that fits in memory for a cross-subject fit.
BANDS: Tuple[Tuple[float, float], ...] = ((8.0, 13.0), (13.0, 30.0))

#: Skipped after a cue before a window counts as that action. Movement does
#: not begin at the cue, and labelling the reaction time as movement teaches
#: the decoder that rest looks like action.
ONSET_LAG = 0.5


@dataclass
class WindowSet:
    """Sliding windows with the action each one covers."""

    #: (windows, channels x bands) log band power.
    x: np.ndarray
    #: (windows, bands, channels, channels) spatial covariance per band. This
    #: is what CSP is fitted from; the diagonal is the band power above.
    cov: np.ndarray
    #: Index into :data:`CLASSES`.
    y: np.ndarray
    #: Window centre, in seconds from the start of its run.
    centre: np.ndarray
    #: Run number each window came from, for leave-one-run-out.
    run: np.ndarray
    subject: str
    channel_names: List[str]
    feature_names: List[str]

    @property
    def n_windows(self) -> int:
        return int(self.x.shape[0])


def _band_power(data: np.ndarray, rate: float) -> np.ndarray:
    """Band-limited signals for every band, shaped (bands, channels, samples)."""
    nyquist = rate / 2.0
    out = np.empty((len(BANDS), *data.shape), dtype=np.float32)
    for index, (low, high) in enumerate(BANDS):
        sos = sp_signal.butter(
            4, [low / nyquist, min(high / nyquist, 0.99)], btype="band", output="sos"
        )
        # Zero-phase, so the power envelope is not shifted in time relative to
        # the annotation that labels it.
        out[index] = sp_signal.sosfiltfilt(sos, data, axis=-1)
    return out


def _label_at(recording: EdfRecording, run: int, time: float) -> Optional[int]:
    """Which action was underway at ``time``, or None if it is ambiguous."""
    task = RUN_TASKS.get(run)
    if task is None:
        return None

    for annotation in recording.annotations:
        start, end = annotation.onset, annotation.onset + annotation.duration
        if not (start <= time < end):
            continue
        if annotation.label == "T0":
            return CLASSES.index("rest")
        if time < start + ONSET_LAG:
            return None
        if annotation.label == "T1":
            return CLASSES.index(task[0])
        if annotation.label == "T2":
            return CLASSES.index(task[1])
    return None


def windows_for_run(path: Path, run: int) -> Optional[WindowSet]:
    """Sliding windows for one recording."""
    recording = read_edf(path)
    rate = recording.sampling_rate

    data = recording.data.astype(np.float64)
    # Common average reference. Without it every channel carries the reference
    # site's activity, which blurs exactly the left/right contrast the decode
    # depends on.
    data -= data.mean(axis=0, keepdims=True)

    filtered = _band_power(data, rate)

    width = int(round(WINDOW * rate))
    hop = int(round(STEP * rate))
    n_channels = data.shape[0]

    rows: List[np.ndarray] = []
    covariances: List[np.ndarray] = []
    labels: List[int] = []
    centres: List[float] = []

    for start in range(0, data.shape[1] - width + 1, hop):
        centre = (start + width / 2.0) / rate
        label = _label_at(recording, run, centre)
        if label is None:
            continue

        block = filtered[:, :, start : start + width]

        # Spatial covariance per band, normalised by its own trace so that a
        # window is described by *how activity is distributed over the scalp*
        # rather than by overall amplitude — which drifts with impedance and
        # would otherwise dominate.
        window_cov = np.empty((len(BANDS), n_channels, n_channels), dtype=np.float32)
        for band in range(len(BANDS)):
            centred = block[band] - block[band].mean(axis=-1, keepdims=True)
            # Accelerate raises spurious divide/overflow/invalid flags on
            # matmul; see app/realdata/reconstruct.py for the check that the
            # results are unaffected.
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                matrix = centred @ centred.T / max(centred.shape[-1] - 1, 1)
            trace = np.trace(matrix)
            window_cov[band] = matrix / trace if trace > 1e-12 else matrix

        # Log power. The log is what makes band power roughly normal and puts
        # multiplicative rhythm changes on an additive scale, which is the
        # scale a linear model can use.
        power = np.log(block.var(axis=-1) + 1e-12)

        rows.append(power.reshape(-1))
        covariances.append(window_cov)
        labels.append(label)
        centres.append(centre)

    if not rows:
        return None

    feature_names = [
        f"{name}_{int(low)}-{int(high)}Hz"
        for low, high in BANDS
        for name in recording.channel_names
    ]

    return WindowSet(
        x=np.stack(rows),
        cov=np.stack(covariances),
        y=np.array(labels, dtype=np.int16),
        centre=np.array(centres),
        run=np.full(len(rows), run, dtype=np.int16),
        subject=recording.subject,
        channel_names=recording.channel_names,
        feature_names=feature_names,
    )


def windows_for_subject(root: Path, subject: str) -> Optional[WindowSet]:
    """Every executed-movement run for one subject, concatenated."""
    parts: List[WindowSet] = []
    for run in sorted(RUN_TASKS):
        path = root / subject / f"{subject}R{run:02d}.edf"
        if not path.exists():
            continue
        part = windows_for_run(path, run)
        if part is not None:
            parts.append(part)

    if not parts:
        return None

    log.info(
        "%s: %d windows over %d runs",
        subject,
        sum(p.n_windows for p in parts),
        len(parts),
    )

    return WindowSet(
        x=np.concatenate([p.x for p in parts]),
        cov=np.concatenate([p.cov for p in parts]),
        y=np.concatenate([p.y for p in parts]),
        centre=np.concatenate([p.centre for p in parts]),
        run=np.concatenate([p.run for p in parts]),
        subject=subject,
        channel_names=parts[0].channel_names,
        feature_names=parts[0].feature_names,
    )


def available_subjects(root: Path) -> List[str]:
    """Subjects with at least one executed-movement run downloaded."""
    found = []
    for directory in sorted(root.glob("S*")):
        if any((directory / f"{directory.name}R{r:02d}.edf").exists() for r in RUN_TASKS):
            found.append(directory.name)
    return found


def load_or_build(root: Path, subject: str, force: bool = False) -> Optional[WindowSet]:
    """Windows for one subject.

    Deliberately not cached. The covariances make a subject roughly 90 MB,
    while rebuilding from the six EDF files takes a few seconds — caching that
    would trade a lot of disk for very little time.
    """
    return windows_for_subject(root, subject)
