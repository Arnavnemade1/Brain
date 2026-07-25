"""Signal synchronization.

The EEG amplifier and the video playback machine keep independent clocks.
They start at different moments and tick at slightly different rates, so over
a few minutes a response drifts away from the frame that caused it. Without
correction, an evoked potential recorded 300 ms late is attributed to the
wrong scene, and every downstream fidelity number is quietly wrong.

Recovering the relationship is a two-parameter fit: a constant offset and a
linear drift.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..models.neural import SyncResult

log = logging.getLogger("mindscape.sync")

#: Markers further apart than this after alignment are treated as unmatched.
MATCH_TOLERANCE_SECONDS = 0.35


#: Typical hardware trigger jitter, seconds. Used to decide whether drift is
#: estimable at all — see `drift_is_identifiable`.
TRIGGER_JITTER_SECONDS = 0.005

#: Don't report a drift estimate whose standard error exceeds this. A number
#: dominated by its own uncertainty is worse than no number.
MAX_DRIFT_STANDARD_ERROR_PPM = 80.0


def drift_is_identifiable(marker_times: Sequence[float]) -> bool:
    """Whether the markers can support a drift estimate at all.

    Drift is a slope fitted across the marker span. Its standard error scales
    as jitter divided by the spread of the markers, so a handful of triggers
    over a short clip produce an estimate whose uncertainty is many times the
    drift being measured — for a 20 s clip with three markers that is roughly
    ±230 ppm against a true drift of tens of ppm.

    Reporting that number anyway would be worse than useless: it looks like a
    measurement. Better to fit offset only and say drift was not estimable.
    """
    if len(marker_times) < 3:
        return False

    times = np.asarray(marker_times, dtype=np.float64)
    spread = float(np.sqrt(np.sum((times - times.mean()) ** 2)))
    if spread < 1e-6:
        return False

    standard_error_ppm = (TRIGGER_JITTER_SECONDS / spread) * 1e6
    return standard_error_ppm <= MAX_DRIFT_STANDARD_ERROR_PPM


def fit_alignment(
    marker_samples: Sequence[int],
    marker_times: Sequence[float],
    sample_rate: int,
) -> SyncResult:
    """Fit offset, and drift when the markers can support it.

    Solves ``recorded = offset + stimulus × (1 + drift)`` by least squares.
    Drift is only fitted when :func:`drift_is_identifiable` says the marker
    geometry makes it measurable; otherwise offset alone is fitted and drift
    is reported as zero.

    The residual is always computed against *every* marker, including any
    rejected as outliers. Reporting residual over the retained subset would
    make a two-point fit look perfect by construction.
    """
    if len(marker_samples) != len(marker_times):
        raise ValueError("marker sample and time counts differ")

    expected = len(marker_times)

    if expected == 0:
        return SyncResult(
            offset_seconds=0.0,
            drift_ppm=0.0,
            residual_ms=float("inf"),
            markers_matched=0,
            markers_expected=0,
        )

    recorded = np.asarray(marker_samples, dtype=np.float64) / float(sample_rate)
    stimulus = np.asarray(marker_times, dtype=np.float64)

    if expected == 1:
        return SyncResult(
            offset_seconds=float(recorded[0] - stimulus[0]),
            drift_ppm=0.0,
            residual_ms=0.0,
            markers_matched=1,
            markers_expected=1,
        )

    fit_drift = drift_is_identifiable(stimulus)

    # Robust fit: least squares, then discard pairs whose residual is an
    # outlier and refit. One mistriggered marker would otherwise tilt the
    # whole alignment.
    keep = np.ones(expected, dtype=bool)
    minimum_kept = 2 if fit_drift else 1

    for _ in range(2):
        if keep.sum() < minimum_kept:
            break
        slope, intercept = _solve(stimulus[keep], recorded[keep], fit_drift)

        residuals = recorded - (intercept + slope * stimulus)
        kept_residuals = residuals[keep]
        scale = np.median(np.abs(kept_residuals - np.median(kept_residuals)))
        threshold = max(1.4826 * scale * 3.0, 0.02)

        updated = np.abs(residuals) <= threshold
        if updated.sum() < minimum_kept or np.array_equal(updated, keep):
            break
        keep = updated

    slope, intercept = _solve(stimulus[keep], recorded[keep], fit_drift)

    # Residual over every marker, not just the retained ones.
    residuals = recorded - (intercept + slope * stimulus)
    residual_ms = float(np.sqrt(np.mean(residuals**2)) * 1000.0)

    return SyncResult(
        offset_seconds=round(intercept, 6),
        drift_ppm=round((slope - 1.0) * 1e6, 3) if fit_drift else 0.0,
        residual_ms=round(residual_ms, 4),
        markers_matched=int(keep.sum()),
        markers_expected=expected,
    )


def _solve(
    stimulus: np.ndarray, recorded: np.ndarray, fit_drift: bool
) -> Tuple[float, float]:
    """Least-squares ``(slope, intercept)``, optionally holding slope at 1."""
    if not fit_drift or stimulus.size < 2:
        # Offset only: the best constant shift is the mean difference.
        return 1.0, float(np.mean(recorded - stimulus))

    design = np.column_stack([stimulus, np.ones(stimulus.size)])
    solution, *_ = np.linalg.lstsq(design, recorded, rcond=None)
    return float(solution[0]), float(solution[1])


class StimulusClock:
    """Converts between stimulus time and recording samples."""

    __slots__ = ("offset", "slope", "sample_rate")

    def __init__(self, sync: SyncResult, sample_rate: int) -> None:
        self.offset = sync.offset_seconds
        self.slope = 1.0 + sync.drift_ppm * 1e-6
        self.sample_rate = sample_rate

    def to_sample(self, stimulus_time: float) -> int:
        """Recording sample index for a moment in the stimulus."""
        return int(round((self.offset + stimulus_time * self.slope) * self.sample_rate))

    def to_stimulus(self, sample: int) -> float:
        """Stimulus time for a recording sample index."""
        return (sample / float(self.sample_rate) - self.offset) / self.slope


def detect_markers(
    data: np.ndarray,
    sample_rate: int,
    channels: List[int],
    threshold_sigma: float = 4.5,
    refractory_seconds: float = 0.6,
) -> List[int]:
    """Locate evoked-response onsets when no trigger channel was recorded.

    A fallback for recordings without hardware triggers: scene transitions
    produce large time-locked posterior deflections, so their onsets can be
    found in the signal itself. Less accurate than real triggers, which the
    residual error will reflect.
    """
    if data.ndim != 2 or not channels:
        return []

    valid = [c for c in channels if 0 <= c < data.shape[0]]
    if not valid:
        return []

    trace = data[valid].mean(axis=0)

    # Band-limit to the evoked-response range before thresholding; ongoing
    # alpha is larger than the transient and would dominate otherwise.
    from .filters import bandpass

    filtered = bandpass(trace[np.newaxis, :], sample_rate, 1.0, 12.0)[0]

    median = np.median(filtered)
    scale = max(float(1.4826 * np.median(np.abs(filtered - median))), 1e-9)
    z = np.abs(filtered - median) / scale

    refractory = int(refractory_seconds * sample_rate)
    onsets: List[int] = []
    last = -refractory

    for index in np.flatnonzero(z > threshold_sigma):
        if index - last < refractory:
            continue
        onsets.append(int(index))
        last = int(index)

    return onsets


def synchronise(
    marker_samples: Sequence[int],
    marker_times: Sequence[float],
    sample_rate: int,
    fallback_data: Optional[np.ndarray] = None,
    fallback_channels: Optional[List[int]] = None,
) -> Tuple[SyncResult, StimulusClock]:
    """Align the recording to the stimulus timeline.

    Uses hardware markers when present, falling back to evoked-response
    detection. Returns the fit and a clock for converting between timelines.
    """
    samples = list(marker_samples)

    if len(samples) < 2 and fallback_data is not None and fallback_channels:
        detected = detect_markers(fallback_data, sample_rate, fallback_channels)
        if len(detected) >= 2:
            log.info("No hardware triggers; recovered %d onsets from signal", len(detected))
            # Match detected onsets to expected stimulus events in order,
            # keeping only as many as there are known events.
            samples = detected[: len(marker_times)]

    usable = min(len(samples), len(marker_times))
    result = fit_alignment(samples[:usable], list(marker_times)[:usable], sample_rate)

    return result, StimulusClock(result, sample_rate)
