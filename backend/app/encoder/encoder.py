"""Temporal Neural Encoder.

Reduces each analysis window to the small set of visual-system measures that
scalp EEG can actually carry about what someone was looking at. Everything
downstream reads only these — the reconstruction never touches raw signal.

Measures are expressed relative to a pre-stimulus baseline recorded before
playback began. Absolute EEG amplitude varies enormously between subjects,
sessions and electrode fits; without baselining, a decoder fitted on one
recording is meaningless on the next.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..models.neural import VisualResponse
from ..processing import montage
from ..processing.bands import BAND_RANGES
from ..processing.filters import bandpass
from ..processing.spectral import welch_psd

#: Ordered feature names the encoder emits. Decoders are fitted against this
#: exact ordering, so it must stay stable.
FEATURE_NAMES: List[str] = [
    "occipital_drive",
    "alpha_suppression",
    "motion_response",
    "transient",
    "detail_response",
    "lateral_balance",
    "vertical_balance",
]


@dataclass
class Baseline:
    """Pre-stimulus reference levels, per measure."""

    occipital_broadband: float
    posterior_alpha: float
    motion_band: float
    gamma: float
    left_power: float
    right_power: float
    dorsal_power: float
    ventral_power: float

    @classmethod
    def neutral(cls) -> "Baseline":
        return cls(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def _band_power(
    freqs: np.ndarray, psd: np.ndarray, channels: List[int], low: float, high: float
) -> float:
    """Mean absolute power in a band across selected channels."""
    if psd.ndim == 1:
        psd = psd[np.newaxis, :]
    valid = [c for c in channels if 0 <= c < psd.shape[0]]
    if not valid:
        return 0.0
    mask = (freqs >= low) & (freqs < high)
    if not mask.any():
        return 0.0
    return float(np.mean(np.trapz(psd[valid][:, mask], freqs[mask], axis=-1)))


class TemporalEncoder:
    """Encodes analysis windows into visual-response measures."""

    def __init__(self, sample_rate: int, channels: List[str]) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.baseline = Baseline.neutral()

        # Resolve montage groups against whatever channels this recording has,
        # so a partial cap still produces defined features.
        self.occipital = self._resolve(montage.OCCIPITAL)
        self.occipito_parietal = self._resolve(montage.OCCIPITO_PARIETAL)
        self.motion_area = self._resolve(montage.MOTION_SENSITIVE)
        self.posterior = self._resolve(montage.POSTERIOR)
        self.left = self._resolve(montage.POSTERIOR_LEFT)
        self.right = self._resolve(montage.POSTERIOR_RIGHT)
        self.dorsal = self._resolve(montage.POSTERIOR_DORSAL)
        self.ventral = self._resolve(montage.POSTERIOR_VENTRAL)

    def _resolve(self, canonical: List[int]) -> List[int]:
        names = {montage.CHANNELS[i] for i in canonical if i < len(montage.CHANNELS)}
        resolved = [i for i, name in enumerate(self.channels) if name in names]
        if resolved:
            return resolved
        # No canonical coverage — fall back to the last third of the montage,
        # which is posterior in every standard ordering.
        start = max(0, (len(self.channels) * 2) // 3)
        return list(range(start, len(self.channels))) or [0]

    # ----------------------------------------------------------------- fitting

    def fit_baseline(self, windows: List[np.ndarray]) -> Baseline:
        """Measure resting levels from pre-stimulus windows."""
        if not windows:
            return self.baseline

        accumulators: Dict[str, List[float]] = {
            key: [] for key in Baseline.__dataclass_fields__
        }

        for window in windows:
            freqs, psd = welch_psd(window, self.sample_rate)
            accumulators["occipital_broadband"].append(
                _band_power(freqs, psd, self.occipital, 4.0, 30.0)
            )
            accumulators["posterior_alpha"].append(
                _band_power(freqs, psd, self.posterior, *BAND_RANGES["alpha"])
            )
            accumulators["motion_band"].append(
                _band_power(freqs, psd, self.motion_area, BAND_RANGES["beta"][0], BAND_RANGES["gamma"][1])
            )
            accumulators["gamma"].append(
                _band_power(freqs, psd, self.occipito_parietal, *BAND_RANGES["gamma"])
            )
            accumulators["left_power"].append(
                _band_power(freqs, psd, self.left, 4.0, 30.0)
            )
            accumulators["right_power"].append(
                _band_power(freqs, psd, self.right, 4.0, 30.0)
            )
            accumulators["dorsal_power"].append(
                _band_power(freqs, psd, self.dorsal, 4.0, 30.0)
            )
            accumulators["ventral_power"].append(
                _band_power(freqs, psd, self.ventral, 4.0, 30.0)
            )

        # Median, not mean: a blink in the baseline period would otherwise
        # inflate the reference and suppress every later measurement.
        self.baseline = Baseline(
            **{
                key: max(float(np.median(values)), 1e-12)
                for key, values in accumulators.items()
            }
        )
        return self.baseline

    # ---------------------------------------------------------------- encoding

    def encode(self, window: np.ndarray) -> VisualResponse:
        """Reduce one cleaned window to visual-response measures."""
        freqs, psd = welch_psd(window, self.sample_rate)
        base = self.baseline

        # --- Luminance drive ---------------------------------------------
        # Occipital broadband power relative to rest. Log-compressed: neural
        # power responds roughly logarithmically to stimulus intensity, and a
        # linear ratio saturates on bright frames.
        broadband = _band_power(freqs, psd, self.occipital, 4.0, 30.0)
        drive = np.log1p(broadband / base.occipital_broadband) / np.log1p(6.0)

        # --- Alpha suppression -------------------------------------------
        # Posterior alpha falls below baseline with visual engagement, so the
        # measure is inverted: more suppression means more visual input.
        alpha = _band_power(freqs, psd, self.posterior, *BAND_RANGES["alpha"])
        suppression = 1.0 - (alpha / base.posterior_alpha)

        # --- Motion ---------------------------------------------------------
        motion_power = _band_power(
            freqs, psd, self.motion_area, BAND_RANGES["beta"][0], BAND_RANGES["gamma"][1]
        )
        motion = np.log1p(motion_power / base.motion_band) / np.log1p(6.0)

        # --- Spatial detail -------------------------------------------------
        gamma = _band_power(freqs, psd, self.occipito_parietal, *BAND_RANGES["gamma"])
        detail = np.log1p(gamma / base.gamma) / np.log1p(6.0)

        # --- Evoked transient -----------------------------------------------
        transient = self._transient_magnitude(window)

        # --- Field layout ----------------------------------------------------
        # Contralateral organisation: right-field brightness drives the left
        # hemisphere. Normalised difference, so overall amplitude cancels.
        left = _band_power(freqs, psd, self.left, 4.0, 30.0) / base.left_power
        right = _band_power(freqs, psd, self.right, 4.0, 30.0) / base.right_power
        lateral = (left - right) / max(left + right, 1e-9)

        dorsal = _band_power(freqs, psd, self.dorsal, 4.0, 30.0) / base.dorsal_power
        ventral = _band_power(freqs, psd, self.ventral, 4.0, 30.0) / base.ventral_power
        vertical = (ventral - dorsal) / max(ventral + dorsal, 1e-9)

        clip01 = lambda v: float(np.clip(v, 0.0, 1.0))  # noqa: E731
        clip11 = lambda v: float(np.clip(v, -1.0, 1.0))  # noqa: E731

        return VisualResponse(
            occipital_drive=round(clip01(drive), 5),
            alpha_suppression=round(clip01(suppression), 5),
            motion_response=round(clip01(motion), 5),
            transient=round(clip01(transient), 5),
            detail_response=round(clip01(detail), 5),
            lateral_balance=round(clip11(lateral * 2.2), 5),
            vertical_balance=round(clip11(vertical * 2.2), 5),
        )

    def _transient_magnitude(self, window: np.ndarray) -> float:
        """Peak time-locked deflection in the evoked-response band.

        Scene changes produce a large 1–12 Hz deflection over posterior sites.
        Measured as peak-to-peak amplitude relative to the window's own robust
        spread, so it responds to *transients* rather than to sustained level.
        """
        valid = [c for c in self.occipito_parietal if c < window.shape[0]]
        if not valid or window.shape[1] < 16:
            return 0.0

        trace = window[valid].mean(axis=0)
        filtered = bandpass(trace[np.newaxis, :], self.sample_rate, 1.0, 12.0)[0]

        spread = float(np.median(np.abs(filtered - np.median(filtered))) * 1.4826)
        if spread < 1e-9:
            return 0.0

        peak_to_peak = float(filtered.max() - filtered.min())
        return float(np.clip(peak_to_peak / (spread * 12.0), 0.0, 1.0))


def response_vector(response: VisualResponse) -> np.ndarray:
    """Encoder output as a flat array in :data:`FEATURE_NAMES` order."""
    return np.array(
        [getattr(response, name) for name in FEATURE_NAMES], dtype=np.float64
    )
