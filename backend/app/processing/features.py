"""Feature extraction — the numeric substrate every inference stage reasons over.

Hjorth descriptors, spectral entropy, connectivity and complexity measures are
computed once per window and passed forward, so no later stage has to touch
the raw signal again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from scipy import signal as sp_signal


@dataclass
class WindowFeatures:
    """Everything the inference stages need from one analysis window."""

    #: Signal variance — proportional to total power.
    activity: float
    #: Hjorth mobility: dominant frequency proxy.
    mobility: float
    #: Hjorth complexity: deviation from a pure sine.
    complexity: float
    #: Shannon entropy of the normalised PSD, 0–1. Low = rhythmic, high = broadband.
    spectral_entropy: float
    #: Mean pairwise correlation across channels, 0–1.
    connectivity: float
    #: Frontal alpha asymmetry (right − left), the valence driver.
    frontal_asymmetry: float
    #: Posterior/anterior alpha ratio — rises with internally directed attention.
    posterior_ratio: float
    #: Zero-crossing rate normalised by sample rate.
    zero_crossing: float
    #: 95% spectral edge frequency in Hz.
    spectral_edge: float
    #: Per-channel 0–1 activation for topography.
    activation: Dict[str, float] = field(default_factory=dict)

    def as_vector(self) -> List[float]:
        """Flat ordering used by the inference models."""
        return [
            self.activity,
            self.mobility,
            self.complexity,
            self.spectral_entropy,
            self.connectivity,
            self.frontal_asymmetry,
            self.posterior_ratio,
            self.zero_crossing,
            self.spectral_edge,
        ]


def hjorth(data: np.ndarray) -> tuple:
    """Hjorth activity, mobility and complexity, averaged across channels."""
    x = data if data.ndim == 2 else data[np.newaxis, :]
    if x.shape[-1] < 3:
        return 0.0, 0.0, 0.0

    d1 = np.diff(x, axis=-1)
    d2 = np.diff(d1, axis=-1)

    var0 = np.var(x, axis=-1)
    var1 = np.var(d1, axis=-1)
    var2 = np.var(d2, axis=-1)

    safe0 = np.maximum(var0, 1e-12)
    safe1 = np.maximum(var1, 1e-12)

    mobility = np.sqrt(var1 / safe0)
    complexity = np.sqrt(var2 / safe1) / np.maximum(mobility, 1e-12)

    return float(np.mean(var0)), float(np.mean(mobility)), float(np.mean(complexity))


def spectral_entropy(psd: np.ndarray) -> float:
    """Normalised Shannon entropy of the power spectrum, 0–1."""
    p = psd.mean(axis=0) if psd.ndim == 2 else psd
    total = float(np.sum(p))
    if total <= 0 or p.size < 2:
        return 0.0
    prob = p / total
    prob = prob[prob > 0]
    entropy = -float(np.sum(prob * np.log(prob)))
    return float(np.clip(entropy / np.log(p.size), 0.0, 1.0))


def connectivity(data: np.ndarray) -> float:
    """Mean absolute pairwise channel correlation, 0–1.

    A crude but stable coherence proxy: high values mean the montage is moving
    together, which the temporal stage reads as a well-formed global state.
    """
    if data.ndim != 2 or data.shape[0] < 2:
        return 0.0

    valid = np.std(data, axis=-1) > 1e-9
    if valid.sum() < 2:
        return 0.0

    corr = np.corrcoef(data[valid])
    upper = corr[np.triu_indices_from(corr, k=1)]
    upper = upper[np.isfinite(upper)]
    if upper.size == 0:
        return 0.0
    return float(np.clip(np.mean(np.abs(upper)), 0.0, 1.0))


def zero_crossing_rate(data: np.ndarray, sample_rate: int) -> float:
    """Mean per-channel zero crossings per second, normalised to 0–1.

    Averaged *per channel* rather than over the montage mean: independent
    channels crossing zero at different instants cancel in a mean trace, which
    collapses the measure toward zero and makes it useless as a feature.
    """
    x = data if data.ndim == 2 else data[np.newaxis, :]
    if x.shape[-1] < 2:
        return 0.0

    seconds = x.shape[-1] / float(sample_rate)
    crossings = np.sum(np.diff(np.signbit(x), axis=-1), axis=-1)
    rate = float(np.mean(crossings)) / seconds

    # A 45 Hz band limit permits at most 90 crossings per second.
    return float(np.clip(rate / 90.0, 0.0, 1.0))


def band_power_at(
    freqs: np.ndarray, psd: np.ndarray, channels: List[int], low: float, high: float
) -> float:
    """Absolute power in ``[low, high)`` across selected channels.

    Combined with a *median* rather than a mean: frontal electrodes carry
    blink artifacts, and one contaminated lead is enough to dominate a
    three-channel mean and destroy the asymmetry estimate that depends on it.
    """
    if psd.ndim == 1:
        psd = psd[np.newaxis, :]
    valid = [c for c in channels if 0 <= c < psd.shape[0]]
    if not valid:
        return 0.0
    mask = (freqs >= low) & (freqs < high)
    if not mask.any():
        return 0.0
    return float(np.median(np.trapz(psd[valid][:, mask], freqs[mask], axis=-1)))


def frontal_alpha_asymmetry(
    freqs: np.ndarray, psd: np.ndarray, left: List[int], right: List[int]
) -> float:
    """ln(right alpha) − ln(left alpha), squashed to −1..1.

    Relatively greater *left* frontal alpha (i.e. less left activation) is the
    established correlate of withdrawal/negative affect; the sign convention
    here makes positive values correspond to approach/positive valence.
    """
    left_power = max(band_power_at(freqs, psd, left, 8.0, 13.0), 1e-12)
    right_power = max(band_power_at(freqs, psd, right, 8.0, 13.0), 1e-12)
    raw = np.log(right_power) - np.log(left_power)
    return float(np.tanh(raw))


def posterior_alpha_ratio(
    freqs: np.ndarray, psd: np.ndarray, posterior: List[int], anterior: List[int]
) -> float:
    """Posterior vs anterior alpha, 0–1.

    Elevated posterior alpha accompanies internally directed attention — the
    state in which autobiographical recall is most often observed.
    """
    post = band_power_at(freqs, psd, posterior, 8.0, 13.0)
    ante = band_power_at(freqs, psd, anterior, 8.0, 13.0)
    total = post + ante
    if total <= 1e-12:
        return 0.5
    return float(np.clip(post / total, 0.0, 1.0))


def envelope(data: np.ndarray) -> np.ndarray:
    """Analytic-signal amplitude envelope, averaged across channels."""
    x = data.mean(axis=0) if data.ndim == 2 else data
    if x.size < 8:
        return np.abs(x)
    return np.abs(sp_signal.hilbert(x))
