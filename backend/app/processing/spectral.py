"""Spectral estimation — Welch PSD and the display spectrum sent to the client."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy import signal

from ..models.neural import Spectrum
from .bands import BAND_RANGES

#: Frequency axis the client renders. Fixed so the spectrum view never has to
#: re-scale between frames.
DISPLAY_MIN_HZ = 0.5
DISPLAY_MAX_HZ = 45.0
DISPLAY_BINS = 96


def welch_psd(
    data: np.ndarray, sample_rate: int, segment_seconds: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density.

    Returns ``(freqs, psd)`` where ``psd`` keeps the channel axis if the input
    had one, so per-channel topography can be derived from the same estimate.
    """
    nperseg = int(min(data.shape[-1], max(64, sample_rate * segment_seconds)))
    noverlap = nperseg // 2
    freqs, psd = signal.welch(
        data,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        window="hann",
        detrend=False,
        axis=-1,
    )
    return freqs, psd


def to_display_spectrum(freqs: np.ndarray, psd: np.ndarray) -> Spectrum:
    """Resample a PSD onto the fixed display axis, in dB."""
    if psd.ndim == 2:
        psd = psd.mean(axis=0)

    axis = np.linspace(DISPLAY_MIN_HZ, DISPLAY_MAX_HZ, DISPLAY_BINS)
    in_range = (freqs >= DISPLAY_MIN_HZ) & (freqs <= DISPLAY_MAX_HZ)
    if not in_range.any():
        flat = [-90.0] * DISPLAY_BINS
        return Spectrum(frequencies=axis.tolist(), magnitudes=flat, peak_frequency=0.0)

    resampled = np.interp(axis, freqs[in_range], psd[in_range])
    magnitudes = 10.0 * np.log10(np.maximum(resampled, 1e-12))

    return Spectrum(
        frequencies=[round(float(f), 3) for f in axis],
        magnitudes=[round(float(m), 3) for m in magnitudes],
        peak_frequency=round(dominant_frequency(axis, resampled), 3),
    )


def dominant_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Frequency of the strongest *oscillatory* peak, in Hz.

    EEG spectra follow a 1/f aperiodic slope, so the raw PSD maximum is always
    the lowest frequency bin regardless of what rhythm actually dominates —
    reporting it as the "peak frequency" is meaningless. The slope is removed
    first by whitening (multiplying by frequency), which is what leaves the
    genuine rhythmic peak — an alpha peak near 10 Hz, say — as the maximum.
    """
    if psd.ndim == 2:
        psd = psd.mean(axis=0)
    if psd.size == 0 or freqs.size == 0:
        return 0.0

    whitened = psd * freqs
    return float(freqs[int(np.argmax(whitened))])


def channel_activation(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """Per-channel 0–1 activation for the scalp topography map.

    Uses total in-band power per channel, normalised across the montage so the
    heatmap shows spatial *contrast* rather than absolute microvolts.
    """
    if psd.ndim == 1:
        psd = psd[np.newaxis, :]

    mask = (freqs >= BAND_RANGES["delta"][0]) & (freqs <= BAND_RANGES["gamma"][1])
    if not mask.any():
        return np.zeros(psd.shape[0])

    power = np.trapz(psd[:, mask], freqs[mask], axis=-1)
    power = np.log10(np.maximum(power, 1e-12))

    lo, hi = float(power.min()), float(power.max())
    if hi - lo < 1e-9:
        return np.full(psd.shape[0], 0.5)
    return (power - lo) / (hi - lo)


def decimate_waveform(data: np.ndarray, target: int = 128) -> List[float]:
    """Decimate a window to a fixed-length trace for the live waveform view.

    The mean across channels is used — the display is a montage overview, not a
    per-channel readout, and the client draws it as a single continuous line.
    """
    trace = data.mean(axis=0) if data.ndim == 2 else data
    if trace.size == 0:
        return [0.0] * target

    if trace.size > target:
        # Bin-average rather than subsample so transients survive as amplitude
        # rather than aliasing into the display.
        edges = np.linspace(0, trace.size, target + 1).astype(int)
        trace = np.array([trace[a:b].mean() if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])])
    elif trace.size < target:
        trace = np.interp(np.linspace(0, trace.size - 1, target), np.arange(trace.size), trace)

    peak = float(np.max(np.abs(trace)))
    if peak > 1e-9:
        trace = trace / peak
    return [round(float(v), 4) for v in trace]


def line_noise_level(freqs: np.ndarray, psd: np.ndarray, line_frequency: float) -> float:
    """0–1 estimate of residual mains contamination after notching."""
    if psd.ndim == 2:
        psd = psd.mean(axis=0)

    near = np.abs(freqs - line_frequency) < 1.5
    if not near.any():
        return 0.0

    baseline = np.median(psd) if psd.size else 1e-12
    peak = float(np.max(psd[near]))
    ratio = peak / max(float(baseline), 1e-12)
    return float(np.clip(np.log10(max(ratio, 1.0)) / 2.0, 0.0, 1.0))
