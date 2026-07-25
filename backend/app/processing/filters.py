"""Signal cleaning: referencing, detrending, band-limiting and notch filtering.

These are the operations behind the ``cleaning`` and ``denoise`` pipeline
stages. Everything works on ``(channels, samples)`` float arrays.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy import signal


def common_average_reference(data: np.ndarray, exclude: List[int] = None) -> np.ndarray:
    """Re-reference every channel to the mean across channels.

    Removes reference-electrode bias and much of the shared environmental
    pickup. Channels in ``exclude`` are left out of the average (but still
    referenced) so a dead channel cannot poison the montage.
    """
    if data.ndim != 2 or data.shape[0] < 2:
        return data

    keep = np.ones(data.shape[0], dtype=bool)
    if exclude:
        keep[list(exclude)] = False
    if not keep.any():
        return data

    reference = data[keep].mean(axis=0, keepdims=True)
    return data - reference


def detrend(data: np.ndarray) -> np.ndarray:
    """Remove the linear trend caused by electrode polarisation drift.

    Implemented with closed-form least squares rather than
    ``scipy.signal.detrend``: the SciPy version routes through BLAS ``matmul``,
    which on Accelerate-backed builds raises spurious floating-point warnings
    on every call. The reduction below stays in NumPy's own loops, and the
    closed form is faster than a general lstsq for the single-segment case.
    """
    x = np.asarray(data, dtype=np.float64)
    n = x.shape[-1]
    if n < 2:
        return x

    t = np.arange(n, dtype=np.float64)
    t_mean = t.mean()
    t_centered = t - t_mean
    denom = float(np.dot(t_centered, t_centered))
    if denom <= 0.0:
        return x - x.mean(axis=-1, keepdims=True)

    x_mean = x.mean(axis=-1, keepdims=True)
    centered = x - x_mean
    slope = np.sum(centered * t_centered, axis=-1, keepdims=True) / denom
    return centered - slope * t_centered


def bandpass(
    data: np.ndarray,
    sample_rate: int,
    low: float = 0.5,
    high: float = 45.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass.

    ``filtfilt`` is used so the reconstruction timeline is not shifted relative
    to the source recording — phase distortion would smear episode boundaries.
    """
    nyquist = sample_rate / 2.0
    high = min(high, nyquist * 0.98)
    if low >= high:
        return data

    sos = signal.butter(order, [low / nyquist, high / nyquist], btype="band", output="sos")
    padlen = min(3 * order * 2, max(0, data.shape[-1] - 1))
    return signal.sosfiltfilt(sos, data, axis=-1, padlen=padlen)


def notch(
    data: np.ndarray,
    sample_rate: int,
    frequency: float = 60.0,
    quality: float = 30.0,
    harmonics: int = 2,
) -> np.ndarray:
    """Remove mains hum at ``frequency`` and its harmonics."""
    nyquist = sample_rate / 2.0
    out = data
    for k in range(1, harmonics + 1):
        f = frequency * k
        if f >= nyquist * 0.95:
            break
        b, a = signal.iirnotch(f / nyquist, quality)
        padlen = min(3 * max(len(a), len(b)), max(0, out.shape[-1] - 1))
        out = signal.filtfilt(b, a, out, axis=-1, padlen=padlen)
    return out


def robust_sigma(data: np.ndarray) -> np.ndarray:
    """Median-absolute-deviation scale estimate, per channel.

    The standard deviation of a window containing a blink is inflated *by* that
    blink, which raises the rejection threshold above the very artifact it is
    meant to catch. MAD is unaffected by a small proportion of extreme samples,
    so the threshold stays anchored to the underlying EEG.
    """
    median = np.median(data, axis=-1, keepdims=True)
    mad = np.median(np.abs(data - median), axis=-1, keepdims=True)
    return np.maximum(1.4826 * mad, 1e-9)


def attenuate_artifacts(
    data: np.ndarray, threshold_sigma: float = 3.5
) -> Tuple[np.ndarray, float]:
    """Soft-limit high-amplitude transients (blinks, jaw clench, cable sway).

    Deleting artifact segments would break temporal continuity, which the
    reconstruction engine depends on. Instead samples beyond
    ``threshold_sigma`` robust deviations are compressed toward the threshold
    with a tanh knee, preserving timing while removing the excursion.

    Returns the cleaned data and the proportion of samples that were affected.
    """
    if data.size == 0:
        return data, 0.0

    limit = threshold_sigma * robust_sigma(data)

    excess = np.abs(data) > limit
    ratio = float(excess.mean())
    if ratio == 0.0:
        return data, 0.0

    magnitude = np.abs(data)
    compressed = np.sign(data) * limit * (1.0 + np.tanh((magnitude - limit) / limit))
    return np.where(excess, compressed, data), ratio


def flat_channels(data: np.ndarray, relative_floor: float = 0.08) -> List[int]:
    """Indices of channels carrying effectively no signal.

    Tested *relative to the montage*, not against an absolute microvolt
    threshold: recordings arrive in wildly different units and amplifier
    gains, so a fixed epsilon either catches everything or nothing. A lead
    sitting below ``relative_floor`` of the median channel amplitude is
    disconnected, saturated to rail, or otherwise unusable.
    """
    if data.ndim != 2 or data.shape[0] == 0:
        return []

    sigmas = np.std(data, axis=-1)
    reference = float(np.median(sigmas))
    if reference <= 1e-12:
        # Whole montage is silent — report every channel rather than none.
        return list(range(data.shape[0]))

    return [i for i, sigma in enumerate(sigmas) if float(sigma) < reference * relative_floor]


def clean_window(
    data: np.ndarray,
    sample_rate: int,
    line_frequency: float = 60.0,
) -> Tuple[np.ndarray, List[int], float]:
    """Full cleaning chain for one analysis window.

    Order matters. Artifacts are attenuated *before* the band-pass, while
    blinks and muscle bursts still carry their full wideband amplitude — after
    filtering they are already partly suppressed and would be under-reported,
    leaving the quality score blind to a genuinely noisy recording.

    Returns ``(cleaned, dropped_channel_indices, artifact_ratio)``.
    """
    dropped = flat_channels(data)
    x = detrend(np.asarray(data, dtype=np.float64))
    x = common_average_reference(x, exclude=dropped)
    x = notch(x, sample_rate, line_frequency)
    x, artifact_ratio = attenuate_artifacts(x)
    x = bandpass(x, sample_rate)
    return x, dropped, artifact_ratio
