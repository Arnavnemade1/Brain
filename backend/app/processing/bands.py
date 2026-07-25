"""Canonical EEG frequency bands and band-derived indices.

Edges follow standard clinical convention and match ``BAND_RANGES`` in
``frontend/src/types/neural.ts``.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

BAND_RANGES: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

BAND_ORDER: List[str] = ["delta", "theta", "alpha", "beta", "gamma"]


def integrate_bands(
    freqs: np.ndarray, psd: np.ndarray, bands: Iterable[str] = BAND_ORDER
) -> Dict[str, float]:
    """Integrate a PSD over each band and return relative power (sums to 1).

    ``psd`` may be 1-D (single channel) or 2-D (channels × frequencies), in
    which case power is averaged across channels first.
    """
    if psd.ndim == 2:
        psd = psd.mean(axis=0)

    absolute: Dict[str, float] = {}
    for name in bands:
        low, high = BAND_RANGES[name]
        mask = (freqs >= low) & (freqs < high)
        if not mask.any():
            absolute[name] = 0.0
            continue
        absolute[name] = float(np.trapz(psd[mask], freqs[mask]))

    total = sum(absolute.values())
    if total <= 0:
        # Degenerate window (flat or fully rejected) — fall back to uniform.
        uniform = 1.0 / max(1, len(absolute))
        return {name: uniform for name in absolute}

    return {name: value / total for name, value in absolute.items()}


def alpha_theta_ratio(bands: Dict[str, float]) -> float:
    """Classic drowsiness / relaxation discriminator."""
    theta = max(bands.get("theta", 0.0), 1e-6)
    return float(bands.get("alpha", 0.0) / theta)


def beta_alpha_ratio(bands: Dict[str, float]) -> float:
    """Engagement proxy — rises with active, externally directed attention."""
    alpha = max(bands.get("alpha", 0.0), 1e-6)
    return float(bands.get("beta", 0.0) / alpha)


def theta_beta_ratio(bands: Dict[str, float]) -> float:
    """Cognitive-load proxy — elevated under sustained effortful processing."""
    beta = max(bands.get("beta", 0.0), 1e-6)
    return float(bands.get("theta", 0.0) / beta)


def engagement_index(bands: Dict[str, float]) -> float:
    """beta / (alpha + theta), normalised to 0–1.

    A standard workload index from the operator-state literature.
    """
    denom = max(bands.get("alpha", 0.0) + bands.get("theta", 0.0), 1e-6)
    raw = bands.get("beta", 0.0) / denom
    return float(np.clip(raw / (raw + 0.6), 0.0, 1.0))


def spectral_edge(freqs: np.ndarray, psd: np.ndarray, fraction: float = 0.95) -> float:
    """Frequency below which ``fraction`` of total power lies (SEF95)."""
    if psd.ndim == 2:
        psd = psd.mean(axis=0)
    total = float(np.sum(psd))
    if total <= 0:
        return 0.0
    cumulative = np.cumsum(psd) / total
    idx = int(np.searchsorted(cumulative, fraction))
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])
