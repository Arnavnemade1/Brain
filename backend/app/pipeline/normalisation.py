"""Feature normalisation.

Every extracted feature has its own natural scale — Hjorth complexity sits
around 2.0, connectivity around 0.2, spectral edge around 30 Hz. Scoring
against raw targets means each prior needs hand-tuned units, and a target
written from intuition rather than measurement can land entirely outside the
feature's real range, silently contributing nothing.

Instead, features are mapped to a robust z-score against their measured
distribution, and every prior downstream is stated in those units: ``-1.0``
means "a standard deviation below typical", regardless of the feature.

The reference statistics come from ``tools/calibrate.py --dump-features``,
sampled across all simulated profiles. They describe the *typical* range of
each measure and only need revisiting if the feature definitions change.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from ..processing.features import WindowFeatures

#: ``feature -> (median, robust scale)``. Scale is half the interquartile
#: range, which is resistant to the long tails several of these have.
FEATURE_STATS: Dict[str, Tuple[float, float]] = {
    "activity": (3.96, 3.85),
    "mobility": (0.327, 0.031),
    "complexity": (2.067, 0.178),
    "spectral_entropy": (0.658, 0.044),
    "connectivity": (0.222, 0.096),
    "frontal_asymmetry": (0.115, 0.235),
    "posterior_ratio": (0.696, 0.076),
    "zero_crossing": (0.180, 0.055),
    "spectral_edge": (31.0, 3.0),
}


#: ``band -> (mean relative power, standard deviation)`` as actually recovered
#: by the analysis chain, measured across every profile.
#:
#: Relative band power does *not* average 0.2 per band. Broadband background
#: fills wide bands more than narrow ones, and gamma carries far less power
#: than its share of the axis. Scoring against a nominal 1/5 therefore hands a
#: permanent bonus to any state weighting gamma negatively — which is exactly
#: what made one emotion absorb most of the distribution before this table
#: existed.
BAND_STATS: Dict[str, Tuple[float, float]] = {
    "delta": (0.235, 0.157),
    "theta": (0.277, 0.087),
    "alpha": (0.181, 0.072),
    "beta": (0.234, 0.101),
    "gamma": (0.074, 0.035),
}


def zband(band: str, value: float) -> float:
    """Band power as a z-score against its recovered distribution."""
    mean, scale = BAND_STATS.get(band, (0.2, 0.1))
    if scale <= 1e-9:
        return 0.0
    return float(np.clip((value - mean) / scale, -4.0, 4.0))


def z(feature: str, value: float) -> float:
    """Robust z-score of ``value`` for ``feature``, clamped to ±4."""
    median, scale = FEATURE_STATS.get(feature, (0.0, 1.0))
    if scale <= 1e-9:
        return 0.0
    return float(np.clip((value - median) / scale, -4.0, 4.0))


def normalise(features: WindowFeatures) -> Dict[str, float]:
    """Every feature as a robust z-score."""
    return {name: z(name, getattr(features, name)) for name in FEATURE_STATS}


def unit(feature: str, value: float) -> float:
    """Feature mapped to 0–1 across roughly its central range.

    Used where a bounded value is needed for display or for scene parameters,
    rather than the signed z-score the scorers use.
    """
    return float(np.clip(0.5 + z(feature, value) / 5.0, 0.0, 1.0))
