"""Sliding-window affective features.

Turns continuous EEG recorded while someone watched a clip into a feature
sequence at a few hertz — one vector per moment. Everything the per-frame
emotion decoder sees comes through here.

The measures are the ones with an established link to affect, chosen so that
a decoding result can be attributed to a named neural quantity rather than to
an opaque embedding:

* **Frontal alpha asymmetry** — the canonical valence correlate. Relatively
  greater left-frontal activity accompanies approach-oriented, positive
  affect; greater right-frontal activity accompanies withdrawal.
* **Beta/alpha ratio** — cortical activation, the standard arousal proxy.
* **Frontal-midline theta** — engagement and emotional salience.
* **Posterior alpha** — visual engagement, suppressed by absorbing content.
* **Gamma** — associated with emotional intensity.
* **Hemispheric differential entropy** — the feature family that performs
  best in the emotion-recognition literature (SEED, DEAP).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal

log = logging.getLogger("mindscape.emotion.features")

#: Analysis window and hop, in seconds. Four seconds is long enough for a
#: stable alpha estimate and short enough to follow an emotional turn; the
#: half-second hop sets the native rate of the decoded emotion track at 2 Hz.
WINDOW_SECONDS = 4.0
HOP_SECONDS = 0.5

BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

FEATURE_NAMES: List[str] = [
    "frontal_asymmetry",
    "frontal_asymmetry_beta",
    "beta_alpha_ratio",
    "frontal_theta",
    "posterior_alpha",
    "gamma_power",
    "frontal_de_left",
    "frontal_de_right",
    "posterior_de_left",
    "posterior_de_right",
    "temporal_asymmetry",
    "total_power",
    "alpha_suppression",
    "theta_beta_ratio",
]


@dataclass
class Regions:
    """Electrode groups resolved from geometry."""

    frontal_left: List[int]
    frontal_right: List[int]
    posterior_left: List[int]
    posterior_right: List[int]
    frontal_midline: List[int]
    posterior: List[int]
    temporal_left: List[int]
    temporal_right: List[int]
    all_eeg: List[int]


def resolve_regions(positions: np.ndarray, eeg_channels: List[int]) -> Regions:
    """Group electrodes by scalp location.

    Geometric rather than by label: this is a 128-channel EGI net whose names
    (E1..E128) carry no anatomical meaning, so thresholds on the coordinates
    are the only honest way to say where a sensor sits.
    """
    x = positions[:, 0]  # anterior +
    y = positions[:, 1]  # left +
    z = positions[:, 2]  # superior +

    def pick(mask) -> List[int]:
        return [i for i in eeg_channels if mask(x[i], y[i], z[i])]

    # Thresholds are fractions of the net radius (~10 units), chosen to give
    # well-populated, non-overlapping groups.
    return Regions(
        frontal_left=pick(lambda a, l, s: a > 3.5 and l > 1.5),
        frontal_right=pick(lambda a, l, s: a > 3.5 and l < -1.5),
        posterior_left=pick(lambda a, l, s: a < -3.0 and l > 1.5),
        posterior_right=pick(lambda a, l, s: a < -3.0 and l < -1.5),
        frontal_midline=pick(lambda a, l, s: a > 2.5 and abs(l) <= 2.5),
        posterior=pick(lambda a, l, s: a < -3.0),
        temporal_left=pick(lambda a, l, s: abs(a) <= 4.0 and l > 5.0),
        temporal_right=pick(lambda a, l, s: abs(a) <= 4.0 and l < -5.0),
        all_eeg=list(eeg_channels),
    )


def _band_power(
    freqs: np.ndarray, psd: np.ndarray, channels: List[int], band: Tuple[float, float]
) -> float:
    """Mean absolute power in a band across a group of channels."""
    valid = [c for c in channels if 0 <= c < psd.shape[0]]
    if not valid:
        return 0.0
    mask = (freqs >= band[0]) & (freqs < band[1])
    if not mask.any():
        return 0.0
    return float(np.mean(np.trapz(psd[valid][:, mask], freqs[mask], axis=-1)))


def _differential_entropy(power: float) -> float:
    """Differential entropy of a band-limited Gaussian.

    Reduces to a log of the power, which is the form that performs best in the
    emotion-recognition literature and is far better behaved than raw power:
    EEG power is heavy-tailed, and a log makes it roughly symmetric.
    """
    return float(np.log(max(power, 1e-12)))


def clean(window: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Band-limit and re-reference one window."""
    nyquist = sampling_rate / 2.0
    sos = sp_signal.butter(
        4, [1.0 / nyquist, min(45.0 / nyquist, 0.99)], btype="band", output="sos"
    )
    filtered = sp_signal.sosfiltfilt(sos, window, axis=-1).astype(np.float32)

    # Common average reference across scalp electrodes.
    filtered -= filtered.mean(axis=0, keepdims=True)

    # Soft-limit extreme excursions. Emotional clips provoke movement and
    # blinks; clipping at a robust threshold keeps those from dominating the
    # power estimate without excising the segment and breaking continuity.
    scale = np.median(np.abs(filtered - np.median(filtered))) * 1.4826
    if scale > 1e-9:
        limit = scale * 8.0
        filtered = np.tanh(filtered / limit) * limit

    return filtered


def extract_window(window: np.ndarray, sampling_rate: float, regions: Regions) -> np.ndarray:
    """One feature vector, in :data:`FEATURE_NAMES` order."""
    cleaned = clean(window, sampling_rate)

    nperseg = min(cleaned.shape[1], int(sampling_rate * 2))
    freqs, psd = sp_signal.welch(
        cleaned, fs=sampling_rate, nperseg=nperseg, axis=-1
    )

    fl, fr = regions.frontal_left, regions.frontal_right
    pl, pr = regions.posterior_left, regions.posterior_right

    alpha_fl = _band_power(freqs, psd, fl, BANDS["alpha"])
    alpha_fr = _band_power(freqs, psd, fr, BANDS["alpha"])
    beta_fl = _band_power(freqs, psd, fl, BANDS["beta"])
    beta_fr = _band_power(freqs, psd, fr, BANDS["beta"])

    # Frontal alpha asymmetry. Alpha is inversely related to cortical
    # activity, so ln(right) - ln(left) is *positive* for greater left
    # activation, which accompanies positive valence.
    asymmetry = _differential_entropy(alpha_fr) - _differential_entropy(alpha_fl)
    asymmetry_beta = _differential_entropy(beta_fr) - _differential_entropy(beta_fl)

    alpha_all = _band_power(freqs, psd, regions.all_eeg, BANDS["alpha"])
    beta_all = _band_power(freqs, psd, regions.all_eeg, BANDS["beta"])
    theta_all = _band_power(freqs, psd, regions.all_eeg, BANDS["theta"])
    gamma_all = _band_power(freqs, psd, regions.all_eeg, BANDS["gamma"])
    delta_all = _band_power(freqs, psd, regions.all_eeg, BANDS["delta"])

    theta_fm = _band_power(freqs, psd, regions.frontal_midline, BANDS["theta"])
    alpha_post = _band_power(freqs, psd, regions.posterior, BANDS["alpha"])

    temporal = _differential_entropy(
        _band_power(freqs, psd, regions.temporal_right, BANDS["gamma"])
    ) - _differential_entropy(
        _band_power(freqs, psd, regions.temporal_left, BANDS["gamma"])
    )

    total = delta_all + theta_all + alpha_all + beta_all + gamma_all

    return np.array(
        [
            asymmetry,
            asymmetry_beta,
            np.log(max(beta_all, 1e-12) / max(alpha_all, 1e-12)),
            _differential_entropy(theta_fm),
            _differential_entropy(alpha_post),
            _differential_entropy(gamma_all),
            _differential_entropy(_band_power(freqs, psd, fl, BANDS["alpha"])),
            _differential_entropy(_band_power(freqs, psd, fr, BANDS["alpha"])),
            _differential_entropy(_band_power(freqs, psd, pl, BANDS["alpha"])),
            _differential_entropy(_band_power(freqs, psd, pr, BANDS["alpha"])),
            temporal,
            _differential_entropy(total),
            -_differential_entropy(alpha_post),
            np.log(max(theta_all, 1e-12) / max(beta_all, 1e-12)),
        ],
        dtype=np.float64,
    )


def extract_sequence(
    recording,
    onset: float,
    duration: float,
    regions: Regions,
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float = HOP_SECONDS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Feature sequence across one trial.

    Returns ``(times, features)`` where ``times`` are seconds from clip onset
    at each window *centre* — the centre, not the leading edge, so a feature
    is attributed to the moment it actually summarises.
    """
    rate = recording.sampling_rate
    window_samples = int(round(window_seconds * rate))
    hop_samples = max(1, int(round(hop_seconds * rate)))

    start_sample = int(round(onset * rate))
    end_sample = int(round((onset + duration) * rate))

    times: List[float] = []
    vectors: List[np.ndarray] = []

    position = start_sample
    while position + window_samples <= end_sample:
        raw = recording.window(position, window_samples)
        vectors.append(extract_window(raw, rate, regions))
        centre = (position + window_samples / 2) / rate - onset
        times.append(centre)
        position += hop_samples

    if not vectors:
        return np.zeros(0), np.zeros((0, len(FEATURE_NAMES)))

    return np.array(times), np.stack(vectors)
