"""Signal quality assessment.

Quality gates the whole product: a low-confidence recording must produce a
visibly fragmented reconstruction rather than a confident-looking wrong one.
Each penalty below corresponds to a distinct, independently observable defect.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..models.neural import SignalQuality

#: Band used to estimate the electrode noise floor. Genuine cortical signal
#: falls off steeply above ~35 Hz, so residual power there is dominated by
#: amplifier and impedance noise rather than brain activity.
NOISE_FLOOR_BAND = (35.0, 45.0)


#: Grade boundaries, calibrated against the composite score's actual
#: distribution rather than an even split of the 0–1 range. Because every
#: penalty term is small for a plausible recording, real scores occupy roughly
#: 0.75–0.99; boundaries spread evenly over 0–1 would grade every session
#: "excellent" and make the readout meaningless.
GRADE_THRESHOLDS = (
    ("excellent", 0.95),
    ("good", 0.90),
    ("fair", 0.84),
)


def grade_for(score: float) -> str:
    for grade, floor in GRADE_THRESHOLDS:
        if score >= floor:
            return grade
    return "poor"


def noisy_lead_ratio(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Severity-weighted proportion of channels with a raised noise floor, 0–1.

    High-impedance electrodes raise the broadband floor on their own channel
    while their neighbours stay clean, so the signature is a *relative*
    outlier rather than a raised montage-wide level.

    The result ramps smoothly from z=2 to z=4 rather than thresholding at a
    single z. With only ~19 channels a hard cut is both statistically shaky
    and visibly unstable — a lead hovering at the boundary would make the
    quality readout flicker between grades on consecutive windows.
    """
    if psd.ndim != 2 or psd.shape[0] < 3:
        return 0.0

    low, high = NOISE_FLOOR_BAND
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        return 0.0

    floor = np.trapz(psd[:, mask], freqs[mask], axis=-1)
    floor = np.log10(np.maximum(floor, 1e-12))

    median = float(np.median(floor))
    deviation = np.median(np.abs(floor - median))
    scale = max(float(1.4826 * deviation), 1e-6)

    z = (floor - median) / scale
    severity = np.clip((z - 2.0) / 2.0, 0.0, 1.0)
    return float(np.mean(severity))


def deviant_channel_ratio(cleaned: np.ndarray) -> float:
    """Severity-weighted proportion of channels with outlying amplitude, 0–1.

    Amplitude thresholding (see ``filters.attenuate_artifacts``) only catches
    *transients*: when a lead loses contact for seconds at a time, the drift
    becomes that window's own statistics and the robust threshold rises with
    it, so the artifact hides itself. The deviation criterion catches exactly
    that case by comparing each channel's amplitude against the montage,
    which a single misbehaving lead cannot shift.
    """
    if cleaned.ndim != 2 or cleaned.shape[0] < 3:
        return 0.0

    sigmas = np.std(cleaned, axis=-1)
    median = float(np.median(sigmas))
    if median <= 1e-12:
        return 1.0

    deviation = np.median(np.abs(sigmas - median))
    scale = max(float(1.4826 * deviation), median * 0.05)

    z = np.abs(sigmas - median) / scale
    return float(np.mean(np.clip((z - 2.5) / 2.5, 0.0, 1.0)))


def assess(
    cleaned: np.ndarray,
    channels: List[str],
    dropped_indices: List[int],
    artifact_ratio: float,
    line_noise: float,
    connectivity: float,
    freqs: np.ndarray = None,
    psd: np.ndarray = None,
) -> SignalQuality:
    """Composite quality score.

    Penalties, in descending weight:

    * channel loss — dead leads reduce spatial coverage outright
    * channel deviation — leads with outlying amplitude (contact instability)
    * artifact load — how much of the window needed amplitude compression
    * noisy leads — channels sitting well above the montage noise floor
    * line noise — residual mains contamination surviving the notch
    * channel agreement — near-zero *or* near-perfect correlation both signal
      a fault (uncorrelated noise, or a bridged/shorted montage)
    * saturation — samples clustered at the rail have lost dynamic range
    """
    total_channels = max(1, len(channels))
    channel_penalty = len(dropped_indices) / total_channels
    deviation_penalty = deviant_channel_ratio(cleaned)
    # Even a badly contaminated window only pushes a few percent of samples
    # past 3.5 robust deviations, so the scale is anchored at 6% = unusable
    # rather than the 100% a naive normalisation would imply.
    artifact_penalty = float(np.clip(artifact_ratio / 0.06, 0.0, 1.0))
    noise_penalty = float(np.clip(line_noise, 0.0, 1.0))

    lead_penalty = 0.0
    if freqs is not None and psd is not None:
        lead_penalty = noisy_lead_ratio(freqs, psd)

    if connectivity < 0.08:
        agreement_penalty = 0.6
    elif connectivity > 0.97:
        agreement_penalty = 0.5
    else:
        agreement_penalty = 0.0

    saturation_penalty = 0.0
    if cleaned.size:
        peak = float(np.max(np.abs(cleaned)))
        if peak > 1e-9:
            at_rail = float(np.mean(np.abs(cleaned) > peak * 0.985))
            saturation_penalty = float(np.clip(at_rail * 12.0, 0.0, 0.7))

    score = 1.0 - (
        0.26 * channel_penalty
        + 0.24 * deviation_penalty
        + 0.20 * artifact_penalty
        + 0.14 * lead_penalty
        + 0.08 * noise_penalty
        + 0.06 * agreement_penalty
        + 0.04 * saturation_penalty
    )
    score = float(np.clip(score, 0.0, 1.0))

    return SignalQuality(
        score=round(score, 4),
        grade=grade_for(score),
        artifact_ratio=round(float(np.clip(artifact_ratio, 0.0, 1.0)), 4),
        dropped_channels=[channels[i] for i in dropped_indices if i < len(channels)],
        line_noise=round(noise_penalty, 4),
    )
