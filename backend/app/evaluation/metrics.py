"""Fidelity evaluation.

Scores a reconstruction against the reference it was decoded from. This is the
deliverable of the whole system — the reconstruction is only interesting to
the extent these numbers say it resembles what the subject actually saw.

Every metric is computed at the native reconstruction resolution against the
downsampled reference, so nothing is flattered by display upsampling.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..models.memory import (
    FidelityMetrics,
    FrameFidelity,
    SceneEstimate,
    VisualFeatures,
)
from ..stimulus.features import layout_vector, luminance_of

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Spatial metrics                                                             #
# --------------------------------------------------------------------------- #


def ssim(a: np.ndarray, b: np.ndarray, window: int = 7) -> float:
    """Structural similarity between two frames, on luminance.

    Local means, variances and covariance over a sliding window, combined with
    the standard stabilising constants. Computed on luminance rather than
    per-channel because the reconstruction's colour is a global tint — scoring
    each channel separately would count the same structural agreement three
    times.
    """
    x = a @ _LUMA if a.ndim == 3 else a.astype(np.float64)
    y = b @ _LUMA if b.ndim == 3 else b.astype(np.float64)

    if x.shape != y.shape or x.size == 0:
        return 0.0

    # Dynamic range is 0–1 for these frames.
    c1 = 0.01**2
    c2 = 0.03**2

    h, w = x.shape
    radius = max(1, min(window // 2, min(h, w) // 2))

    def local_mean(image: np.ndarray) -> np.ndarray:
        # Box filter via cumulative sums — separable and exact.
        padded = np.pad(image, radius, mode="reflect")
        cumulative = padded.cumsum(axis=0).cumsum(axis=1)
        cumulative = np.pad(cumulative, ((1, 0), (1, 0)), mode="constant")
        size = 2 * radius + 1
        total = (
            cumulative[size:, size:]
            - cumulative[:-size, size:]
            - cumulative[size:, :-size]
            + cumulative[:-size, :-size]
        )
        return total / (size * size)

    mu_x = local_mean(x)
    mu_y = local_mean(y)
    mu_x2 = mu_x**2
    mu_y2 = mu_y**2
    mu_xy = mu_x * mu_y

    sigma_x = local_mean(x * x) - mu_x2
    sigma_y = local_mean(y * y) - mu_y2
    sigma_xy = local_mean(x * y) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2)

    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.where(denominator > 1e-12, numerator / denominator, 0.0)

    return float(np.clip(np.mean(similarity), -1.0, 1.0))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB, for frames in 0–1."""
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse < 1e-12:
        return 60.0
    return float(10.0 * np.log10(1.0 / mse))


def layout_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation of coarse block-luminance layout between two frames."""
    va = layout_vector(a.astype(np.float32))
    vb = layout_vector(b.astype(np.float32))
    return _correlation(va, vb)


# --------------------------------------------------------------------------- #
# Temporal metrics                                                            #
# --------------------------------------------------------------------------- #


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, returning 0 for a degenerate input.

    A constant series has no correlation with anything. Returning 0 rather
    than NaN is deliberate: a reconstruction that outputs a fixed value should
    score as no better than chance, not as undefined.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    if a.size < 2 or b.size < 2 or a.size != b.size:
        return 0.0
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0

    return float(np.clip(np.corrcoef(a, b)[0, 1], -1.0, 1.0))


def scene_cut_f1_by_time(
    predicted_times: Sequence[float],
    truth_times: Sequence[float],
    tolerance: float = 1.25,
) -> float:
    """F1 of detected boundaries, matching predicted to true times greedily.

    The tolerance covers one analysis window: the evoked response marking a
    cut is spread over hundreds of milliseconds and windows are seconds long,
    so a boundary reported one window late is a hit.
    """
    if not truth_times:
        return 1.0 if not predicted_times else 0.0
    if not predicted_times:
        return 0.0

    unmatched = list(truth_times)
    true_positives = 0

    for candidate in predicted_times:
        best = None
        best_distance = tolerance
        for actual in unmatched:
            distance = abs(candidate - actual)
            if distance <= best_distance:
                best = actual
                best_distance = distance
        if best is not None:
            true_positives += 1
            unmatched.remove(best)

    precision = true_positives / len(predicted_times)
    recall = true_positives / len(truth_times)

    if precision + recall < 1e-9:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def scene_cut_f1(
    predicted: Sequence[bool],
    truth: Sequence[bool],
    times: Sequence[float],
    tolerance: float = 1.0,
) -> float:
    """F1 of detected scene boundaries, matched within a time tolerance.

    Exact index matching would be far too strict: the evoked response marking
    a cut is spread over a few hundred milliseconds and the analysis window is
    seconds long, so a boundary detected one window late is a hit, not a miss.
    """
    predicted_times = [times[i] for i, flag in enumerate(predicted) if flag]
    truth_times = [times[i] for i, flag in enumerate(truth) if flag]

    if not truth_times:
        return 1.0 if not predicted_times else 0.0
    if not predicted_times:
        return 0.0

    unmatched = list(truth_times)
    true_positives = 0

    for candidate in predicted_times:
        best = None
        best_distance = tolerance
        for actual in unmatched:
            distance = abs(candidate - actual)
            if distance <= best_distance:
                best = actual
                best_distance = distance
        if best is not None:
            true_positives += 1
            unmatched.remove(best)

    precision = true_positives / len(predicted_times)
    recall = true_positives / len(truth_times)

    if precision + recall < 1e-9:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


# --------------------------------------------------------------------------- #
# Aggregate                                                                   #
# --------------------------------------------------------------------------- #

#: Weights for the composite score. Luminance and scene boundaries dominate
#: because they are what the signal genuinely supports; colour is weighted
#: lightly because recovering it at all is close to a bonus.
_COMPOSITE_WEIGHTS = {
    "ssim": 0.24,
    "luminance": 0.24,
    "scene_cut": 0.18,
    "motion": 0.14,
    "layout": 0.12,
    "color": 0.08,
}


def evaluate(
    reconstructed: Sequence[np.ndarray],
    reference: Sequence[np.ndarray],
    estimates: Sequence[SceneEstimate],
    truth: Sequence[VisualFeatures],
    all_truth: Optional[Sequence[VisualFeatures]] = None,
) -> Tuple[FidelityMetrics, List[FrameFidelity]]:
    """Score a complete reconstruction against its reference.

    ``truth`` is the reference sampled at reconstruction times, used for the
    continuous metrics. ``all_truth`` is the *complete* per-frame reference and
    must be supplied for scene-boundary scoring: cuts are instantaneous events
    on a 10 fps timeline, while reconstructions land roughly one per second, so
    a sampled truth series drops most real cuts and scores detection against a
    ground truth that has itself lost the answer.
    """
    count = min(len(reconstructed), len(reference), len(estimates), len(truth))
    if count == 0:
        return (
            FidelityMetrics(
                ssim=0.0,
                psnr=0.0,
                luminance_correlation=0.0,
                color_correlation=0.0,
                motion_correlation=0.0,
                scene_cut_f1=0.0,
                layout_correlation=0.0,
                composite=0.0,
            ),
            [],
        )

    per_frame: List[FrameFidelity] = []
    ssim_values: List[float] = []
    psnr_values: List[float] = []
    layout_values: List[float] = []

    for i in range(count):
        frame_ssim = ssim(reconstructed[i], reference[i])
        frame_psnr = psnr(reconstructed[i], reference[i])
        frame_layout = layout_correlation(reconstructed[i], reference[i])

        ssim_values.append(frame_ssim)
        psnr_values.append(frame_psnr)
        layout_values.append(frame_layout)

        per_frame.append(
            FrameFidelity(
                index=i,
                t=round(estimates[i].t, 4),
                ssim=round(frame_ssim, 5),
                psnr=round(frame_psnr, 3),
                layout_correlation=round(frame_layout, 5),
            )
        )

    # Temporal series
    predicted_luminance = np.array([e.luminance for e in estimates[:count]])
    true_luminance = np.array([f.luminance for f in truth[:count]])

    predicted_motion = np.array([e.motion for e in estimates[:count]])
    true_motion = np.array([f.motion for f in truth[:count]])

    predicted_color = np.array([e.color for e in estimates[:count]])
    true_color = np.array([f.color for f in truth[:count]])

    color_correlation = float(
        np.mean(
            [_correlation(predicted_color[:, c], true_color[:, c]) for c in range(3)]
        )
    )

    # Score boundaries against every reference frame, matching on time.
    reference_truth = all_truth if all_truth is not None else truth
    cut_f1 = scene_cut_f1_by_time(
        [e.t for e in estimates[:count] if e.scene_cut],
        [f.t for f in reference_truth if f.scene_cut],
    )

    metrics = FidelityMetrics(
        ssim=round(float(np.mean(ssim_values)), 5),
        psnr=round(float(np.mean(psnr_values)), 3),
        luminance_correlation=round(
            _correlation(predicted_luminance, true_luminance), 5
        ),
        color_correlation=round(color_correlation, 5),
        motion_correlation=round(_correlation(predicted_motion, true_motion), 5),
        scene_cut_f1=round(cut_f1, 5),
        layout_correlation=round(float(np.mean(layout_values)), 5),
        composite=0.0,
    )

    metrics.composite = round(composite_score(metrics), 5)
    return metrics, per_frame


def composite_score(metrics: FidelityMetrics) -> float:
    """Weighted 0–1 summary of the individual metrics.

    Correlations are rectified to 0 before weighting: a negative correlation
    means the reconstruction moved the wrong way, which is a failure, not a
    negative contribution that could offset a success elsewhere.
    """
    rectify = lambda v: max(0.0, float(v))  # noqa: E731

    return float(
        np.clip(
            _COMPOSITE_WEIGHTS["ssim"] * rectify(metrics.ssim)
            + _COMPOSITE_WEIGHTS["luminance"] * rectify(metrics.luminance_correlation)
            + _COMPOSITE_WEIGHTS["scene_cut"] * rectify(metrics.scene_cut_f1)
            + _COMPOSITE_WEIGHTS["motion"] * rectify(metrics.motion_correlation)
            + _COMPOSITE_WEIGHTS["layout"] * rectify(metrics.layout_correlation)
            + _COMPOSITE_WEIGHTS["color"] * rectify(metrics.color_correlation),
            0.0,
            1.0,
        )
    )


def baseline_metrics(
    reference: Sequence[np.ndarray], truth: Sequence[VisualFeatures]
) -> FidelityMetrics:
    """Score a trivial baseline: the mean frame, held constant.

    Every reported result should be read against this. A reconstruction that
    cannot beat "output the average frame forever" has recovered nothing,
    however respectable its SSIM looks in isolation.
    """
    count = len(reference)
    if count == 0:
        return FidelityMetrics(
            ssim=0.0,
            psnr=0.0,
            luminance_correlation=0.0,
            color_correlation=0.0,
            motion_correlation=0.0,
            scene_cut_f1=0.0,
            layout_correlation=0.0,
            composite=0.0,
        )

    mean_frame = np.mean(np.stack([np.asarray(f) for f in reference]), axis=0)

    ssim_values = [ssim(mean_frame, reference[i]) for i in range(count)]
    psnr_values = [psnr(mean_frame, reference[i]) for i in range(count)]
    layout_values = [layout_correlation(mean_frame, reference[i]) for i in range(count)]

    metrics = FidelityMetrics(
        ssim=round(float(np.mean(ssim_values)), 5),
        psnr=round(float(np.mean(psnr_values)), 3),
        luminance_correlation=0.0,
        color_correlation=0.0,
        motion_correlation=0.0,
        scene_cut_f1=0.0,
        layout_correlation=round(float(np.mean(layout_values)), 5),
        composite=0.0,
    )
    metrics.composite = round(composite_score(metrics), 5)
    return metrics
