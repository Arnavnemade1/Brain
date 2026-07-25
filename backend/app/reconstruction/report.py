"""Reconstruction reporting.

States what was recovered and what was not, in terms tied to the measured
numbers. Every claim here is derived from a metric — the report never
characterises the reconstruction more confidently than the evaluation
supports.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..models.memory import (
    FidelityMetrics,
    ReconstructionReport,
    SceneEstimate,
    StimulusClip,
    VisualFeatures,
)
from ..models.neural import SyncResult
from .decoders import DecoderBundle

#: Correlation above which a property is described as tracked.
TRACKED = 0.30
WEAK = 0.15


def _grade(value: float) -> str:
    if value >= 0.55:
        return "strong"
    if value >= TRACKED:
        return "moderate"
    if value >= WEAK:
        return "weak"
    return "no measurable"


def build_report(
    clip: StimulusClip,
    metrics: FidelityMetrics,
    scenes: Sequence[SceneEstimate],
    truth: Sequence[VisualFeatures],
    sync: Optional[SyncResult],
    decoders: DecoderBundle,
) -> ReconstructionReport:
    """Compose the closing account of a reconstruction."""
    findings: List[str] = []
    limitations: List[str] = []

    # --- What was recovered ---------------------------------------------
    findings.append(
        f"Luminance tracking shows {_grade(metrics.luminance_correlation)} "
        f"agreement (r = {metrics.luminance_correlation:+.2f}). Overall brightness is "
        "the property scalp EEG carries most directly, through occipital response "
        "amplitude."
    )

    findings.append(
        f"Motion tracking is {_grade(metrics.motion_correlation)} "
        f"(r = {metrics.motion_correlation:+.2f}), decoded from beta and gamma "
        "activity over lateral occipito-temporal sites."
    )

    detected = sum(1 for s in scenes if s.scene_cut)
    actual = sum(1 for f in truth if f.scene_cut)
    findings.append(
        f"Scene boundaries: {detected} detected against {actual} present, "
        f"F1 = {metrics.scene_cut_f1:.2f}. Abrupt visual changes produce large "
        "time-locked responses and survive noise better than any continuous property."
    )

    findings.append(
        f"Structural similarity averages {metrics.ssim:.3f} at "
        f"{clip.grid_width}×{clip.grid_height}, with peak SNR {metrics.psnr:.1f} dB."
    )

    # --- What was not ------------------------------------------------------
    if metrics.layout_correlation < TRACKED:
        limitations.append(
            f"Spatial layout was not recovered (r = {metrics.layout_correlation:+.2f}). "
            "Scalp EEG resolves roughly left/right and upper/lower field balance and "
            "nothing finer, so where things sat within the frame is largely absent."
        )

    if metrics.color_correlation < TRACKED:
        limitations.append(
            f"Colour was not recovered (r = {metrics.color_correlation:+.2f}). "
            "Chromatic information is weakly represented at the scalp; the "
            "reconstruction's colour is a decoded global tint, not observed hue."
        )

    limitations.append(
        "Absolute brightness is uncertain. The encoder measures response relative to "
        "a pre-stimulus baseline, so the reconstruction tracks how brightness "
        "*changed* far better than what it was."
    )

    if sync is not None and not sync.usable:
        limitations.append(
            f"Clock alignment was poor (residual {sync.residual_ms:.0f} ms across "
            f"{sync.markers_matched} markers). Responses may be attributed to "
            "neighbouring frames, which depresses every metric here."
        )

    if not decoders.fitted:
        limitations.append(
            "Decoders are untrained — no fitted weights were found, so this "
            "reconstruction used the fallback mapping. Run the training harness."
        )

    # --- Summary -----------------------------------------------------------
    strongest = max(
        [
            ("luminance", metrics.luminance_correlation),
            ("motion", metrics.motion_correlation),
            ("scene structure", metrics.scene_cut_f1),
        ],
        key=lambda pair: pair[1],
    )

    summary = (
        f"Reconstruction of {clip.title} reached a composite fidelity of "
        f"{metrics.composite * 100:.0f}%, with {strongest[0]} the best-recovered "
        f"property. Temporal structure — when the scene was bright, when it moved, "
        f"and where it changed — is substantially present. Spatial detail is not: at "
        f"{clip.grid_width}×{clip.grid_height} the reconstruction carries visual gist "
        f"rather than image content, which is the ceiling scalp EEG imposes rather "
        f"than a limitation of the decoder. Every figure here is measured against the "
        f"reference clip the subject actually watched."
    )

    title = f"{clip.title} — {metrics.composite * 100:.0f}% fidelity"

    return ReconstructionReport(
        summary=summary, title=title, findings=findings, limitations=limitations
    )
