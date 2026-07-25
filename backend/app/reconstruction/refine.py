"""Video Refinement.

Frames decoded independently from noisy windows flicker badly: consecutive
estimates of the same static scene can differ more than two genuinely
different scenes do. Refinement enforces temporal continuity *within* a scene
while deliberately preserving discontinuity *at* boundaries — smoothing
straight through a cut would erase the one thing the system recovers most
reliably.

Nothing here invents spatial detail. Refinement only redistributes information
already present across neighbouring frames in time.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..models.memory import SceneEstimate


class TemporalRefiner:
    """Streaming temporal smoother with scene-boundary awareness."""

    def __init__(self, strength: float = 0.62) -> None:
        #: 0 = no smoothing, →1 = heavy. Applied as an exponential blend.
        self.strength = float(np.clip(strength, 0.0, 0.95))
        self._previous: np.ndarray = None  # type: ignore[assignment]
        self._scene_index: int = -1

    def reset(self) -> None:
        self._previous = None  # type: ignore[assignment]
        self._scene_index = -1

    def refine(self, frame: np.ndarray, scene: SceneEstimate) -> np.ndarray:
        """Blend a frame with recent history, resetting at scene boundaries."""
        boundary = scene.scene_cut or scene.scene_index != self._scene_index
        self._scene_index = scene.scene_index

        if boundary or self._previous is None:
            # Start the new scene clean; carrying the previous scene across a
            # cut would smear two unrelated images together.
            self._previous = frame.copy()
            return frame

        # Weight smoothing by confidence: a low-confidence frame should lean
        # harder on history, a high-confidence one should assert itself.
        alpha = self.strength * (1.0 - 0.5 * scene.confidence)
        blended = self._previous * alpha + frame * (1.0 - alpha)

        self._previous = blended
        return np.clip(blended, 0.0, 1.0)


def upsample(frame: np.ndarray, factor: int = 8) -> np.ndarray:
    """Enlarge a frame for display without inventing detail.

    Bilinear, and applied only for presentation. Fidelity is always measured
    at the native reconstruction resolution — scoring an upsampled frame would
    reward the interpolator rather than the decoder.
    """
    if factor <= 1:
        return frame

    h, w = frame.shape[:2]
    target_h, target_w = h * factor, w * factor

    ys = np.linspace(0, h - 1, target_h)
    xs = np.linspace(0, w - 1, target_w)

    y0 = np.floor(ys).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    wy = (ys - y0)[:, None, None]

    x0 = np.floor(xs).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    wx = (xs - x0)[None, :, None]

    top = frame[y0][:, x0] * (1 - wx) + frame[y0][:, x1] * wx
    bottom = frame[y1][:, x0] * (1 - wx) + frame[y1][:, x1] * wx
    return top * (1 - wy) + bottom * wy


def smooth_scene_estimates(
    estimates: Sequence[SceneEstimate], window: int = 3
) -> List[SceneEstimate]:
    """Median-filter continuous scene quantities within each scene.

    A median rather than a mean: single-window outliers from a blink or a
    muscle burst should be rejected outright, not averaged in.
    """
    if len(estimates) < 3:
        return list(estimates)

    half = max(1, window // 2)
    out: List[SceneEstimate] = []

    for i, estimate in enumerate(estimates):
        # Only pool neighbours from the same scene.
        neighbours = [
            other
            for j, other in enumerate(estimates)
            if abs(j - i) <= half and other.scene_index == estimate.scene_index
        ]
        if len(neighbours) < 2:
            out.append(estimate)
            continue

        out.append(
            estimate.model_copy(
                update={
                    "luminance": round(
                        float(np.median([n.luminance for n in neighbours])), 5
                    ),
                    "contrast": round(
                        float(np.median([n.contrast for n in neighbours])), 5
                    ),
                    "motion": round(float(np.median([n.motion for n in neighbours])), 5),
                }
            )
        )

    return out
