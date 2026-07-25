"""Visual feature extraction from reference frames.

This is the ground truth of the whole system. The simulator turns these
features into EEG; the evaluator scores the reconstruction against them. Both
read the same function, so there is no possibility of the two disagreeing
about what the reference actually contained.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..models.memory import VisualFeatures

#: Perceptual luminance weights (Rec. 709).
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

#: Frame-to-frame luminance change above which a cut is declared. Set from the
#: separation between within-scene variation and between-scene jumps in the
#: clip library; see `tools/calibrate.py`.
SCENE_CUT_THRESHOLD = 0.13


def luminance_of(frames: np.ndarray) -> np.ndarray:
    """Per-pixel luminance, shape ``(n, h, w)``."""
    return frames @ _LUMA


def extract(
    frames: np.ndarray, fps: float, scene_indices: Optional[List[int]] = None
) -> List[VisualFeatures]:
    """Per-frame ground-truth features for a rendered clip."""
    luma = luminance_of(frames)
    n = frames.shape[0]

    mean_luminance = luma.reshape(n, -1).mean(axis=1)
    contrast = luma.reshape(n, -1).std(axis=1)
    mean_color = frames.reshape(n, -1, 3).mean(axis=1)

    # Spatial detail: mean absolute gradient magnitude.
    gy = np.abs(np.diff(luma, axis=1)).mean(axis=(1, 2))
    gx = np.abs(np.diff(luma, axis=2)).mean(axis=(1, 2))
    edge_density = np.clip((gx + gy) * 6.0, 0.0, 1.0)

    # Motion: mean absolute temporal difference, plus a coarse direction from
    # the horizontal/vertical components of that difference.
    motion = np.zeros(n, dtype=np.float32)
    direction: List[Optional[float]] = [None] * n

    if n > 1:
        delta = np.abs(np.diff(luma, axis=0))
        motion[1:] = np.clip(delta.mean(axis=(1, 2)) * 9.0, 0.0, 1.0)
        motion[0] = motion[1]

        for i in range(1, n):
            difference = luma[i] - luma[i - 1]
            if np.abs(difference).mean() < 0.004:
                continue
            # Weight each pixel's displacement contribution by where the
            # change happened, giving a dominant flow direction.
            weight = np.abs(difference)
            total = weight.sum()
            if total < 1e-6:
                continue
            h, w = difference.shape
            ys, xs = np.mgrid[0:h, 0:w]
            cx = (weight * xs).sum() / total / max(w - 1, 1) - 0.5
            cy = (weight * ys).sum() / total / max(h - 1, 1) - 0.5
            if abs(cx) + abs(cy) > 0.02:
                direction[i] = float(np.arctan2(cy, cx))

    # Scene cuts: a large jump in luminance *or* colour between consecutive
    # frames. Using both catches transitions between scenes that happen to
    # share a brightness level but differ in hue.
    cuts = np.zeros(n, dtype=bool)
    if scene_indices is not None:
        # Ground truth is known from the clip structure.
        for i in range(1, n):
            cuts[i] = scene_indices[i] != scene_indices[i - 1]
        cuts[0] = True
    elif n > 1:
        luminance_jump = np.abs(np.diff(mean_luminance))
        color_jump = np.abs(np.diff(mean_color, axis=0)).mean(axis=1)
        cuts[1:] = (luminance_jump + color_jump) > SCENE_CUT_THRESHOLD
        cuts[0] = True

    # Derive scene index from cuts when not supplied.
    if scene_indices is None:
        indices: List[int] = []
        current = -1
        for i in range(n):
            if cuts[i]:
                current += 1
            indices.append(max(0, current))
        scene_indices = indices

    return [
        VisualFeatures(
            t=round(i / fps, 4),
            luminance=round(float(mean_luminance[i]), 5),
            contrast=round(float(contrast[i]), 5),
            motion=round(float(motion[i]), 5),
            motion_direction=(
                round(float(direction[i]), 4) if direction[i] is not None else None
            ),
            edge_density=round(float(edge_density[i]), 5),
            color=(
                round(float(mean_color[i][0]), 5),
                round(float(mean_color[i][1]), 5),
                round(float(mean_color[i][2]), 5),
            ),
            scene_cut=bool(cuts[i]),
            scene_index=int(scene_indices[i]),
        )
        for i in range(n)
    ]


def layout_vector(frame: np.ndarray, blocks: Tuple[int, int] = (6, 4)) -> np.ndarray:
    """Coarse block-average luminance layout of one frame.

    Used for the spatial-layout fidelity metric. Deliberately coarser than the
    reconstruction grid: comparing at full grid resolution measures noise, and
    what the signal plausibly carries is which regions were bright.
    """
    luma = frame @ _LUMA
    h, w = luma.shape
    bw, bh = blocks

    ys = np.linspace(0, h, bh + 1).astype(int)
    xs = np.linspace(0, w, bw + 1).astype(int)

    out = np.zeros(bh * bw, dtype=np.float32)
    for j in range(bh):
        for i in range(bw):
            block = luma[ys[j] : ys[j + 1], xs[i] : xs[i + 1]]
            out[j * bw + i] = block.mean() if block.size else 0.0
    return out
