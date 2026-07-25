"""Scene Reconstruction.

Decodes the latent trajectory into scene-level visual properties: brightness,
contrast, motion, dominant colour, and where the scene changed.

Scene boundaries are handled separately from the continuous quantities. They
are events, not levels, and they are also the most reliably recoverable thing
in the whole system — an abrupt visual change produces a large time-locked
response that survives a great deal of noise.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..models.memory import SceneEstimate
from ..models.neural import LatentPoint
from ..encoder.latent import stack
from .decoders import DecoderBundle, SCENE_TARGETS


def detect_cuts(
    latents: List[LatentPoint], threshold: float, refractory: float = 1.2
) -> np.ndarray:
    """Boolean per-latent scene-boundary flags.

    Uses the evoked-transient dimension of the latent, which spikes when the
    visual field changes abruptly. A refractory period suppresses the trailing
    windows of the same evoked response — the transient spans several windows,
    and without it one cut is reported as three.
    """
    n = len(latents)
    cuts = np.zeros(n, dtype=bool)
    if n == 0:
        return cuts

    vectors = stack(latents)
    # Index 3 is `transient`; index 13 is `transient_peak` over the context.
    transient = vectors[:, 3]
    peak = vectors[:, 13] if vectors.shape[1] > 13 else transient

    # A boundary is a transient that is both absolutely large and locally
    # prominent — a sustained high level is a bright scene, not a cut.
    prominence = transient - np.concatenate([[0.0], transient[:-1]])
    score = transient * 0.6 + peak * 0.2 + np.clip(prominence, 0, None) * 0.6

    last_cut = -1e9
    for i in range(n):
        if score[i] < threshold:
            continue
        if latents[i].t - last_cut < refractory:
            continue
        cuts[i] = True
        last_cut = latents[i].t

    # The first window always begins a scene.
    cuts[0] = True
    return cuts


def decode_scenes(
    latents: List[LatentPoint], decoders: DecoderBundle
) -> List[SceneEstimate]:
    """Decode the latent trajectory into per-window scene estimates."""
    if not latents:
        return []

    vectors = stack(latents)
    predictions = decoders.scene.predict(vectors)

    cuts = detect_cuts(latents, decoders.cut_threshold)

    # Running scene index from detected boundaries.
    scene_index = -1
    indices: List[int] = []
    for i in range(len(latents)):
        if cuts[i]:
            scene_index += 1
        indices.append(max(0, scene_index))

    column = {name: i for i, name in enumerate(SCENE_TARGETS)}
    estimates: List[SceneEstimate] = []

    for i, point in enumerate(latents):
        row = predictions[i]

        luminance = float(np.clip(row[column["luminance"]], 0.0, 1.0))
        contrast = float(np.clip(row[column["contrast"]], 0.0, 1.0))
        motion = float(np.clip(row[column["motion"]], 0.0, 1.0))

        color = (
            float(np.clip(row[column["color_r"]], 0.0, 1.0)),
            float(np.clip(row[column["color_g"]], 0.0, 1.0)),
            float(np.clip(row[column["color_b"]], 0.0, 1.0)),
        )

        # Direction is carried by the lateral balance dimension, which encodes
        # which side of the field the change occurred on. That supports a
        # left/right call and nothing finer, so anything below a clear
        # threshold is reported as no direction rather than guessed.
        lateral = point.vector[5] if len(point.vector) > 5 else 0.0
        direction: Optional[float] = None
        if motion > 0.2 and abs(lateral) > 0.12:
            direction = 0.0 if lateral > 0 else float(np.pi)

        estimates.append(
            SceneEstimate(
                t=point.t,
                luminance=round(luminance, 5),
                contrast=round(contrast, 5),
                motion=round(motion, 5),
                motion_direction=direction,
                color=(round(color[0], 5), round(color[1], 5), round(color[2], 5)),
                scene_cut=bool(cuts[i]),
                scene_index=indices[i],
                confidence=point.confidence,
            )
        )

    return estimates
