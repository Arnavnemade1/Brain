"""Neural Memory Latent Space.

The interface between neural measurement and visual reconstruction. Encoder
output for a window, plus a short window of temporal context, is projected
into a fixed-dimensional vector. Everything after this stage reads only the
latent.

Temporal context matters more than it might appear. A single 2-second window
cannot distinguish "bright scene beginning" from "bright scene ending", and
the evoked transient that marks a scene boundary spans several windows. The
latent therefore carries recent history and local derivatives alongside the
current measurement.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

import numpy as np

from ..models.neural import LatentPoint, VisualResponse
from .encoder import FEATURE_NAMES, response_vector

#: Windows of history the latent carries.
CONTEXT_WINDOWS = 4

#: Dimensionality of the latent space. Matches ``LATENT_DIMENSIONS`` in the
#: models and the TypeScript contract.
LATENT_DIMENSIONS = 16


def latent_component_names() -> List[str]:
    """Human-readable name for each latent dimension.

    The latent is interpretable by construction — a black box here would make
    every fidelity result impossible to diagnose.
    """
    return [
        *FEATURE_NAMES,  # 7 — current measurement
        "drive_delta",  # rate of luminance change
        "motion_delta",
        "transient_delta",
        "drive_mean",  # short-term context
        "motion_mean",
        "suppression_mean",
        "drive_variance",  # local stability
        "transient_peak",  # strongest transient in context
        "context_confidence",
    ]


class LatentProjector:
    """Builds latent trajectories from a stream of encoder outputs."""

    def __init__(self) -> None:
        self._history: Deque[np.ndarray] = deque(maxlen=CONTEXT_WINDOWS)
        self._confidences: Deque[float] = deque(maxlen=CONTEXT_WINDOWS)

    def reset(self) -> None:
        self._history.clear()
        self._confidences.clear()

    def project(self, response: VisualResponse, confidence: float, t: float) -> LatentPoint:
        """Project one encoder output into the latent space."""
        current = response_vector(response)
        self._history.append(current)
        self._confidences.append(float(np.clip(confidence, 0.0, 1.0)))

        history = np.array(self._history)
        previous = history[-2] if len(history) > 1 else current

        index = {name: i for i, name in enumerate(FEATURE_NAMES)}

        drive_delta = current[index["occipital_drive"]] - previous[index["occipital_drive"]]
        motion_delta = current[index["motion_response"]] - previous[index["motion_response"]]
        transient_delta = current[index["transient"]] - previous[index["transient"]]

        drive_mean = float(history[:, index["occipital_drive"]].mean())
        motion_mean = float(history[:, index["motion_response"]].mean())
        suppression_mean = float(history[:, index["alpha_suppression"]].mean())
        drive_variance = float(history[:, index["occipital_drive"]].std())
        transient_peak = float(history[:, index["transient"]].max())
        context_confidence = float(np.mean(self._confidences))

        vector = np.concatenate(
            [
                current,
                [
                    drive_delta,
                    motion_delta,
                    transient_delta,
                    drive_mean,
                    motion_mean,
                    suppression_mean,
                    drive_variance,
                    transient_peak,
                    context_confidence,
                ],
            ]
        )

        assert vector.size == LATENT_DIMENSIONS, (
            f"latent is {vector.size}d, expected {LATENT_DIMENSIONS}"
        )

        return LatentPoint(
            t=round(t, 4),
            vector=[round(float(v), 5) for v in vector],
            confidence=round(float(np.clip(confidence, 0.0, 1.0)), 4),
        )


def stack(points: List[LatentPoint]) -> np.ndarray:
    """Latent trajectory as a ``(n, LATENT_DIMENSIONS)`` array."""
    if not points:
        return np.zeros((0, LATENT_DIMENSIONS))
    return np.array([p.vector for p in points], dtype=np.float64)
