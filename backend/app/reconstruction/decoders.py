"""Fitted decoders mapping the latent space to visual properties.

Ridge regression, fitted offline against known reference clips by
``tools/train_decoders.py`` and loaded here. Deliberately linear and
inspectable: with a 16-dimensional latent carrying a handful of genuinely
recoverable quantities, a larger model would fit noise, and every fidelity
number would become impossible to attribute to anything.

Coefficients ship as JSON next to this module. If they are missing, decoding
falls back to a documented identity mapping that is honest about being
untrained rather than silently producing confident-looking output.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("mindscape.decoders")

WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.json"

#: Scene-level quantities decoded from the latent, in fitted order.
SCENE_TARGETS: List[str] = [
    "luminance",
    "contrast",
    "motion",
    "color_r",
    "color_g",
    "color_b",
]

#: Coarse spatial layout resolution. Matches ``features.layout_vector``.
LAYOUT_BLOCKS: Tuple[int, int] = (6, 4)
LAYOUT_SIZE = LAYOUT_BLOCKS[0] * LAYOUT_BLOCKS[1]


@dataclass
class RidgeModel:
    """A fitted linear map with input standardisation."""

    #: (features,) input means and scales used at fit time.
    input_mean: np.ndarray
    input_scale: np.ndarray
    #: (features, targets) coefficients and (targets,) intercepts.
    coefficients: np.ndarray
    intercepts: np.ndarray
    target_names: List[str]
    #: Held-out R² per target, recorded at fit time. Reported, never used.
    r2: Dict[str, float]

    def predict(self, latents: np.ndarray) -> np.ndarray:
        """Predict targets for ``(n, features)`` latents."""
        if latents.ndim == 1:
            latents = latents[np.newaxis, :]
        standardised = (latents - self.input_mean) / self.input_scale
        return standardised @ self.coefficients + self.intercepts

    def to_dict(self) -> dict:
        return {
            "inputMean": self.input_mean.tolist(),
            "inputScale": self.input_scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercepts": self.intercepts.tolist(),
            "targetNames": self.target_names,
            "r2": self.r2,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RidgeModel":
        return cls(
            input_mean=np.array(payload["inputMean"], dtype=np.float64),
            input_scale=np.array(payload["inputScale"], dtype=np.float64),
            coefficients=np.array(payload["coefficients"], dtype=np.float64),
            intercepts=np.array(payload["intercepts"], dtype=np.float64),
            target_names=list(payload["targetNames"]),
            r2=dict(payload.get("r2", {})),
        )


def fit_ridge(
    latents: np.ndarray,
    targets: np.ndarray,
    target_names: List[str],
    alpha: float = 1.0,
) -> RidgeModel:
    """Fit a ridge model from latents to targets.

    Ridge rather than ordinary least squares because latent dimensions are
    strongly collinear by construction — context means and deltas are derived
    from the same measurements as the raw features — and an unregularised fit
    produces enormous cancelling coefficients that generalise poorly.
    """
    if latents.ndim != 2 or targets.ndim != 2:
        raise ValueError("latents and targets must both be 2-D")
    if latents.shape[0] != targets.shape[0]:
        raise ValueError("latent and target counts differ")

    mean = latents.mean(axis=0)
    scale = latents.std(axis=0)
    scale[scale < 1e-8] = 1.0

    x = (latents - mean) / scale
    y_mean = targets.mean(axis=0)
    y = targets - y_mean

    n_features = x.shape[1]
    gram = x.T @ x + alpha * np.eye(n_features)
    coefficients = np.linalg.solve(gram, x.T @ y)

    return RidgeModel(
        input_mean=mean,
        input_scale=scale,
        coefficients=coefficients,
        intercepts=y_mean,
        target_names=target_names,
        r2={},
    )


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Per-target coefficient of determination."""
    residual = ((actual - predicted) ** 2).sum(axis=0)
    total = ((actual - actual.mean(axis=0)) ** 2).sum(axis=0)
    total = np.where(total < 1e-12, 1e-12, total)
    return 1.0 - residual / total


@dataclass
class DecoderBundle:
    """Everything needed to turn a latent trajectory into frames."""

    scene: RidgeModel
    layout: RidgeModel
    #: Latent transient level above which a scene boundary is declared.
    cut_threshold: float
    #: Measured held-out correlation of the spatial layout decoder, 0-1.
    #: Frame reconstruction scales its spatial variation by this, so the
    #: output only asserts structure to the degree the decoder recovers it.
    layout_reliability: float
    #: Provenance, so a report can state what produced these numbers.
    trained_on: List[str]
    trained_frames: int
    #: True when weights were loaded from disk rather than defaulted.
    fitted: bool = True

    def save(self, path: Path = WEIGHTS_PATH) -> None:
        payload = {
            "scene": self.scene.to_dict(),
            "layout": self.layout.to_dict(),
            "cutThreshold": self.cut_threshold,
            "layoutReliability": self.layout_reliability,
            "trainedOn": self.trained_on,
            "trainedFrames": self.trained_frames,
        }
        path.write_text(json.dumps(payload, indent=1))

    @classmethod
    def load(cls, path: Path = WEIGHTS_PATH) -> "DecoderBundle":
        if not path.exists():
            log.warning(
                "No fitted decoder weights at %s — reconstruction will use the "
                "untrained fallback. Run `python -m tools.train_decoders`.",
                path,
            )
            return untrained()

        payload = json.loads(path.read_text())
        return cls(
            scene=RidgeModel.from_dict(payload["scene"]),
            layout=RidgeModel.from_dict(payload["layout"]),
            cut_threshold=float(payload.get("cutThreshold", 0.55)),
            layout_reliability=float(payload.get("layoutReliability", 0.0)),
            trained_on=list(payload.get("trainedOn", [])),
            trained_frames=int(payload.get("trainedFrames", 0)),
            fitted=True,
        )


def untrained() -> DecoderBundle:
    """Fallback used when no fitted weights are available.

    Maps the two latent dimensions with an unambiguous physical meaning —
    occipital drive to luminance, motion response to motion — and leaves
    everything else at its mean. Fidelity will be poor, which is the correct
    and visible outcome for an untrained system.
    """
    from ..encoder.latent import LATENT_DIMENSIONS

    scene_coefficients = np.zeros((LATENT_DIMENSIONS, len(SCENE_TARGETS)))
    scene_coefficients[0, 0] = 0.18  # occipital_drive → luminance
    scene_coefficients[2, 2] = 0.18  # motion_response → motion

    scene = RidgeModel(
        input_mean=np.zeros(LATENT_DIMENSIONS),
        input_scale=np.ones(LATENT_DIMENSIONS),
        coefficients=scene_coefficients,
        intercepts=np.array([0.3, 0.15, 0.2, 0.3, 0.3, 0.3]),
        target_names=SCENE_TARGETS,
        r2={},
    )

    layout = RidgeModel(
        input_mean=np.zeros(LATENT_DIMENSIONS),
        input_scale=np.ones(LATENT_DIMENSIONS),
        coefficients=np.zeros((LATENT_DIMENSIONS, LAYOUT_SIZE)),
        intercepts=np.full(LAYOUT_SIZE, 0.3),
        target_names=[f"block_{i}" for i in range(LAYOUT_SIZE)],
        r2={},
    )

    return DecoderBundle(
        scene=scene,
        layout=layout,
        cut_threshold=0.55,
        layout_reliability=0.0,
        trained_on=[],
        trained_frames=0,
        fitted=False,
    )


_bundle: Optional[DecoderBundle] = None


def get_decoders() -> DecoderBundle:
    """Process-wide decoder bundle."""
    global _bundle
    if _bundle is None:
        _bundle = DecoderBundle.load()
    return _bundle


def reload_decoders() -> DecoderBundle:
    global _bundle
    _bundle = DecoderBundle.load()
    return _bundle
