"""Frame Reconstruction.

Turns a latent point plus its decoded scene properties into a coarse image.

The resolution ceiling here is not an implementation shortcut. Scalp EEG
carries a handful of spatially-resolved quantities about the visual field —
essentially left/right and upper/lower balance — so the decoder predicts a
6×4 block layout and that layout is interpolated up to the display grid.
Predicting per-pixel detail would be predicting noise, and the fidelity
metrics would correctly punish it.
"""

from __future__ import annotations

import base64
from typing import List, Tuple

import numpy as np

from ..models.memory import SceneEstimate
from ..models.neural import LatentPoint
from ..stimulus.clips import GRID_HEIGHT, GRID_WIDTH
from .decoders import DecoderBundle, LAYOUT_BLOCKS


def _interpolate_layout(layout: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bilinearly expand a block layout to the reconstruction grid."""
    blocks_x, blocks_y = LAYOUT_BLOCKS
    grid = layout.reshape(blocks_y, blocks_x)

    # Sample block centres, then interpolate along each axis in turn.
    ys = np.linspace(0, blocks_y - 1, height)
    xs = np.linspace(0, blocks_x - 1, width)

    y0 = np.floor(ys).astype(int)
    y1 = np.minimum(y0 + 1, blocks_y - 1)
    wy = (ys - y0)[:, None]

    x0 = np.floor(xs).astype(int)
    x1 = np.minimum(x0 + 1, blocks_x - 1)
    wx = (xs - x0)[None, :]

    top = grid[y0][:, x0] * (1 - wx) + grid[y0][:, x1] * wx
    bottom = grid[y1][:, x0] * (1 - wx) + grid[y1][:, x1] * wx
    return top * (1 - wy) + bottom * wy


def reconstruct_frame(
    latent: LatentPoint,
    scene: SceneEstimate,
    decoders: DecoderBundle,
    width: int = GRID_WIDTH,
    height: int = GRID_HEIGHT,
) -> np.ndarray:
    """Reconstruct one frame as float RGB in 0–1, shape ``(h, w, 3)``."""
    vector = np.array(latent.vector, dtype=np.float64)[np.newaxis, :]
    layout = decoders.layout.predict(vector)[0]

    spatial = _interpolate_layout(layout, width, height)

    # Re-centre the layout on the separately decoded mean luminance. The
    # layout decoder is good at *relative* brightness across the field and
    # poor at absolute level; the scene decoder is the reverse, so each is
    # used for what it actually recovers.
    deviation = spatial - spatial.mean()

    # Scale spatial variation by how well the layout decoder was *measured* to
    # work, and by this window's own confidence.
    #
    # Without this the frame asserts spatial structure at full decoded
    # contrast regardless of whether any was recovered — and since scalp EEG
    # resolves little more than field balance, that structure is mostly wrong.
    # Asserting it made reconstructions score *below* a flat mean frame on
    # SSIM: confidently wrong detail is worse than honest smoothness. Where
    # layout is unrecoverable the frame correctly collapses toward a uniform
    # field at the decoded brightness, which is exactly what the system knows.
    reliability = float(np.clip(decoders.layout_reliability, 0.0, 1.0))
    spread = float(deviation.std())
    if spread > 1e-6 and reliability > 1e-3:
        target_spread = scene.contrast * reliability * (0.4 + 0.6 * scene.confidence)
        deviation = deviation * (target_spread / spread)
    else:
        deviation = deviation * 0.0

    spatial = np.clip(scene.luminance + deviation, 0.0, 1.0)

    # Apply the decoded colour as a tint, preserving the luminance layout.
    color = np.array(scene.color, dtype=np.float64)
    color_luma = float(color @ np.array([0.2126, 0.7152, 0.0722]))
    if color_luma > 1e-6:
        tint = color / color_luma
    else:
        tint = np.ones(3)

    frame = spatial[..., None] * tint[None, None, :]
    return np.clip(frame, 0.0, 1.0)


def encode_frame(frame: np.ndarray) -> str:
    """Frame as base64 RGB bytes for transport."""
    payload = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8).tobytes()
    return base64.b64encode(payload).decode("ascii")


def decode_frame(encoded: str, width: int, height: int) -> np.ndarray:
    """Inverse of :func:`encode_frame`."""
    raw = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    return raw.reshape(height, width, 3).astype(np.float64) / 255.0


def reference_at(frames: np.ndarray, t: float, fps: float) -> np.ndarray:
    """Reference frame nearest a given stimulus time."""
    index = int(round(t * fps))
    index = max(0, min(index, frames.shape[0] - 1))
    return frames[index].astype(np.float64)
