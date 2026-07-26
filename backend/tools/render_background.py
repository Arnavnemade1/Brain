"""Render the site's background loop.

A slow field of light moving like sun through deep water. Generated rather
than sourced from stock so the loop is guaranteed seamless, the palette
matches the design tokens exactly, and there is no licensing ambiguity in the
repository.

Seamlessness is the whole trick. Every animated term is a function of
``sin``/``cos`` of ``2π·k·t/T`` for integer ``k``, so at ``t = T`` every term
has completed a whole number of cycles and the last frame flows into the
first with no cut.

Run with::

    backend/.venv/bin/python -m tools.render_background
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent.parent / "frontend" / "public" / "ambient.mp4"

WIDTH, HEIGHT = 1280, 720
FPS = 24
#: Loop length in seconds. Long enough not to read as repetitive.
PERIOD = 16.0

#: Deep, desaturated base tones. Kept dark so foreground type stays legible
#: at the low opacity the background is shown at.
DEEP = np.array([6, 9, 16], dtype=np.float64)
MID = np.array([24, 44, 60], dtype=np.float64)
LIFT = np.array([74, 118, 132], dtype=np.float64)
GLOW = np.array([158, 200, 196], dtype=np.float64)


def _field(shape, t: float) -> np.ndarray:
    """Layered caustic-like interference at phase ``t`` in 0-1."""
    h, w = shape
    y, x = np.mgrid[0:h, 0:w]
    x = x / w
    y = y / h

    tau = 2 * math.pi * t
    value = np.zeros((h, w), dtype=np.float64)

    # Each wave uses an integer number of cycles per loop, which is what makes
    # the final frame continuous with the first.
    waves = [
        # (amplitude, spatial fx, spatial fy, cycles per loop, phase)
        (0.50, 1.1, 0.7, 1, 0.0),
        (0.32, 1.9, 1.4, 2, 1.3),
        (0.22, 0.7, 2.3, 1, 2.6),
        (0.16, 3.1, 2.0, 3, 0.7),
        (0.10, 4.3, 3.4, 2, 4.1),
    ]

    for amplitude, fx, fy, cycles, phase in waves:
        value += amplitude * np.sin(
            2 * math.pi * (fx * x + fy * y) + cycles * tau + phase
        )

    # A slow vertical drift, so the field feels like it is descending.
    value += 0.28 * np.sin(2 * math.pi * (2.2 * y - 0.6 * x) + tau)

    # Normalise to 0-1 with a soft knee, which keeps the highlights from
    # clipping into flat white.
    value = value / 1.6
    return 1.0 / (1.0 + np.exp(-2.4 * value))


def _frame(t: float) -> np.ndarray:
    field = _field((HEIGHT, WIDTH), t)

    # Vertical gradient: darker at the top so headline text sits on calm.
    y = np.linspace(0.0, 1.0, HEIGHT)[:, None]
    depth = np.clip(0.25 + y * 0.85, 0, 1)

    base = DEEP[None, None, :] + (MID - DEEP)[None, None, :] * depth[..., None]

    caustic = np.clip((field - 0.42) / 0.58, 0, 1) ** 1.6
    lift = (LIFT - MID)[None, None, :] * caustic[..., None] * depth[..., None]

    highlight = np.clip((field - 0.80) / 0.20, 0, 1) ** 2.2
    glow = (GLOW - LIFT)[None, None, :] * highlight[..., None] * 0.55

    image = base + lift + glow

    # Vignette, so the edges fall away and content floats above the centre.
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    radius = np.hypot((xx / WIDTH - 0.5) * 1.15, (yy / HEIGHT - 0.5) * 1.3)
    image *= np.clip(1.15 - radius * 1.25, 0.28, 1.0)[..., None]

    # Fine grain. Large dark gradients band badly on 8-bit displays, and a
    # little noise is what keeps this looking like light rather than steps.
    rng = np.random.default_rng(int(t * 100000) % 65536)
    image += rng.normal(0, 1.4, image.shape)

    return np.clip(image, 0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    import imageio.v2 as imageio

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = int(PERIOD * FPS)
    writer = imageio.get_writer(
        path,
        fps=FPS,
        codec="libx264",
        quality=6,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "30"],
    )
    try:
        for index in range(n_frames):
            writer.append_data(_frame(index / n_frames))
    finally:
        writer.close()

    size = path.stat().st_size / 1e6
    print(f"wrote {path}  ({size:.1f} MB, {n_frames} frames, {PERIOD:.0f}s loop)")

    # Verify the seam: the last frame should be close to the first.
    difference = float(
        np.mean(np.abs(_frame(0.0).astype(float) - _frame((n_frames - 1) / n_frames).astype(float)))
    )
    print(f"loop seam mean |Δ| = {difference:.2f} / 255 (lower is smoother)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
