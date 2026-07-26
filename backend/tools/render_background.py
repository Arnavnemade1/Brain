"""Render the site's background loop: an aurora over a night mountain range.

Curtains of aurora drifting above layered ridges, with mist pooling between
the ridges and a field of quietly breathing stars overhead.

Three things make the aurora read as an aurora rather than a coloured smear,
and all three matter: the lower edge of each curtain sweeps in long arcs, a
much higher-frequency field breaks the light into vertical rays that drift at
their own rate, and a horizontal envelope keeps each curtain local — bright
in places, absent in others — instead of spanning the frame.

Generated rather than sourced from stock. Three reasons, in order of how much
they matter:

1. There is no licensing question. Nothing here is anyone else's footage.
2. The loop is seamless by construction rather than by careful cutting.
3. The palette is the design tokens, so the site and its background agree.

Seamlessness is the whole trick. Every animated term is a function of
``sin``/``cos`` of ``2π·k·t`` for integer ``k``, or a horizontal drift of a
whole number of periods, so at ``t = 1`` every term has completed an integer
number of cycles and the last frame flows into the first with no cut. The
ridges themselves never move, which is both what a mountain does and what
keeps the eye off the loop.

Run with::

    backend/.venv/bin/python -m tools.render_background
    backend/.venv/bin/python -m tools.render_background --still /tmp/look.png
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
PERIOD = 20.0

# --- Palette -----------------------------------------------------------------
# Kept dark. This sits behind body copy at high opacity, so the brightest
# thing in frame — the horizon glow — still has to read as a backdrop, not a
# subject. Tones come from the design tokens: ice-blue neutrals with a single
# warm note low on the horizon.

ZENITH = np.array([5, 7, 13], dtype=np.float64)
UPPER = np.array([10, 16, 28], dtype=np.float64)
HORIZON = np.array([54, 79, 92], dtype=np.float64)
#: A trace of `--color-memory-400` where the last light sits.
EMBER = np.array([96, 78, 62], dtype=np.float64)
MIST = np.array([74, 100, 112], dtype=np.float64)

# Aurora colours run bottom to top the way a real curtain does: oxygen green
# at the base, through cyan, into a faint violet where it thins out. Pulled
# toward the site's ice-blue tokens so it belongs to the palette rather than
# looking like a postcard dropped onto it.
AURORA_BASE = np.array([64, 196, 148], dtype=np.float64)
AURORA_MID = np.array([72, 168, 190], dtype=np.float64)
AURORA_TOP = np.array([104, 92, 176], dtype=np.float64)

#: Curtains, back to front.
#:
#: Every drift figure is an integer number of periods per loop, which is what
#: keeps the whole thing seamless: at ``t = 1`` each curtain has travelled a
#: whole number of frame widths and lands exactly where it started.
AURORA_BANDS = [
    # (base y, height, undulation amp, shape seed, drift, ray freq, ray drift, strength)
    (0.545, 0.44, 0.105, 3, 1, 22, 2, 1.00),
    (0.470, 0.34, 0.090, 17, 2, 34, 3, 0.70),
    (0.590, 0.27, 0.075, 29, 1, 16, 1, 0.52),
]

#: Farthest ridge first. Haze lifts the distant ones toward the sky colour and
#: drains their contrast, which is the entire cue for depth here.
RIDGES = [
    # (base height as fraction of frame, relief amplitude, colour, seed)
    (0.615, 0.030, np.array([44, 60, 72], dtype=np.float64), 11),
    (0.665, 0.042, np.array([34, 48, 59], dtype=np.float64), 23),
    (0.730, 0.052, np.array([24, 35, 45], dtype=np.float64), 37),
    (0.815, 0.062, np.array([15, 23, 31], dtype=np.float64), 53),
    (0.920, 0.075, np.array([8, 12, 18], dtype=np.float64), 71),
]

#: Where the sky stops and the land begins, as a fraction of frame height.
HORIZON_Y = 0.62


def _periodic_fbm(x: np.ndarray, seed: int, octaves: int = 6) -> np.ndarray:
    """Fractal noise that is exactly periodic over ``x`` in [0, 1).

    Built from integer-frequency sines, so shifting ``x`` by a whole number
    wraps perfectly — which is what lets the drifting layers loop.
    """
    rng = np.random.default_rng(seed)
    value = np.zeros_like(x)
    amplitude, total = 1.0, 0.0
    for octave in range(octaves):
        frequency = 2**octave
        phase = rng.uniform(0, 2 * math.pi)
        value += amplitude * np.sin(2 * math.pi * frequency * x + phase)
        total += amplitude
        amplitude *= 0.5
    return value / total


def _ridge_tops() -> list[np.ndarray]:
    """Static skyline for each layer, as a y-fraction per column."""
    x = np.linspace(0.0, 1.0, WIDTH, endpoint=False)
    tops = []
    for base, amplitude, _, seed in RIDGES:
        profile = _periodic_fbm(x, seed)
        # Sharpen the peaks: mountains are not sine waves. Folding the noise
        # around zero gives ridged rather than rolling terrain.
        profile = (1.0 - np.abs(profile)) ** 1.7
        tops.append(base - amplitude * (profile * 2.0 - 0.6))
    return tops


def _sky() -> np.ndarray:
    """Static vertical gradient, zenith through horizon glow."""
    y = np.linspace(0.0, 1.0, HEIGHT)[:, None, None]
    sky = np.empty((HEIGHT, WIDTH, 3), dtype=np.float64)

    # Two-stop ramp so the upper sky stays genuinely dark instead of washing
    # linearly into the horizon.
    upper = np.clip(y / HORIZON_Y, 0, 1) ** 2.4
    sky[:] = ZENITH + (UPPER - ZENITH) * upper

    glow = np.clip(1.0 - np.abs(y - HORIZON_Y) / 0.30, 0, 1) ** 2.6
    sky += (HORIZON - UPPER) * glow

    # The warm note is narrow and sits just under the horizon line.
    ember = np.clip(1.0 - np.abs(y - (HORIZON_Y - 0.012)) / 0.075, 0, 1) ** 3.0
    sky += (EMBER - UPPER) * ember * 0.30
    return sky


def _stars(count: int = 420):
    """Fixed star field above the horizon, with per-star twinkle parameters."""
    rng = np.random.default_rng(9)
    sx = rng.uniform(0, WIDTH, count).astype(int)
    # Denser high up, thinning toward the haze.
    sy = (rng.beta(1.5, 3.0, count) * HORIZON_Y * HEIGHT).astype(int)
    brightness = rng.uniform(0.25, 1.0, count)
    # Integer cycles per loop, so the twinkle wraps with everything else.
    cycles = rng.integers(1, 4, count).astype(float)
    phase = rng.uniform(0, 2 * math.pi, count)
    return sx, sy, brightness, cycles, phase


def _aurora(t: float, y_norm: np.ndarray, x_line: np.ndarray) -> np.ndarray:
    """Additive aurora light for loop phase ``t``.

    Each curtain is a ribbon whose lower edge undulates across the frame.
    Light rises from that edge and fades out with height, broken into vertical
    rays by a second, much higher-frequency noise field drifting at its own
    rate — that shearing between the ribbon and its rays is what reads as an
    aurora rather than as a coloured smear.
    """
    tau = 2 * math.pi * t
    light = np.zeros((y_norm.shape[0], x_line.shape[0], 3), dtype=np.float64)

    for base_y, height, amp, seed, drift, ray_freq, ray_drift, strength in AURORA_BANDS:
        # Lower edge of the curtain. Few octaves, so it sweeps in long arcs
        # rather than jittering.
        edge = base_y + amp * _periodic_fbm(x_line - drift * t, seed, octaves=3)[None, :]

        # Horizontal envelope. Without this every curtain spans the full frame
        # and the three of them stack into one flat band of glow — the single
        # thing that most stops this reading as an aurora. Real curtains are
        # local: bright over here, absent over there.
        envelope = _periodic_fbm(x_line * 2 - drift * t, seed + 11, octaves=3)
        envelope = np.clip(envelope * 1.7 + 0.36, 0, 1)[None, :] ** 1.9

        # Curtains are not the same height all the way along. Varying it with
        # its own noise field is what keeps the tops ragged; a constant height
        # gives every band a flat ceiling and the whole sky looks cropped.
        local_height = height * (0.55 + 0.75 * (0.5 + 0.5 * _periodic_fbm(
            x_line * 3 - drift * t, seed + 23, octaves=3
        )))[None, :]

        # Height above that edge, normalised: 0 at the base, 1 at the top.
        rise = (edge - y_norm) / local_height
        inside = rise > 0
        u = np.clip(rise, 0, 1)

        # Brightest just above the base, thinning upward. The `u / 0.08` term
        # softens the bottom edge so the curtain does not begin on a hard line.
        profile = np.exp(-2.4 * u) * (1.0 - u) ** 1.8 * np.clip(u / 0.08, 0, 1)
        profile = np.where(inside, profile, 0.0)

        # Vertical rays. Passing `x * ray_freq` keeps every component frequency
        # an integer, so the field stays periodic across the frame.
        rays = _periodic_fbm(x_line * ray_freq - ray_drift * t, seed + 5, octaves=4)
        rays = (0.34 + 0.66 * np.clip(0.5 + 0.5 * rays, 0, 1) ** 1.4)[None, :]

        # Slow breathing, one cycle per loop.
        pulse = 0.78 + 0.22 * math.sin(tau + seed)

        # Colour ramp with height: green, cyan, violet.
        low = np.clip(u / 0.45, 0, 1)[..., None]
        high = np.clip((u - 0.45) / 0.55, 0, 1)[..., None]
        colour = AURORA_BASE + (AURORA_MID - AURORA_BASE) * low
        colour = colour + (AURORA_TOP - AURORA_MID) * high

        amount = (profile * rays * envelope * strength * pulse)[..., None]
        light += colour * amount

        # A wider, softer copy of the same curtain standing in for bloom. Real
        # aurora spills light into the sky around it; without this the curtains
        # read as cut paper.
        halo = np.exp(-1.1 * u) * (1.0 - u) ** 1.6 * np.clip(u / 0.25, 0, 1)
        halo = np.where(inside, halo, 0.0)
        light += colour * (halo * envelope * np.sqrt(rays) * strength * pulse * 0.11)[..., None]

    return light * 0.78


def _frame(t: float, sky: np.ndarray, tops: list[np.ndarray], stars) -> np.ndarray:
    """One frame at loop phase ``t`` in [0, 1)."""
    tau = 2 * math.pi * t
    image = sky.copy()

    y_norm = np.linspace(0.0, 1.0, HEIGHT)[:, None]
    x_line = np.linspace(0.0, 1.0, WIDTH, endpoint=False)

    # --- Stars --------------------------------------------------------------
    sx, sy, brightness, cycles, phase = stars
    twinkle = brightness * (0.62 + 0.38 * np.sin(cycles * tau + phase))
    # Fade them out as they approach the haze.
    depth = np.clip(1.0 - (sy / (HORIZON_Y * HEIGHT)) ** 1.5, 0, 1)
    image[sy, sx] += (twinkle * depth * 150)[:, None]

    # --- Aurora -------------------------------------------------------------
    image += _aurora(t, y_norm, x_line)

    # --- Ridges, far to near, with mist pooling in front of each ------------
    for index, ((_, _, colour, _), top) in enumerate(zip(RIDGES, tops)):
        top_row = top[None, :]

        # Mist sits just above the skyline it belongs to. Drifting by a whole
        # number of widths per loop keeps it seamless.
        drift = _periodic_fbm(x_line - (index + 1) * t, 100 + index)[None, :]
        centre = top_row - 0.020
        spread = 0.030 + 0.012 * index
        density = np.clip(0.45 + 0.55 * drift, 0, 1)
        alpha = np.exp(-(((y_norm - centre) / spread) ** 2)) * density
        alpha = alpha * (0.34 - 0.045 * index)
        image += (MIST - image) * alpha[..., None]

        # The ridge itself, opaque.
        mask = y_norm >= top_row
        image[np.broadcast_to(mask, image.shape[:2])] = colour

        # A thin rim of haze catching the horizon light along the crest, which
        # is what separates one silhouette from the next.
        rim = np.exp(-(((y_norm - top_row) / 0.006) ** 2)) * ~mask
        image += (HORIZON - colour)[None, None, :] * (rim * 0.22)[..., None]

    # --- Atmosphere ---------------------------------------------------------
    # Vignette, so the edges fall away and page content floats above centre.
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    radius = np.hypot((xx / WIDTH - 0.5) * 1.05, (yy / HEIGHT - 0.5) * 1.25)
    image *= np.clip(1.18 - radius * 1.15, 0.30, 1.0)[..., None]

    # Fine grain. Large dark gradients band badly on 8-bit displays, and a
    # little noise is what keeps this looking like light rather than steps.
    rng = np.random.default_rng(int(t * 100000) % 65536)
    image += rng.normal(0, 1.5, image.shape)

    return np.clip(image, 0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--still", default="", help="write a single PNG and stop")
    args = parser.parse_args()

    import imageio.v2 as imageio

    sky, tops, stars = _sky(), _ridge_tops(), _stars()

    if args.still:
        imageio.imwrite(args.still, _frame(0.3, sky, tops, stars))
        print(f"wrote still {args.still}")
        return 0

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = int(PERIOD * FPS)
    writer = imageio.get_writer(
        path,
        fps=FPS,
        codec="libx264",
        quality=6,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "28"],
    )
    try:
        for index in range(n_frames):
            writer.append_data(_frame(index / n_frames, sky, tops, stars))
    finally:
        writer.close()

    size = path.stat().st_size / 1e6
    print(f"wrote {path}  ({size:.1f} MB, {n_frames} frames, {PERIOD:.0f}s loop)")

    first = _frame(0.0, sky, tops, stars).astype(float)
    last = _frame((n_frames - 1) / n_frames, sky, tops, stars).astype(float)
    print(f"loop seam mean |Δ| = {float(np.mean(np.abs(first - last))):.2f} / 255")
    print(f"mean luminance     = {float(first.mean()):.1f} / 255")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
