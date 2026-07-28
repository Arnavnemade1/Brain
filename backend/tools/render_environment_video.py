"""Render an environment reconstructed from EEG, as video.

Two landscapes side by side. Both use the same terrain, the same camera and
the same sun — all fixed before any brain activity is read. What differs is
appearance: light, haze, colour, water, barrenness.

* **Left** is graded from the true visual statistics of the photograph that
  was on screen.
* **Right** is graded from those same statistics as recovered from EEG alone,
  held out throughout.

So the left panel is the environment the image *should* produce, and the right
is the one the brain activity produces. Where they agree, the decode worked.

Run with::

    backend/.venv/bin/python -m tools.render_environment_video --seconds 60
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.realdata.images import load_catalogue, thumbnail
from app.scene.from_eeg import BINDINGS, decode_visual, grade_from
from app.scene.terrain import Grade, Sky, Terrain, compose, geometry, shade

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "ds004018"
OUT_DIR = DATA_ROOT / "derivatives"

WIDTH, HEIGHT, FPS = 1120, 620, 25
PANEL_W, PANEL_H = 520, 336
PANEL_Y = 84
FRAMES_PER_MOMENT = 12

BACKGROUND = (9, 11, 16)
INK = (238, 241, 244)
INK_MUTED = (150, 160, 172)
INK_FAINT = (104, 114, 126)
TRUTH_TINT = (150, 200, 235)
DECODED_TINT = (118, 214, 170)


def _font(size: int):
    from PIL import ImageFont

    for name in ("/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _panel_x(index: int) -> int:
    gap = WIDTH - 2 * PANEL_W - 64
    return 32 + index * (PANEL_W + gap)


def _lerp_grade(a: Grade, b: Grade, t: float) -> Grade:
    """Interpolate two appearances.

    Presentations are 200 ms apart; cutting between them makes the sequence
    strobe. Easing between consecutive grades is a rendering choice about
    legibility and changes nothing about what was decoded.
    """
    t = float(np.clip(t, 0.0, 1.0))
    t = t * t * (3.0 - 2.0 * t)

    def mix(x: float, y: float) -> float:
        return x + (y - x) * t

    def mix3(x, y):
        return tuple(mix(float(p), float(q)) for p, q in zip(x, y))

    return Grade(
        daylight=mix(a.daylight, b.daylight),
        haze=mix(a.haze, b.haze),
        water_level=mix(a.water_level, b.water_level),
        saturation=mix(a.saturation, b.saturation),
        sun_tint=mix3(a.sun_tint, b.sun_tint),
        horizon=mix3(a.horizon, b.horizon),
        zenith=mix3(a.zenith, b.zenith),
        barrenness=mix(a.barrenness, b.barrenness),
        exposure=mix(a.exposure, b.exposure),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="sub-01")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=float, default=3.0)
    parser.add_argument("--steps", type=int, default=210)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    print(f"decoding visual properties for {args.subject}…")
    decoding = decode_visual(DATA_ROOT, args.subject)
    usable = decoding.usable()
    for name, what in BINDINGS:
        index = decoding.index(name)
        r = decoding.correlation[index] if index is not None else float("nan")
        state = "drives" if name in usable else "HELD AT PRIOR"
        print(f"  {name:<14} r={r:5.3f}  {state:<14} {what}")

    # Consolidate per stimulus: every repetition's prediction is held out, and
    # averaging them is the same "revisited memory" condition used elsewhere.
    numbers = sorted(set(int(s) for s in decoding.stimulus))
    truth_by: Dict[int, np.ndarray] = {}
    decoded_by: Dict[int, np.ndarray] = {}
    for n in numbers:
        mask = decoding.stimulus == n
        truth_by[n] = decoding.truth[mask].mean(axis=0)
        decoded_by[n] = decoding.predicted[mask].mean(axis=0)

    # --- the fixed world ----------------------------------------------------
    terrain = Terrain(amplitude=1150, wavelength=1500, ridged=0.95, octaves=10, seed=args.seed)
    camera, base_water = compose(terrain, distance=5600, clearance=110, fov=0.75)
    camera.pitch = -0.05

    print("building geometry (once)…")
    buffer = geometry(PANEL_W, PANEL_H, terrain, camera, Sky(), steps=args.steps, far=20000)

    catalogue = load_catalogue(DATA_ROOT / "stimuli")
    order = np.argsort(decoding.onset, kind="stable")
    n_moments = int(args.seconds * FPS / FRAMES_PER_MOMENT)
    moments = [int(i) for i in order[:n_moments]]

    grades: List[Tuple[Grade, Grade]] = []
    for i in moments:
        n = int(decoding.stimulus[i])
        grades.append(
            (
                grade_from(truth_by[n], decoding, base_water, allowed=usable),
                grade_from(decoded_by[n], decoding, base_water, allowed=usable),
            )
        )

    # Agreement per moment: correlation between the true and decoded property
    # vectors over the properties that are actually driving the render.
    driving = [decoding.index(name) for name, _ in BINDINGS if name in usable]
    agreement = []
    for i in moments:
        n = int(decoding.stimulus[i])
        a = truth_by[n][driving]
        b = decoded_by[n][driving]
        agreement.append(float(np.corrcoef(a, b)[0, 1]) if len(driving) > 1 else 0.0)

    overall = float(np.mean(agreement))
    print(f"\nmean per-moment property agreement: r = {overall:.3f}")

    from PIL import Image, ImageDraw
    import imageio.v2 as imageio

    small, medium = _font(13), _font(17)
    out = Path(args.out) if args.out else OUT_DIR / f"{args.subject}_environment.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        out,
        fps=FPS,
        codec="libx264",
        quality=7,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "22"],
    )

    thumb_cache: Dict[str, np.ndarray] = {}

    try:
        for position, index in enumerate(moments):
            nxt = min(position + 1, len(moments) - 1)
            stimulus = catalogue.get(int(decoding.stimulus[index]))

            for step in range(FRAMES_PER_MOMENT):
                t = step / FRAMES_PER_MOMENT
                left = shade(buffer, _lerp_grade(grades[position][0], grades[nxt][0], t))
                right = shade(buffer, _lerp_grade(grades[position][1], grades[nxt][1], t))

                frame = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
                draw = ImageDraw.Draw(frame)

                draw.text((32, 20), "ENVIRONMENT RECONSTRUCTED FROM EEG", fill=INK_FAINT, font=small)
                draw.text(
                    (32, 42),
                    f"{decoding.subject} · terrain, camera and sun are fixed; only appearance is decoded",
                    fill=INK,
                    font=medium,
                )

                for panel, (image, title, tint) in enumerate(
                    (
                        (left, "FROM THE PHOTOGRAPH", TRUTH_TINT),
                        (right, "FROM THEIR EEG ALONE", DECODED_TINT),
                    )
                ):
                    px = _panel_x(panel)
                    draw.text((px, PANEL_Y - 20), title, fill=tint, font=small)
                    frame.paste(
                        Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8)),
                        (px, PANEL_Y),
                    )
                    draw.rectangle(
                        [px - 1, PANEL_Y - 1, px + PANEL_W, PANEL_Y + PANEL_H],
                        outline=(38, 46, 58),
                    )

                # The photograph actually on screen, small, in the corner.
                if stimulus is not None:
                    if stimulus.filename not in thumb_cache:
                        thumb_cache[stimulus.filename] = thumbnail(
                            DATA_ROOT / "stimuli" / stimulus.filename, size=72
                        )
                    art = thumb_cache[stimulus.filename]
                    ax, ay = _panel_x(0) + 8, PANEL_Y + PANEL_H - 80
                    frame.paste(
                        Image.fromarray((art * 255).astype(np.uint8)), (ax, ay)
                    )
                    draw.rectangle([ax - 1, ay - 1, ax + 72, ay + 72], outline=(90, 100, 115))
                    draw.text((ax + 80, ay + 28), stimulus.label, fill=INK_MUTED, font=small)

                # --- property readout ---------------------------------------
                ry = PANEL_Y + PANEL_H + 26
                draw.text((32, ry), "DECODED VISUAL PROPERTIES", fill=INK_FAINT, font=small)
                n = int(decoding.stimulus[index])
                for slot, (name, _what) in enumerate(BINDINGS):
                    j = decoding.index(name)
                    if j is None:
                        continue
                    cx = 32 + slot * 155
                    bar_y = ry + 24
                    draw.text((cx, bar_y), name[:12], fill=INK_MUTED, font=small)

                    # Two ticks on a shared axis: truth and decoded.
                    axis_y = bar_y + 20
                    draw.line([cx, axis_y, cx + 130, axis_y], fill=(46, 54, 66), width=1)
                    for value, colour in (
                        (truth_by[n][j], TRUTH_TINT),
                        (decoded_by[n][j], DECODED_TINT),
                    ):
                        pos = cx + 65 + float(np.clip(value, -2.5, 2.5)) * 26
                        draw.line([pos, axis_y - 7, pos, axis_y + 7], fill=colour, width=2)
                    draw.text(
                        (cx, axis_y + 12),
                        f"r={decoding.correlation[j]:.2f}",
                        fill=INK_FAINT,
                        font=small,
                    )

                # --- agreement trace ----------------------------------------
                ty = ry + 82
                draw.text((32, ty), "PROPERTY AGREEMENT", fill=INK_FAINT, font=small)
                x0, x1 = 200, WIDTH - 32
                draw.rectangle([x0, ty - 2, x1, ty + 14], fill=(16, 20, 28))
                window = 80
                lo = max(0, position - window)
                series = agreement[lo : position + 1]
                if len(series) > 1:
                    pts = [
                        (
                            x0 + (x1 - x0) * k / max(window, 1),
                            ty + 12 - float(np.clip(v, -1, 1)) * 12,
                        )
                        for k, v in enumerate(series)
                    ]
                    draw.line(pts, fill=DECODED_TINT, width=2)

                draw.text(
                    (32, HEIGHT - 42),
                    f"Held out throughout. Appearance is bound to {len(usable)} visual properties "
                    f"EEG recovers (r = 0.47 to 0.69); mean agreement r = {overall:.2f}.",
                    fill=INK_MUTED,
                    font=small,
                )
                draw.text(
                    (32, HEIGHT - 26),
                    "The landform, camera and sunlight are a fixed prior and are not decoded. "
                    "This is how the scene looked, not what was in it.",
                    fill=INK_FAINT,
                    font=small,
                )

                writer.append_data(np.asarray(frame))

            if position % 25 == 0:
                print(f"  {position}/{len(moments)} moments")
    finally:
        writer.close()

    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB, {len(moments)} moments)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
