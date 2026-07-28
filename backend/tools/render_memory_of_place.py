"""Render a brain-state trajectory as a place you move through.

One continuous recording of a person having an experience, turned into a
journey across terrain. The viewpoint travels forward; how fast, how bright,
how clear the air and how bare the ground all follow the recording moment to
moment.

Everything driving the picture is measured from the EEG rather than inferred
about the world — see ``app/scene/cognitive.py`` for why that distinction is
the whole point, and which of the readings are interpretation rather than
measurement.

Run with::

    backend/.venv/bin/python -m tools.render_memory_of_place --seconds 60
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.emotion import dataset
from app.scene.cognitive import CHANNELS, build_track, grade_from_state, travel_rate
from app.scene.terrain import (
    Camera,
    Grade,
    Sky,
    Terrain,
    geometry,
    sea_level,
    shade,
)

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "ds003751"
OUT_DIR = DATA_ROOT / "derivatives"

WIDTH, HEIGHT, FPS = 1120, 620, 25
VIEW_W, VIEW_H = 1056, 392
VIEW_X, VIEW_Y = 32, 74

BACKGROUND = (9, 11, 16)
INK = (238, 241, 244)
INK_MUTED = (150, 160, 172)
INK_FAINT = (104, 114, 126)
MEASURED = (128, 206, 240)
INTERPRETED = (206, 176, 122)


def _font(size: int):
    from PIL import ImageFont

    for name in ("/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _path_cameras(
    terrain: Terrain, count: int, span: float = 5200.0
) -> List[Camera]:
    """Viewpoints along a slow forward dolly, each grounded on the terrain.

    Geometry is precomputed per viewpoint, so the number of these sets both
    the cost of the render and how smooth the travel looks.
    """
    cameras = []
    for k in range(count):
        t = k / max(count - 1, 1)
        # A gentle arc rather than a straight line, so the ridges reveal
        # themselves by parallax instead of only growing.
        x = -2600.0 + span * t
        z = -3400.0 + span * 0.62 * t + 380.0 * math.sin(t * math.pi * 1.1)
        yaw = 0.30 - 0.22 * t
        camera = Camera(position=(x, 0.0, z), yaw=yaw, pitch=-0.045, fov=0.80)
        cameras.append(camera.grounded(terrain, clearance=115.0))
    return cameras


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=float, default=3.0)
    parser.add_argument("--keyframes", type=int, default=44)
    parser.add_argument("--steps", type=int, default=190)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    cache_dir = DATA_ROOT / "derivatives" / "emotion_features"
    available = sorted(p.name.split("_")[0] for p in cache_dir.glob("*_features.npz"))
    if not available:
        raise SystemExit("no cached emotion features; run tools/fetch_emotion.py")
    subject = args.subject or available[0]

    trials = dataset.load(dataset.cache_path(DATA_ROOT, subject))
    track = build_track(trials, subject)
    print(
        f"{subject}: {track.values.shape[0]} state windows over "
        f"{track.times[-1]:.0f}s, {len(set(track.trial.tolist()))} trials"
    )

    terrain = Terrain(amplitude=1150, wavelength=1500, ridged=0.95, octaves=10, seed=args.seed)
    base_water = sea_level(terrain)
    cameras = _path_cameras(terrain, args.keyframes)

    print(f"building {len(cameras)} geometry keyframes…")
    buffers = []
    for k, camera in enumerate(cameras):
        buffers.append(
            geometry(VIEW_W, VIEW_H, terrain, camera, Sky(), steps=args.steps, far=19000)
        )
        if k % 8 == 0:
            print(f"  {k}/{len(cameras)}")

    total_frames = int(args.seconds * FPS)

    # Walk the state track at a rate set by arousal, and accumulate travel so
    # the viewpoint advances faster where the recording is more alert.
    n_windows = track.values.shape[0]
    window_at = np.linspace(0, n_windows - 1, total_frames)

    rates = np.array([travel_rate(track, int(round(w))) for w in window_at])
    travel = np.cumsum(rates)
    travel = travel / travel[-1] * (len(buffers) - 1)

    from PIL import Image, ImageDraw
    import imageio.v2 as imageio

    small, medium = _font(13), _font(17)
    out = Path(args.out) if args.out else OUT_DIR / f"{subject}_memory_of_place.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        out,
        fps=FPS,
        codec="libx264",
        quality=7,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "22"],
    )

    print("shading…")
    try:
        for frame_index in range(total_frames):
            window = int(round(window_at[frame_index]))
            buffer = buffers[int(round(travel[frame_index]))]
            grade = grade_from_state(track, window, base_water)

            image = shade(buffer, grade)

            frame = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
            draw = ImageDraw.Draw(frame)
            frame.paste(
                Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8)),
                (VIEW_X, VIEW_Y),
            )
            draw.rectangle(
                [VIEW_X - 1, VIEW_Y - 1, VIEW_X + VIEW_W, VIEW_Y + VIEW_H],
                outline=(38, 46, 58),
            )

            draw.text((32, 18), "A MEMORY OF PLACE, FROM BRAIN STATE", fill=INK_FAINT, font=small)
            draw.text(
                (32, 40),
                f"{subject} · {track.times[window]:6.1f}s · continuous EEG",
                fill=INK,
                font=medium,
            )
            draw.text(
                (WIDTH - 300, 24),
                f"travel x{travel_rate(track, window):.2f}",
                fill=INK_MUTED,
                font=small,
            )

            # --- state channels ---------------------------------------------
            ry = VIEW_Y + VIEW_H + 22
            draw.text((32, ry), "BRAIN STATE DRIVING THE SCENE", fill=INK_FAINT, font=small)
            draw.text(
                (330, ry), "measured", fill=MEASURED, font=small
            )
            draw.text(
                (430, ry), "interpretation", fill=INTERPRETED, font=small
            )

            for slot, channel in enumerate(CHANNELS):
                column, row = slot % 4, slot // 4
                cx = 32 + column * 268
                cy = ry + 22 + row * 40
                colour = MEASURED if channel.status == "measured" else INTERPRETED

                value = track.at(window, channel.feature)
                draw.text((cx, cy), channel.label[:13], fill=colour, font=small)

                # A signed bar around a fixed centre, in standard deviations.
                axis = cx + 162
                draw.line([axis - 48, cy + 8, axis + 48, cy + 8], fill=(40, 48, 60), width=1)
                extent = float(np.clip(value, -2.5, 2.5)) * 18
                draw.rectangle(
                    [min(axis, axis + extent), cy + 4, max(axis, axis + extent), cy + 12],
                    fill=colour,
                )
                draw.line([axis, cy + 1, axis, cy + 15], fill=(96, 106, 120), width=1)

            draw.text(
                (32, HEIGHT - 40),
                "Band powers and intensity are read directly from the recording, not inferred. "
                "The cognitive readings are convention.",
                fill=INK_MUTED,
                font=small,
            )
            draw.text(
                (32, HEIGHT - 22),
                "The terrain is a fixed prior. This subject was watching video in a lab, "
                "not on a mountain: the place is not decoded, the state is.",
                fill=INK_FAINT,
                font=small,
            )

            writer.append_data(np.asarray(frame))

            if frame_index % 250 == 0:
                print(f"  frame {frame_index}/{total_frames}")
    finally:
        writer.close()

    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
