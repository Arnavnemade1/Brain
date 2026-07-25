"""Render a reconstruction as an actual video file.

Produces a real `.mp4`: one frame per moment of the clip, carrying the decoded
emotional state at that instant, the trajectory so far, and the self-reported
ground truth for comparison.

What this video is, precisely: a *per-frame emotional reconstruction* of a
viewing. It is not a reconstruction of the images the person saw — scalp EEG
cannot support that, as the identification work in `REAL_DATA.md` measures.
Every frame is labelled with the confidence the validation actually supports,
so the file cannot be mistaken for more than it is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("mindscape.emotion.video")

WIDTH = 960
HEIGHT = 540
FPS = 25

#: Circumplex colours. Warm for high arousal, cool for low; saturated for
#: extreme valence, desaturated near neutral.
BACKGROUND = (10, 13, 22)
INK = (238, 242, 249)
INK_MUTED = (150, 160, 180)
INK_FAINT = (92, 102, 122)


def _emotion_colour(valence: float, arousal: float) -> Tuple[int, int, int]:
    """Map a circumplex point to a colour.

    Valence drives hue (amber through teal), arousal drives brightness. The
    mapping is presentational only — nothing downstream reads it.
    """
    v = float(np.clip(valence, -1, 1))
    a = float(np.clip(arousal, -1, 1))

    negative = np.array([224, 96, 88], dtype=float)   # warm red
    neutral = np.array([120, 130, 155], dtype=float)  # slate
    positive = np.array([90, 200, 170], dtype=float)  # teal

    if v >= 0:
        base = neutral + (positive - neutral) * v
    else:
        base = neutral + (negative - neutral) * (-v)

    brightness = 0.55 + 0.45 * (a + 1) / 2
    return tuple(int(np.clip(c * brightness, 0, 255)) for c in base)


@dataclass
class EmotionTrack:
    """The decoded emotional trajectory for one viewing."""

    times: np.ndarray
    valence: np.ndarray
    arousal: np.ndarray
    subject: str
    stimulus: str
    duration: float
    #: Self-reported ratings, mapped to -1..+1.
    reported_valence: float
    reported_arousal: float
    reported_emotion: str = ""
    click_times: Sequence[float] = ()
    #: How much the validation supports this track. Shown on every frame.
    confidence_note: str = ""

    def at(self, t: float) -> Tuple[float, float]:
        """Interpolated (valence, arousal) at a moment in the clip."""
        if self.times.size == 0:
            return 0.0, 0.0
        return (
            float(np.interp(t, self.times, self.valence)),
            float(np.interp(t, self.times, self.arousal)),
        )


def _draw(track: EmotionTrack, t: float, font) -> np.ndarray:
    """One rendered frame at clip time ``t``."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    valence, arousal = track.at(t)
    colour = _emotion_colour(valence, arousal)

    small, medium, large = font

    # --- Header ---------------------------------------------------------
    draw.text((28, 22), "MEMORY RECONSTRUCTION — EMOTIONAL TRACK", fill=INK_FAINT, font=small)
    draw.text((28, 44), f"{track.subject} · {track.stimulus}", fill=INK, font=medium)
    draw.text(
        (WIDTH - 150, 30), f"{t:5.1f}s / {track.duration:.0f}s", fill=INK_MUTED, font=small
    )

    # --- Circumplex -----------------------------------------------------
    cx, cy, radius = 250, 300, 150

    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius], outline=(46, 54, 72), width=1
    )
    draw.line([cx - radius, cy, cx + radius, cy], fill=(38, 45, 60), width=1)
    draw.line([cx, cy - radius, cx, cy + radius], fill=(38, 45, 60), width=1)

    draw.text((cx + radius + 8, cy - 7), "pleasant", fill=INK_FAINT, font=small)
    draw.text((cx - radius - 62, cy - 7), "unpleasant", fill=INK_FAINT, font=small)
    draw.text((cx - 22, cy - radius - 20), "aroused", fill=INK_FAINT, font=small)
    draw.text((cx - 16, cy + radius + 8), "calm", fill=INK_FAINT, font=small)

    # Trajectory up to now, fading into the past.
    visible = track.times <= t
    if visible.sum() > 1:
        points = [
            (cx + track.valence[i] * radius, cy - track.arousal[i] * radius)
            for i in np.nonzero(visible)[0]
        ]
        for i in range(1, len(points)):
            fade = i / len(points)
            shade = tuple(int(c * (0.25 + 0.55 * fade)) for c in colour)
            draw.line([points[i - 1], points[i]], fill=shade, width=2)

    # Current position.
    px = cx + valence * radius
    py = cy - arousal * radius
    draw.ellipse([px - 16, py - 16, px + 16, py + 16], outline=colour, width=1)
    draw.ellipse([px - 7, py - 7, px + 7, py + 7], fill=colour)

    # Reported ground truth, as a hollow marker.
    rx = cx + track.reported_valence * radius
    ry = cy - track.reported_arousal * radius
    draw.ellipse([rx - 9, ry - 9, rx + 9, ry + 9], outline=(150, 160, 180), width=2)
    draw.text((rx + 12, ry - 8), "reported", fill=INK_FAINT, font=small)

    # --- Readouts -------------------------------------------------------
    from .decoder import label_for, quadrant_of

    x0 = 470
    draw.text((x0, 150), "DECODED NOW", fill=INK_FAINT, font=small)
    draw.text((x0, 172), label_for(valence, arousal).upper(), fill=colour, font=large)

    for i, (name, value) in enumerate(
        [("valence", valence), ("arousal", arousal)]
    ):
        y = 235 + i * 62
        draw.text((x0, y), name, fill=INK_FAINT, font=small)
        draw.text((x0, y + 18), f"{value:+.2f}", fill=INK, font=medium)

        # Bar from centre.
        bar_x, bar_w = x0 + 110, 300
        draw.rectangle([bar_x, y + 20, bar_x + bar_w, y + 30], fill=(30, 36, 48))
        mid = bar_x + bar_w / 2
        draw.line([mid, y + 16, mid, y + 34], fill=(60, 68, 86), width=1)
        extent = value * bar_w / 2
        draw.rectangle(
            [min(mid, mid + extent), y + 20, max(mid, mid + extent), y + 30], fill=colour
        )

    draw.text((x0, 365), "QUADRANT", fill=INK_FAINT, font=small)
    draw.text((x0, 385), quadrant_of(valence, arousal), fill=INK, font=medium)

    if track.reported_emotion:
        draw.text((x0 + 160, 365), "SUBJECT REPORTED", fill=INK_FAINT, font=small)
        draw.text((x0 + 160, 385), track.reported_emotion, fill=INK_MUTED, font=medium)

    # --- Timeline -------------------------------------------------------
    ty, tx0, tx1 = 460, 28, WIDTH - 28
    draw.rectangle([tx0, ty, tx1, ty + 26], fill=(20, 25, 36))

    if track.times.size > 1:
        for i in range(track.times.size):
            frac = track.times[i] / max(track.duration, 1e-6)
            x = tx0 + frac * (tx1 - tx0)
            band = _emotion_colour(track.valence[i], track.arousal[i])
            width_px = max(2, (tx1 - tx0) / track.times.size + 1)
            draw.rectangle([x, ty, x + width_px, ty + 26], fill=band)

    # Playhead and click markers.
    head = tx0 + (t / max(track.duration, 1e-6)) * (tx1 - tx0)
    draw.line([head, ty - 5, head, ty + 31], fill=INK, width=2)

    for click in track.click_times:
        x = tx0 + (click / max(track.duration, 1e-6)) * (tx1 - tx0)
        draw.line([x, ty - 9, x, ty + 35], fill=(240, 200, 90), width=2)
        draw.text((x - 12, ty - 26), "felt", fill=(240, 200, 90), font=small)

    # --- Honesty footer --------------------------------------------------
    if track.confidence_note:
        draw.text((28, HEIGHT - 32), track.confidence_note, fill=INK_FAINT, font=small)

    return np.asarray(image)


def _load_fonts():
    from PIL import ImageFont

    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return (
                    ImageFont.truetype(path, 13),
                    ImageFont.truetype(path, 20),
                    ImageFont.truetype(path, 34),
                )
            except OSError:
                continue

    default = ImageFont.load_default()
    return default, default, default


def render(track: EmotionTrack, path: Path, fps: int = FPS) -> Path:
    """Write the reconstruction to an MP4."""
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    fonts = _load_fonts()

    n_frames = max(1, int(round(track.duration * fps)))

    writer = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    try:
        for index in range(n_frames):
            t = index / fps
            writer.append_data(_draw(track, t, fonts))
    finally:
        writer.close()

    log.info("wrote %s (%d frames, %.1fs)", path.name, n_frames, track.duration)
    return path
