"""Render a real-life memory reconstruction as a video.

Three panels moving together in the order the person actually saw them: the
photograph that was on screen, what one viewing of it recovered, and what
eight held-out viewings recovered. Underneath, the ranked candidate set and a
running trace of how close each reconstruction was in meaning.

Showing one viewing next to eight is the honest framing. One viewing is what a
headset would have in real time, and it is wrong most of the time. Eight is
what repeated exposure buys, which is a real effect and a large one — and it
is also why this is a reconstruction of a *revisited* memory rather than a
live read.

Nothing is drawn that was not decoded. Where the retrieval is uncertain the
panel is veiled rather than replaced, so the picture is always the system's
actual answer.

Run with::

    backend/.venv/bin/python -m tools.render_reallife_video --seconds 60
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.realdata.epochs import available_subjects
from app.realdata.images import thumbnail
from app.realdata.stream import Moment, StreamReconstruction, reconstruct_stream

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "ds004018"
OUT_DIR = DATA_ROOT / "derivatives"

WIDTH, HEIGHT, FPS = 1120, 620, 25

#: Presentations are 200 ms apart. Held for twice that so the labels can be
#: read; the header says so rather than letting the viewer assume real time.
FRAMES_PER_MOMENT = 10

BACKGROUND = (9, 11, 16)
INK = (238, 241, 244)
INK_MUTED = (150, 160, 172)
INK_FAINT = (104, 114, 126)
TRUTH_TINT = (150, 200, 235)
CORRECT = (104, 214, 158)
WRONG = (198, 92, 84)

#: Outline colour and weight per semantic tier. A different photograph of the
#: same object is a good answer, not a miss, and marking it the same grey as
#: an unrelated object was hiding most of what the decoder gets right.
TIER_STYLE = {
    "exact": ((104, 214, 158), 2),
    "concept": ((118, 188, 236), 2),
    "category": ((96, 116, 146), 1),
    "animacy": ((58, 68, 82), 1),
    "unrelated": ((40, 46, 58), 1),
}

PANEL_W = 328
PANEL_GAP = 36
IMAGE = 216
#: Thumbnail size for the ranked candidate strip.
CHIP = 44

# Explicit vertical budget. The first version let each block choose its own
# offset and the trace ended up drawn through the footer.
TOP = 108
Y_LABEL = TOP + IMAGE + 12          # 336
Y_NOTE = Y_LABEL + 24               # 360
Y_RANK_TITLE = Y_NOTE + 26          # 386
Y_CHIPS = Y_RANK_TITLE + 18         # 404
Y_RANK = Y_CHIPS + CHIP + 6         # 454
Y_TRACE_TITLE = Y_RANK + 22         # 476
Y_TRACE = Y_TRACE_TITLE + 16        # 492
TRACE_H = 48
Y_FOOTER = Y_TRACE + TRACE_H + 14   # 554

#: Evidence z-scores span roughly this range; dimming is normalised to it so
#: the panel brightness actually varies instead of sitting at full for
#: everything. See `solid_confidence_auc` for how much it is worth.
EVIDENCE_LO, EVIDENCE_HI = 2.0, 2.8


def _font(size: int):
    from PIL import ImageFont

    for name in (
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/Library/Fonts/Menlo.ttc",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _panel_x(index: int) -> int:
    left = (WIDTH - (3 * PANEL_W + 2 * PANEL_GAP)) // 2
    return left + index * (PANEL_W + PANEL_GAP)


def _paste(image, array: np.ndarray, x: int, y: int, dim: float = 1.0) -> None:
    """Draw an RGB float array, optionally darkened."""
    from PIL import Image

    scaled = np.clip(array * dim, 0, 1)
    image.paste(Image.fromarray((scaled * 255).astype(np.uint8)), (x, y))


def _evidence(z: float) -> float:
    """Evidence z-score mapped to 0-1 over its observed range."""
    return float(np.clip((z - EVIDENCE_LO) / (EVIDENCE_HI - EVIDENCE_LO), 0.0, 1.0))


def _veil(draw, x: int, y: int, size: int, confidence: float) -> None:
    """Scanlines over an uncertain reconstruction.

    Drawn *over* the retrieved image, never in place of it: the veil qualifies
    the answer, it does not replace it with a blank.
    """
    if confidence >= 0.95:
        return
    gap = max(2, int(2 + (1.0 - confidence) * 6))
    for row in range(y + 1, y + size, gap):
        draw.line([x + 1, row, x + size - 1, row], fill=BACKGROUND, width=1)


class _Thumbs:
    """Stimulus thumbnails, loaded once each."""

    def __init__(self, stimuli_dir: Path, size: int) -> None:
        self.dir = stimuli_dir
        self.size = size
        self._cache: Dict[str, np.ndarray] = {}

    def get(self, filename: str) -> Optional[np.ndarray]:
        if not filename:
            return None
        if filename not in self._cache:
            path = self.dir / filename
            if not path.exists():
                return None
            self._cache[filename] = thumbnail(path, size=self.size)
        return self._cache[filename]


def _filename_for(number: int, catalogue: Dict[int, object]) -> str:
    entry = catalogue.get(int(number))
    return entry.filename if entry is not None else ""


def _draw_frame(
    moments: Sequence[Moment],
    position: int,
    thumbs: _Thumbs,
    chips: _Thumbs,
    catalogue: Dict[int, object],
    space,
    small,
    medium,
    large,
    metrics: Dict[str, float],
    subject: str,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    moment = moments[position]

    # --- header -------------------------------------------------------------
    draw.text((32, 20), "MEMORY RECONSTRUCTION · REAL-WORLD IMAGERY", fill=INK_FAINT, font=small)
    draw.text((32, 42), f"{subject} · rapid serial viewing", fill=INK, font=medium)
    draw.text(
        (WIDTH - 300, 24),
        f"{moment.onset:7.1f}s   half speed",
        fill=INK_MUTED,
        font=small,
    )

    titles = ("WHAT THEY SAW", "FROM ONE VIEWING", "FROM EIGHT VIEWINGS")
    colours = (TRUTH_TINT, INK_FAINT, INK_FAINT)
    entries = (
        (moment.truth_filename, moment.truth_label, 1.0, None),
        (moment.live_filename, moment.live_label, moment.live_confidence, moment.live_rank),
        (moment.solid_filename, moment.solid_label, moment.solid_confidence, moment.solid_rank),
    )

    for index, (title, colour, (filename, label, confidence, rank)) in enumerate(
        zip(titles, colours, entries)
    ):
        px = _panel_x(index)
        draw.text((px, TOP - 22), title, fill=colour, font=small)

        ix = px + (PANEL_W - IMAGE) // 2
        draw.rectangle([ix - 1, TOP - 1, ix + IMAGE, TOP + IMAGE], outline=(30, 36, 46))

        art = thumbs.get(filename)
        if art is None:
            draw.text((ix + 12, TOP + IMAGE // 2), "not decoded", fill=INK_FAINT, font=small)
        else:
            # Evidence dims the two reconstruction panels; the photograph they
            # actually saw is never dimmed.
            strength = 1.0 if index == 0 else _evidence(confidence)
            _paste(image, art, ix, TOP, dim=0.5 + 0.5 * strength)
            if index > 0:
                _veil(draw, ix, TOP, IMAGE, strength)

        # Label, and whether this panel got it right.
        if index == 0:
            draw.text((px, Y_LABEL), label, fill=INK, font=medium)
        else:
            hit = rank == 0
            draw.text((px, Y_LABEL), label or "—", fill=CORRECT if hit else INK_MUTED, font=medium)
            note = "correct" if hit else f"true object ranked {rank + 1} of 200"
            draw.text((px, Y_NOTE), note, fill=CORRECT if hit else WRONG, font=small)

    # --- ranked candidates from the consolidated track ----------------------
    # Shown as pictures, not just names: the true object lands in this strip
    # far more often than it wins outright, and a text list hides that.
    draw.text(
        (32, Y_RANK_TITLE),
        "RANKED CANDIDATES (EIGHT VIEWINGS)   green = the exact photo   "
        "blue = the same object, different photo",
        fill=INK_FAINT,
        font=small,
    )
    for slot, (number, label, _probability) in enumerate(moment.solid_top[:5]):
        cx = 32 + slot * 211
        tier = space.tier(moment.truth_number, number)
        colour, weight = TIER_STYLE[tier]

        chip = chips.get(_filename_for(number, catalogue))
        if chip is not None:
            _paste(image, chip, cx, Y_CHIPS, dim=1.0)
        draw.rectangle(
            [cx - 1, Y_CHIPS - 1, cx + CHIP, Y_CHIPS + CHIP],
            outline=colour,
            width=weight,
        )
        draw.text(
            (cx + CHIP + 8, Y_CHIPS + CHIP // 2 - 8),
            f"{slot + 1}. {label[:12]}",
            fill=colour if tier in ("exact", "concept") else INK_MUTED,
            font=small,
        )

    # --- running semantic similarity ---------------------------------------
    draw.text(
        (32, Y_TRACE_TITLE), "SEMANTIC SIMILARITY TO WHAT WAS SEEN", fill=INK_FAINT, font=small
    )

    x0, x1 = 32, WIDTH - 190
    window = 90
    lo = max(0, position - window)
    span = x1 - x0
    chance = metrics.get("chance_similarity", 0.137)

    draw.rectangle([x0, Y_TRACE, x1, Y_TRACE + TRACE_H], fill=(16, 20, 28))
    chance_row = Y_TRACE + TRACE_H - int(chance * TRACE_H)
    draw.line([x0, chance_row, x1, chance_row], fill=(70, 78, 90), width=1)

    for series, colour in (
        ([m.live_similarity for m in moments[lo : position + 1]], (120, 140, 165)),
        ([m.solid_similarity for m in moments[lo : position + 1]], CORRECT),
    ):
        if len(series) < 2:
            continue
        points = [
            (
                x0 + span * k / max(window, 1),
                Y_TRACE + TRACE_H - int(np.clip(value, 0, 1) * TRACE_H),
            )
            for k, value in enumerate(series)
        ]
        draw.line(points, fill=colour, width=2)

    draw.text((x1 + 8, Y_TRACE - 2), "8 views", fill=CORRECT, font=small)
    draw.text((x1 + 8, Y_TRACE + 18), "1 view", fill=(120, 140, 165), font=small)
    draw.text((x1 + 8, Y_TRACE + 38), "chance", fill=INK_FAINT, font=small)

    # --- footer -------------------------------------------------------------
    # Kept under ~135 characters a line: at 13pt monospace anything longer
    # runs off the right edge of the frame.
    lines = [
        (
            f"Held out throughout, {int(metrics['candidates'])} candidates. "
            f"One viewing: {metrics['live_exact'] * 100:.1f}% exact, "
            f"{metrics['live_top5'] * 100:.1f}% top-5. "
            f"Eight viewings: {metrics['solid_exact'] * 100:.1f}% exact, "
            f"{metrics['solid_top5'] * 100:.1f}% top-5.",
            INK_MUTED,
        ),
        (
            f"Chance {metrics['chance_exact'] * 100:.1f}% / {metrics['chance_top5'] * 100:.1f}%. "
            f"The same object appears in the top 5 "
            f"{metrics['solid_top5_concept'] * 100:.1f}% of the time (chance 10%). "
            "Shuffled labels score at chance.",
            INK_MUTED,
        ),
        (
            "Retrieval over real photographs, not generated pixels. Semantic similarity "
            f"{metrics['solid_similarity']:.3f} vs {metrics['chance_similarity']:.3f} chance; "
            f"brightness tracks evidence (AUC {metrics['solid_confidence_auc']:.2f}).",
            INK_FAINT,
        ),
    ]
    for row, (text, colour) in enumerate(lines):
        draw.text((32, Y_FOOTER + row * 20), text, fill=colour, font=small)

    return np.asarray(image)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    subjects = available_subjects(DATA_ROOT)
    if not subjects:
        raise SystemExit("no subjects downloaded; run tools/fetch_dataset.py")
    subject = args.subject or subjects[0]

    print(f"reconstructing {subject}…")
    stream = reconstruct_stream(DATA_ROOT, subject)
    metrics = stream.metrics

    print(
        f"  one viewing   exact {metrics['live_exact']*100:5.1f}%  "
        f"top5 {metrics['live_top5']*100:5.1f}%  sim {metrics['live_similarity']:.3f}"
    )
    print(
        f"  eight viewings exact {metrics['solid_exact']*100:5.1f}%  "
        f"top5 {metrics['solid_top5']*100:5.1f}%  sim {metrics['solid_similarity']:.3f}"
    )
    print(f"  chance         exact {metrics['chance_exact']*100:5.1f}%  "
          f"top5 {metrics['chance_top5']*100:5.1f}%  sim {metrics['chance_similarity']:.3f}")

    n_moments = int(args.seconds * FPS / FRAMES_PER_MOMENT)
    moments = stream.moments[:n_moments]
    if not moments:
        raise SystemExit("no moments reconstructed")

    from app.realdata.images import load_catalogue
    from app.realdata.semantics import SemanticSpace

    catalogue = load_catalogue(DATA_ROOT / "stimuli")
    space = SemanticSpace.from_catalogue(catalogue)
    thumbs = _Thumbs(DATA_ROOT / "stimuli", IMAGE)
    chips = _Thumbs(DATA_ROOT / "stimuli", CHIP)
    small, medium, large = _font(13), _font(17), _font(22)

    out = Path(args.out) if args.out else OUT_DIR / f"{subject}_reallife_video.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    import imageio.v2 as imageio

    writer = imageio.get_writer(
        out,
        fps=FPS,
        codec="libx264",
        quality=7,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "23"],
    )
    try:
        for position in range(len(moments)):
            frame = _draw_frame(
                moments, position, thumbs, chips, catalogue, space,
                small, medium, large, metrics, subject
            )
            for _ in range(FRAMES_PER_MOMENT):
                writer.append_data(frame)
    finally:
        writer.close()

    size = out.stat().st_size / 1e6
    print(f"\nwrote {out}  ({size:.1f} MB, {len(moments)} moments)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
