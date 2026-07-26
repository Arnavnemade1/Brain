"""Reconstruct a memory as a continuous video.

Slides a window across a whole recording, decodes what was happening at every
moment from EEG alone, and renders the world twice side by side: as it
actually was, and as the brain activity recovered it.

No event log is used at inference. The decoder is not told when anything
happened — it decides at every step whether the player is firing, being
rewarded, crashing, or nothing at all.

Run with::

    backend/.venv/bin/python -m tools.render_memory_video --seconds 60
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="  %(message)s")

from app.gameplay.continuous import (  # noqa: E402
    CLASSES,
    ContinuousScene,
    build_windows,
    reconstruct_session,
)
from app.gameplay.events import DATA_ROOT, available_subjects, paths_for, read_events  # noqa: E402
from app.gameplay.scene import (  # noqa: E402
    PANEL_H,
    PANEL_W,
    draw_scene,
    state_from_decoded,
    state_from_events,
)

WIDTH, HEIGHT, FPS = 1120, 620, 25

INK = (238, 242, 249)
INK_MUTED = (150, 160, 180)
INK_FAINT = (92, 102, 122)
BACKGROUND = (10, 13, 22)

CLASS_COLOUR = {
    "idle": (70, 80, 100),
    "shoot": (96, 170, 235),
    "reward": (96, 214, 156),
    "crash": (232, 104, 96),
}


def _fonts():
    from PIL import ImageFont

    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, 13), ImageFont.truetype(path, 18)
            except OSError:
                continue
    default = ImageFont.load_default()
    return default, default


def _frame(
    t: float,
    onsets: np.ndarray,
    categories: np.ndarray,
    scene: ContinuousScene,
    fonts,
    stats: Dict[str, float],
) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    small, medium = fonts

    draw.text((28, 18), "MEMORY RECONSTRUCTION", fill=INK_FAINT, font=small)
    draw.text((28, 38), f"{scene.subject} · Escape from Asteroid Axon", fill=INK, font=medium)
    draw.text((WIDTH - 170, 24), f"{t:6.1f}s", fill=INK_MUTED, font=small)

    top = 82
    left_x, right_x = 28, WIDTH - PANEL_W - 28

    draw.text((left_x, top - 20), "WHAT HAPPENED", fill=INK_FAINT, font=small)
    draw.text((right_x, top - 20), "RECONSTRUCTED FROM EEG", fill=(150, 200, 235), font=small)

    draw_scene(
        draw, left_x, top, state_from_events(t, onsets, categories), False, small
    )
    draw_scene(
        draw,
        right_x,
        top,
        state_from_decoded(t, scene.times, scene.probabilities, CLASSES),
        True,
        small,
    )

    # --- Decoded class strip ---------------------------------------------
    strip_y = top + PANEL_H + 26
    draw.text((28, strip_y - 20), "DECODED STATE OVER TIME", fill=INK_FAINT, font=small)

    # The strip stops short of the right margin so the row titles drawn at
    # ``x1 + 8`` land inside the canvas. Running it to WIDTH - 28 pushed
    # "actual"/"decoded" off the edge, where they rendered as "actu"/"decc".
    x0, x1 = 28, WIDTH - 96
    span = x1 - x0
    window = 30.0
    lo, hi = t - window * 0.7, t + window * 0.3

    for row, (title, series) in enumerate(
        [("actual", scene.actual), ("decoded", scene.predicted)]
    ):
        y = strip_y + row * 40
        draw.rectangle([x0, y, x1, y + 30], fill=(18, 22, 32))
        draw.text((x0 - 0, y + 32), "", fill=INK_FAINT, font=small)

        visible = np.nonzero((scene.times >= lo) & (scene.times <= hi))[0]
        for i in visible:
            frac = (scene.times[i] - lo) / max(hi - lo, 1e-6)
            x = x0 + frac * span
            category = series[i]
            if category == "idle":
                continue
            colour = CLASS_COLOUR.get(category, INK_MUTED)
            width = max(3, span / max(visible.size, 1) * 0.8)
            draw.rectangle([x - width / 2, y + 4, x + width / 2, y + 26], fill=colour)

        draw.text((x1 + 8, y + 9), title, fill=INK_FAINT, font=small)

    head = x0 + 0.7 * span
    draw.line([head, strip_y - 4, head, strip_y + 74], fill=INK, width=2)

    # --- Footer -----------------------------------------------------------
    # The legend gets its own row above the caption. Both used to be drawn on
    # the same baseline, so the class swatches sat on top of the accuracy
    # figures and neither was readable.
    legend_y = HEIGHT - 78
    for i, (name, colour) in enumerate(CLASS_COLOUR.items()):
        lx = 28 + i * 96
        draw.rectangle([lx, legend_y, lx + 10, legend_y + 10], fill=colour)
        draw.text((lx + 15, legend_y - 2), name, fill=INK_MUTED, font=small)

    draw.text(
        (28, HEIGHT - 54),
        f"Continuous decoding, no event log at inference. "
        f"Balanced accuracy {stats['balanced'] * 100:.0f}% over 4 classes "
        f"(chance 25%). Event detection recall {stats['recall'] * 100:.0f}%, "
        f"precision {stats['precision'] * 100:.0f}%.",
        fill=INK_MUTED,
        font=small,
    )
    draw.text(
        (28, HEIGHT - 34),
        "Leave-one-subject-out. The right panel dims where the signal is weak; "
        "nothing is drawn that was not decoded.",
        fill=INK_FAINT,
        font=small,
    )

    return np.asarray(image)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="")
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()

    subjects = available_subjects(DATA_ROOT)
    if len(subjects) < 2:
        print("Need at least 2 subjects. Run: python -m tools.fetch_gameplay")
        return 1

    print("=" * 78)
    print("Continuous memory reconstruction — no event log at inference")
    print("=" * 78)

    # Sliding-window extraction reads a 350 MB recording per subject and
    # takes minutes; cache it so re-rendering is quick.
    cache: Dict[str, Tuple] = {}
    cache_dir = DATA_ROOT / "derivatives" / "continuous_windows"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for subject in subjects:
        cached = cache_dir / f"{subject}.npz"
        if cached.exists():
            blob = np.load(cached, allow_pickle=False)
            cache[subject] = (
                blob["times"],
                blob["x"].astype(np.float64),
                blob["y"].astype(object),
                float(blob["duration"]),
                [],
            )
            continue
        set_path, events_path = paths_for(DATA_ROOT, subject)
        times, x, y, duration, names = build_windows(set_path, events_path, subject)
        np.savez_compressed(
            cached,
            times=times,
            x=x.astype(np.float32),
            y=np.array([str(v) for v in y]),
            duration=duration,
        )
        cache[subject] = (times, x, y, duration, names)

    scenes: Dict[str, ContinuousScene] = {}
    summary: List[Dict] = []

    for held_out in subjects:
        train = [(cache[s][1], cache[s][2]) for s in subjects if s != held_out]
        times, x, y, duration, _ = cache[held_out]

        scene = reconstruct_session(train, x, y, times, held_out, duration)
        scenes[held_out] = scene

        recall, precision = scene.detection_rate()
        summary.append(
            {
                "subject": held_out,
                "balanced": scene.accuracy(),
                "event_only": scene.event_accuracy(),
                "recall": recall,
                "precision": precision,
            }
        )
        print(
            f"  {held_out}  4-class {scene.accuracy() * 100:5.1f}%   "
            f"events-only {scene.event_accuracy() * 100:5.1f}%   "
            f"detect r={recall * 100:4.1f}% p={precision * 100:4.1f}%"
        )

    mean = {
        key: float(np.mean([s[key] for s in summary]))
        for key in ("balanced", "event_only", "recall", "precision")
    }
    print()
    print(f"  4-class balanced   {mean['balanced'] * 100:5.1f}%   (chance 25%)")
    print(f"  events only        {mean['event_only'] * 100:5.1f}%   (chance 33%)")
    print(
        f"  detection          recall {mean['recall'] * 100:.1f}%  "
        f"precision {mean['precision'] * 100:.1f}%"
    )

    subject = args.subject or max(scenes, key=lambda s: scenes[s].accuracy())
    scene = scenes[subject]

    _, events_path = paths_for(DATA_ROOT, subject)
    events = read_events(events_path)
    onsets = np.array([e.onset for e in events])
    categories = np.array([e.category for e in events])

    # Start where the session is busiest, so the clip is representative.
    if onsets.size > 40:
        counts = [((onsets >= o) & (onsets < o + args.seconds)).sum() for o in onsets[:-20]]
        start = float(onsets[int(np.argmax(counts))])
    else:
        start = float(scene.times[0])

    stats = {
        "balanced": scene.accuracy(),
        "recall": scene.detection_rate()[0],
        "precision": scene.detection_rate()[1],
    }

    output = DATA_ROOT / "derivatives" / f"{subject}_memory_video.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n  rendering {subject} from {start:.0f}s ({args.seconds:.0f}s)…")

    import imageio.v2 as imageio

    fonts = _fonts()
    writer = imageio.get_writer(
        output, fps=FPS, codec="libx264", quality=8, macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    try:
        for index in range(int(args.seconds * FPS)):
            writer.append_data(
                _frame(start + index / FPS, onsets, categories, scene, fonts, stats)
            )
    finally:
        writer.close()

    print(f"  wrote {output}  ({output.stat().st_size / 1e6:.1f} MB)")

    (DATA_ROOT / "derivatives" / "continuous_results.json").write_text(
        json.dumps({"per_subject": summary, "mean": mean, "rendered": subject}, indent=1)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
