"""Reconstruct someone's gameplay from their EEG, and render it as video.

Validates leave-one-subject-out, then writes a real `.mp4` replaying the
session: what actually happened on the top track, what the brain activity
alone recovered on the bottom, side by side in time.

Run with::

    backend/.venv/bin/python -m tools.reconstruct_gameplay
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="  %(message)s")

from app.gameplay.decode import (  # noqa: E402
    Features,
    Reconstruction,
    balanced_accuracy,
    build,
    classify_pair,
    reconstruct,
)
from app.gameplay.events import DATA_ROOT, available_subjects, load_or_extract  # noqa: E402

WIDTH, HEIGHT, FPS = 1100, 560, 25

COLOURS = {
    "reward": (90, 210, 150),
    "crash": (228, 96, 88),
    "shoot": (96, 170, 235),
}
BACKGROUND = (10, 13, 22)
INK = (238, 242, 249)
INK_MUTED = (150, 160, 180)
INK_FAINT = (92, 102, 122)

LABELS = {
    "reward": "COLLECTED",
    "crash": "CRASHED",
    "shoot": "FIRED",
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
                return (
                    ImageFont.truetype(path, 13),
                    ImageFont.truetype(path, 19),
                    ImageFont.truetype(path, 40),
                )
            except OSError:
                continue
    default = ImageFont.load_default()
    return default, default, default


def _frame(rec: Reconstruction, t: float, window: float, fonts, accuracy: float):
    """One rendered frame at session time ``t``."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    small, medium, large = fonts

    draw.text((28, 20), "GAMEPLAY RECONSTRUCTED FROM EEG", fill=INK_FAINT, font=small)
    draw.text((28, 42), f"{rec.subject} · Escape from Asteroid Axon", fill=INK, font=medium)
    draw.text((WIDTH - 190, 26), f"{t:6.1f}s / {rec.duration:.0f}s", fill=INK_MUTED, font=small)

    # --- Live readout: the most recent event before now -------------------
    past = np.nonzero(rec.onsets <= t)[0]
    if past.size:
        i = int(past[-1])
        age = t - rec.onsets[i]
        if age < 1.2:
            actual, predicted = rec.actual[i], rec.predicted[i]
            hit = actual == predicted

            draw.text((28, 110), "BRAIN SAYS", fill=INK_FAINT, font=small)
            draw.text(
                (28, 132),
                LABELS.get(predicted, predicted.upper()),
                fill=COLOURS.get(predicted, INK),
                font=large,
            )
            draw.text((28, 190), f"confidence {rec.confidence[i] * 100:.0f}%", fill=INK_MUTED, font=small)

            draw.text((360, 110), "ACTUALLY HAPPENED", fill=INK_FAINT, font=small)
            draw.text(
                (360, 132),
                LABELS.get(actual, actual.upper()),
                fill=COLOURS.get(actual, INK),
                font=large,
            )
            draw.text(
                (360, 190),
                "correct" if hit else "wrong",
                fill=(90, 210, 150) if hit else (228, 96, 88),
                font=small,
            )

    # --- Scrolling tracks -------------------------------------------------
    x0, x1 = 28, WIDTH - 28
    span = x1 - x0
    lo, hi = t - window * 0.75, t + window * 0.25

    for row, (title, series) in enumerate(
        [("ACTUAL", rec.actual), ("RECONSTRUCTED FROM EEG", rec.predicted)]
    ):
        y = 260 + row * 110
        draw.text((x0, y - 22), title, fill=INK_FAINT, font=small)
        draw.rectangle([x0, y, x1, y + 62], fill=(18, 22, 32))

        visible = np.nonzero((rec.onsets >= lo) & (rec.onsets <= hi))[0]
        for i in visible:
            frac = (rec.onsets[i] - lo) / max(hi - lo, 1e-6)
            x = x0 + frac * span
            category = series[i]
            colour = COLOURS.get(category, INK_MUTED)

            # Mark disagreements on the reconstructed track so errors are
            # visible rather than buried in an aggregate.
            wrong = row == 1 and rec.actual[i] != rec.predicted[i]
            height = 44 if not wrong else 30
            draw.rectangle([x - 3, y + 62 - height, x + 3, y + 58], fill=colour)
            if wrong:
                draw.line([x - 6, y + 12, x + 6, y + 24], fill=(228, 96, 88), width=2)
                draw.line([x - 6, y + 24, x + 6, y + 12], fill=(228, 96, 88), width=2)

    # Playhead across both tracks.
    head = x0 + 0.75 * span
    draw.line([head, 250, head, 445], fill=INK, width=2)

    # --- Footer -----------------------------------------------------------
    draw.text(
        (28, HEIGHT - 62),
        f"Balanced accuracy {accuracy * 100:.0f}% over 3 classes (chance 33%). "
        "Leave-one-subject-out: this player's data was never used to fit the model.",
        fill=INK_MUTED,
        font=small,
    )
    draw.text(
        (28, HEIGHT - 38),
        "Event timing comes from the game log. What is decoded from EEG is which "
        "event each moment was.",
        fill=INK_FAINT,
        font=small,
    )

    legend_x = WIDTH - 300
    for i, (name, colour) in enumerate(COLOURS.items()):
        draw.rectangle(
            [legend_x + i * 95, HEIGHT - 60, legend_x + i * 95 + 10, HEIGHT - 50], fill=colour
        )
        draw.text((legend_x + i * 95 + 16, HEIGHT - 62), LABELS[name], fill=INK_MUTED, font=small)

    return np.asarray(image)


def render(rec: Reconstruction, path: Path, start: float, length: float, accuracy: float) -> Path:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    fonts = _fonts()

    writer = imageio.get_writer(
        path, fps=FPS, codec="libx264", quality=8, macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    try:
        for index in range(int(length * FPS)):
            writer.append_data(_frame(rec, start + index / FPS, 24.0, fonts, accuracy))
    finally:
        writer.close()
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="")
    parser.add_argument("--seconds", type=float, default=45.0)
    args = parser.parse_args()

    subjects = available_subjects(DATA_ROOT)
    if len(subjects) < 2:
        print("Need at least 2 subjects. Run: python -m tools.fetch_gameplay")
        return 1

    print("=" * 78)
    print("Gameplay reconstruction from EEG — ds003517 (Cavanagh et al. 2016)")
    print("=" * 78)

    features: Dict[str, Features] = {}
    for subject in subjects:
        features[subject] = build(load_or_extract(DATA_ROOT, subject))

    total = sum(f.x.shape[0] for f in features.values())
    print(f"{len(subjects)} subjects · {total} events · {features[subjects[0]].x.shape[1]} features")
    print("Leave-one-subject-out; balanced accuracy throughout.\n")

    rng = np.random.default_rng(5)
    results: Dict[str, List[float]] = {"reward_crash": [], "shoot_reward": [], "three_way": []}
    controls: Dict[str, List[float]] = {"reward_crash": [], "three_way": []}
    reconstructions: Dict[str, Reconstruction] = {}

    for held_out in subjects:
        train = [features[s] for s in subjects if s != held_out]
        test = features[held_out]

        pair = classify_pair(train, test, "reward", "crash")
        results["reward_crash"].append(pair["accuracy"])
        controls["reward_crash"].append(
            classify_pair(train, test, "reward", "crash", shuffle=True, rng=rng)["accuracy"]
        )
        results["shoot_reward"].append(
            classify_pair(train, test, "shoot", "reward")["accuracy"]
        )

        rec = reconstruct(train, test)
        reconstructions[held_out] = rec
        results["three_way"].append(rec.accuracy)

        shuffled = reconstruct(
            [
                Features(
                    x=f.x,
                    categories=rng.permutation(f.categories),
                    onsets=f.onsets,
                    subject=f.subject,
                    duration=f.duration,
                )
                for f in train
            ],
            test,
        )
        controls["three_way"].append(shuffled.accuracy)

        print(
            f"  {held_out}  reward/crash {pair['accuracy'] * 100:5.1f}%   "
            f"shoot/reward {results['shoot_reward'][-1] * 100:5.1f}%   "
            f"3-way {rec.accuracy * 100:5.1f}%   (n={rec.actual.size})"
        )

    print()
    print("  Across all subjects (balanced accuracy)")
    print(
        f"    reward vs crash   {np.mean(results['reward_crash']) * 100:5.1f}%   "
        f"(chance 50%, shuffled {np.mean(controls['reward_crash']) * 100:.1f}%)"
    )
    print(
        f"    shoot vs reward   {np.mean(results['shoot_reward']) * 100:5.1f}%   (chance 50%)"
    )
    print(
        f"    three-way         {np.mean(results['three_way']) * 100:5.1f}%   "
        f"(chance 33%, shuffled {np.mean(controls['three_way']) * 100:.1f}%)"
    )

    # --- Render -----------------------------------------------------------
    subject = args.subject or max(reconstructions, key=lambda s: reconstructions[s].accuracy)
    rec = reconstructions[subject]

    # Start where the action is densest, so the clip is representative.
    if rec.onsets.size > 40:
        counts = [
            ((rec.onsets >= o) & (rec.onsets < o + args.seconds)).sum()
            for o in rec.onsets[:-20]
        ]
        start = float(rec.onsets[int(np.argmax(counts))])
    else:
        start = float(rec.onsets[0])

    output = DATA_ROOT / "derivatives" / f"{subject}_gameplay_reconstruction.mp4"
    print(f"\n  rendering {subject} from {start:.0f}s ({args.seconds:.0f}s)…")
    render(rec, output, start, args.seconds, rec.accuracy)
    print(f"  wrote {output}  ({output.stat().st_size / 1e6:.1f} MB)")

    summary = DATA_ROOT / "derivatives" / "gameplay_results.json"
    summary.write_text(
        json.dumps(
            {
                "subjects": subjects,
                "reward_crash": float(np.mean(results["reward_crash"])),
                "reward_crash_shuffled": float(np.mean(controls["reward_crash"])),
                "shoot_reward": float(np.mean(results["shoot_reward"])),
                "three_way": float(np.mean(results["three_way"])),
                "three_way_shuffled": float(np.mean(controls["three_way"])),
                "rendered": subject,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
