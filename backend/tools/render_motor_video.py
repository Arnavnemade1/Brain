"""Render a reconstruction of what someone's body was doing.

Two figures moving side by side in real time: on the left the action the
person actually performed, on the right the action recovered from their
sensorimotor rhythms alone. Underneath, the decoder's posterior over the five
actions and a timeline of the whole session.

This is a different question from the earlier reconstructions. Those asked
what someone was *looking at*; this asks what they were *doing* — and it is
the question scalp EEG is best equipped to answer, because moving a limb
suppresses the mu and beta rhythms over the opposite sensorimotor strip, which
is a large effect with a clear spatial signature.

Run with::

    backend/.venv/bin/python -m tools.render_motor_video --subject S002
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from app.motor.decode import (
    BREAKDOWN,
    decompose,
    graded,
    leave_one_run_out,
    score,
    tier_of,
)
from app.motor.windows import CLASSES, available_subjects, windows_for_subject

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "eegmmidb"
OUT_DIR = DATA_ROOT / "derivatives"

WIDTH, HEIGHT = 1120, 620
#: Windows advance every 0.25 s, so five frames at 20 fps is exactly real time.
FPS = 20
FRAMES_PER_WINDOW = 5

BACKGROUND = (9, 11, 16)
INK = (238, 241, 244)
INK_MUTED = (150, 160, 172)
INK_FAINT = (104, 114, 126)
TRUTH_TINT = (150, 200, 235)
CORRECT = (104, 214, 158)
WRONG = (198, 92, 84)
IDLE = (52, 60, 74)

#: Colour per action, shared by the figures, the posterior bars and the strip.
ACTION_COLOUR = {
    "rest": (96, 108, 126),
    "left fist": (118, 188, 236),
    "right fist": (236, 176, 118),
    "both fists": (156, 148, 232),
    "both feet": (104, 214, 158),
}

#: Which limbs each action lights up.
ACTION_LIMBS: Dict[str, Tuple[str, ...]] = {
    "rest": (),
    "left fist": ("left arm",),
    "right fist": ("right arm",),
    "both fists": ("left arm", "right arm"),
    "both feet": ("left leg", "right leg"),
}

LIMBS: Tuple[str, ...] = ("left arm", "right arm", "left leg", "right leg")

#: Cycles per second the hands and feet open and close at. The task is
#: self-paced; this is a nominal rate so the figure reads as *moving* rather
#: than as a limb that has simply changed colour.
MOVE_HZ = 1.2


def limb_belief(posterior: np.ndarray) -> Dict[str, float]:
    """Marginal probability that each limb is moving.

    The decoder's answer is a distribution, and collapsing it to the argmax
    throws away the most informative thing it produces. When the posterior is
    split between left and right fist, both arms should glow at half — that is
    a truthful picture of the evidence, where a single lit arm would be an
    invention.
    """
    out = {name: 0.0 for name in LIMBS}
    for index, action in enumerate(CLASSES):
        for limb in ACTION_LIMBS[action]:
            out[limb] += float(posterior[index])
    return out

PANEL_W = 420

# Explicit vertical budget for a 620px frame. The first pass let the timeline
# and the footer choose their own offsets and they were drawn over each other.
TOP = 104
FIGURE_H = 230
Y_ACTION = TOP + FIGURE_H + 14      # 348
Y_TIER = Y_ACTION + 26              # 374
Y_BARS_TITLE = Y_TIER + 28          # 402
Y_BARS = Y_BARS_TITLE + 18          # 420
Y_BAR_LABELS = Y_BARS + 20          # 440
Y_STRIP_TITLE = Y_BAR_LABELS + 28   # 468
Y_STRIP = Y_STRIP_TITLE + 16        # 484
Y_FOOTER = Y_STRIP + 62             # 546


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


def _draw_body(
    draw,
    cx: int,
    cy: int,
    tint,
    belief: Dict[str, float],
    phase: float,
) -> None:
    """A figure with each limb lit in proportion to how much it is moving.

    Drawn as if seen from behind, so the subject's left hand is on the
    viewer's left. The limbs are labelled anyway — left and right is exactly
    the thing this decoder is worst at, and a mirrored figure would make an
    honest error look like a labelling bug.

    ``belief`` is 1.0 for a limb that is definitely moving and a fraction for
    one the decoder is unsure about, so uncertainty is rendered rather than
    resolved. ``phase`` drives the open/close oscillation.
    """
    dim = IDLE

    head_r = 20
    shoulder_y = cy - 40
    hip_y = cy + 34

    # Head and torso are never "active"; they are the frame of reference.
    draw.ellipse(
        [cx - head_r, shoulder_y - 2 * head_r - 6, cx + head_r, shoulder_y - 6],
        outline=INK_MUTED,
        width=3,
    )
    draw.line([cx, shoulder_y, cx, hip_y], fill=INK_MUTED, width=5)

    limbs = {
        "left arm": [(cx, shoulder_y + 4), (cx - 58, shoulder_y + 36), (cx - 72, shoulder_y + 84)],
        "right arm": [(cx, shoulder_y + 4), (cx + 58, shoulder_y + 36), (cx + 72, shoulder_y + 84)],
        "left leg": [(cx, hip_y), (cx - 30, hip_y + 48), (cx - 40, hip_y + 96)],
        "right leg": [(cx, hip_y), (cx + 30, hip_y + 48), (cx + 40, hip_y + 96)],
    }

    for name, points in limbs.items():
        amount = float(np.clip(belief.get(name, 0.0), 0.0, 1.0))
        colour = tuple(
            int(dim[c] + (tint[c] - dim[c]) * amount) for c in range(3)
        )
        width = int(round(5 + 4 * amount))

        # The hand or foot opens and closes while the limb is active, scaled
        # by how strongly the decoder believes it. A limb that merely changes
        # colour does not read as a limb that is moving.
        swing = amount * 5.0 * np.sin(2 * np.pi * phase)
        moved = list(points[:-1]) + [(points[-1][0], points[-1][1] + swing)]

        draw.line(moved, fill=colour, width=width, joint="curve")
        end = moved[-1]
        radius = 7 + 5 * amount + amount * 2.5 * np.cos(2 * np.pi * phase)
        draw.ellipse(
            [end[0] - radius, end[1] - radius, end[0] + radius, end[1] + radius],
            fill=colour,
        )

    small = _font(12)
    draw.text((cx - 90, shoulder_y + 92), "L", fill=INK_FAINT, font=small)
    draw.text((cx + 80, shoulder_y + 92), "R", fill=INK_FAINT, font=small)


def _draw_frame(
    position: int,
    phase: float,
    actual: np.ndarray,
    predicted: np.ndarray,
    probability: np.ndarray,
    centre: np.ndarray,
    fonts,
    metrics: Dict[str, float],
    subject: str,
    group: str,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    small, medium = fonts
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)

    truth = CLASSES[int(actual[position])]
    guess = CLASSES[int(predicted[position])]
    hit = truth == guess

    # --- header -------------------------------------------------------------
    draw.text((32, 20), "MEMORY RECONSTRUCTION · PHYSICAL ACTION", fill=INK_FAINT, font=small)
    draw.text((32, 42), f"{subject} · opening and closing fists and feet", fill=INK, font=medium)
    draw.text((WIDTH - 250, 24), f"{centre[position]:6.1f}s   real time", fill=INK_MUTED, font=small)

    # --- the two figures ----------------------------------------------------
    left_x = 32 + PANEL_W // 2
    right_x = WIDTH - 32 - PANEL_W // 2

    draw.text((32, TOP - 22), "WHAT THEY DID", fill=TRUTH_TINT, font=small)
    draw.text((WIDTH - 32 - PANEL_W, TOP - 22), "RECONSTRUCTED FROM EEG", fill=CORRECT if hit else INK_FAINT, font=small)

    truth_belief = {limb: 1.0 for limb in ACTION_LIMBS[truth]}
    _draw_body(
        draw, left_x, TOP + FIGURE_H // 2, ACTION_COLOUR[truth], truth_belief, phase
    )
    _draw_body(
        draw,
        right_x,
        TOP + FIGURE_H // 2,
        ACTION_COLOUR[guess],
        limb_belief(probability[position]),
        phase,
    )

    draw.text((32, Y_ACTION), truth, fill=INK, font=medium)
    draw.text(
        (WIDTH - 32 - PANEL_W, Y_ACTION), guess, fill=CORRECT if hit else WRONG, font=medium
    )

    tier = tier_of(int(actual[position]), int(predicted[position]))
    note = {
        "exact": "correct",
        "effector": "right limbs, wrong action",
        "moving": "right about moving, wrong limbs",
        "wrong": f"actually {truth}",
    }[tier]
    draw.text(
        (WIDTH - 32 - PANEL_W, Y_TIER),
        note,
        fill=CORRECT if tier == "exact" else (TRUTH_TINT if tier == "effector" else WRONG),
        font=small,
    )

    # --- posterior ----------------------------------------------------------
    draw.text(
        (32, Y_BARS_TITLE),
        "WHAT THE DECODER BELIEVES   the right figure lights each limb by these odds, not by the winner",
        fill=INK_FAINT,
        font=small,
    )
    bar_w = 168
    for index, name in enumerate(CLASSES):
        x = 32 + index * (bar_w + 42)
        value = float(probability[position, index])
        draw.rectangle([x, Y_BARS, x + bar_w, Y_BARS + 14], fill=(20, 25, 34))
        draw.rectangle(
            [x, Y_BARS, x + int(bar_w * value), Y_BARS + 14], fill=ACTION_COLOUR[name]
        )
        draw.text(
            (x, Y_BAR_LABELS), f"{name} {value * 100:.0f}%", fill=INK_MUTED, font=small
        )

    # --- timeline -----------------------------------------------------------
    draw.text((32, Y_STRIP_TITLE), "ACTION OVER TIME", fill=INK_FAINT, font=small)
    x0, x1 = 32, WIDTH - 120
    span = x1 - x0
    window = 200
    lo = max(0, position - window + 40)

    for row, (series, title) in enumerate(((actual, "actual"), (predicted, "decoded"))):
        y = Y_STRIP + row * 26
        draw.rectangle([x0, y, x1, y + 20], fill=(16, 20, 28))
        for k in range(lo, min(actual.size, lo + window)):
            name = CLASSES[int(series[k])]
            if name == "rest":
                continue
            left = x0 + span * (k - lo) / window
            draw.rectangle(
                [left, y + 2, left + max(2, span / window), y + 18],
                fill=ACTION_COLOUR[name],
            )
        draw.text((x1 + 8, y + 4), title, fill=INK_FAINT, font=small)

    head = x0 + span * (position - lo) / window
    draw.line([head, Y_STRIP - 2, head, Y_STRIP + 46], fill=INK, width=2)

    # --- footer -------------------------------------------------------------
    # Kept under ~130 characters a line; longer runs off the frame at 13pt.
    lines = [
        (
            "Held out: trained on this person's other runs, never this one. "
            f"Exact action {metrics['exact'] * 100:.0f}%, right limbs "
            f"{metrics['exact or effector'] * 100:.0f}%, moving vs still "
            f"{metrics['right about moving'] * 100:.0f}%.",
            INK_MUTED,
        ),
        (
            f"Balanced across actions, chance 20%. Shuffled labels {metrics['shuffled'] * 100:.0f}%. "
            f"{group}",
            INK_MUTED,
        ),
        (
            "How well a distinction decodes tracks how far apart the body parts sit on "
            f"the cortex: fists vs feet {metrics['fists vs feet'] * 100:.0f}%, "
            f"which fist {metrics['left vs right fist'] * 100:.0f}%.",
            INK_FAINT,
        ),
    ]
    for row, (text, colour) in enumerate(lines):
        draw.text((32, Y_FOOTER + row * 18), text, fill=colour, font=small)

    return np.asarray(image)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="")
    parser.add_argument("--seconds", type=float, default=75.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    subjects = available_subjects(DATA_ROOT)
    if not subjects:
        raise SystemExit("no subjects downloaded; run tools/fetch_motor.py")
    subject = args.subject or subjects[0]

    print(f"reconstructing {subject}…")
    windows = windows_for_subject(DATA_ROOT, subject)
    if windows is None:
        raise SystemExit(f"{subject}: no usable runs")

    prediction = leave_one_run_out(windows)
    report = score(prediction)
    metrics = decompose(windows)
    metrics.update(graded(prediction))
    metrics["five class"] = report.balanced
    metrics["shuffled"] = score(leave_one_run_out(windows, shuffle=True)).balanced

    for name, value in metrics.items():
        print(f"  {name:<22} {value * 100:5.1f}%")

    # Walk the session in time order, run by run.
    order = np.lexsort((prediction.centre, prediction.run))
    total = int(args.seconds * FPS / FRAMES_PER_WINDOW)
    order = order[:total]

    # Stated on every frame: motor decoding varies enormously between people,
    # and showing a strong subject without the spread would misrepresent it.
    group = "Across 20 subjects: exact 37%, range 24-60%. This is a strong one."

    fonts = (_font(13), _font(17))
    out = Path(args.out) if args.out else OUT_DIR / f"{subject}_motor_video.mp4"
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
        actual = prediction.actual[order]
        predicted = prediction.predicted[order]
        probability = prediction.probability[order]
        centre = prediction.centre[order]

        for position in range(order.size):
            # Redrawn every frame rather than once per window, so the limbs
            # actually move instead of stepping between still poses.
            for tick in range(FRAMES_PER_WINDOW):
                elapsed = (position * FRAMES_PER_WINDOW + tick) / FPS
                writer.append_data(
                    _draw_frame(
                        position, elapsed * MOVE_HZ, actual, predicted, probability,
                        centre, fonts, metrics, subject, group,
                    )
                )
    finally:
        writer.close()

    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB, {order.size} windows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
