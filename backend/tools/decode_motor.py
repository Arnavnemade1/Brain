"""Validate the physical-action reconstruction.

Reports every accuracy **balanced** across actions. This matters more here
than anywhere else in the project: rest occupies 54% of every recording, so a
decoder that answered "rest" and nothing else would score 54% raw. Balanced
accuracy puts any constant answer at 1/5.

Three things are reported, and the gap between them is the finding:

* the exact action, five ways;
* whether the right *limbs* were identified, even if the action was not —
  answering "both fists" for "right fist" is a different quality of error
  from answering "both feet";
* whether the decoder was right about the body moving at all.

Alongside a shuffled-label control through the identical pipeline.

Run with::

    backend/.venv/bin/python -m tools.decode_motor --subjects 12
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from app.motor.decode import decompose, graded, leave_one_run_out, score
from app.motor.windows import RUN_TASKS, available_subjects, windows_for_subject

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "eegmmidb"


def complete_subjects(limit: int) -> List[str]:
    """Subjects with all six executed-movement runs on disk."""
    out = []
    for subject in available_subjects(DATA_ROOT):
        present = sum(
            (DATA_ROOT / subject / f"{subject}R{run:02d}.edf").exists()
            for run in RUN_TASKS
        )
        if present == len(RUN_TASKS):
            out.append(subject)
        if len(out) >= limit:
            break
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, default=12)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    subjects = complete_subjects(args.subjects)
    if not subjects:
        raise SystemExit("no complete subjects; run tools/fetch_motor.py")

    real: List[Dict[str, float]] = []
    shuffled: List[Dict[str, float]] = []

    for subject in subjects:
        windows = windows_for_subject(DATA_ROOT, subject)
        if windows is None:
            continue

        prediction = leave_one_run_out(windows)
        metrics = dict(graded(prediction))
        metrics.update(decompose(windows))
        real.append(metrics)

        control = leave_one_run_out(windows, shuffle=True)
        shuffled.append(dict(graded(control)))

        print(
            f"  {subject}  exact {metrics['exact'] * 100:4.1f}%  "
            f"limbs {metrics['exact or effector'] * 100:4.1f}%  "
            f"moving {metrics['right about moving'] * 100:4.1f}%"
        )

    def mean(rows: List[Dict[str, float]], key: str) -> float:
        values = [r[key] for r in rows if key in r]
        return float(np.mean(values)) if values else float("nan")

    def spread(key: str) -> str:
        values = [r[key] for r in real if key in r]
        return f"{min(values) * 100:.0f}-{max(values) * 100:.0f}%"

    print(f"\n{len(real)} subjects, leave-one-run-out\n")
    print(f"  {'':<26}{'real':>9}{'shuffled':>11}{'chance':>9}{'range':>11}")
    for key, chance in (
        ("exact", 0.20),
        ("exact or effector", None),
        ("right about moving", None),
    ):
        chance_text = f"{chance * 100:.0f}%" if chance else "—"
        print(
            f"  {key:<26}{mean(real, key) * 100:8.1f}%"
            f"{mean(shuffled, key) * 100:10.1f}%"
            f"{chance_text:>9}{spread(key):>11}"
        )

    print(f"\n  {'sub-question':<26}{'real':>9}{'chance':>11}{'range':>11}")
    for key, chance in (
        ("moving vs rest", 0.50),
        ("fists vs feet", 0.50),
        ("left vs right fist", 0.50),
        ("which of four limbs", 0.25),
    ):
        print(
            f"  {key:<26}{mean(real, key) * 100:8.1f}%"
            f"{chance * 100:10.0f}%{spread(key):>11}"
        )

    print(
        "\n  Read the spread, not just the mean. Motor rhythms differ enough\n"
        "  between people that some subjects sit at chance throughout — a\n"
        "  well-documented split, and the reason a single-subject figure from\n"
        "  this dataset means very little on its own."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
