"""Render a person's viewing as a video reconstructed from their EEG.

Fits the emotion decoder on that subject's *other* trials, decodes the held-out
one moment by moment, and writes a real `.mp4`.

Within-subject calibration is used deliberately. Cross-subject transfer did not
reach significance on this dataset (see `EMOTION.md`), and within-subject is
also the realistic deployment case: anyone wearing the headset for a bowling
night would calibrate on themselves first.

Run with::

    backend/.venv/bin/python -m tools.render_memory --subject sub-mit003
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="  %(message)s")

from app.emotion.dataset import DATA_ROOT, load_all  # noqa: E402
from app.emotion.decoder import decode_trial, fit, normalise_per_subject, to_unit  # noqa: E402
from app.emotion.features import FEATURE_NAMES  # noqa: E402
from app.emotion.video import EmotionTrack, render  # noqa: E402

NOTE = (
    "Emotional track decoded from EEG. Within-subject calibration. "
    "Arousal r=+0.19 on held-out trials; valence did not reach significance."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="")
    parser.add_argument("--trial", type=int, default=-1, help="-1 picks the most emotive")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    trials = load_all()
    if not trials:
        print("No data. Run: python -m tools.fetch_emotion")
        return 1

    subject = args.subject or sorted({t.subject for t in trials})[0]
    own = [t for t in trials if t.subject == subject]
    if not own:
        print(f"No trials for {subject}")
        return 1

    if args.trial >= 0:
        selected = next((t for t in own if t.trial_index == args.trial), own[0])
    else:
        # Prefer a trial the subject actually marked, so the video has a
        # felt-emotion moment to show against the decoded track.
        marked = [t for t in own if t.click_times]
        pool = marked or own
        selected = max(pool, key=lambda t: abs(to_unit(t.valence)) + abs(to_unit(t.arousal)))

    train = [t for t in own if t is not selected]
    if len(train) < 3:
        print("Not enough trials to calibrate on")
        return 1

    model, _ = fit(train, FEATURE_NAMES, alpha=50.0)
    stats = normalise_per_subject(own)
    valence, arousal = decode_trial(model, selected, stats)

    if not valence.size:
        print("No decodable windows in that trial")
        return 1

    track = EmotionTrack(
        times=selected.times,
        valence=valence,
        arousal=arousal,
        subject=subject,
        stimulus=selected.stimulus,
        duration=float(selected.duration),
        reported_valence=to_unit(selected.valence),
        reported_arousal=to_unit(selected.arousal),
        reported_emotion=selected.emotion,
        click_times=list(selected.click_times),
        confidence_note=NOTE,
    )

    output = (
        Path(args.out)
        if args.out
        else DATA_ROOT / "derivatives" / f"{subject}_trial{selected.trial_index}_memory.mp4"
    )

    print(f"  subject   {subject}")
    print(f"  stimulus  {selected.stimulus}  ({selected.duration:.1f}s)")
    print(
        f"  reported  valence {selected.valence:.1f}/9  arousal {selected.arousal:.1f}/9"
        + (f"  ({selected.emotion})" if selected.emotion else "")
    )
    print(
        f"  decoded   valence {np.mean(valence):+.2f}  arousal {np.mean(arousal):+.2f}  "
        f"({valence.size} windows)"
    )
    if selected.click_times:
        print(f"  felt at   {', '.join(f'{c:.1f}s' for c in selected.click_times)}")

    render(track, output)
    size = output.stat().st_size / 1e6
    print(f"\n  wrote {output}  ({size:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
