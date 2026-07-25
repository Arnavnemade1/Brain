"""Calibration harness for the analysis pipeline.

Runs every simulated profile through the full chain and reports how well the
pipeline recovers the structure the generator scripted. Nothing is passed
between the two — this measures genuine round-trip accuracy.

Run with::

    backend/.venv/bin/python -m tools.calibrate
"""

from __future__ import annotations

import sys
import warnings
from collections import Counter
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, ".")

from app.pipeline.analyzer import WindowAnalyzer  # noqa: E402
from app.processing.bands import BAND_ORDER  # noqa: E402
from app.simulation.generator import generate_session  # noqa: E402
from app.simulation.profiles import PROFILES  # noqa: E402

SAMPLE_RATE = 256
WINDOW = 512
HOP = 256
DURATION = 90.0
SEEDS = (3, 17, 41)


def phase_at(profile, t: float, duration: float):
    """The scripted phase covering session time ``t``."""
    phases = profile.normalised_phases()
    acc = 0.0
    for phase in phases:
        acc += phase.weight * duration
        if t <= acc:
            return phase
    return phases[-1]


def run() -> int:
    band_errors: List[float] = []
    cognitive_hits = 0
    cognitive_total = 0
    valence_pairs: List[tuple] = []
    arousal_by_phase: Dict[str, List[float]] = {}
    emotion_confusion: Dict[str, Counter] = {}

    print("=" * 78)
    print("MindScape pipeline calibration")
    print("=" * 78)

    for pid, profile in PROFILES.items():
        p_band_err: List[float] = []
        p_cog_hit = 0
        p_cog_total = 0
        arousals: List[float] = []
        target_arousals: List[float] = []

        for seed in SEEDS:
            rec = generate_session(pid, DURATION, SAMPLE_RATE, seed=seed)
            analyzer = WindowAnalyzer(f"cal-{pid}", SAMPLE_RATE, rec.channels, 60.0)

            for start in range(0, rec.total_samples - WINDOW, HOP):
                t = start / SAMPLE_RATE
                result = analyzer.analyze(rec.window(start, WINDOW), t)
                frame = result.frame
                phase = phase_at(profile, t, DURATION)

                total = sum(phase.bands.values()) or 1.0
                target = {b: phase.bands.get(b, 0.0) / total for b in BAND_ORDER}
                err = float(
                    np.mean([abs(frame.band_powers[b] - target[b]) for b in BAND_ORDER])
                )
                p_band_err.append(err)

                p_cog_total += 1
                if frame.cognitive.state == phase.cognitive:
                    p_cog_hit += 1

                valence_pairs.append((phase.valence, frame.emotion.valence))
                arousals.append(frame.emotion.arousal)
                target_arousals.append(phase.arousal)

                emotion_confusion.setdefault(phase.emotion, Counter())[
                    frame.emotion.primary
                ] += 1

        band_errors.extend(p_band_err)
        cognitive_hits += p_cog_hit
        cognitive_total += p_cog_total
        arousal_by_phase[pid] = arousals

        print(
            f"\n{profile.name:<26} band MAE {np.mean(p_band_err):.3f}   "
            f"cognitive {p_cog_hit / max(1, p_cog_total) * 100:5.1f}%   "
            f"arousal {np.mean(arousals):.2f} (target {np.mean(target_arousals):.2f})"
        )

    print("\n" + "-" * 78)
    print(f"Band power MAE          {np.mean(band_errors):.4f}  (0 = perfect recovery)")
    print(
        f"Cognitive state accuracy {cognitive_hits / max(1, cognitive_total) * 100:5.1f}%  "
        f"(chance = {100 / 7:.1f}%)"
    )

    scripted = np.array([p[0] for p in valence_pairs])
    recovered = np.array([p[1] for p in valence_pairs])
    if scripted.std() > 1e-9 and recovered.std() > 1e-9:
        r = float(np.corrcoef(scripted, recovered)[0, 1])
        print(f"Valence correlation      {r:+.3f}  (scripted vs recovered)")

    all_arousal = np.concatenate([np.array(v) for v in arousal_by_phase.values()])
    print(
        f"Arousal range            {all_arousal.min():.2f} – {all_arousal.max():.2f}  "
        f"mean {all_arousal.mean():.2f}"
    )

    print("\nEmotion confusion (scripted → most frequent recovered):")
    for scripted_emotion, counter in sorted(emotion_confusion.items()):
        total = sum(counter.values())
        top = ", ".join(f"{k} {v / total * 100:.0f}%" for k, v in counter.most_common(3))
        match = counter.get(scripted_emotion, 0) / total * 100
        print(f"  {scripted_emotion:<12} exact {match:5.1f}%   top: {top}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
