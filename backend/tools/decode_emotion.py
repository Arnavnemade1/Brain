"""Validate the continuous emotion decoder on real human EEG.

Leave-one-subject-out throughout: the model is fitted on some people and
tested on a person it has never seen, which is the only split that answers
whether this would work on someone new.

Every number is reported beside its chance level and a shuffled control run
through the identical pipeline.

Run with::

    backend/.venv/bin/python -m tools.decode_emotion
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

from dataclasses import replace

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="  %(message)s")

from app.emotion.dataset import DATA_ROOT, load_all  # noqa: E402
from app.emotion.decoder import (  # noqa: E402
    EmotionModel,
    TrialFeatures,
    build_matrix,
    decode_trial,
    fit,
    normalise_per_subject,
    quadrant_of,
    to_unit,
)
from app.emotion.features import FEATURE_NAMES  # noqa: E402


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def balanced_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Mean of the two per-class accuracies.

    The honest metric when classes are imbalanced: a decoder that always says
    "low valence" scores its raw accuracy at the class prior (62% here) but
    50% balanced, which is what it deserves.
    """
    predicted = np.asarray(predicted, dtype=bool)
    actual = np.asarray(actual, dtype=bool)
    if predicted.size == 0:
        return 0.0
    rates = []
    for value in (True, False):
        mask = actual == value
        if mask.any():
            rates.append(float(np.mean(predicted[mask] == value)))
    return float(np.mean(rates)) if rates else 0.0


def within_subject(
    trials: List[TrialFeatures],
    alpha: float = 50.0,
    shuffle: bool = False,
    seed: int = 0,
) -> Dict:
    """Leave-one-trial-out inside each subject.

    A different and easier question than cross-subject transfer: "given a
    calibration session from this person, can their emotion be read?" Most
    published emotion-recognition accuracies are of this kind, so it is
    reported separately rather than blended with the cross-subject numbers.

    ``shuffle`` permutes the ratings within each subject. This control was
    missing when the regime was first reported, and it is not optional here:
    leave-one-out over ten trials biases the correlation *downward* on its own,
    because dropping one sample pulls the training mean away from the held-out
    value. Without a control there is no way to tell how much of any number is
    that bias rather than signal, and "chance = 0" is an assumption rather than
    a measurement.
    """
    predicted_v, actual_v, predicted_a, actual_a = [], [], [], []
    rng = np.random.default_rng(seed)

    for subject in sorted({t.subject for t in trials}):
        own = [t for t in trials if t.subject == subject]
        if len(own) < 4:
            continue

        if shuffle:
            # Reassign this subject's ratings among their own trials, so the
            # per-subject design and rating distribution both survive.
            ratings = [(t.valence, t.arousal) for t in own]
            rng.shuffle(ratings)
            relabelled = []
            for trial, (valence, arousal) in zip(own, ratings):
                clone = replace(trial, valence=valence, arousal=arousal)
                relabelled.append(clone)
            own = relabelled

        for held in range(len(own)):
            train = [t for i, t in enumerate(own) if i != held]
            test = own[held]
            try:
                model, stats = fit(train, FEATURE_NAMES, alpha=alpha)
            except RuntimeError:
                continue
            valence, arousal = decode_trial(model, test, stats)
            if not valence.size:
                continue
            predicted_v.append(float(np.mean(valence)))
            predicted_a.append(float(np.mean(arousal)))
            actual_v.append(to_unit(test.valence))
            actual_a.append(to_unit(test.arousal))

    pv, av = np.array(predicted_v), np.array(actual_v)
    pa, aa = np.array(predicted_a), np.array(actual_a)
    return {
        "valence_r": correlation(pv, av),
        "arousal_r": correlation(pa, aa),
        "valence_accuracy": balanced_accuracy(pv >= 0, av >= 0),
        "arousal_accuracy": balanced_accuracy(pa >= 0, aa >= 0),
        "n_trials": len(predicted_v),
    }


def leave_one_subject_out(
    trials: List[TrialFeatures], alpha: float = 50.0, shuffle: bool = False
) -> Dict:
    """Fit on all-but-one subject, predict the held-out one, repeat."""
    subjects = sorted({t.subject for t in trials})
    rng = np.random.default_rng(3)

    predicted_valence: List[float] = []
    predicted_arousal: List[float] = []
    actual_valence: List[float] = []
    actual_arousal: List[float] = []
    quadrant_hits: List[bool] = []
    valence_sign_hits: List[bool] = []
    arousal_sign_hits: List[bool] = []

    for held_out in subjects:
        train = [t for t in trials if t.subject != held_out]
        test = [t for t in trials if t.subject == held_out]
        if not train or not test:
            continue

        if shuffle:
            # Control: permute the labels across training trials, leaving the
            # features and every other step untouched.
            train = list(train)
            ratings = [(t.valence, t.arousal) for t in train]
            rng.shuffle(ratings)
            shuffled = []
            for trial, (valence, arousal) in zip(train, ratings):
                clone = TrialFeatures(**{**trial.__dict__})
                clone.valence, clone.arousal = valence, arousal
                shuffled.append(clone)
            train = shuffled

        model, _ = fit(train, FEATURE_NAMES, alpha=alpha)

        # The held-out subject's own feature statistics are used to
        # standardise. This is legitimate — it needs no labels, only a few
        # minutes of that person's resting signal, which any deployment has.
        test_stats = normalise_per_subject(test)

        for trial in test:
            valence, arousal = decode_trial(model, trial, test_stats)
            if not valence.size:
                continue

            mean_valence = float(np.mean(valence))
            mean_arousal = float(np.mean(arousal))

            predicted_valence.append(mean_valence)
            predicted_arousal.append(mean_arousal)
            actual_valence.append(to_unit(trial.valence))
            actual_arousal.append(to_unit(trial.arousal))

            quadrant_hits.append(
                quadrant_of(mean_valence, mean_arousal) == trial.quadrant
            )
            valence_sign_hits.append(
                (mean_valence >= 0) == (to_unit(trial.valence) >= 0)
            )
            arousal_sign_hits.append(
                (mean_arousal >= 0) == (to_unit(trial.arousal) >= 0)
            )

    pv = np.array(predicted_valence)
    pa = np.array(predicted_arousal)
    av = np.array(actual_valence)
    aa = np.array(actual_arousal)

    return {
        "valence_r": correlation(pv, av),
        "arousal_r": correlation(pa, aa),
        # Balanced accuracy, not raw. With 62% of trials in one class, raw
        # accuracy rewards a decoder that has learned only the class prior;
        # balanced accuracy averages the two per-class rates and sits at 50%
        # for any constant prediction.
        "valence_accuracy": balanced_accuracy(pv >= 0, av >= 0),
        "arousal_accuracy": balanced_accuracy(pa >= 0, aa >= 0),
        "quadrant_accuracy": float(np.mean(quadrant_hits)) if quadrant_hits else 0.0,
        "n_trials": len(predicted_valence),
        "n_subjects": len(subjects),
    }


def click_alignment(trials: List[TrialFeatures], alpha: float = 50.0) -> Dict:
    """Does the decoded emotion intensify near the moment the subject clicked?

    This is the only genuinely *per-moment* test available. Supervision is
    trial-level, so nothing in fitting ever saw a click; if decoded intensity
    is systematically higher near clicks than at random times within the same
    trials, the decoder is tracking something that moves inside a trial rather
    than merely reproducing its average.
    """
    subjects = sorted({t.subject for t in trials})
    rng = np.random.default_rng(11)

    at_click: List[float] = []
    at_random: List[float] = []

    for held_out in subjects:
        train = [t for t in trials if t.subject != held_out]
        test = [t for t in trials if t.subject == held_out and t.click_times]
        if not train or not test:
            continue

        model, _ = fit(train, FEATURE_NAMES, alpha=alpha)
        test_stats = normalise_per_subject([t for t in trials if t.subject == held_out])

        for trial in test:
            valence, arousal = decode_trial(model, trial, test_stats)
            if valence.size < 4:
                continue

            # Emotional intensity: distance from neutral on the circumplex.
            intensity = np.hypot(valence, arousal)

            for click in trial.click_times:
                index = int(np.argmin(np.abs(trial.times - click)))
                at_click.append(float(intensity[index]))

                # A matched control drawn from the same trial, so trial-level
                # differences cannot explain any gap.
                candidates = [
                    i
                    for i in range(intensity.size)
                    if abs(trial.times[i] - click) > 3.0
                ]
                if candidates:
                    at_random.append(float(intensity[int(rng.choice(candidates))]))

    if not at_click or not at_random:
        return {"available": False}

    click_mean = float(np.mean(at_click))
    random_mean = float(np.mean(at_random))
    pooled = float(np.sqrt((np.var(at_click) + np.var(at_random)) / 2)) or 1e-9

    return {
        "available": True,
        "n_clicks": len(at_click),
        "intensity_at_click": click_mean,
        "intensity_elsewhere": random_mean,
        "difference": click_mean - random_mean,
        "effect_size": (click_mean - random_mean) / pooled,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=50.0)
    args = parser.parse_args()

    trials = load_all()
    if not trials:
        print("No data. Run: python -m tools.fetch_emotion")
        return 1

    subjects = sorted({t.subject for t in trials})
    windows = sum(t.features.shape[0] for t in trials)

    print("=" * 78)
    print("Continuous emotion decoding — real human EEG (DENS)")
    print("=" * 78)
    print(
        f"{len(trials)} trials · {len(subjects)} subjects · {windows} feature windows "
        f"· {len(FEATURE_NAMES)} features"
    )
    print(
        "Validation: leave-one-subject-out — every result is on a person the "
        "model never saw.\n"
    )

    real = leave_one_subject_out(trials, alpha=args.alpha)
    control = leave_one_subject_out(trials, alpha=args.alpha, shuffle=True)

    print("  Cross-subject (leave-one-subject-out) — agreement with self-report")
    print(
        f"    valence   r = {real['valence_r']:+.3f}   "
        f"binary {real['valence_accuracy'] * 100:5.1f}%   (chance 50%)"
    )
    print(
        f"    arousal   r = {real['arousal_r']:+.3f}   "
        f"binary {real['arousal_accuracy'] * 100:5.1f}%   (chance 50%)"
    )
    print(
        f"    quadrant  {real['quadrant_accuracy'] * 100:5.1f}%   (chance 25%, 4-way)"
    )
    print()
    print("  Shuffled-label control (must collapse to chance)")
    print(
        f"    valence   r = {control['valence_r']:+.3f}   "
        f"binary {control['valence_accuracy'] * 100:5.1f}%"
    )
    print(
        f"    arousal   r = {control['arousal_r']:+.3f}   "
        f"binary {control['arousal_accuracy'] * 100:5.1f}%"
    )
    print(f"    quadrant  {control['quadrant_accuracy'] * 100:5.1f}%")

    print("\n  Within-subject (leave-one-trial-out; easier question, reported separately)")
    within = within_subject(trials, alpha=args.alpha)
    within_control = within_subject(trials, alpha=args.alpha, shuffle=True)
    print(
        f"    valence   r = {within['valence_r']:+.3f}   "
        f"balanced {within['valence_accuracy'] * 100:5.1f}%"
    )
    print(
        f"    arousal   r = {within['arousal_r']:+.3f}   "
        f"balanced {within['arousal_accuracy'] * 100:5.1f}%   (n={within['n_trials']})"
    )
    print("  within-subject, shuffled control:")
    print(
        f"    valence   r = {within_control['valence_r']:+.3f}   "
        f"balanced {within_control['valence_accuracy'] * 100:5.1f}%"
    )
    print(
        f"    arousal   r = {within_control['arousal_r']:+.3f}   "
        f"balanced {within_control['arousal_accuracy'] * 100:5.1f}%"
    )

    print("\n  Per-moment check — decoded intensity around the felt-emotion click")
    alignment = click_alignment(trials, alpha=args.alpha)
    if alignment.get("available"):
        print(
            f"    at click   {alignment['intensity_at_click']:.3f}\n"
            f"    elsewhere  {alignment['intensity_elsewhere']:.3f}\n"
            f"    difference {alignment['difference']:+.3f}  "
            f"(effect size {alignment['effect_size']:+.2f}, n={alignment['n_clicks']})"
        )
    else:
        print("    no click markers available")

    output = DATA_ROOT / "derivatives" / "emotion_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "real": real,
                "shuffled": control,
                "within_subject": within,
                "within_subject_shuffled": within_control,
                "click_alignment": alignment,
            },
            indent=1
        )
    )
    print(f"\n  wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
