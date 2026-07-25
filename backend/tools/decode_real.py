"""Decode real human EEG and report identification accuracy.

Answers the question the platform exists to answer: from EEG recorded while a
person looked at one of 200 objects, how often can the right object be
identified?

Every number is reported next to its chance level and next to a shuffled
control run through the identical pipeline. Both splits are reported: the
conventional repetition-fold cross-validation, and a strict time split that
rules out temporal leakage between overlapping RSVP epochs.

Run with::

    backend/.venv/bin/python -m tools.decode_real --subjects sub-01,sub-02
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="  %(message)s")

from app.realdata.decoding import (  # noqa: E402
    build_features,
    classify,
    identify_by_template,
    repetition_folds,
    time_resolved_accuracy,
    time_split,
)
from app.realdata.epochs import available_subjects, load_or_extract  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "ds004018"

#: Averaging levels reported. 1 is a single presentation; higher values model
#: someone seeing the same content repeatedly, which is the realistic case
#: for a memory rather than a one-off glance.
AVERAGING = [1, 4, 10]


def run_subject(subject: str, rng: np.random.Generator) -> Dict:
    print(f"\n{'=' * 78}\n{subject}\n{'=' * 78}")

    epochs = load_or_extract(ROOT, subject)
    features = build_features(epochs)

    print(
        f"  {features.x.shape[0]} trials · {features.n_channels} channels × "
        f"{features.n_times} timepoints · "
        f"{np.unique(features.stimulus).size} stimuli"
    )

    folds = repetition_folds(features.stimulus, n_folds=5)
    test_fold = folds[0]
    train_fold = np.concatenate(folds[1:])

    strict_train, strict_test = time_split(features.x.shape[0])

    results: Dict = {"subject": subject}

    # --- Semantic decoding ------------------------------------------------
    print("\n  Semantic decoding (repetition-fold CV)")
    animacy = classify(
        features, features.level_a, train_fold, test_fold, "animate vs inanimate"
    )
    category = classify(
        features, features.level_b, train_fold, test_fold, "category (10-way)"
    )
    animacy_shuffled = classify(
        features,
        features.level_a,
        train_fold,
        test_fold,
        "  ↳ shuffled control",
        shuffle=True,
        rng=rng,
    )
    for r in (animacy, category, animacy_shuffled):
        print("    " + r.summary())

    print("\n  Semantic decoding (strict time split)")
    animacy_strict = classify(
        features, features.level_a, strict_train, strict_test, "animate vs inanimate"
    )
    category_strict = classify(
        features, features.level_b, strict_train, strict_test, "category (10-way)"
    )
    for r in (animacy_strict, category_strict):
        print("    " + r.summary())

    results["animacy"] = animacy.accuracy
    results["animacy_strict"] = animacy_strict.accuracy
    results["animacy_shuffled"] = animacy_shuffled.accuracy
    results["category"] = category.accuracy
    results["category_strict"] = category_strict.accuracy

    # --- 200-way identification -------------------------------------------
    print("\n  200-way identification (repetition-fold CV)")
    identification = {}
    for k in AVERAGING:
        result = identify_by_template(
            features,
            train_fold,
            test_fold,
            trials_averaged=k,
            label=f"{k} trial(s) averaged",
            rng=rng,
        )
        print("    " + result.summary())
        identification[k] = result

    control = identify_by_template(
        features,
        train_fold,
        test_fold,
        trials_averaged=max(AVERAGING),
        label="  ↳ shuffled control",
        shuffle=True,
        rng=rng,
    )
    print("    " + control.summary())

    print("\n  200-way identification (strict time split)")
    strict_identification = {}
    for k in AVERAGING:
        result = identify_by_template(
            features,
            strict_train,
            strict_test,
            trials_averaged=k,
            label=f"{k} trial(s) averaged",
            rng=rng,
        )
        print("    " + result.summary())
        strict_identification[k] = result

    results["identification"] = {
        k: {
            "top1": v.top1,
            "top5": v.top5,
            "pairwise": v.pairwise,
            "mean_rank": v.mean_rank,
        }
        for k, v in identification.items()
    }
    results["identification_strict"] = {
        k: {"top1": v.top1, "top5": v.top5, "pairwise": v.pairwise}
        for k, v in strict_identification.items()
    }
    results["identification_shuffled"] = {"top1": control.top1, "pairwise": control.pairwise}

    # --- Time-resolved sanity check ---------------------------------------
    print("\n  Time-resolved animacy decoding (must be at chance before 0 ms)")
    times, curve = time_resolved_accuracy(
        epochs, features.level_a, train_fold, test_fold, step=3
    )
    pre = curve[times < 0]
    peak_index = int(np.argmax(curve))
    print(
        f"    pre-stimulus mean {pre.mean() * 100:.1f}%  "
        f"(max {pre.max() * 100:.1f}%)  →  "
        f"peak {curve[peak_index] * 100:.1f}% at {times[peak_index] * 1000:+.0f} ms"
    )
    results["pre_stimulus"] = float(pre.mean())
    results["peak_accuracy"] = float(curve[peak_index])
    results["peak_latency_ms"] = float(times[peak_index] * 1000)
    results["curve_times"] = times.tolist()
    results["curve"] = curve.tolist()

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", default="", help="Comma-separated, or all")
    args = parser.parse_args()

    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        subjects = available_subjects(ROOT)

    if not subjects:
        print("No subjects available. Run: python -m tools.fetch_dataset")
        return 1

    rng = np.random.default_rng(7)
    all_results = [run_subject(subject, rng) for subject in subjects]

    print(f"\n{'=' * 78}\nSummary across {len(all_results)} subject(s)\n{'=' * 78}")

    def mean_of(key: str) -> float:
        return float(np.mean([r[key] for r in all_results]))

    print(f"  animacy (2-way, chance 50%)          {mean_of('animacy') * 100:5.1f}%")
    print(f"    strict time split                  {mean_of('animacy_strict') * 100:5.1f}%")
    print(f"    shuffled control                   {mean_of('animacy_shuffled') * 100:5.1f}%")
    print(f"  category (10-way, chance 10%)        {mean_of('category') * 100:5.1f}%")
    print(f"    strict time split                  {mean_of('category_strict') * 100:5.1f}%")
    print()
    print("  200-way identification (chance: top-1 0.5%, pairwise 50%)")
    for k in AVERAGING:
        top1 = float(np.mean([r["identification"][k]["top1"] for r in all_results]))
        top5 = float(np.mean([r["identification"][k]["top5"] for r in all_results]))
        pw = float(np.mean([r["identification"][k]["pairwise"] for r in all_results]))
        strict = float(np.mean([r["identification_strict"][k]["top1"] for r in all_results]))
        print(
            f"    {k:>2} trial(s)   top-1 {top1 * 100:5.1f}%  top-5 {top5 * 100:5.1f}%  "
            f"pairwise {pw * 100:5.1f}%   (strict top-1 {strict * 100:4.1f}%)"
        )
    shuffled = float(np.mean([r["identification_shuffled"]["top1"] for r in all_results]))
    print(f"    shuffled control  top-1 {shuffled * 100:.2f}%")

    print()
    print(
        f"  Pre-stimulus decoding {mean_of('pre_stimulus') * 100:.1f}% "
        f"(must be ~50%) → peak {mean_of('peak_accuracy') * 100:.1f}% "
        f"at {mean_of('peak_latency_ms'):+.0f} ms"
    )

    output = ROOT / "derivatives" / "decoding_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    import json

    output.write_text(json.dumps(all_results, indent=1))
    print(f"\n  wrote {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
