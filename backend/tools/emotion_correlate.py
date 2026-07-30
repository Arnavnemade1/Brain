"""Correlate every EEG feature against emotion, and control for having looked.

The prior work here classified trials into affective quadrants and failed: the
shuffled control scored 38.9% against the real model's 34.7% (see EMOTION.md).
This asks the question a different way. Continuous correlation against the raw
1-9 ratings is a more sensitive test than four-way classification, which throws
away most of the rating's information by binning it.

The search is deliberately wide — 14 features x 7 within-trial aggregates x
several targets — and that width is exactly why it needs the machinery below.
Running 300 correlations at p < 0.05 yields about 15 "significant" results from
noise alone, and reporting the largest of those as a finding is the standard
way emotion-EEG results fail to replicate.

So four things guard it:

* **Correlation within subject, then pooled.** Absolute EEG power differs more
  between two skulls than between any two states in one, so a correlation
  computed across pooled subjects would largely measure who was wearing the
  net. Each subject's own trials are correlated separately and combined by
  Fisher z, weighted by degrees of freedom.
* **A permutation null that respects that structure.** Ratings are shuffled
  *within* subject, so the null keeps the per-subject design and destroys only
  the association being tested.
* **Benjamini-Hochberg across the whole family.** Every test run is counted,
  including the ones that found nothing.
* **A trial-order control.** With ten trials a feature that merely drifts
  across the session can correlate with anything that happens to drift too.
  Every correlation is recomputed with trial index partialled out.

Run with::

    backend/.venv/bin/python -m tools.emotion_correlate
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

from app.emotion import dataset
from app.emotion.features import FEATURE_NAMES

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "ds003751"

#: Ways of reducing a trial's feature sequence to one number. Emotion may be
#: carried by a level, by how much it varies, or by how it moves over the clip
#: — a mean alone would miss the last two.
AGGREGATES: Dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda v: float(np.mean(v)),
    "std": lambda v: float(np.std(v)),
    "slope": lambda v: float(np.polyfit(np.arange(v.size), v, 1)[0]) if v.size > 2 else 0.0,
    "onset": lambda v: float(np.mean(v[: max(1, v.size // 3)])),
    "late": lambda v: float(np.mean(v[-max(1, v.size // 3) :])),
    "shift": lambda v: float(
        np.mean(v[-max(1, v.size // 3) :]) - np.mean(v[: max(1, v.size // 3)])
    ),
    "range": lambda v: float(np.max(v) - np.min(v)),
}

#: Permutations for the empirical null.
PERMUTATIONS = 2000


@dataclass
class Trial:
    subject: str
    index: int
    values: Dict[str, float]
    valence: float
    arousal: float
    quadrant: str


def collect(root: Path, subjects: Sequence[str]) -> List[Trial]:
    """Trial-level predictors and targets for every subject."""
    out: List[Trial] = []
    for subject in subjects:
        for trial in dataset.load(dataset.cache_path(root, subject)):
            if trial.features.size == 0:
                continue
            values: Dict[str, float] = {}
            for column, name in enumerate(FEATURE_NAMES):
                series = trial.features[:, column].astype(np.float64)
                for label, reduce in AGGREGATES.items():
                    values[f"{name}::{label}"] = reduce(series)
            out.append(
                Trial(
                    subject=subject,
                    index=int(trial.trial_index),
                    values=values,
                    valence=float(trial.valence),
                    arousal=float(trial.arousal),
                    quadrant=str(trial.quadrant),
                )
            )
    return out


def _residualise(y: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    """Remove a linear trend in ``covariate`` from ``y``."""
    if covariate.size < 3 or covariate.std() < 1e-12:
        return y - y.mean()
    slope, intercept = np.polyfit(covariate, y, 1)
    return y - (slope * covariate + intercept)


def pooled_correlation(
    predictor: Dict[str, np.ndarray],
    target: Dict[str, np.ndarray],
    order: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[float, float, int]:
    """Fisher-z pooled within-subject correlation.

    Returns the pooled r, the pooled z, and the total degrees of freedom.
    Combining z rather than averaging r is the standard correction for r's
    compressed tail; averaging r would understate a strong effect.
    """
    weights, zs = [], []
    for subject, x in predictor.items():
        y = target[subject]
        if x.size < 5:
            continue

        if order is not None:
            x = _residualise(x, order[subject])
            y = _residualise(y, order[subject])

        if x.std() < 1e-12 or y.std() < 1e-12:
            continue

        r = float(np.corrcoef(x, y)[0, 1])
        r = float(np.clip(r, -0.9999, 0.9999))
        # A partialled correlation loses one more degree of freedom.
        dof = x.size - 3 - (1 if order is not None else 0)
        if dof <= 0:
            continue
        zs.append(np.arctanh(r))
        weights.append(dof)

    if not zs:
        return 0.0, 0.0, 0

    weights_array = np.array(weights, dtype=float)
    pooled_z = float(np.sum(np.array(zs) * weights_array) / np.sum(weights_array))
    total = int(np.sum(weights_array))
    return float(np.tanh(pooled_z)), pooled_z, total


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """FDR-adjusted p-values, in the input order."""
    p = np.asarray(p_values, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # Enforce monotonicity from the largest p downward.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.clip(ranked, 0, 1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    subjects = sorted(
        Path(p).name.split("_")[0]
        for p in glob.glob(str(DATA_ROOT / "derivatives/emotion_features/*.npz"))
    )
    if not subjects:
        raise SystemExit("no cached emotion features; run tools/fetch_emotion.py")

    trials = collect(DATA_ROOT, subjects)
    names = sorted(trials[0].values)
    print(f"{len(subjects)} subjects, {len(trials)} trials, {len(names)} predictors\n")

    # Group by subject once; every test reuses these.
    by_subject: Dict[str, List[Trial]] = {}
    for trial in trials:
        by_subject.setdefault(trial.subject, []).append(trial)

    order = {s: np.array([t.index for t in v], dtype=float) for s, v in by_subject.items()}
    targets = {
        "valence": {s: np.array([t.valence for t in v]) for s, v in by_subject.items()},
        "arousal": {s: np.array([t.arousal for t in v]) for s, v in by_subject.items()},
        # How strongly the clip was felt at all, regardless of direction. A
        # plausible reading of "intensity" that neither axis captures alone.
        "intensity": {
            s: np.array([abs(t.valence - 5.0) for t in v]) for s, v in by_subject.items()
        },
    }

    predictors = {
        name: {s: np.array([t.values[name] for t in v]) for s, v in by_subject.items()}
        for name in names
    }

    rng = np.random.default_rng(args.seed)

    # Pre-draw the permutations so every test faces the identical null. Drawing
    # fresh ones per test would let a lucky draw favour one feature.
    shuffles: List[Dict[str, np.ndarray]] = []
    for _ in range(args.permutations):
        shuffles.append(
            {s: rng.permutation(len(v)) for s, v in by_subject.items()}
        )

    results = []
    for target_name, target in targets.items():
        for name in names:
            r, z, dof = pooled_correlation(predictors[name], target)
            r_partial, z_partial, _ = pooled_correlation(
                predictors[name], target, order=order
            )

            # Empirical two-sided p from the permutation null.
            null = np.empty(args.permutations)
            for k, shuffle in enumerate(shuffles):
                permuted = {s: target[s][idx] for s, idx in shuffle.items()}
                _, null_z, _ = pooled_correlation(predictors[name], permuted)
                null[k] = null_z
            p = float((np.sum(np.abs(null) >= abs(z)) + 1) / (args.permutations + 1))

            results.append(
                {
                    "target": target_name,
                    "predictor": name,
                    "r": r,
                    "rPartial": r_partial,
                    "z": z,
                    "p": p,
                    "dof": dof,
                    "nullSd": float(null.std()),
                }
            )
        print(f"  {target_name}: {len(names)} tests done")

    adjusted = benjamini_hochberg([entry["p"] for entry in results])
    for entry, q in zip(results, adjusted):
        entry["q"] = float(q)

    results.sort(key=lambda e: e["p"])

    print(f"\n{'target':<11} {'predictor':<34} {'r':>7} {'partial':>8} {'p':>8} {'q':>8}")
    for entry in results[:20]:
        mark = "  <-- survives FDR" if entry["q"] < 0.05 else ""
        print(
            f"{entry['target']:<11} {entry['predictor']:<34} {entry['r']:>7.3f} "
            f"{entry['rPartial']:>8.3f} {entry['p']:>8.4f} {entry['q']:>8.3f}{mark}"
        )

    survivors = [e for e in results if e["q"] < 0.05]
    print(f"\n{len(survivors)} of {len(results)} tests survive FDR at q < 0.05.")
    if not survivors:
        strongest = results[0]
        print(
            f"Strongest raw association: {strongest['predictor']} vs "
            f"{strongest['target']}, r = {strongest['r']:.3f}, p = {strongest['p']:.4f}, "
            f"q = {strongest['q']:.3f}."
        )
        print(
            f"Expected false positives at uncorrected p < 0.05: "
            f"{0.05 * len(results):.0f}. Observed: "
            f"{sum(1 for e in results if e['p'] < 0.05)}."
        )

    destination = DATA_ROOT / "derivatives" / "emotion_correlations.json"
    destination.write_text(
        json.dumps(
            {
                "subjects": subjects,
                "trials": len(trials),
                "permutations": args.permutations,
                "tests": len(results),
                "survivors": len(survivors),
                "uncorrectedHits": sum(1 for e in results if e["p"] < 0.05),
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
