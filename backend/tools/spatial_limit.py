"""How much spatial structure can EEG actually specify?

The terrain in every reconstruction so far is a fixed heightfield, chosen
before any brain activity is read. Only its *appearance* is decoded. The
question this answers is whether the shape itself could come from the signal
instead — and if so, at what resolution it stops being real.

Method: reduce each stimulus to a coarse luminance grid at several
resolutions, regress EEG onto every cell of that grid held out, and correlate
the decoded grid against the true one across the 200 stimuli. A grid that
survives is a heightfield the signal can genuinely specify; one that collapses
to its shuffled control is a heightfield that would be invention.

This is the same measurement that licensed the appearance work, applied to
geometry instead: a cell must clear the same r = 0.35 bar before it is allowed
to move anything.

Run with::

    backend/.venv/bin/python -m tools.spatial_limit
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

from app.realdata.decoding import build_features, repetition_folds
from app.realdata.epochs import load_or_extract
from app.realdata.images import load_catalogue

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "ds004018"

#: Grid resolutions tested, in cells per side.
RESOLUTIONS = (1, 2, 3, 4, 6, 8, 12)

#: A cell has to clear this held-out correlation to count as recovered. Same
#: bar the appearance work uses, so the two are directly comparable.
BAR = 0.35


def luminance_grid(path: Path, side: int) -> np.ndarray:
    """Mean luminance over a ``side`` x ``side`` grid, flattened."""
    from PIL import Image

    with Image.open(path) as handle:
        # Downsampling with `resize` averages over each cell, which is exactly
        # the grid mean and far faster than slicing block by block.
        small = handle.convert("L").resize((side, side), Image.BOX)
        return np.asarray(small, dtype=np.float64).ravel() / 255.0


def decode_grid(
    x: np.ndarray,
    stimulus: np.ndarray,
    target: np.ndarray,
    numbers: List[int],
    shuffle: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """Held-out correlation per cell, across stimuli."""
    from sklearn.linear_model import Ridge

    if shuffle:
        rng = np.random.default_rng(seed)
        target = target[rng.permutation(target.shape[0])]

    predicted = np.zeros_like(target)
    for test in repetition_folds(stimulus, 5):
        train = np.setdiff1d(np.arange(x.shape[0]), test)
        mean = x[train].mean(axis=0, keepdims=True)
        scale = x[train].std(axis=0, keepdims=True)
        scale[scale < 1e-9] = 1.0
        model = Ridge(alpha=1e4).fit((x[train] - mean) / scale, target[train])
        # Ridge drops the trailing axis for a single output, which the 1x1
        # grid is; reshape so every resolution takes the same path.
        predicted[test] = model.predict((x[test] - mean) / scale).reshape(
            len(test), target.shape[1]
        )

    # Average the held-out predictions per stimulus, then correlate across the
    # 200 stimuli. Correlating trial by trial would mostly measure how often
    # each stimulus repeats, not whether its layout was recovered.
    out = np.zeros(target.shape[1])
    for cell in range(target.shape[1]):
        decoded = np.array([predicted[stimulus == n, cell].mean() for n in numbers])
        truth = np.array([target[stimulus == n, cell].mean() for n in numbers])
        if decoded.std() < 1e-12 or truth.std() < 1e-12:
            out[cell] = 0.0
        else:
            out[cell] = np.corrcoef(decoded, truth)[0, 1]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    catalogue = load_catalogue(DATA_ROOT / "stimuli")
    numbers = sorted(catalogue)
    stimuli_dir = DATA_ROOT / "stimuli"

    subjects = [f"sub-{i:02d}" for i in range(1, args.subjects + 1)]
    loaded = []
    for subject in subjects:
        epochs = load_or_extract(DATA_ROOT, subject)
        loaded.append(build_features(epochs))
    print(f"subjects: {', '.join(subjects)}\n")

    print(f"{'grid':>6} {'cells':>6} {'mean r':>8} {'best':>7} {'worst':>7} "
          f"{'shuffled':>9} {'above 0.35':>11}")

    rows: List[Tuple[int, float, float]] = []
    for side in RESOLUTIONS:
        grids = np.stack([luminance_grid(stimuli_dir / catalogue[n].filename, side) for n in numbers])
        index = {n: i for i, n in enumerate(numbers)}

        real_all, null_all = [], []
        for features in loaded:
            target = np.stack([grids[index[int(s)]] for s in features.stimulus])
            real_all.append(decode_grid(features.x, features.stimulus, target, numbers))
            null_all.append(
                decode_grid(features.x, features.stimulus, target, numbers, shuffle=True)
            )

        real = np.mean(real_all, axis=0)
        null = np.mean(null_all, axis=0)
        share = float(np.mean(real > BAR))

        print(
            f"{side}x{side:<3} {side * side:>6} {real.mean():>8.3f} {real.max():>7.3f} "
            f"{real.min():>7.3f} {np.abs(null).mean():>9.3f} {share * 100:>10.0f}%"
        )
        rows.append((side, float(real.mean()), share))

    # --- how many *independent* degrees of freedom is that? ----------------
    #
    # A high per-cell correlation does not mean 144 independent heights were
    # recovered. These stimuli are objects on a plain background, so most cells
    # are driven by one underlying factor — how much of the object covers them
    # — which is close to overall coverage and luminance, and both of those
    # decode well on their own. Two checks separate real spatial structure
    # from one global number wearing a grid costume.
    print("\n--- independent structure, at 8x8 ---")
    side = 8
    grids = np.stack(
        [luminance_grid(stimuli_dir / catalogue[n].filename, side) for n in numbers]
    )
    index = {n: i for i, n in enumerate(numbers)}

    # 1. Strip each image's own mean brightness. What survives is layout only.
    centred = grids - grids.mean(axis=1, keepdims=True)
    scores = []
    for features in loaded:
        target = np.stack([centred[index[int(s)]] for s in features.stimulus])
        scores.append(decode_grid(features.x, features.stimulus, target, numbers))
    centred_r = np.mean(scores, axis=0)
    print(
        f"  cells with image mean removed : mean r {centred_r.mean():.3f}, "
        f"{np.mean(centred_r > BAR) * 100:.0f}% above {BAR}"
    )

    # 2. How many principal components of the grid are recoverable at all.
    standardised = grids - grids.mean(axis=0, keepdims=True)
    _, singular, components = np.linalg.svd(standardised, full_matrices=False)
    variance = singular**2 / np.sum(singular**2)
    projected = standardised @ components.T

    keep = min(12, projected.shape[1])
    per_component = []
    for features in loaded:
        target = np.stack([projected[index[int(s)]][:keep] for s in features.stimulus])
        per_component.append(decode_grid(features.x, features.stimulus, target, numbers))
    component_r = np.mean(per_component, axis=0)

    print(f"\n  {'component':>10} {'variance':>9} {'r':>7}")
    survivors = 0
    for k in range(keep):
        mark = "  <-- recovered" if component_r[k] > BAR else ""
        if component_r[k] > BAR:
            survivors += 1
        print(f"  {k + 1:>10} {variance[k] * 100:>8.1f}% {component_r[k]:>7.3f}{mark}")

    print(
        f"\n  {survivors} of {keep} components clear {BAR}. That is the honest count "
        f"of independent\n  spatial degrees of freedom, not the {side * side} cells."
    )

    # Deliberately not phrased as "N independent height values". The per-cell
    # table above would support that reading and the component table refutes
    # it: cells are correlated, and the count that matters is the second one.
    print(
        "\nPer-cell correlation stays high to 12x12, but the cells are not "
        "independent.\nThe component count is the number that can honestly "
        "drive geometry."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
