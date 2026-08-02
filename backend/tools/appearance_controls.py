"""Is the appearance result cortical, or is it the eyes?

The terrain reconstruction rests entirely on one measurement: EEG regressed
onto the visual statistics of what was on screen, at r = 0.47 to 0.69. That
number has never had an artifact control, and the motor work already showed why
that is not good enough — there, the failure mode was never a result that
collapses but one that looks excellent while decoding somebody's jaw.

Here the specific worry is different and more likely. Visual stimuli evoke eye
movements and blinks; the corneo-retinal dipole is an order of magnitude larger
than cortical signal and lands hardest on frontal electrodes. A decoder handed
Fp1 and Fp2 could read luminance off the eyes and never touch visual cortex.
Brightness in particular drives pupil and lid behaviour directly, so "luminance
decodes at r = 0.62" is exactly the result an ocular artifact would produce.

Two controls, both by construction rather than by cleaning:

* **Spatial.** Fit the same regression on electrode groups separately. If this
  is vision, occipital and parieto-occipital sites carry it and frontal ones
  add little. If it is ocular, the frontal pole wins.
* **Spectral.** Ocular artifacts are low-frequency — a blink is a fraction of a
  second of large deflection, concentrated below about 4 Hz.

  This one turned out to have little power here, and the reason is worth
  keeping. The motor controls could lean on frequency because mu and beta
  desynchronisation is an *oscillation*, so splitting by band separates it from
  broadband muscle. A visual evoked response is a transient waveform, and a
  transient is broadband by construction — so a flat profile across bands is
  what genuine ERP decoding looks like, not evidence either way. It is reported
  because it was run, not because it settles anything.

Run with::

    backend/.venv/bin/python -m tools.appearance_controls
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

from scipy import signal as sp_signal
from sklearn.linear_model import Ridge

from app.realdata.decoding import DECODE_WINDOW, repetition_folds
from app.realdata.epochs import load_or_extract
from app.realdata.images import embed_image, load_catalogue, FEATURE_NAMES

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "ds004018"

#: The properties the environment render is actually allowed to use.
TRACKED = ("luminance", "contrast", "saturation", "mean_r", "edge_density", "coverage")

#: Electrode groups. Named by what they would implicate, not by convenience.
GROUPS: Dict[str, Tuple[str, ...]] = {
    "occipital": ("O1", "Oz", "O2", "PO7", "PO3", "POz", "PO4", "PO8"),
    "parietal": ("P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8"),
    "central": ("C5", "C3", "C1", "Cz", "C2", "C4", "C6", "FCz"),
    "frontal pole (ocular)": ("Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8"),
    "temporal (muscle)": ("T7", "T8", "FT7", "FT8", "FT9", "FT10", "TP7", "TP8"),
}

#: Bands. Blinks and saccades dominate the lowest one; a visual evoked response
#: is broader and reaches higher.
BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("0.5-4 Hz (ocular)", 0.5, 4.0),
    ("4-8 Hz", 4.0, 8.0),
    ("8-15 Hz", 8.0, 15.0),
    ("15-40 Hz", 15.0, 40.0),
)


def visual_targets(root: Path) -> Tuple[Dict[int, np.ndarray], List[str]]:
    catalogue = load_catalogue(root / "stimuli")
    numbers = sorted(catalogue)
    matrix = np.stack([embed_image(root / "stimuli" / catalogue[n].filename) for n in numbers])
    matrix = (matrix - matrix.mean(0)) / np.clip(matrix.std(0), 1e-9, None)
    return {n: matrix[i] for i, n in enumerate(numbers)}, numbers


def decode(x: np.ndarray, stimulus: np.ndarray, target: np.ndarray,
           numbers: Sequence[int]) -> np.ndarray:
    """Held-out correlation per target column, across stimuli."""
    predicted = np.zeros_like(target)
    for test in repetition_folds(stimulus, 5):
        train = np.setdiff1d(np.arange(x.shape[0]), test)
        mean = x[train].mean(0, keepdims=True)
        scale = np.clip(x[train].std(0, keepdims=True), 1e-9, None)
        model = Ridge(alpha=1e4).fit((x[train] - mean) / scale, target[train])
        predicted[test] = model.predict((x[test] - mean) / scale).reshape(
            len(test), target.shape[1]
        )

    out = np.zeros(target.shape[1])
    for column in range(target.shape[1]):
        a = np.array([predicted[stimulus == n, column].mean() for n in numbers])
        b = np.array([target[stimulus == n, column].mean() for n in numbers])
        out[column] = 0.0 if a.std() < 1e-12 else np.corrcoef(a, b)[0, 1]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    lookup, numbers = visual_targets(DATA_ROOT)
    columns = [FEATURE_NAMES.index(t) for t in TRACKED]

    subjects = [f"sub-{i:02d}" for i in range(1, args.subjects + 1)]
    loaded = []
    for subject in subjects:
        epochs = load_or_extract(DATA_ROOT, subject)
        loaded.append(epochs)
    print(f"subjects: {', '.join(subjects)}\n")

    names = list(loaded[0].channel_names)
    index = {n: i for i, n in enumerate(names)}
    times = loaded[0].times
    mask = (times >= DECODE_WINDOW[0]) & (times <= DECODE_WINDOW[1])
    picks = np.linspace(np.nonzero(mask)[0][0], np.nonzero(mask)[0][-1], 18).astype(int)

    report: Dict = {"subjects": subjects, "tracked": list(TRACKED), "spatial": {}, "spectral": {}}

    # ---------------------------------------------------------------- spatial
    print("SPATIAL — which electrodes carry it")
    print(f"  {'group':<24} {'ch':>3}  " + "".join(f"{t[:9]:>10}" for t in TRACKED))
    for label, wanted in GROUPS.items():
        rows = [index[c] for c in wanted if c in index]
        if not rows:
            continue
        scores = []
        for epochs in loaded:
            data = epochs.data[:, rows][:, :, picks]
            x = data.reshape(data.shape[0], -1).astype(np.float64)
            stimulus = np.asarray(epochs.stimulus, dtype=np.int32)
            target = np.stack([lookup[int(s)][columns] for s in stimulus])
            scores.append(decode(x, stimulus, target, numbers))
        mean = np.mean(scores, axis=0)
        report["spatial"][label] = {"channels": len(rows), "r": mean.tolist()}
        print(f"  {label:<24} {len(rows):>3}  " + "".join(f"{v:>10.3f}" for v in mean))

    # --------------------------------------------------------------- spectral
    print("\nSPECTRAL — which frequencies carry it")
    print(f"  {'band':<24} {'':>3}  " + "".join(f"{t[:9]:>10}" for t in TRACKED))
    for label, low, high in BANDS:
        scores = []
        for epochs in loaded:
            rate = float(epochs.sampling_rate) if hasattr(epochs, "sampling_rate") else 250.0
            nyq = rate / 2.0
            sos = sp_signal.butter(
                4, [low / nyq, min(high / nyq, 0.99)], btype="band", output="sos"
            )
            filtered = sp_signal.sosfiltfilt(sos, epochs.data.astype(np.float64), axis=-1)
            data = filtered[:, :, picks]
            x = data.reshape(data.shape[0], -1)
            stimulus = np.asarray(epochs.stimulus, dtype=np.int32)
            target = np.stack([lookup[int(s)][columns] for s in stimulus])
            scores.append(decode(x, stimulus, target, numbers))
        mean = np.mean(scores, axis=0)
        report["spectral"][label] = {"r": mean.tolist()}
        print(f"  {label:<24} {'':>3}  " + "".join(f"{v:>10.3f}" for v in mean))

    # ----------------------------------------------------------- how many
    # If eight occipital electrodes carry most of it, the practical question
    # changes: a research cap is not required to reproduce this, and the
    # channel budget is worth knowing before anyone buys hardware.
    print("\nCHANNEL BUDGET — what a smaller montage would get")
    budgets: Dict[str, Tuple[str, ...]] = {
        "all 63": tuple(names),
        "10-20 nineteen": (
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
            "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
        ),
        "occipital 8": GROUPS["occipital"],
        "occipital 4": ("O1", "Oz", "O2", "POz"),
        "occipital 2": ("O1", "O2"),
    }
    print(f"  {'montage':<18} {'ch':>3}  " + "".join(f"{t[:9]:>10}" for t in TRACKED))
    for label, wanted in budgets.items():
        rows = [index[c] for c in wanted if c in index]
        if not rows:
            continue
        scores = []
        for epochs in loaded:
            data = epochs.data[:, rows][:, :, picks]
            x = data.reshape(data.shape[0], -1).astype(np.float64)
            stimulus = np.asarray(epochs.stimulus, dtype=np.int32)
            target = np.stack([lookup[int(s)][columns] for s in stimulus])
            scores.append(decode(x, stimulus, target, numbers))
        mean = np.mean(scores, axis=0)
        report.setdefault("budget", {})[label] = {
            "channels": len(rows), "r": mean.tolist()
        }
        print(f"  {label:<18} {len(rows):>3}  " + "".join(f"{v:>10.3f}" for v in mean))

    destination = DATA_ROOT / "derivatives" / "appearance_controls.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
