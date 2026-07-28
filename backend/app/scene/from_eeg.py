"""Driving an environment from decoded EEG.

What this does and does not claim
---------------------------------

It does **not** claim the person saw this landscape. They were looking at
photographs of objects. The terrain is a fixed prior — its shape, the camera,
the position of the sun and every shadow are decided before any EEG is read,
and none of them vary.

What varies is **appearance**: how bright the scene is, how hazy, how warm or
cool the light, how saturated, how much water lies in the valleys, how barren
the ground. Each of those is bound to a visual property that EEG measurably
recovers, and to nothing else.

That binding is the honest part, and it rests on a measurement rather than on
taste. Regressing EEG onto the visual statistics of what was on screen — held
out, averaged over repetitions, correlated across the two hundred stimuli —
gives correlations that are much higher than anything object identification
achieved:

    edge density        r = 0.69
    mean red            r = 0.63
    luminance           r = 0.60
    saturation          r = 0.59
    contrast            r = 0.51
    coverage            r = 0.52
    spatial frequency   r = 0.49

with shuffled controls at |r| < 0.13 throughout. Identification of *which*
object was on screen was 7.6% correct. The lesson is consistent with
everything else in this project: scalp EEG carries **how a scene looked** far
better than **what was in it**, and an environment is exactly the kind of
output that low-level visual statistics can honestly drive.

Only properties above :data:`MIN_CORRELATION` are allowed to move anything.
The rest are held at their prior value and reported as such, so a viewer can
never mistake a rendered detail for a decoded one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..realdata.images import FEATURE_NAMES, embed_image, load_catalogue
from .terrain import Grade

log = logging.getLogger("mindscape.scene.from_eeg")

#: Held-out correlation a property must reach before it is allowed to drive
#: any part of the render.
MIN_CORRELATION = 0.35

#: Ridge penalty for the EEG-to-feature regression. Large, because the design
#: matrix is 1,134 correlated spatiotemporal samples against a few thousand
#: trials.
RIDGE_ALPHA = 1e4

#: Properties bound to the render, and what each one moves. Everything not
#: listed here is prior and stays fixed no matter what the EEG says.
BINDINGS: Tuple[Tuple[str, str], ...] = (
    ("luminance", "how bright the scene is"),
    ("contrast", "how much haze sits in the distance"),
    ("saturation", "colour intensity"),
    ("mean_r", "warmth of the light"),
    ("mean_b", "coolness of the sky"),
    ("edge_density", "how bare the ground is"),
    ("coverage", "how high the water stands"),
)


@dataclass
class VisualDecoding:
    """Held-out predictions of visual properties, in z-units."""

    #: (trials, features) predicted and true, standardised.
    predicted: np.ndarray
    truth: np.ndarray
    stimulus: np.ndarray
    onset: np.ndarray
    #: Held-out correlation per feature, across stimuli.
    correlation: np.ndarray
    feature_names: List[str]
    subject: str

    def index(self, name: str) -> Optional[int]:
        try:
            return self.feature_names.index(name)
        except ValueError:
            return None

    def usable(self) -> Dict[str, float]:
        """Properties that cleared the threshold, with their correlations."""
        out: Dict[str, float] = {}
        for name, _ in BINDINGS:
            i = self.index(name)
            if i is not None and self.correlation[i] >= MIN_CORRELATION:
                out[name] = float(self.correlation[i])
        return out


def _stimulus_features(root: Path) -> Tuple[np.ndarray, Dict[int, int], List[str]]:
    """Standardised visual features for every stimulus.

    Features with no variance across the catalogue are dropped rather than
    carried as NaN — every stimulus is square, so `aspect` is constant and
    correlating against it produces a nan that silently poisons any summary
    it lands in.
    """
    catalogue = load_catalogue(root / "stimuli")
    numbers = sorted(catalogue)
    raw = np.stack(
        [embed_image(root / "stimuli" / catalogue[n].filename) for n in numbers]
    )

    spread = raw.std(axis=0)
    keep = np.nonzero(spread > 1e-8)[0]
    dropped = [FEATURE_NAMES[i] for i in range(raw.shape[1]) if i not in set(keep)]
    if dropped:
        log.info("dropped constant features: %s", ", ".join(dropped))

    values = (raw[:, keep] - raw[:, keep].mean(axis=0)) / spread[keep]
    return values, {n: i for i, n in enumerate(numbers)}, [FEATURE_NAMES[i] for i in keep]


def decode_visual(
    root: Path, subject: str, n_folds: int = 5, shuffle: bool = False, seed: int = 0
) -> VisualDecoding:
    """Regress EEG onto visual properties, held out throughout."""
    from sklearn.linear_model import Ridge

    from ..realdata.decoding import build_features, repetition_folds
    from ..realdata.epochs import load_or_extract

    table, position, names = _stimulus_features(root)

    epochs = load_or_extract(root, subject)
    features = build_features(epochs)
    x, stimulus = features.x, features.stimulus

    truth = np.stack([table[position[int(s)]] for s in stimulus])
    if shuffle:
        rng = np.random.default_rng(seed)
        truth = truth[rng.permutation(truth.shape[0])]

    predicted = np.zeros_like(truth)
    for test in repetition_folds(stimulus, n_folds):
        train = np.setdiff1d(np.arange(x.shape[0]), test)
        mean = x[train].mean(axis=0, keepdims=True)
        scale = x[train].std(axis=0, keepdims=True)
        scale[scale < 1e-9] = 1.0
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            model = Ridge(alpha=RIDGE_ALPHA).fit((x[train] - mean) / scale, truth[train])
            predicted[test] = model.predict((x[test] - mean) / scale)

    # Correlation across *stimuli*, not across trials. Per-trial correlation
    # would be inflated by there being only two hundred distinct answers.
    unique = sorted(set(int(s) for s in stimulus))
    correlation = np.zeros(truth.shape[1])
    for j in range(truth.shape[1]):
        a = np.array([predicted[stimulus == n, j].mean() for n in unique])
        b = np.array([truth[stimulus == n, j].mean() for n in unique])
        correlation[j] = float(np.corrcoef(a, b)[0, 1])

    return VisualDecoding(
        predicted=predicted,
        truth=truth,
        stimulus=stimulus,
        onset=epochs.onset_s,
        correlation=correlation,
        feature_names=names,
        subject=subject,
    )


def _squash(value: float, limit: float = 2.2) -> float:
    """Clamp a z-score into a bounded range, smoothly."""
    return float(np.tanh(np.clip(value, -limit, limit) / limit))


def grade_from(
    vector: np.ndarray,
    decoding: VisualDecoding,
    base_water: float,
    allowed: Optional[Dict[str, float]] = None,
) -> Grade:
    """Turn one z-scored visual feature vector into an appearance.

    ``allowed`` restricts which properties may move; anything absent keeps its
    prior value. Passing the result of :meth:`VisualDecoding.usable` is what
    guarantees an under-decoded property cannot quietly shape the picture.
    """
    allowed = decoding.usable() if allowed is None else allowed

    def get(name: str) -> float:
        index = decoding.index(name)
        if index is None or name not in allowed:
            return 0.0
        return _squash(float(vector[index]))

    luminance = get("luminance")
    contrast = get("contrast")
    saturation = get("saturation")
    warm = get("mean_r")
    cool = get("mean_b")
    edges = get("edge_density")
    coverage = get("coverage")

    return Grade(
        daylight=float(np.clip(1.0 + 0.80 * luminance, 0.28, 2.0)),
        exposure=float(np.clip(1.42 + 0.48 * luminance, 0.85, 2.05)),
        # High contrast reads as clear air; low contrast as a hazy day.
        haze=float(np.clip(1.10 - 0.80 * contrast, 0.30, 2.1)),
        saturation=float(np.clip(1.35 + 0.80 * saturation, 0.40, 2.2)),
        sun_tint=(
            float(np.clip(1.0 + 0.24 * warm, 0.72, 1.28)),
            0.96,
            float(np.clip(0.88 + 0.24 * cool, 0.60, 1.16)),
        ),
        horizon=(
            float(np.clip(0.62 + 0.18 * warm, 0.38, 0.86)),
            0.72,
            float(np.clip(0.84 + 0.18 * cool, 0.58, 1.0)),
        ),
        zenith=(0.26, 0.44, float(np.clip(0.72 + 0.10 * cool, 0.55, 0.9))),
        barrenness=float(np.clip(0.42 + 0.55 * edges, 0.0, 1.0)),
        water_level=float(base_water + 85.0 * coverage),
    )
