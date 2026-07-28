"""Brain state, rendered as a place.

Everything driving the environment here is **measured, not inferred**, and the
distinction carries the whole argument.

When this project decoded *what someone was looking at*, there was an
inference gap and it had to be quantified: object identity 7.6% correct,
visual properties r = 0.47 to 0.69, emotion across subjects a documented
failure. Every one of those numbers is an error bar on a guess about the
world.

Band power is not a guess. Alpha power over occipital cortex **is** alpha
power over occipital cortex, computed from the recording by Welch's method.
Total power is the amplitude that was there. The five rhythms, the intensity,
and the way they move over time are all direct readings, and binding a
landscape to them is a faithful rendering of the signal rather than a claim
about the world.

What is *not* measured, and is labelled as interpretation wherever it appears:

* that posterior alpha indexes visual disengagement,
* that beta over alpha indexes arousal,
* that theta over beta indexes inattention,
* that frontal alpha asymmetry indexes affective valence.

These are standard readings in the literature and each is contested at the
edges. This project has already found frontal asymmetry wanting: cross-subject
emotion decoding on this very dataset failed, with a shuffled control scoring
higher than the real model (see ``EMOTION.md``). So valence gets the smallest
influence of anything here, and the caption says the mapping is convention
rather than a result.

The terrain, as always, is prior. Nobody in this dataset was on a mountain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..emotion.features import FEATURE_NAMES
from .terrain import Grade

log = logging.getLogger("mindscape.scene.cognitive")


@dataclass(frozen=True)
class Channel:
    """One measured quantity and what it moves in the world."""

    feature: str
    label: str
    drives: str
    #: "measured" for a direct reading; "interpretation" where a cognitive
    #: meaning is being assumed on top of one.
    status: str


#: The bindings, in the order they are shown. Each names the feature it reads,
#: so nothing can be moved by a quantity that is not on screen.
CHANNELS: Tuple[Channel, ...] = (
    Channel("total_power", "signal intensity", "how vivid the scene is", "measured"),
    Channel("posterior_alpha", "posterior alpha", "how far the world recedes", "interpretation"),
    Channel("alpha_suppression", "alpha supp.", "clarity of the air", "interpretation"),
    Channel("beta_alpha_ratio", "beta / alpha", "colour, and speed of travel", "interpretation"),
    Channel("frontal_theta", "frontal theta", "how bare the ground is", "interpretation"),
    Channel("theta_beta_ratio", "theta / beta", "how the light fades", "interpretation"),
    Channel("gamma_power", "gamma", "sharpness of detail", "measured"),
    Channel("frontal_asymmetry", "frontal asym.", "warmth of the light", "interpretation"),
)


@dataclass
class StateTrack:
    """A continuous brain-state trajectory, standardised per subject."""

    #: (windows, features) z-scored within this subject.
    values: np.ndarray
    #: Seconds from the start of the concatenated session.
    times: np.ndarray
    #: Which trial each window came from.
    trial: np.ndarray
    #: Stimulus label per window, for the readout.
    stimulus: List[str]
    subject: str
    feature_names: List[str]

    def index(self, name: str) -> Optional[int]:
        try:
            return self.feature_names.index(name)
        except ValueError:
            return None

    def at(self, window: int, name: str) -> float:
        i = self.index(name)
        return 0.0 if i is None else float(self.values[window, i])


def build_track(trials: Sequence, subject: str) -> StateTrack:
    """Concatenate a subject's trials into one continuous state trajectory.

    Standardisation is **within subject**, which is not optional: absolute EEG
    power differs more between two people's skulls than between any two states
    in the same person. Without it the render would be showing head size.
    """
    ordered = sorted(trials, key=lambda t: t.trial_index)

    blocks, times, trial_ids, labels = [], [], [], []
    clock = 0.0
    for trial in ordered:
        if trial.features.size == 0:
            continue
        blocks.append(trial.features)
        times.append(np.asarray(trial.times, dtype=float) + clock)
        trial_ids.append(np.full(trial.features.shape[0], trial.trial_index))
        labels.extend([str(trial.stimulus)] * trial.features.shape[0])
        clock += float(trial.duration) if trial.duration else float(trial.times[-1])

    values = np.concatenate(blocks).astype(np.float64)
    mean = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    scale[scale < 1e-9] = 1.0

    return StateTrack(
        values=(values - mean) / scale,
        times=np.concatenate(times),
        trial=np.concatenate(trial_ids),
        stimulus=labels,
        subject=subject,
        feature_names=list(FEATURE_NAMES),
    )


def _squash(value: float, limit: float = 2.2) -> float:
    return float(np.tanh(np.clip(value, -limit, limit) / limit))


def grade_from_state(track: StateTrack, window: int, base_water: float) -> Grade:
    """One frame's appearance, from one moment of brain state."""
    intensity = _squash(track.at(window, "total_power"))
    alpha = _squash(track.at(window, "posterior_alpha"))
    suppression = _squash(track.at(window, "alpha_suppression"))
    arousal = _squash(track.at(window, "beta_alpha_ratio"))
    theta = _squash(track.at(window, "frontal_theta"))
    inattention = _squash(track.at(window, "theta_beta_ratio"))
    gamma = _squash(track.at(window, "gamma_power"))
    valence = _squash(track.at(window, "frontal_asymmetry"))

    return Grade(
        # Vividness from raw signal intensity, dimmed by inattention.
        daylight=float(np.clip(1.0 + 0.55 * intensity - 0.35 * inattention, 0.30, 1.95)),
        exposure=float(np.clip(1.42 + 0.34 * intensity, 0.90, 2.0)),
        # Alpha rising is the classic signature of the visual world dropping
        # away; here that is distance filling with haze.
        haze=float(np.clip(1.05 + 0.75 * alpha - 0.45 * suppression, 0.30, 2.2)),
        saturation=float(np.clip(1.35 + 0.60 * arousal, 0.45, 2.2)),
        # Valence moves the least of anything, because it is the one reading
        # this project has already tested and found unreliable.
        sun_tint=(
            float(np.clip(1.0 + 0.10 * valence, 0.88, 1.12)),
            0.96,
            float(np.clip(0.88 - 0.08 * valence, 0.76, 1.0)),
        ),
        horizon=(
            float(np.clip(0.62 + 0.10 * valence, 0.46, 0.78)),
            0.72,
            float(np.clip(0.84 + 0.12 * alpha, 0.66, 0.98)),
        ),
        zenith=(0.26, 0.44, float(np.clip(0.72 + 0.10 * alpha, 0.58, 0.88))),
        barrenness=float(np.clip(0.42 + 0.40 * theta - 0.25 * gamma, 0.0, 1.0)),
        water_level=float(base_water + 45.0 * alpha),
    )


def travel_rate(track: StateTrack, window: int) -> float:
    """How fast the viewpoint moves through the memory at this moment.

    Bound to arousal, so a alert stretch of the recording covers more ground
    than a flat one. Returns a multiplier around 1.
    """
    arousal = _squash(track.at(window, "beta_alpha_ratio"))
    inattention = _squash(track.at(window, "theta_beta_ratio"))
    return float(np.clip(1.0 + 0.85 * arousal - 0.45 * inattention, 0.15, 2.2))
