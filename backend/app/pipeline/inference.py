"""Cognitive state and emotion inference.

Both models are transparent, weight-based scorers over the spectral and
feature descriptors produced upstream. They are deliberately *not* black
boxes: every score can be traced to a named neural measure, which is what
makes the evidence strings on memory fragments truthful rather than decorative.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..models.neural import CognitiveReading, EmotionReading
from ..processing.bands import engagement_index, theta_beta_ratio
from ..processing.features import WindowFeatures
from .normalisation import z, zband

# --------------------------------------------------------------------------- #
# Cognitive state                                                             #
# --------------------------------------------------------------------------- #

#: Per-state band affinity. Rows are states, values weight the relative band
#: power. Derived from standard band–state associations rather than fitted.
_STATE_BAND_WEIGHTS: Dict[str, Dict[str, float]] = {
    "focused":    {"delta": -0.6, "theta": -0.1, "alpha": -0.8, "beta": 1.6, "gamma": 0.7},
    "alert":      {"delta": -0.9, "theta": -0.5, "alpha": -0.6, "beta": 1.4, "gamma": 1.1},
    "relaxed":    {"delta": -0.2, "theta": 0.2, "alpha": 1.5, "beta": -0.5, "gamma": -0.4},
    "meditative": {"delta": 0.2, "theta": 1.0, "alpha": 1.3, "beta": -0.8, "gamma": -0.5},
    "drowsy":     {"delta": 1.5, "theta": 1.0, "alpha": -0.2, "beta": -0.9, "gamma": -0.7},
    "dreaming":   {"delta": 0.3, "theta": 1.6, "alpha": -0.6, "beta": 0.2, "gamma": 0.9},
    "recalling":  {"delta": -0.2, "theta": 1.4, "alpha": 0.7, "beta": -0.1, "gamma": 0.1},
}

#: Additional feature-space evidence, applied on top of the band score.
#:
#: Each entry is ``(feature, target z-score, weight)``. Targets are expressed
#: in robust z units against the measured feature distribution (see
#: ``normalisation.FEATURE_STATS``) rather than raw values — a target written
#: in raw units from intuition can fall outside the feature's actual range and
#: contribute nothing at all, which is exactly what happened to an earlier
#: version of this table.
_STATE_FEATURE_PRIORS: Dict[str, List[Tuple[str, float, float]]] = {
    # Externally directed attention: coupled montage, broadband spectrum.
    "focused":    [("connectivity", 1.0, 0.5), ("spectral_entropy", 0.3, 0.3)],
    # Vigilant: fast crossings, high spectral edge.
    "alert":      [("zero_crossing", 1.2, 0.5), ("spectral_edge", 0.8, 0.4)],
    # Posterior-dominant with an unremarkable montage.
    "relaxed":    [("posterior_ratio", 0.6, 0.6), ("connectivity", -0.4, 0.2)],
    # Deeper posterior dominance with elevated coupling.
    "meditative": [("posterior_ratio", 1.1, 0.7), ("connectivity", 0.5, 0.3)],
    # Slow and simple: low mobility, low spectral edge, low entropy.
    "drowsy":     [("mobility", -1.2, 0.6), ("spectral_edge", -1.3, 0.5)],
    # Broadband and decoupled — the REM signature.
    "dreaming":   [("spectral_entropy", 0.9, 0.6), ("connectivity", -1.0, 0.5)],
    # Internally directed: posterior alpha with moderate coupling.
    "recalling":  [("posterior_ratio", 0.4, 0.6), ("complexity", -0.6, 0.4)],
}


def _softmax(scores: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    if not scores:
        return {}
    values = np.array(list(scores.values()), dtype=np.float64) / max(temperature, 1e-6)
    values -= values.max()
    exp = np.exp(values)
    total = exp.sum()
    if total <= 0:
        uniform = 1.0 / len(scores)
        return {k: uniform for k in scores}
    return {k: float(v) for k, v in zip(scores.keys(), exp / total)}


def infer_cognitive(bands: Dict[str, float], features: WindowFeatures) -> CognitiveReading:
    """Score every cognitive state, then softmax into a distribution."""
    scores: Dict[str, float] = {}

    for state, weights in _STATE_BAND_WEIGHTS.items():
        # Band evidence in z-space, so a weight means "this many standard
        # deviations above what this band typically shows" rather than a
        # comparison against a nominal even spectrum the data never matches.
        score = sum(
            weights.get(band, 0.0) * zband(band, bands.get(band, 0.0)) for band in weights
        )
        score *= 0.55

        # Feature evidence: proximity to the state's expected profile, in
        # z-space so one Gaussian width is meaningful for every feature.
        for attribute, target_z, weight in _STATE_FEATURE_PRIORS.get(state, []):
            actual_z = z(attribute, getattr(features, attribute, 0.0))
            score += weight * math.exp(-(((actual_z - target_z) / 1.4) ** 2))

        scores[state] = score

    distribution = _softmax(scores, temperature=0.8)
    state = max(distribution, key=lambda k: distribution[k])

    # Confidence is the winning margin, not the winning probability: a state
    # winning 0.30 to 0.28 is a genuinely ambiguous window and should drive a
    # fragmented reconstruction, even though 0.30 looks respectable alone.
    ordered = sorted(distribution.values(), reverse=True)
    margin = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
    confidence = float(np.clip(0.35 + margin * 2.6, 0.0, 1.0))

    load = float(np.clip(theta_beta_ratio(bands) / 4.0, 0.0, 1.0))

    return CognitiveReading(
        state=state,
        confidence=round(confidence, 4),
        distribution={k: round(v, 4) for k, v in distribution.items()},
        engagement=round(engagement_index(bands), 4),
        load=round(load, 4),
    )


# --------------------------------------------------------------------------- #
# Emotion                                                                     #
# --------------------------------------------------------------------------- #

#: Each emotion's anchor in the valence–arousal plane, plus a band affinity
#: that separates emotions sharing a quadrant (e.g. wonder vs joy).
#: Anchors are spread deliberately wide. Several of these states are genuine
#: neighbours in the affective plane (calm/nostalgia, stress/fear,
#: joy/wonder), so the band affinity — not the plane distance — is what has
#: to separate them, and it is weighted accordingly below.
_EMOTION_ANCHORS: Dict[str, Dict[str, object]] = {
    "calm":       {"valence": 0.45, "arousal": 0.12, "bands": {"alpha": 1.7, "theta": -0.4, "beta": -0.8}},
    "nostalgia":  {"valence": 0.35, "arousal": 0.45, "bands": {"theta": 1.8, "alpha": 0.4, "gamma": -0.5}},
    "wonder":     {"valence": 0.60, "arousal": 0.72, "bands": {"gamma": 1.6, "theta": 0.9, "beta": -0.3}},
    "joy":        {"valence": 0.85, "arousal": 0.60, "bands": {"beta": 1.1, "alpha": 0.5, "delta": -0.9}},
    "curiosity":  {"valence": 0.25, "arousal": 0.62, "bands": {"beta": 1.3, "gamma": 0.3, "theta": -0.6}},
    "melancholy": {"valence": -0.40, "arousal": 0.22, "bands": {"delta": 1.4, "theta": 0.6, "beta": -0.7}},
    "stress":     {"valence": -0.55, "arousal": 0.75, "bands": {"beta": 1.6, "alpha": -1.2, "gamma": 0.2}},
    "fear":       {"valence": -0.80, "arousal": 0.95, "bands": {"gamma": 1.9, "beta": 0.5, "delta": -0.8}},
}


# Fitted affect coefficients (tools/fit_affect.py). Arousal R² 0.34, r +0.58;
# valence-from-asymmetry R² 0.30, r +0.55.
VALENCE_GAIN = 0.476
AROUSAL_INTERCEPT = 0.308
AROUSAL_FAST_SHARE = 0.759
AROUSAL_MOBILITY = 0.035
AROUSAL_EDGE = 0.015
AROUSAL_ENTROPY = -0.063

# A least-squares fit is the minimum-variance estimator, so its predictions
# shrink toward the mean by roughly the correlation coefficient. That is the
# right answer for a point estimate and the wrong one here: the discrete
# emotions are anchored across the *full* valence–arousal plane, and a
# readout compressed into the middle third can never reach `calm` at the low
# end or `fear` at the high end. Expanding deviations by 1/r restores the
# spread of the underlying quantity at the cost of some pointwise accuracy —
# the standard variance-matching correction for a shrunken predictor.
AROUSAL_CENTRE = 0.55
AROUSAL_EXPANSION = 1.0 / 0.583
VALENCE_EXPANSION = 1.0 / 0.547


def infer_emotion(bands: Dict[str, float], features: WindowFeatures) -> EmotionReading:
    """Estimate valence and arousal, then resolve to a discrete distribution.

    Valence comes from frontal alpha asymmetry — greater relative right-frontal
    alpha (less right activation) accompanies approach-oriented affect. Arousal
    combines beta/gamma dominance with signal mobility.
    """
    # Coefficients below are fitted, not hand-picked — see `tools/fit_affect.py`,
    # which regresses these readouts against the simulator's scripted ground
    # truth. Re-run it if the feature definitions change.

    # Valence: frontal alpha asymmetry is the one physiologically grounded
    # predictor here, so it is used alone. Regression finds raw alpha power
    # correlates slightly better, but only because relaxed alpha-rich phases
    # happen to carry positive valence in the profile library — fitting that
    # would encode a quirk of the test data rather than a real relationship.
    # No intercept: zero asymmetry must mean neutral valence, and the fitted
    # offset only reflects the profile mix skewing positive.
    valence = float(
        np.clip(features.frontal_asymmetry * VALENCE_GAIN * VALENCE_EXPANSION, -1.0, 1.0)
    )

    # Arousal: share of total power in the fast bands, modulated by signal
    # mobility, spectral edge and entropy. A ratio rather than a difference,
    # so the measure stays bounded and centred.
    total_power = sum(bands.values()) or 1.0
    fast_share = (bands.get("beta", 0.0) + bands.get("gamma", 0.0)) / total_power

    arousal_fitted = (
        AROUSAL_INTERCEPT
        + AROUSAL_FAST_SHARE * fast_share
        + AROUSAL_MOBILITY * z("mobility", features.mobility)
        + AROUSAL_EDGE * z("spectral_edge", features.spectral_edge)
        + AROUSAL_ENTROPY * z("spectral_entropy", features.spectral_entropy)
    )
    arousal = float(
        np.clip(
            AROUSAL_CENTRE + (arousal_fitted - AROUSAL_CENTRE) * AROUSAL_EXPANSION,
            0.0,
            1.0,
        )
    )

    scores: Dict[str, float] = {}
    for emotion, anchor in _EMOTION_ANCHORS.items():
        target_v = float(anchor["valence"])
        target_a = float(anchor["arousal"])

        # Distance in the affective plane — the dominant term.
        distance = math.hypot((valence - target_v) * 0.75, arousal - target_a)
        score = 3.4 * math.exp(-((distance / 0.42) ** 2))

        # Band affinity separates emotions that share an affective quadrant.
        # Weighted above the plane term: with eight states over two bounded
        # axes, plane distance alone cannot resolve neighbours.
        band_weights = anchor["bands"]
        assert isinstance(band_weights, dict)
        score += 0.62 * sum(
            w * zband(b, bands.get(b, 0.0)) for b, w in band_weights.items()
        )

        scores[emotion] = score

    distribution = _softmax(scores, temperature=0.65)
    ranked = sorted(distribution.items(), key=lambda kv: kv[1], reverse=True)

    primary = ranked[0][0]
    secondary = ranked[1][0] if len(ranked) > 1 and ranked[1][1] > 0.12 else None

    # Intensity blends arousal with how far valence sits from neutral, scaled
    # by how decisively one emotion won.
    decisiveness = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)
    intensity = float(
        np.clip(0.35 * arousal + 0.35 * abs(valence) + 0.55 * decisiveness, 0.0, 1.0)
    )

    return EmotionReading(
        primary=primary,
        secondary=secondary,
        intensity=round(intensity, 4),
        valence=round(valence, 4),
        arousal=round(arousal, 4),
        distribution={k: round(v, 4) for k, v in distribution.items()},
    )


# --------------------------------------------------------------------------- #
# Memory attributes                                                           #
# --------------------------------------------------------------------------- #


class MemoryAttributes:
    """Latent episodic attributes inferred from one window.

    These are the bridge between neural measures and scene parameters: the
    context stage reads only these, never the raw spectrum.
    """

    __slots__ = (
        "spatial_scale",
        "familiarity",
        "population",
        "object_salience",
        "vividness",
        "temporal_depth",
        "self_reference",
    )

    def __init__(
        self,
        spatial_scale: float,
        familiarity: float,
        population: float,
        object_salience: float,
        vividness: float,
        temporal_depth: float,
        self_reference: float,
    ) -> None:
        self.spatial_scale = spatial_scale
        self.familiarity = familiarity
        self.population = population
        self.object_salience = object_salience
        self.vividness = vividness
        self.temporal_depth = temporal_depth
        self.self_reference = self_reference

    def as_dict(self) -> Dict[str, float]:
        return {slot: round(getattr(self, slot), 4) for slot in self.__slots__}

    @classmethod
    def aggregate(
        cls, samples: List["MemoryAttributes"], weights: Optional[List[float]] = None
    ) -> "MemoryAttributes":
        """Combine per-window attributes into one description of the memory.

        A weighted *median* rather than a mean: a handful of artifact-hit
        windows can drag a mean far enough to change which biome is selected,
        and the choice of setting should reflect the bulk of the evidence.
        Weights (typically per-window confidence) are applied by resampling
        the cumulative distribution.
        """
        if not samples:
            return cls(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        if len(samples) == 1:
            return samples[0]

        w = np.asarray(weights if weights else [1.0] * len(samples), dtype=np.float64)
        w = np.maximum(w, 1e-6)

        values = {}
        for slot in cls.__slots__:
            column = np.array([getattr(s, slot) for s in samples], dtype=np.float64)
            order = np.argsort(column)
            sorted_values = column[order]
            cumulative = np.cumsum(w[order])
            half = cumulative[-1] / 2.0
            index = int(np.searchsorted(cumulative, half))
            values[slot] = float(sorted_values[min(index, len(sorted_values) - 1)])

        return cls(**values)


def infer_memory_attributes(
    bands: Dict[str, float],
    features: WindowFeatures,
    cognitive: CognitiveReading,
    emotion: EmotionReading,
) -> MemoryAttributes:
    """Map spectral and feature descriptors onto episodic memory attributes.

    Each mapping has a stated rationale:

    * **spatial scale** — low-frequency dominance and low connectivity
      accompany broad, diffuse spatial representations; tight fast-band
      activity accompanies near-field, detailed ones.
    * **familiarity** — frontal-midline theta is the canonical correlate of
      successful retrieval of well-consolidated material.
    * **population** — sustained beta with high inter-channel coupling tracks
      social and externally directed attention.
    * **object salience** — gamma and spectral edge track fine-detail binding.
    * **vividness** — gamma with high coherence marks strongly reinstated
      sensory content.
    * **temporal depth** — delta with low mobility accompanies remote,
      time-distant retrieval.
    * **self-reference** — posterior alpha with internally directed attention
      is the default-mode signature of self-referential processing.
    """
    # Each attribute is a weighted sum of z-scored evidence passed through a
    # logistic. Working in z-space and squashing at the end keeps every
    # attribute centred near 0.5 with usable spread across recordings.
    #
    # Raw band values with additive constants do not: relative power is
    # bounded and skewed, so hand-chosen coefficients saturate almost every
    # attribute near 1.0 and the biome selector downstream collapses onto
    # whichever setting expects the highest values.
    zb = lambda band: zband(band, bands.get(band, 0.0))  # noqa: E731
    zf = lambda name: z(name, getattr(features, name))  # noqa: E731

    def squash(x: float) -> float:
        return float(np.clip(1.0 / (1.0 + math.exp(-x)), 0.0, 1.0))

    # Theta deliberately absent: it is the retrieval/familiarity marker used
    # just below, and letting it drive spatial extent as well double-counts it.
    # An interior memory is theta-rich *and* small-scale, so a shared term
    # makes those two attributes contradict each other and no interior setting
    # can ever be selected.
    spatial_scale = squash(
        0.60 * zb("delta") - 0.55 * zb("beta") - 0.45 * zf("connectivity")
    )
    familiarity = squash(
        0.75 * zb("theta") + 0.30 * zf("posterior_ratio") - 0.35 * zb("gamma")
    )
    population = squash(
        0.65 * zb("beta") + 0.45 * zf("connectivity") - 0.40 * zb("alpha")
    )
    object_salience = squash(0.70 * zb("gamma") + 0.45 * zf("spectral_edge"))
    vividness = squash(
        0.55 * zb("gamma")
        + 0.35 * zf("connectivity")
        + 1.6 * (emotion.intensity - 0.45)
    )
    temporal_depth = squash(
        0.70 * zb("delta") - 0.45 * zf("mobility") + 0.25 * zb("theta")
    )
    self_reference = squash(
        0.55 * zb("alpha") + 0.55 * zf("posterior_ratio") - 0.30 * zf("zero_crossing")
    )

    clip = lambda v: float(np.clip(v, 0.0, 1.0))  # noqa: E731

    # Recall-like states sharpen familiarity and self-reference; dreaming
    # loosens spatial constraint and raises vividness.
    if cognitive.state == "recalling":
        familiarity = clip(familiarity + 0.12 * cognitive.confidence)
        self_reference = clip(self_reference + 0.10 * cognitive.confidence)
    elif cognitive.state == "dreaming":
        spatial_scale = clip(spatial_scale + 0.14 * cognitive.confidence)
        vividness = clip(vividness + 0.12 * cognitive.confidence)
    elif cognitive.state in ("focused", "alert"):
        object_salience = clip(object_salience + 0.10 * cognitive.confidence)

    return MemoryAttributes(
        spatial_scale=spatial_scale,
        familiarity=familiarity,
        population=population,
        object_salience=object_salience,
        vividness=vividness,
        temporal_depth=temporal_depth,
        self_reference=self_reference,
    )
