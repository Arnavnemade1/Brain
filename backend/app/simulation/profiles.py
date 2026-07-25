"""Memory profiles that drive the EEG simulator.

A profile is a *script*: a sequence of phases, each with a target spectral
composition and an affective/cognitive character. The generator synthesises
signal matching those targets, and the analysis pipeline then recovers the
structure independently — nothing is passed between them out of band.

This matters: the reconstruction the user explores is genuinely derived from
the spectral content of the simulated recording, not read off the script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

BandWeights = Dict[str, float]


@dataclass(frozen=True)
class Phase:
    """One segment of a session's neural trajectory."""

    #: Fraction of total session duration this phase occupies.
    weight: float
    #: Relative band power targets; normalised by the generator.
    bands: BandWeights
    #: Ground-truth character of the phase — used only to shape the signal.
    cognitive: str
    emotion: str
    #: 0–1 affective activation, drives amplitude modulation depth.
    arousal: float
    #: −1..1, drives frontal alpha asymmetry.
    valence: float
    #: 0–1 likelihood the pipeline should find an episode boundary here.
    salience: float = 0.5
    label: str = ""


@dataclass(frozen=True)
class MemoryProfile:
    """A complete simulated recording scenario."""

    id: str
    name: str
    description: str
    phases: Sequence[Phase]
    #: Baseline artifact load, 0–1 (electrode quality of the notional session).
    artifact_level: float = 0.12
    #: 0–1 how much the montage moves together — affects recovered coherence.
    global_coupling: float = 0.55
    tags: List[str] = field(default_factory=list)

    def normalised_phases(self) -> List[Phase]:
        total = sum(p.weight for p in self.phases) or 1.0
        return [
            Phase(
                weight=p.weight / total,
                bands=p.bands,
                cognitive=p.cognitive,
                emotion=p.emotion,
                arousal=p.arousal,
                valence=p.valence,
                salience=p.salience,
                label=p.label,
            )
            for p in self.phases
        ]


# --------------------------------------------------------------------------- #
# Profile library                                                             #
# --------------------------------------------------------------------------- #

_COASTAL = MemoryProfile(
    id="coastal_recall",
    name="Coastal Recall",
    description=(
        "Sustained posterior alpha with intermittent theta bursts — the "
        "signature of relaxed autobiographical recall in an open outdoor setting."
    ),
    global_coupling=0.62,
    artifact_level=0.14,
    tags=["autobiographical", "outdoor", "low-arousal"],
    phases=[
        Phase(0.18, {"delta": 0.14, "theta": 0.18, "alpha": 0.42, "beta": 0.19, "gamma": 0.07},
              "relaxed", "calm", 0.30, 0.42, 0.30, "settling"),
        Phase(0.26, {"delta": 0.12, "theta": 0.30, "alpha": 0.36, "beta": 0.16, "gamma": 0.06},
              "recalling", "nostalgia", 0.48, 0.55, 0.85, "retrieval onset"),
        Phase(0.30, {"delta": 0.10, "theta": 0.26, "alpha": 0.40, "beta": 0.17, "gamma": 0.07},
              "recalling", "nostalgia", 0.55, 0.62, 0.70, "elaboration"),
        Phase(0.26, {"delta": 0.16, "theta": 0.20, "alpha": 0.44, "beta": 0.14, "gamma": 0.06},
              "relaxed", "calm", 0.26, 0.48, 0.35, "resolution"),
    ],
)

_CHILDHOOD = MemoryProfile(
    id="childhood_interior",
    name="Childhood Interior",
    description=(
        "Strong theta-band dominance with frontal midline activity — highly "
        "familiar interior spatial memory with dense episodic anchoring."
    ),
    global_coupling=0.58,
    artifact_level=0.30,
    tags=["autobiographical", "interior", "high-familiarity"],
    phases=[
        Phase(0.14, {"delta": 0.18, "theta": 0.24, "alpha": 0.34, "beta": 0.18, "gamma": 0.06},
              "relaxed", "calm", 0.32, 0.20, 0.25, "orientation"),
        Phase(0.30, {"delta": 0.14, "theta": 0.38, "alpha": 0.28, "beta": 0.15, "gamma": 0.05},
              "recalling", "nostalgia", 0.58, 0.44, 0.92, "spatial retrieval"),
        Phase(0.24, {"delta": 0.16, "theta": 0.34, "alpha": 0.26, "beta": 0.18, "gamma": 0.06},
              "recalling", "melancholy", 0.52, -0.18, 0.78, "affective shift"),
        Phase(0.32, {"delta": 0.20, "theta": 0.30, "alpha": 0.30, "beta": 0.14, "gamma": 0.06},
              "meditative", "nostalgia", 0.40, 0.30, 0.55, "consolidation"),
    ],
)

_URBAN = MemoryProfile(
    id="urban_transit",
    name="Urban Transit",
    description=(
        "Elevated beta with low alpha and frequent artifact — externally "
        "directed attention in a dense, high-stimulus environment."
    ),
    global_coupling=0.44,
    artifact_level=0.52,
    tags=["episodic", "urban", "high-arousal"],
    phases=[
        Phase(0.22, {"delta": 0.10, "theta": 0.16, "alpha": 0.20, "beta": 0.42, "gamma": 0.12},
              "alert", "stress", 0.72, -0.34, 0.60, "arrival"),
        Phase(0.30, {"delta": 0.08, "theta": 0.14, "alpha": 0.18, "beta": 0.46, "gamma": 0.14},
              "focused", "stress", 0.80, -0.42, 0.88, "navigation"),
        Phase(0.26, {"delta": 0.10, "theta": 0.20, "alpha": 0.26, "beta": 0.34, "gamma": 0.10},
              "focused", "curiosity", 0.64, 0.16, 0.72, "orientation shift"),
        Phase(0.22, {"delta": 0.14, "theta": 0.22, "alpha": 0.32, "beta": 0.24, "gamma": 0.08},
              "relaxed", "curiosity", 0.46, 0.28, 0.44, "settling"),
    ],
)

_LUCID = MemoryProfile(
    id="lucid_dream",
    name="Lucid Dream Sequence",
    description=(
        "REM-like architecture: mixed theta and gamma over low muscle tone, "
        "with sharp state transitions and unstable spatial structure."
    ),
    global_coupling=0.36,
    artifact_level=0.05,
    tags=["dream", "REM", "non-linear"],
    phases=[
        Phase(0.20, {"delta": 0.26, "theta": 0.32, "alpha": 0.20, "beta": 0.14, "gamma": 0.08},
              "drowsy", "calm", 0.28, 0.10, 0.30, "descent"),
        Phase(0.26, {"delta": 0.12, "theta": 0.40, "alpha": 0.16, "beta": 0.18, "gamma": 0.14},
              "dreaming", "wonder", 0.62, 0.48, 0.90, "scene formation"),
        Phase(0.28, {"delta": 0.10, "theta": 0.34, "alpha": 0.14, "beta": 0.22, "gamma": 0.20},
              "dreaming", "wonder", 0.76, 0.36, 0.95, "lucidity"),
        Phase(0.26, {"delta": 0.16, "theta": 0.36, "alpha": 0.18, "beta": 0.18, "gamma": 0.12},
              "dreaming", "fear", 0.68, -0.30, 0.80, "destabilisation"),
    ],
)

_SUMMIT = MemoryProfile(
    id="summit_ascent",
    name="Summit Ascent",
    description=(
        "Broadband activation with high gamma coupling — an intensely vivid "
        "memory dominated by spatial scale and awe."
    ),
    global_coupling=0.68,
    artifact_level=0.38,
    tags=["episodic", "outdoor", "high-scale"],
    phases=[
        Phase(0.24, {"delta": 0.12, "theta": 0.22, "alpha": 0.26, "beta": 0.30, "gamma": 0.10},
              "focused", "curiosity", 0.60, 0.34, 0.50, "approach"),
        Phase(0.22, {"delta": 0.10, "theta": 0.18, "alpha": 0.22, "beta": 0.36, "gamma": 0.14},
              "alert", "stress", 0.78, -0.10, 0.70, "exertion"),
        Phase(0.32, {"delta": 0.08, "theta": 0.20, "alpha": 0.30, "beta": 0.24, "gamma": 0.18},
              "recalling", "wonder", 0.82, 0.72, 0.96, "summit"),
        Phase(0.22, {"delta": 0.16, "theta": 0.24, "alpha": 0.38, "beta": 0.16, "gamma": 0.06},
              "meditative", "calm", 0.38, 0.60, 0.42, "descent"),
    ],
)

_RUINS = MemoryProfile(
    id="ruins_wander",
    name="Ruins Wander",
    description=(
        "Slow-wave dominance with sparse alpha — a fragmentary memory with "
        "weak retrieval cues and long uncertain intervals."
    ),
    global_coupling=0.40,
    artifact_level=0.72,
    tags=["fragmentary", "low-confidence", "exploratory"],
    phases=[
        Phase(0.30, {"delta": 0.32, "theta": 0.28, "alpha": 0.20, "beta": 0.14, "gamma": 0.06},
              "drowsy", "melancholy", 0.32, -0.24, 0.35, "fragmentary onset"),
        Phase(0.24, {"delta": 0.22, "theta": 0.34, "alpha": 0.22, "beta": 0.16, "gamma": 0.06},
              "recalling", "curiosity", 0.48, 0.12, 0.75, "partial retrieval"),
        Phase(0.22, {"delta": 0.28, "theta": 0.26, "alpha": 0.24, "beta": 0.16, "gamma": 0.06},
              "drowsy", "melancholy", 0.36, -0.30, 0.40, "decay"),
        Phase(0.24, {"delta": 0.18, "theta": 0.32, "alpha": 0.26, "beta": 0.17, "gamma": 0.07},
              "recalling", "wonder", 0.56, 0.26, 0.82, "late recovery"),
    ],
)

_DEEP_FOCUS = MemoryProfile(
    id="deep_focus",
    name="Deep Focus Interval",
    description=(
        "Sustained beta with suppressed alpha and tight inter-channel coupling "
        "— a procedural memory recorded under high cognitive load."
    ),
    global_coupling=0.72,
    artifact_level=0.09,
    tags=["procedural", "high-load", "indoor"],
    phases=[
        Phase(0.20, {"delta": 0.10, "theta": 0.24, "alpha": 0.28, "beta": 0.30, "gamma": 0.08},
              "focused", "curiosity", 0.56, 0.22, 0.45, "engagement"),
        Phase(0.34, {"delta": 0.08, "theta": 0.28, "alpha": 0.16, "beta": 0.38, "gamma": 0.10},
              "focused", "stress", 0.74, -0.12, 0.80, "sustained load"),
        Phase(0.24, {"delta": 0.08, "theta": 0.32, "alpha": 0.18, "beta": 0.32, "gamma": 0.10},
              "focused", "curiosity", 0.68, 0.30, 0.72, "insight"),
        Phase(0.22, {"delta": 0.14, "theta": 0.22, "alpha": 0.36, "beta": 0.20, "gamma": 0.08},
              "relaxed", "joy", 0.44, 0.58, 0.50, "release"),
    ],
)

PROFILES: Dict[str, MemoryProfile] = {
    p.id: p
    for p in (_COASTAL, _CHILDHOOD, _URBAN, _LUCID, _SUMMIT, _RUINS, _DEEP_FOCUS)
}

PROFILE_IDS: List[str] = list(PROFILES.keys())


def get_profile(profile_id: str) -> MemoryProfile:
    """Look up a profile, falling back to coastal recall for unknown ids."""
    return PROFILES.get(profile_id, _COASTAL)
