"""Context reconstruction and scene parameter synthesis.

Takes the accumulated memory attributes, affect and confidence for a session
and commits to a concrete, reproducible description of a place: biome,
palette, lighting, weather, structural regions and soundscape.

Every decision is seeded, so the same neural evidence always rebuilds the same
world — a reconstruction that shuffled between reloads would not be a
reconstruction.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..models.memory import (
    AtmosphereParams,
    LightingParams,
    Palette,
    RegionParams,
    SceneParameters,
    SoundLayer,
)
from ..models.neural import CognitiveReading, EmotionReading
from ..utils.ids import region_id
from .inference import MemoryAttributes

# --------------------------------------------------------------------------- #
# Biome selection                                                             #
# --------------------------------------------------------------------------- #

#: Each biome's expected memory-attribute profile. Scoring is by distance in
#: attribute space, so a biome is chosen because the inferred memory *looks
#: like* that kind of place, not by lookup from an emotion label.
_BIOME_PROFILES: Dict[str, Dict[str, float]] = {
    #                     scale  famil  popul  object vivid  tdepth selfref
    "childhood_home":     {"spatial_scale": 0.22, "familiarity": 0.88, "population": 0.30,
                           "object_salience": 0.62, "vividness": 0.55, "temporal_depth": 0.80,
                           "self_reference": 0.78},
    "school_hallway":     {"spatial_scale": 0.34, "familiarity": 0.74, "population": 0.62,
                           "object_salience": 0.48, "vividness": 0.50, "temporal_depth": 0.72,
                           "self_reference": 0.58},
    "beach_sunset":       {"spatial_scale": 0.78, "familiarity": 0.55, "population": 0.24,
                           "object_salience": 0.35, "vividness": 0.72, "temporal_depth": 0.55,
                           "self_reference": 0.66},
    "forest":             {"spatial_scale": 0.55, "familiarity": 0.45, "population": 0.12,
                           "object_salience": 0.66, "vividness": 0.58, "temporal_depth": 0.42,
                           "self_reference": 0.60},
    "city_streets":       {"spatial_scale": 0.48, "familiarity": 0.40, "population": 0.88,
                           "object_salience": 0.82, "vividness": 0.62, "temporal_depth": 0.30,
                           "self_reference": 0.32},
    "mountains":          {"spatial_scale": 0.94, "familiarity": 0.28, "population": 0.08,
                           "object_salience": 0.30, "vividness": 0.70, "temporal_depth": 0.48,
                           "self_reference": 0.55},
    "ancient_ruins":      {"spatial_scale": 0.68, "familiarity": 0.22, "population": 0.10,
                           "object_salience": 0.55, "vividness": 0.42, "temporal_depth": 0.92,
                           "self_reference": 0.38},
    "dreamscape":         {"spatial_scale": 0.72, "familiarity": 0.18, "population": 0.30,
                           "object_salience": 0.45, "vividness": 0.85, "temporal_depth": 0.35,
                           "self_reference": 0.72},
    "ocean":              {"spatial_scale": 0.90, "familiarity": 0.30, "population": 0.05,
                           "object_salience": 0.18, "vividness": 0.60, "temporal_depth": 0.52,
                           "self_reference": 0.68},
    "floating_islands":   {"spatial_scale": 0.82, "familiarity": 0.15, "population": 0.18,
                           "object_salience": 0.52, "vividness": 0.88, "temporal_depth": 0.28,
                           "self_reference": 0.62},
    "abstract_space":     {"spatial_scale": 0.60, "familiarity": 0.08, "population": 0.10,
                           "object_salience": 0.22, "vividness": 0.75, "temporal_depth": 0.40,
                           "self_reference": 0.85},
    "grand_architecture": {"spatial_scale": 0.70, "familiarity": 0.35, "population": 0.45,
                           "object_salience": 0.70, "vividness": 0.55, "temporal_depth": 0.65,
                           "self_reference": 0.40},
}

#: Emotional affinity, applied as a secondary term after attribute distance.
_BIOME_EMOTION_AFFINITY: Dict[str, Dict[str, float]] = {
    "childhood_home": {"nostalgia": 1.0, "melancholy": 0.7, "calm": 0.5},
    "school_hallway": {"nostalgia": 0.8, "stress": 0.7, "curiosity": 0.4},
    "beach_sunset": {"calm": 1.0, "nostalgia": 0.7, "joy": 0.6},
    "forest": {"calm": 0.8, "curiosity": 0.7, "melancholy": 0.5},
    "city_streets": {"stress": 1.0, "curiosity": 0.7, "joy": 0.4},
    "mountains": {"wonder": 1.0, "calm": 0.6, "melancholy": 0.4},
    "ancient_ruins": {"wonder": 0.8, "melancholy": 0.9, "curiosity": 0.6},
    "dreamscape": {"wonder": 0.9, "fear": 0.8, "curiosity": 0.6},
    "ocean": {"calm": 0.9, "melancholy": 0.8, "wonder": 0.5},
    "floating_islands": {"wonder": 1.0, "joy": 0.7, "curiosity": 0.6},
    "abstract_space": {"fear": 1.0, "stress": 0.7, "wonder": 0.5},
    "grand_architecture": {"wonder": 0.8, "stress": 0.6, "nostalgia": 0.4},
}

_ATTRIBUTE_KEYS = (
    "spatial_scale",
    "familiarity",
    "population",
    "object_salience",
    "vividness",
    "temporal_depth",
    "self_reference",
)


def _profile_statistics() -> Dict[str, Tuple[float, float]]:
    """Mean and spread of each attribute *across the biome library*."""
    stats: Dict[str, Tuple[float, float]] = {}
    for key in _ATTRIBUTE_KEYS:
        values = np.array([p[key] for p in _BIOME_PROFILES.values()])
        stats[key] = (float(values.mean()), max(float(values.std()), 1e-6))
    return stats


_PROFILE_STATS = _profile_statistics()

#: Spread of the attributes the inference stage actually produces. They are
#: logistic-squashed and so cluster near the middle of the range, far more
#: tightly than the biome profiles, which are written across the full 0–1 span.
_OBSERVED_SPREAD = 0.17


def score_biomes(
    attributes: MemoryAttributes, emotion: EmotionReading
) -> List[Tuple[str, float]]:
    """Rank every biome for this memory. Highest score first.

    Matching is on the *pattern* of the attribute vector, not its absolute
    position. Inferred attributes cluster near the middle of the range while
    biome profiles span the full 0–1 axis, so nearest-neighbour scoring in raw
    units simply returns whichever biome sits closest to the centre of the
    library — mid-range settings win every time and the extremes (mountains,
    abstract space) become unreachable regardless of the evidence.

    Both vectors are therefore standardised against their own spread and
    compared by cosine similarity, which asks "is this memory *shaped* like
    that kind of place" — high scale and low familiarity together, say — and
    is invariant to the two scales differing.
    """
    observed = np.array(
        [
            (getattr(attributes, key) - 0.5) / _OBSERVED_SPREAD
            for key in _ATTRIBUTE_KEYS
        ]
    )
    observed_norm = float(np.linalg.norm(observed))

    scored: List[Tuple[str, float]] = []

    for biome, profile in _BIOME_PROFILES.items():
        reference = np.array(
            [
                (profile[key] - _PROFILE_STATS[key][0]) / _PROFILE_STATS[key][1]
                for key in _ATTRIBUTE_KEYS
            ]
        )
        reference_norm = float(np.linalg.norm(reference))

        if observed_norm < 1e-6 or reference_norm < 1e-6:
            similarity = 0.0
        else:
            similarity = float(np.dot(observed, reference) / (observed_norm * reference_norm))

        # Cosine sits in [-1, 1]; shift to [0, 1] so the emotional term below
        # cannot flip an otherwise negative match into the lead.
        score = (similarity + 1.0) / 2.0

        affinity = _BIOME_EMOTION_AFFINITY.get(biome, {})
        emotional = sum(
            weight * emotion.distribution.get(label, 0.0)
            for label, weight in affinity.items()
        )
        score += 0.35 * emotional

        scored.append((biome, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


def select_biome(
    attributes: MemoryAttributes,
    emotion: EmotionReading,
    confidence: float,
    rng: random.Random,
) -> str:
    """Choose a setting by sampling the ranked candidates, not by argmax.

    The top scores are typically within a few percent of each other — the
    model genuinely cannot distinguish "an open shoreline" from "an open water
    surface" on this evidence. Taking the maximum presents that coin-flip as a
    determination, and collapses every ambiguous session onto whichever biome
    sits nearest the centre of the library.

    Sampling instead is both a more honest reflection of the posterior and the
    thing that keeps distinct-but-equally-supported readings reachable. The
    draw is seeded, so a given recording still always rebuilds the same world.

    Confidence sharpens the choice: a well-evidenced session concentrates on
    its leading candidate, a poorly-evidenced one stays closer to a coin toss.
    """
    ranked = score_biomes(attributes, emotion)
    if not ranked:
        return "abstract_space"

    candidates = ranked[:4]
    temperature = float(np.clip(0.14 - 0.09 * confidence, 0.035, 0.14))

    scores = np.array([score for _, score in candidates])
    scores = (scores - scores.max()) / temperature
    weights = np.exp(scores)
    weights /= weights.sum()

    draw = rng.random()
    cumulative = 0.0
    for (biome, _), weight in zip(candidates, weights):
        cumulative += float(weight)
        if draw <= cumulative:
            return biome
    return candidates[0][0]


# --------------------------------------------------------------------------- #
# Palette                                                                     #
# --------------------------------------------------------------------------- #

#: Emotion-driven colour grades. Restrained by design — the reconstruction
#: should read as a graded photograph, not a saturated game level.
_EMOTION_PALETTES: Dict[str, Palette] = {
    "calm": Palette(key="#cfe9ec", fill="#6fa8b4", rim="#a9dce4", fog="#8fb6c2",
                    sky="#93bdd0", ground="#5c7f8a", accent="#5fd4c4"),
    "nostalgia": Palette(key="#f4d9a8", fill="#c08f52", rim="#f0c477", fog="#c9a882",
                         sky="#e0b57e", ground="#7a6047", accent="#e8b464"),
    "wonder": Palette(key="#d9d2ff", fill="#7a6ac4", rim="#c0b1fb", fog="#8f86c9",
                      sky="#9d92e0", ground="#4a4374", accent="#a48bf2"),
    "stress": Palette(key="#e6c4bd", fill="#8a5b55", rim="#e0a196", fog="#93726c",
                      sky="#a8827a", ground="#4f3f3c", accent="#e0736b"),
    "curiosity": Palette(key="#c8e9f5", fill="#4f93ad", rim="#8fd4ea", fog="#7ba6bb",
                         sky="#87b8cf", ground="#3f5f6e", accent="#52c4ea"),
    "melancholy": Palette(key="#c6cfe6", fill="#5b6b96", rim="#93a4cc", fog="#75839f",
                          sky="#7e8daf", ground="#3d465e", accent="#6b83c9"),
    "joy": Palette(key="#fce4b0", fill="#d4a24f", rim="#f6cf82", fog="#d4b58a",
                   sky="#f0c98f", ground="#7d6647", accent="#f2c14e"),
    "fear": Palette(key="#ddc6e8", fill="#6f4f80", rim="#bb96cc", fog="#7d6389",
                    sky="#8a6d99", ground="#402f4a", accent="#b26bc9"),
}


def _blend_palette(primary: str, secondary: Optional[str], weight: float) -> Palette:
    """Interpolate between two emotion palettes in sRGB."""
    base = _EMOTION_PALETTES.get(primary, _EMOTION_PALETTES["calm"])
    if not secondary or secondary not in _EMOTION_PALETTES or weight <= 0.01:
        return base.model_copy()

    other = _EMOTION_PALETTES[secondary]
    k = float(np.clip(weight, 0.0, 0.5))

    def mix(a: str, b: str) -> str:
        ai = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
        bi = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
        return "#" + "".join(f"{round(x + (y - x) * k):02x}" for x, y in zip(ai, bi))

    return Palette(
        key=mix(base.key, other.key),
        fill=mix(base.fill, other.fill),
        rim=mix(base.rim, other.rim),
        fog=mix(base.fog, other.fog),
        sky=mix(base.sky, other.sky),
        ground=mix(base.ground, other.ground),
        accent=mix(base.accent, other.accent),
    )


# --------------------------------------------------------------------------- #
# Lighting and atmosphere                                                     #
# --------------------------------------------------------------------------- #

_TIME_ELEVATION: Dict[str, Tuple[float, float]] = {
    "dawn": (4.0, 0.45),
    "morning": (28.0, 0.80),
    "noon": (68.0, 1.00),
    "afternoon": (44.0, 0.90),
    "golden": (11.0, 0.72),
    "dusk": (2.0, 0.42),
    "night": (-14.0, 0.16),
}


def _choose_time_of_day(
    attributes: MemoryAttributes, emotion: EmotionReading, rng: random.Random
) -> str:
    """Time of day follows affect and temporal depth.

    Remote, nostalgic memories skew toward low sun angles — the warm,
    long-shadow light that dominates recalled autobiographical imagery.
    """
    weights: Dict[str, float] = {
        "dawn": 0.10 + attributes.self_reference * 0.20,
        "morning": 0.14 + max(0.0, emotion.valence) * 0.25,
        "noon": 0.10 + emotion.arousal * 0.22,
        "afternoon": 0.16 + attributes.familiarity * 0.20,
        "golden": 0.18 + attributes.temporal_depth * 0.55,
        "dusk": 0.12 + max(0.0, -emotion.valence) * 0.30,
        "night": 0.08 + max(0.0, -emotion.valence) * 0.35 + attributes.vividness * 0.10,
    }
    if emotion.primary in ("nostalgia", "melancholy"):
        weights["golden"] += 0.45
        weights["dusk"] += 0.25
    if emotion.primary in ("fear", "stress"):
        weights["night"] += 0.30
    if emotion.primary in ("calm", "joy"):
        weights["morning"] += 0.20

    total = sum(weights.values())
    draw = rng.random() * total
    acc = 0.0
    for label, weight in weights.items():
        acc += weight
        if draw <= acc:
            return label
    return "golden"


def _build_lighting(
    time_of_day: str,
    attributes: MemoryAttributes,
    emotion: EmotionReading,
    confidence: float,
    rng: random.Random,
) -> LightingParams:
    elevation, base_intensity = _TIME_ELEVATION[time_of_day]

    # Warm for positive, remote, familiar memories; cool for negative ones.
    temperature = 5200.0
    temperature -= emotion.valence * 900.0 * -1.0  # positive valence → warmer
    temperature -= attributes.temporal_depth * 700.0
    if emotion.primary == "nostalgia":
        temperature -= 600.0
    if emotion.primary in ("fear", "stress"):
        temperature += 900.0
    if time_of_day in ("golden", "dusk", "dawn"):
        temperature -= 700.0
    temperature = float(np.clip(temperature, 2400.0, 8200.0))

    return LightingParams(
        time_of_day=time_of_day,
        sun_elevation=round(elevation + rng.uniform(-3.0, 3.0), 2),
        sun_azimuth=round(rng.uniform(0.0, 360.0), 2),
        # Confident reconstructions are lit more decisively.
        intensity=round(float(np.clip(base_intensity * (0.55 + 0.55 * confidence), 0.0, 2.0)), 3),
        temperature=round(temperature, 0),
        ambient=round(float(np.clip(0.22 + (1.0 - confidence) * 0.30, 0.0, 2.0)), 3),
        volumetric=round(
            float(np.clip(0.25 + attributes.vividness * 0.45 + emotion.intensity * 0.25, 0.0, 1.0)),
            3,
        ),
    )


def _build_atmosphere(
    emotion: EmotionReading,
    attributes: MemoryAttributes,
    confidence: float,
    rng: random.Random,
) -> AtmosphereParams:
    """Weather and particles follow affect, per the emotional design language."""
    weather = "clear"
    particles = "motes"

    if emotion.primary == "stress":
        weather = "fog" if rng.random() < 0.6 else "overcast"
        particles = "none"
    elif emotion.primary == "melancholy":
        weather = "rain" if rng.random() < 0.5 else "overcast"
        particles = "rain" if weather == "rain" else "motes"
    elif emotion.primary == "nostalgia":
        weather = "clear"
        particles = "dust"
    elif emotion.primary == "calm":
        weather = "clear"
        particles = "pollen" if rng.random() < 0.5 else "motes"
    elif emotion.primary == "wonder":
        weather = "clear"
        particles = "motes"
    elif emotion.primary == "fear":
        weather = "fog"
        particles = "ashfall" if rng.random() < 0.3 else "none"
        if particles == "ashfall":
            weather = "ashfall"
            particles = "embers"
    elif emotion.primary == "joy":
        weather = "clear"
        particles = "pollen"
    elif emotion.primary == "curiosity":
        weather = "overcast" if rng.random() < 0.3 else "clear"
        particles = "motes"

    # Low confidence thickens the air — uncertainty reads as haze rather than
    # as missing geometry alone.
    fog_density = 0.10 + (1.0 - confidence) * 0.45
    if weather == "fog":
        fog_density += 0.30
    elif weather in ("rain", "overcast", "ashfall"):
        fog_density += 0.15

    return AtmosphereParams(
        weather=weather,
        fog_density=round(float(np.clip(fog_density, 0.0, 1.0)), 3),
        particles=particles,
        particle_density=round(
            float(np.clip(0.15 + attributes.vividness * 0.55 + emotion.intensity * 0.20, 0.0, 1.0)),
            3,
        ),
        wind=round(float(np.clip(0.12 + emotion.arousal * 0.65, 0.0, 1.0)), 3),
    )


# --------------------------------------------------------------------------- #
# Regions                                                                     #
# --------------------------------------------------------------------------- #

#: Structural vocabulary per biome: which archetypes appear and how often.
_BIOME_ARCHETYPES: Dict[str, List[Tuple[str, float]]] = {
    "childhood_home": [("building", 0.55), ("corridor", 0.25), ("grove", 0.20)],
    "school_hallway": [("corridor", 0.65), ("building", 0.35)],
    "beach_sunset": [("terrain", 0.45), ("water", 0.40), ("monument", 0.15)],
    "forest": [("grove", 0.70), ("terrain", 0.25), ("water", 0.05)],
    "city_streets": [("building", 0.60), ("corridor", 0.30), ("monument", 0.10)],
    "mountains": [("terrain", 0.80), ("monument", 0.20)],
    "ancient_ruins": [("monument", 0.55), ("building", 0.25), ("terrain", 0.20)],
    "dreamscape": [("void", 0.35), ("monument", 0.30), ("building", 0.20), ("terrain", 0.15)],
    "ocean": [("water", 0.75), ("terrain", 0.25)],
    "floating_islands": [("terrain", 0.50), ("void", 0.30), ("monument", 0.20)],
    "abstract_space": [("void", 0.70), ("monument", 0.30)],
    "grand_architecture": [("building", 0.50), ("monument", 0.30), ("corridor", 0.20)],
}

#: Human-readable region names per archetype, chosen to read as an AI's
#: tentative labelling rather than as level-design nouns.
_REGION_LABELS: Dict[str, List[str]] = {
    "building": ["Primary Structure", "Adjacent Interior", "Partial Facade", "Upper Floors"],
    "corridor": ["Connecting Passage", "Long Corridor", "Threshold", "Narrow Approach"],
    "terrain": ["Open Ground", "Rising Slope", "Distant Ridge", "Lower Basin"],
    "monument": ["Central Landmark", "Standing Form", "Fragmented Monument", "Marker"],
    "grove": ["Dense Growth", "Clearing", "Treeline", "Overgrown Margin"],
    "water": ["Water Surface", "Shoreline", "Shallows", "Far Reach"],
    "void": ["Unresolved Region", "Empty Volume", "Undefined Space", "Discontinuity"],
}


def _pick_archetype(biome: str, rng: random.Random) -> str:
    options = _BIOME_ARCHETYPES.get(biome, [("terrain", 1.0)])
    total = sum(weight for _, weight in options)
    draw = rng.random() * total
    acc = 0.0
    for archetype, weight in options:
        acc += weight
        if draw <= acc:
            return archetype
    return options[0][0]


def build_regions(
    session_id: str,
    biome: str,
    attributes: MemoryAttributes,
    confidence: float,
    rng: random.Random,
) -> List[RegionParams]:
    """Lay out the structural regions of the reconstruction.

    Region confidence is deliberately *uneven*. A memory does not come back at
    a uniform resolution, and the renderer needs genuinely fragmented areas to
    contrast against resolved ones — a world that is uniformly 70% confident
    reads as low-quality rather than as partially remembered.
    """
    count = int(round(4 + attributes.spatial_scale * 5 + attributes.object_salience * 3))
    count = max(4, min(12, count))

    regions: List[RegionParams] = []
    used_labels: Dict[str, int] = {}

    for i in range(count):
        archetype = _pick_archetype(biome, rng)

        labels = _REGION_LABELS.get(archetype, ["Region"])
        seen = used_labels.get(archetype, 0)
        label = labels[seen % len(labels)]
        if seen >= len(labels):
            label = f"{label} {seen // len(labels) + 1}"
        used_labels[archetype] = seen + 1

        # Radial layout: the first region sits at the origin (where the user
        # spawns and where evidence is strongest), later ones further out and
        # progressively less certain.
        if i == 0:
            angle = 0.0
            radius_from_centre = 0.0
        else:
            angle = (i / max(1, count - 1)) * math.tau + rng.uniform(-0.35, 0.35)
            radius_from_centre = (18.0 + 62.0 * attributes.spatial_scale) * (
                0.35 + 0.75 * (i / max(1, count - 1))
            )

        elevation = 0.0
        if biome in ("floating_islands", "dreamscape", "abstract_space"):
            elevation = rng.uniform(-8.0, 26.0) * attributes.vividness

        # Distance from the centre of attention costs confidence, then noise
        # breaks up any residual regularity.
        falloff = 1.0 - (i / max(1, count)) * 0.55
        local = confidence * falloff * rng.uniform(0.62, 1.28)

        regions.append(
            RegionParams(
                id=region_id(session_id, i),
                label=label,
                position=(
                    round(math.cos(angle) * radius_from_centre, 2),
                    round(elevation, 2),
                    round(math.sin(angle) * radius_from_centre, 2),
                ),
                radius=round(8.0 + rng.uniform(0.0, 14.0) * (0.5 + attributes.spatial_scale), 2),
                confidence=round(float(np.clip(local, 0.05, 1.0)), 3),
                archetype=archetype,
                locked_by=None,
            )
        )

    return regions


# --------------------------------------------------------------------------- #
# Soundscape                                                                  #
# --------------------------------------------------------------------------- #

_BIOME_SOUNDS: Dict[str, List[str]] = {
    "childhood_home": ["hum", "voices", "chimes"],
    "school_hallway": ["voices", "hum"],
    "beach_sunset": ["waves", "wind", "birds"],
    "forest": ["birds", "wind"],
    "city_streets": ["traffic", "voices"],
    "mountains": ["wind"],
    "ancient_ruins": ["wind", "chimes"],
    "dreamscape": ["hum", "chimes", "heartbeat"],
    "ocean": ["waves", "wind"],
    "floating_islands": ["wind", "chimes"],
    "abstract_space": ["hum", "heartbeat"],
    "grand_architecture": ["hum", "voices"],
}


def build_soundscape(
    biome: str, emotion: EmotionReading, weather: str, confidence: float, rng: random.Random
) -> List[SoundLayer]:
    layers: List[SoundLayer] = []
    kinds = list(_BIOME_SOUNDS.get(biome, ["wind"]))

    if weather == "rain" and "rain" not in kinds:
        kinds.append("rain")
    if emotion.primary in ("fear", "stress") and "heartbeat" not in kinds:
        kinds.append("heartbeat")

    for i, kind in enumerate(kinds):
        # Later layers are quieter and less certain — the model is confident
        # about the dominant sound of a place, much less about the rest.
        gain = float(np.clip(0.72 - i * 0.18 + rng.uniform(-0.08, 0.08), 0.05, 1.0))
        layer_confidence = float(np.clip(confidence * (0.9 - i * 0.16), 0.05, 1.0))
        layers.append(
            SoundLayer(
                kind=kind, gain=round(gain, 3), confidence=round(layer_confidence, 3)
            )
        )

    return layers


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def synthesise_scene(
    session_id: str,
    seed: int,
    attributes: MemoryAttributes,
    emotion: EmotionReading,
    cognitive: CognitiveReading,
    confidence: float,
    coherence: float,
) -> SceneParameters:
    """Commit the interpretation to a concrete, reproducible scene."""
    rng = random.Random(seed)

    biome = select_biome(attributes, emotion, confidence, rng)

    secondary_weight = 0.0
    if emotion.secondary:
        secondary_weight = emotion.distribution.get(emotion.secondary, 0.0)
    palette = _blend_palette(emotion.primary, emotion.secondary, secondary_weight)

    time_of_day = _choose_time_of_day(attributes, emotion, rng)
    lighting = _build_lighting(time_of_day, attributes, emotion, confidence, rng)
    atmosphere = _build_atmosphere(emotion, attributes, confidence, rng)
    regions = build_regions(session_id, biome, attributes, confidence, rng)
    soundscape = build_soundscape(biome, emotion, atmosphere.weather, confidence, rng)

    # Dream-like and low-coherence states bend the geometry.
    distortion = 0.10 + (1.0 - coherence) * 0.40
    if cognitive.state == "dreaming":
        distortion += 0.28
    if biome in ("dreamscape", "abstract_space", "floating_islands"):
        distortion += 0.20

    return SceneParameters(
        seed=seed,
        biome=biome,
        palette=palette,
        lighting=lighting,
        atmosphere=atmosphere,
        spatial_scale=round(attributes.spatial_scale, 3),
        population_density=round(attributes.population, 3),
        object_density=round(attributes.object_salience, 3),
        emotional_intensity=round(emotion.intensity, 3),
        movement=round(float(np.clip(0.12 + emotion.arousal * 0.55, 0.0, 1.0)), 3),
        # The threshold rises with overall confidence: as the model gets more
        # certain overall, it also gets stricter about what counts as resolved.
        fragment_threshold=round(float(np.clip(0.30 + confidence * 0.28, 0.0, 1.0)), 3),
        distortion=round(float(np.clip(distortion, 0.0, 1.0)), 3),
        regions=regions,
        soundscape=soundscape,
    )
