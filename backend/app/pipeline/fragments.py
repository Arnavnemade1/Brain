"""Memory fragment extraction.

Fragments are the discrete episodes the model believes it found inside a
recording. Each is anchored to a real moment in the signal and carries the
evidence that produced it, so opening one in the environment shows *why* the
model thinks that moment mattered — not a decorative caption.

Detection is by salience: a window earns a fragment when it stands out from
its neighbours in affect, spectral character or coherence.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..models.memory import MemoryFragment, RegionParams
from ..models.neural import NeuralFrame
from ..processing.bands import BAND_ORDER
from ..utils.ids import fragment_id
from .inference import MemoryAttributes

# --------------------------------------------------------------------------- #
# Descriptive vocabulary                                                      #
# --------------------------------------------------------------------------- #

_EMOTION_OPENERS: Dict[str, List[str]] = {
    "calm": [
        "A settled interval with little happening",
        "An unhurried stretch of time",
        "A pause with nothing demanding attention",
    ],
    "nostalgia": [
        "A returning impression of somewhere familiar",
        "Something recognised rather than newly seen",
        "A place the signal treats as already known",
    ],
    "wonder": [
        "An opening-out, something larger than expected",
        "A moment of scale beyond the immediate",
        "Attention pulled outward and upward",
    ],
    "stress": [
        "A tightening, attention narrowing quickly",
        "An interval of pressure with little slack",
        "Something demanding an immediate response",
    ],
    "curiosity": [
        "Attention caught by something unresolved",
        "An interval of searching",
        "Something drawing attention sideways",
    ],
    "melancholy": [
        "A slow, weighted passage",
        "Something absent more than present",
        "An interval turned inward and downward",
    ],
    "joy": [
        "A bright, unguarded interval",
        "Something lifting without effort",
        "An easy, open moment",
    ],
    "fear": [
        "A sharp, unresolved alarm",
        "Something approaching that cannot be placed",
        "An interval the signal refuses to settle",
    ],
}

_SENSORY_BY_EMOTION: Dict[str, List[str]] = {
    "calm": ["still air", "even light", "distant water", "low warmth", "unhurried sound"],
    "nostalgia": ["warm dust", "worn surfaces", "late light", "familiar proportions", "faint voices"],
    "wonder": ["cold clear air", "great distance", "open sky", "thin light", "echo without source"],
    "stress": ["close air", "hard edges", "overlapping noise", "narrow sightlines", "cold surfaces"],
    "curiosity": ["shifting light", "half-seen movement", "uneven ground", "an opening ahead", "changing acoustics"],
    "melancholy": ["grey light", "damp air", "empty space", "slow sound", "distance without detail"],
    "joy": ["strong light", "movement nearby", "warm air", "bright sound", "open ground"],
    "fear": ["thick air", "unstable footing", "sound behind", "failing light", "closing space"],
}

_TITLE_NOUNS: Dict[str, List[str]] = {
    "childhood_home": ["Interior", "Doorway", "Room", "Stair", "Window"],
    "school_hallway": ["Corridor", "Threshold", "Row of Doors", "Stairwell"],
    "beach_sunset": ["Shoreline", "Low Sun", "Open Water", "Tideline"],
    "forest": ["Treeline", "Clearing", "Undergrowth", "Canopy"],
    "city_streets": ["Crossing", "Street", "Frontage", "Corner"],
    "mountains": ["Ridge", "Slope", "High Ground", "Far Range"],
    "ancient_ruins": ["Standing Stone", "Broken Arch", "Old Ground", "Fallen Span"],
    "dreamscape": ["Unstable Ground", "Shifting Form", "Impossible Turn", "Recurring Shape"],
    "ocean": ["Open Surface", "Far Horizon", "Deep Water", "Swell"],
    "floating_islands": ["Suspended Ground", "Gap", "Upper Terrace", "Drifting Mass"],
    "abstract_space": ["Undefined Region", "Pure Interval", "Unplaced Feeling", "Bare Signal"],
    "grand_architecture": ["Great Hall", "Colonnade", "High Vault", "Long Approach"],
}

_TIME_ADJECTIVES = [
    "Early", "Late", "Brief", "Extended", "Repeated", "Partial", "Sudden", "Quiet",
]


# --------------------------------------------------------------------------- #
# Salience detection                                                          #
# --------------------------------------------------------------------------- #


def _band_vector(frame: NeuralFrame) -> np.ndarray:
    return np.array([frame.band_powers.get(band, 0.0) for band in BAND_ORDER])


def salience_trace(frames: Sequence[NeuralFrame]) -> np.ndarray:
    """Per-window salience, 0–1.

    Combines three independent signals of "something happened here":

    * **spectral novelty** — how far this window's band profile sits from the
      local running average, which marks state transitions
    * **affective intensity** — emotionally charged windows are the ones that
      become episodes
    * **evidence quality** — a striking window backed by poor signal is more
      likely an artifact than a memory
    """
    if not frames:
        return np.array([])

    matrix = np.array([_band_vector(f) for f in frames])
    novelty = np.zeros(len(frames))

    window = 6
    for i in range(len(frames)):
        low = max(0, i - window)
        if i - low < 2:
            continue
        baseline = matrix[low:i].mean(axis=0)
        novelty[i] = float(np.linalg.norm(matrix[i] - baseline))

    if novelty.max() > 1e-9:
        novelty = novelty / novelty.max()

    intensity = np.array([f.emotion.intensity for f in frames])
    evidence = np.array([f.signal_quality.score * f.neural_confidence for f in frames])

    return np.clip(0.45 * novelty + 0.35 * intensity + 0.20 * evidence, 0.0, 1.0)


def _peak_indices(trace: np.ndarray, target: int, min_separation: int) -> List[int]:
    """Highest-salience windows, spaced out so fragments do not cluster."""
    if trace.size == 0:
        return []

    order = np.argsort(trace)[::-1]
    chosen: List[int] = []

    for index in order:
        if len(chosen) >= target:
            break
        if all(abs(int(index) - c) >= min_separation for c in chosen):
            chosen.append(int(index))

    return sorted(chosen)


# --------------------------------------------------------------------------- #
# Fragment construction                                                       #
# --------------------------------------------------------------------------- #


def _describe(
    frame: NeuralFrame,
    biome: str,
    attributes: MemoryAttributes,
    rng: random.Random,
) -> str:
    """Compose the fragment's narrative description.

    Deliberately hedged. The model is reporting an inference from a two-second
    window of scalp voltage, and the copy should never read as though it
    recovered a photograph.
    """
    emotion = frame.emotion.primary
    opener = rng.choice(_EMOTION_OPENERS.get(emotion, _EMOTION_OPENERS["calm"]))

    if attributes.familiarity > 0.62:
        placement = "The surroundings register as previously known"
    elif attributes.familiarity < 0.38:
        placement = "Nothing here reads as familiar"
    else:
        placement = "The setting is only partly recognised"

    if attributes.population > 0.60:
        people = "Other people are present, though none resolve individually"
    elif attributes.population > 0.35:
        people = "There may be someone else nearby"
    else:
        people = "No one else appears to be here"

    if frame.coherence > 0.55:
        stability = "The impression holds steady across the interval"
    elif frame.coherence > 0.30:
        stability = "The impression drifts, reforming more than once"
    else:
        stability = "The impression does not hold; it is gone almost immediately"

    return f"{opener}. {placement}. {people}. {stability}."


def _evidence_summary(frame: NeuralFrame) -> str:
    """Plain-language account of the measurements behind this fragment."""
    dominant = max(frame.band_powers, key=lambda b: frame.band_powers[b])
    share = frame.band_powers[dominant]

    parts = [
        f"{dominant} dominant at {share * 100:.0f}% of band power",
        f"peak {frame.spectrum.peak_frequency:.1f} Hz",
    ]

    if frame.emotion.valence > 0.2:
        parts.append("frontal asymmetry positive")
    elif frame.emotion.valence < -0.2:
        parts.append("frontal asymmetry negative")

    if frame.cognitive.confidence > 0.6:
        parts.append(f"state {frame.cognitive.state} held confidently")
    else:
        parts.append(f"state ambiguous, nearest {frame.cognitive.state}")

    parts.append(f"signal {frame.signal_quality.grade}")

    return "; ".join(parts) + "."


def _title(biome: str, frame: NeuralFrame, index: int, rng: random.Random) -> str:
    nouns = _TITLE_NOUNS.get(biome, ["Interval"])
    noun = nouns[index % len(nouns)]
    if rng.random() < 0.55:
        return f"{rng.choice(_TIME_ADJECTIVES)} {noun}"
    return noun


def build_fragments(
    session_id: str,
    frames: Sequence[NeuralFrame],
    attributes: MemoryAttributes,
    biome: str,
    regions: Sequence[RegionParams],
    seed: int,
    duration: float,
    target: Optional[int] = None,
) -> List[MemoryFragment]:
    """Extract fragments from an analysed session."""
    if not frames:
        return []

    rng = random.Random(seed ^ 0x5EED)
    trace = salience_trace(frames)

    if target is None:
        # More vivid, more object-dense memories yield more distinct episodes.
        target = int(round(4 + attributes.vividness * 5 + attributes.object_salience * 3))
    target = max(3, min(14, target, len(frames)))

    separation = max(1, len(frames) // (target * 2))
    indices = _peak_indices(trace, target, separation)

    fragments: List[MemoryFragment] = []

    for order, index in enumerate(indices):
        frame = frames[index]

        # Anchor the fragment near a region, so exploring the world leads to
        # them rather than scattering them across empty space.
        if regions:
            region = regions[order % len(regions)]
            angle = rng.uniform(0.0, math.tau)
            offset = region.radius * rng.uniform(0.25, 0.75)
            position = (
                round(region.position[0] + math.cos(angle) * offset, 2),
                round(region.position[1] + rng.uniform(1.2, 3.4), 2),
                round(region.position[2] + math.sin(angle) * offset, 2),
            )
        else:
            angle = rng.uniform(0.0, math.tau)
            radius = rng.uniform(10.0, 45.0)
            position = (
                round(math.cos(angle) * radius, 2),
                round(rng.uniform(1.2, 4.0), 2),
                round(math.sin(angle) * radius, 2),
            )

        sensory_pool = _SENSORY_BY_EMOTION.get(frame.emotion.primary, _SENSORY_BY_EMOTION["calm"])
        detail_count = 2 if frame.neural_confidence < 0.5 else 3
        sensory = rng.sample(sensory_pool, min(detail_count, len(sensory_pool)))

        # Fragment confidence is the evidence behind *this window*, not the
        # session average — some episodes are recovered far better than others.
        confidence = float(
            np.clip(
                0.35 * frame.signal_quality.score
                + 0.35 * frame.neural_confidence
                + 0.30 * frame.coherence,
                0.05,
                1.0,
            )
        )

        # Low-confidence fragments gate a region: opening them is what makes
        # the corresponding area of the world resolve.
        unlocks: Optional[str] = None
        if regions and confidence < 0.62 and order < len(regions):
            candidate = regions[(order + 1) % len(regions)]
            if candidate.confidence < 0.55:
                unlocks = candidate.id

        fragments.append(
            MemoryFragment(
                id=fragment_id(session_id, order),
                title=_title(biome, frame, order, rng),
                description=_describe(frame, biome, attributes, rng),
                position=position,
                emotion=frame.emotion.primary,
                confidence=round(confidence, 3),
                sensory_details=sensory,
                related_ids=[],
                timeline_position=round(
                    float(np.clip(frame.t / max(duration, 1e-6), 0.0, 1.0)), 4
                ),
                neural_evidence=_evidence_summary(frame),
                source_time=round(frame.t, 2),
                unlocks_region=unlocks,
                revealed=False,
            )
        )

    _link_related(fragments)
    return fragments


def _link_related(fragments: List[MemoryFragment]) -> None:
    """Cross-link fragments that share affect or sit close in time.

    This is what makes the memory feel like one episode rather than a list:
    opening a fragment surfaces the others the model believes belong with it.
    """
    for i, fragment in enumerate(fragments):
        scores: List[Tuple[float, str]] = []

        for j, other in enumerate(fragments):
            if i == j:
                continue

            score = 0.0
            if other.emotion == fragment.emotion:
                score += 1.0

            temporal_gap = abs(other.timeline_position - fragment.timeline_position)
            score += max(0.0, 1.0 - temporal_gap * 3.0)

            shared = set(other.sensory_details) & set(fragment.sensory_details)
            score += 0.5 * len(shared)

            scores.append((score, other.id))

        scores.sort(reverse=True)
        fragment.related_ids = [fid for score, fid in scores[:3] if score > 0.35]
