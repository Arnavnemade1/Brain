"""Session narrative generation.

Produces the closing account of a reconstruction: what the model believes it
found, and — equally important — what it could not recover.

The local generator is the default and is fully self-sufficient. When an
OpenAI-compatible endpoint is configured it is used instead, with the local
result as the fallback on any failure. Nothing here ever blocks a session.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List

from ..models.memory import MemoryNarrative, SceneParameters
from ..models.neural import CognitiveReading, EmotionReading
from ..utils.ids import seed_from
from .inference import MemoryAttributes

log = logging.getLogger("mindscape.narrative")

# --------------------------------------------------------------------------- #
# Vocabulary                                                                  #
# --------------------------------------------------------------------------- #

_BIOME_PHRASE: Dict[str, str] = {
    "childhood_home": "an interior with high familiarity weighting",
    "school_hallway": "a repeating institutional corridor",
    "beach_sunset": "an open shoreline under a low sun",
    "forest": "dense vertical cover with diffuse overhead light",
    "city_streets": "a dense street-level environment",
    "mountains": "high ground at considerable spatial scale",
    "ancient_ruins": "fragmentary monumental structures",
    "dreamscape": "a setting that does not hold a stable layout",
    "ocean": "an open water surface with few landmarks",
    "floating_islands": "suspended terrain without visible support",
    "abstract_space": "no literal setting — affect rendered directly",
    "grand_architecture": "a monumental interior volume",
}

_EMOTION_PHRASE: Dict[str, str] = {
    "calm": "settled, low-arousal affect",
    "nostalgia": "nostalgia, with strong familiarity weighting",
    "wonder": "wonder, oriented outward and upward",
    "stress": "sustained stress with narrowed attention",
    "curiosity": "active curiosity and unresolved search",
    "melancholy": "melancholy, weighted toward absence",
    "joy": "positive, unguarded affect",
    "fear": "unresolved alarm",
}

_STATE_PHRASE: Dict[str, str] = {
    "focused": "sustained external focus",
    "alert": "heightened vigilance",
    "relaxed": "relaxed wakefulness",
    "meditative": "inwardly directed, low-effort attention",
    "drowsy": "descending arousal approaching sleep onset",
    "dreaming": "REM-like architecture",
    "recalling": "active autobiographical retrieval",
}

_TITLE_TIME: Dict[str, str] = {
    "dawn": "First Light",
    "morning": "Morning",
    "noon": "Midday",
    "afternoon": "Late Afternoon",
    "golden": "Low Sun",
    "dusk": "Dusk",
    "night": "After Dark",
}

_TITLE_PLACE: Dict[str, str] = {
    "childhood_home": "a Familiar Interior",
    "school_hallway": "a Long Corridor",
    "beach_sunset": "an Open Shore",
    "forest": "Dense Cover",
    "city_streets": "Crowded Streets",
    "mountains": "High Ground",
    "ancient_ruins": "Broken Structures",
    "dreamscape": "Unstable Ground",
    "ocean": "Open Water",
    "floating_islands": "Suspended Terrain",
    "abstract_space": "an Unplaced Interval",
    "grand_architecture": "a Great Interior",
}


# --------------------------------------------------------------------------- #
# Local generator                                                             #
# --------------------------------------------------------------------------- #


def _confidence_phrase(confidence: float) -> str:
    if confidence >= 0.75:
        return "The reconstruction is well supported across most of its extent"
    if confidence >= 0.55:
        return "The reconstruction is supported in its central regions and thins toward the edges"
    if confidence >= 0.35:
        return "The reconstruction is only partially supported; large areas rest on weak evidence"
    return "The reconstruction rests on sparse evidence throughout"


def _build_observations(
    scene: SceneParameters,
    attributes: MemoryAttributes,
    emotion: EmotionReading,
    cognitive: CognitiveReading,
) -> List[str]:
    observations: List[str] = []

    observations.append(
        f"Dominant cognitive signature: {_STATE_PHRASE.get(cognitive.state, cognitive.state)} "
        f"({cognitive.confidence * 100:.0f}% confidence)."
    )
    observations.append(
        f"Affective reading: {_EMOTION_PHRASE.get(emotion.primary, emotion.primary)}, "
        f"valence {emotion.valence:+.2f}, arousal {emotion.arousal:.2f}."
    )

    if attributes.familiarity > 0.62:
        observations.append(
            "High familiarity weighting — the setting was rendered as previously known, "
            "with architecture favoured over open ground."
        )
    elif attributes.familiarity < 0.38:
        observations.append(
            "Low familiarity weighting — no recognisable setting was recovered, so the "
            "environment was generated from spatial statistics alone."
        )

    if attributes.temporal_depth > 0.60:
        observations.append(
            "Temporal depth is high; the memory presents as remote rather than recent, "
            "which drove the lighting toward a low sun angle."
        )

    if attributes.population > 0.58:
        observations.append(
            "Elevated population inference — other people are represented, but no "
            "individual identity could be resolved from the signal."
        )

    if scene.atmosphere.fog_density > 0.45:
        observations.append(
            "Atmospheric density was raised to represent uncertainty: the haze marks "
            "where evidence runs out rather than describing recalled weather."
        )

    if scene.distortion > 0.45:
        observations.append(
            "Geometry departs measurably from euclidean expectations, consistent with "
            "the low temporal coherence observed across windows."
        )

    return observations


def _build_uncertainties(
    scene: SceneParameters, confidence: float, coherence: float
) -> List[str]:
    uncertainties: List[str] = []

    weak = [r for r in scene.regions if r.confidence < scene.fragment_threshold]
    if weak:
        names = ", ".join(r.label.lower() for r in weak[:3])
        uncertainties.append(
            f"{len(weak)} of {len(scene.regions)} regions fell below the resolution "
            f"threshold and remain fragmented — including {names}."
        )

    if coherence < 0.45:
        uncertainties.append(
            "The spectral signature did not stabilise across windows, so the "
            "interpretation may represent several distinct moments merged together."
        )

    if confidence < 0.55:
        uncertainties.append(
            "Overall neural confidence stayed low. Treat the specific contents as "
            "one plausible reading rather than a recovered record."
        )

    low_sound = [layer for layer in scene.soundscape if layer.confidence < 0.4]
    if low_sound:
        uncertainties.append(
            "Several soundscape layers were inferred from weak evidence and may not "
            "belong to this memory at all."
        )

    uncertainties.append(
        "No part of this reconstruction should be read as a literal record. It is the "
        "model's best fit to the neural evidence available."
    )

    return uncertainties


def generate_local(
    scene: SceneParameters,
    attributes: MemoryAttributes,
    emotion: EmotionReading,
    cognitive: CognitiveReading,
    confidence: float,
    coherence: float,
    seed: int,
) -> MemoryNarrative:
    """Deterministic narrative from the reconstruction's own parameters."""
    rng = random.Random(seed ^ 0x2A17)

    biome_phrase = _BIOME_PHRASE.get(scene.biome, "an unclassified setting")
    emotion_phrase = _EMOTION_PHRASE.get(emotion.primary, emotion.primary)
    state_phrase = _STATE_PHRASE.get(cognitive.state, cognitive.state)

    secondary = ""
    if emotion.secondary:
        secondary = f", shading toward {emotion.secondary}"

    openers = [
        "Neural activity across this session is consistent with",
        "The recorded signal is most consistent with",
        "Analysis of this session points to",
    ]

    summary = (
        f"{rng.choice(openers)} an episode characterised by {emotion_phrase}{secondary}, "
        f"recorded under {state_phrase}. The context model resolved the setting as "
        f"{biome_phrase}, lit at {scene.lighting.time_of_day.replace('_', ' ')} "
        f"with {scene.atmosphere.weather} conditions. "
        f"{_confidence_phrase(confidence)}; "
        f"{sum(1 for r in scene.regions if r.confidence < scene.fragment_threshold)} of "
        f"{len(scene.regions)} regions remain below the resolution threshold and are "
        f"rendered as incomplete geometry. Areas of the environment that appear "
        f"unfinished are not stylistic — they mark where the neural evidence was "
        f"insufficient to commit to a structure."
    )

    title = (
        f"{_TITLE_TIME.get(scene.lighting.time_of_day, 'Interval')}, "
        f"{_TITLE_PLACE.get(scene.biome, 'an Unplaced Setting')}"
    )

    return MemoryNarrative(
        summary=summary,
        title=title,
        observations=_build_observations(scene, attributes, emotion, cognitive),
        uncertainties=_build_uncertainties(scene, confidence, coherence),
        generated_by_model=False,
    )


# --------------------------------------------------------------------------- #
# Optional LLM path                                                           #
# --------------------------------------------------------------------------- #


def build_prompt(
    scene: SceneParameters,
    attributes: MemoryAttributes,
    emotion: EmotionReading,
    cognitive: CognitiveReading,
    confidence: float,
    coherence: float,
) -> str:
    """Prompt describing the reconstruction for an external model."""
    weak = [r.label for r in scene.regions if r.confidence < scene.fragment_threshold]

    return (
        "You are the interpretation layer of a neural memory reconstruction system. "
        "Write a single paragraph, 60–90 words, describing what the following "
        "reconstruction represents.\n\n"
        "Rules: be specific and restrained. Never claim the memory was recovered "
        "literally — it is an inference from scalp EEG. Name what could not be "
        "resolved. No second person, no marketing tone, no em dashes.\n\n"
        f"Setting: {scene.biome}\n"
        f"Lighting: {scene.lighting.time_of_day}, {scene.lighting.temperature:.0f}K\n"
        f"Weather: {scene.atmosphere.weather}, fog {scene.atmosphere.fog_density:.2f}\n"
        f"Primary emotion: {emotion.primary} (valence {emotion.valence:+.2f}, "
        f"arousal {emotion.arousal:.2f})\n"
        f"Cognitive state: {cognitive.state} ({cognitive.confidence:.2f})\n"
        f"Familiarity: {attributes.familiarity:.2f}, temporal depth: "
        f"{attributes.temporal_depth:.2f}, population: {attributes.population:.2f}\n"
        f"Overall confidence: {confidence:.2f}, coherence: {coherence:.2f}\n"
        f"Unresolved regions: {', '.join(weak) if weak else 'none'}\n"
    )


async def generate(
    scene: SceneParameters,
    attributes: MemoryAttributes,
    emotion: EmotionReading,
    cognitive: CognitiveReading,
    confidence: float,
    coherence: float,
    seed: int,
) -> MemoryNarrative:
    """Narrative for a completed session.

    Uses the configured LLM when available, falling back to the local
    generator on any error, timeout or missing configuration. The local result
    is always computed first so the fallback path costs nothing.
    """
    local = generate_local(
        scene, attributes, emotion, cognitive, confidence, coherence, seed
    )

    from ..config import get_settings
    from ..services.llm import complete

    settings = get_settings()
    if not settings.llm_enabled:
        return local

    prompt = build_prompt(scene, attributes, emotion, cognitive, confidence, coherence)
    text = await complete(prompt)

    if not text:
        return local

    return MemoryNarrative(
        summary=text.strip(),
        title=local.title,
        observations=local.observations,
        uncertainties=local.uncertainties,
        generated_by_model=True,
    )
