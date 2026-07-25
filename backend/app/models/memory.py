"""Memory reconstruction models — mirrors ``frontend/src/types/memory.ts``."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from pydantic import Field
from typing_extensions import Literal

from .base import CamelModel
from .neural import CognitiveState, EmotionLabel

Biome = Literal[
    "childhood_home",
    "school_hallway",
    "beach_sunset",
    "forest",
    "city_streets",
    "mountains",
    "ancient_ruins",
    "dreamscape",
    "ocean",
    "floating_islands",
    "abstract_space",
    "grand_architecture",
]

BIOMES: List[str] = [
    "childhood_home",
    "school_hallway",
    "beach_sunset",
    "forest",
    "city_streets",
    "mountains",
    "ancient_ruins",
    "dreamscape",
    "ocean",
    "floating_islands",
    "abstract_space",
    "grand_architecture",
]

WeatherKind = Literal["clear", "overcast", "fog", "rain", "snow", "dust", "ashfall"]
TimeOfDay = Literal["dawn", "morning", "noon", "afternoon", "golden", "dusk", "night"]
ParticleKind = Literal["dust", "pollen", "embers", "snow", "rain", "motes", "none"]
SoundLayerKind = Literal[
    "wind", "waves", "rain", "birds", "traffic", "voices", "hum", "chimes", "heartbeat"
]
RegionArchetype = Literal["building", "terrain", "monument", "corridor", "grove", "water", "void"]
ReconstructionStatus = Literal[
    "idle", "acquiring", "processing", "reconstructing", "stabilised", "error"
]
SessionSource = Literal["simulated", "uploaded", "device"]


class SoundLayer(CamelModel):
    kind: SoundLayerKind
    gain: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class Palette(CamelModel):
    key: str
    fill: str
    rim: str
    fog: str
    sky: str
    ground: str
    accent: str


class LightingParams(CamelModel):
    time_of_day: TimeOfDay
    sun_elevation: float
    sun_azimuth: float
    intensity: float = Field(ge=0.0, le=2.0)
    temperature: float
    ambient: float = Field(ge=0.0, le=2.0)
    volumetric: float = Field(ge=0.0, le=1.0)


class AtmosphereParams(CamelModel):
    weather: WeatherKind
    fog_density: float = Field(ge=0.0, le=1.0)
    particles: ParticleKind
    particle_density: float = Field(ge=0.0, le=1.0)
    wind: float = Field(ge=0.0, le=1.0)


class RegionParams(CamelModel):
    id: str
    label: str
    position: Tuple[float, float, float]
    radius: float
    confidence: float = Field(ge=0.0, le=1.0)
    archetype: RegionArchetype
    locked_by: Optional[str] = None


class SceneParameters(CamelModel):
    """The deterministic render descriptor handed to the 3D engine."""

    seed: int
    biome: Biome
    palette: Palette
    lighting: LightingParams
    atmosphere: AtmosphereParams
    spatial_scale: float = Field(ge=0.0, le=1.0)
    population_density: float = Field(ge=0.0, le=1.0)
    object_density: float = Field(ge=0.0, le=1.0)
    emotional_intensity: float = Field(ge=0.0, le=1.0)
    movement: float = Field(ge=0.0, le=1.0)
    fragment_threshold: float = Field(ge=0.0, le=1.0)
    distortion: float = Field(ge=0.0, le=1.0)
    regions: List[RegionParams] = Field(default_factory=list)
    soundscape: List[SoundLayer] = Field(default_factory=list)


class MemoryFragment(CamelModel):
    id: str
    title: str
    description: str
    position: Tuple[float, float, float]
    emotion: EmotionLabel
    confidence: float = Field(ge=0.0, le=1.0)
    sensory_details: List[str] = Field(default_factory=list)
    related_ids: List[str] = Field(default_factory=list)
    timeline_position: float = Field(ge=0.0, le=1.0)
    neural_evidence: str
    source_time: float
    unlocks_region: Optional[str] = None
    revealed: bool = False


class MemoryNarrative(CamelModel):
    summary: str
    title: str
    observations: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    generated_by_model: bool = False


class ConfidenceSample(CamelModel):
    t: float
    confidence: float
    coherence: float


class EmotionSample(CamelModel):
    t: float
    emotion: EmotionLabel
    intensity: float


class SessionVersion(CamelModel):
    version: int
    created_at: str
    confidence: float
    coherence: float
    fragment_count: int
    delta: str


class SessionSummary(CamelModel):
    id: str
    title: str
    created_at: str
    duration_seconds: float
    source: SessionSource
    biome: Biome
    primary_emotion: EmotionLabel
    dominant_cognitive_state: CognitiveState
    confidence: float
    coherence: float
    fragment_count: int
    bookmarked: bool = False
    version: int = 1


class SessionRecord(SessionSummary):
    scene: SceneParameters
    fragments: List[MemoryFragment] = Field(default_factory=list)
    narrative: MemoryNarrative
    confidence_trace: List[ConfidenceSample] = Field(default_factory=list)
    emotion_trace: List[EmotionSample] = Field(default_factory=list)
    history: List[SessionVersion] = Field(default_factory=list)


class ReconstructionState(CamelModel):
    status: ReconstructionStatus = "idle"
    progress: float = 0.0
    confidence: float = 0.0
    coherence: float = 0.0
    region_confidence: Dict[str, float] = Field(default_factory=dict)
    scene: Optional[SceneParameters] = None
    fragments: List[MemoryFragment] = Field(default_factory=list)
