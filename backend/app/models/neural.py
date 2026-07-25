"""Neural domain models — mirrors ``frontend/src/types/neural.ts``."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import Field
from typing_extensions import Literal

from .base import CamelModel

FrequencyBand = Literal["delta", "theta", "alpha", "beta", "gamma"]

CognitiveState = Literal[
    "focused",
    "relaxed",
    "drowsy",
    "dreaming",
    "recalling",
    "alert",
    "meditative",
]

EmotionLabel = Literal[
    "calm",
    "nostalgia",
    "wonder",
    "stress",
    "curiosity",
    "melancholy",
    "joy",
    "fear",
]

SignalQualityGrade = Literal["excellent", "good", "fair", "poor"]

COGNITIVE_STATES: List[str] = [
    "focused",
    "relaxed",
    "drowsy",
    "dreaming",
    "recalling",
    "alert",
    "meditative",
]

EMOTIONS: List[str] = [
    "calm",
    "nostalgia",
    "wonder",
    "stress",
    "curiosity",
    "melancholy",
    "joy",
    "fear",
]


class SignalQuality(CamelModel):
    score: float = Field(ge=0.0, le=1.0)
    grade: SignalQualityGrade
    artifact_ratio: float = Field(ge=0.0, le=1.0)
    dropped_channels: List[str] = Field(default_factory=list)
    line_noise: float = Field(ge=0.0, le=1.0)


class Spectrum(CamelModel):
    frequencies: List[float]
    magnitudes: List[float]
    peak_frequency: float


class CognitiveReading(CamelModel):
    state: CognitiveState
    confidence: float = Field(ge=0.0, le=1.0)
    distribution: Dict[str, float] = Field(default_factory=dict)
    engagement: float = Field(ge=0.0, le=1.0)
    load: float = Field(ge=0.0, le=1.0)


class EmotionReading(CamelModel):
    primary: EmotionLabel
    secondary: Optional[EmotionLabel] = None
    intensity: float = Field(ge=0.0, le=1.0)
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    distribution: Dict[str, float] = Field(default_factory=dict)


class NeuralFrame(CamelModel):
    """One decoded analysis window — the atomic unit the pipeline emits."""

    session_id: str
    index: int
    t: float
    band_powers: Dict[str, float]
    spectrum: Spectrum
    topography: Dict[str, float]
    waveform: List[float]
    signal_quality: SignalQuality
    cognitive: CognitiveReading
    emotion: EmotionReading
    neural_confidence: float = Field(ge=0.0, le=1.0)
    coherence: float = Field(ge=0.0, le=1.0)
