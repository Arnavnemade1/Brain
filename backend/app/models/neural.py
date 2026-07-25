"""Neural signal models — mirrors ``frontend/src/types/neural.ts``."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import Field
from typing_extensions import Literal

from .base import CamelModel

FrequencyBand = Literal["delta", "theta", "alpha", "beta", "gamma"]
SignalQualityGrade = Literal["excellent", "good", "fair", "poor"]


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


class VisualResponse(CamelModel):
    """Visual-system measures the encoder reads.

    These are the quantities scalp EEG can actually carry about visual input.
    Everything the reconstruction knows comes through here.
    """

    occipital_drive: float = Field(ge=0.0, le=1.0)
    alpha_suppression: float = Field(ge=0.0, le=1.0)
    motion_response: float = Field(ge=0.0, le=1.0)
    transient: float = Field(ge=0.0, le=1.0)
    detail_response: float = Field(ge=0.0, le=1.0)
    lateral_balance: float = Field(ge=-1.0, le=1.0)
    vertical_balance: float = Field(ge=-1.0, le=1.0)


class NeuralFrame(CamelModel):
    session_id: str
    index: int
    #: Seconds on the *stimulus* timeline, after synchronization.
    t: float
    band_powers: Dict[str, float]
    spectrum: Spectrum
    topography: Dict[str, float]
    waveform: List[float]
    signal_quality: SignalQuality
    visual: VisualResponse
    confidence: float = Field(ge=0.0, le=1.0)


class SyncResult(CamelModel):
    """Alignment between the recording clock and the stimulus clock."""

    offset_seconds: float
    drift_ppm: float
    residual_ms: float
    markers_matched: int
    markers_expected: int

    @property
    def usable(self) -> bool:
        """Whether alignment is tight enough to attribute responses to frames."""
        return self.markers_matched >= 2 and self.residual_ms < 120.0


class LatentPoint(CamelModel):
    t: float
    vector: List[float]
    confidence: float = Field(ge=0.0, le=1.0)


#: Dimensionality of the memory latent space. Small on purpose: the encoder
#: is summarising a handful of genuinely recoverable visual quantities, and a
#: wider latent would mostly encode noise.
LATENT_DIMENSIONS = 16

_EMPTY: Optional[List[float]] = None
