"""Reconstruction and fidelity models — mirrors ``frontend/src/types/memory.ts``."""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import Field
from typing_extensions import Literal

from .base import CamelModel

StimulusSource = Literal["synthetic", "uploaded"]

ReconstructionStatus = Literal[
    "idle", "synchronising", "encoding", "reconstructing", "refining", "complete", "error"
]


class StimulusClip(CamelModel):
    id: str
    title: str
    description: str
    duration_seconds: float
    fps: float
    #: Reconstruction grid. Deliberately coarse — set by what EEG supports.
    grid_width: int
    grid_height: int
    source: StimulusSource
    tags: List[str] = Field(default_factory=list)
    scene_count: int


class VisualFeatures(CamelModel):
    """Ground-truth visual properties of one reference frame."""

    t: float
    luminance: float
    contrast: float
    motion: float
    motion_direction: Optional[float] = None
    edge_density: float
    color: Tuple[float, float, float]
    scene_cut: bool
    scene_index: int


class SceneEstimate(CamelModel):
    """Scene-level properties decoded from the latent."""

    t: float
    luminance: float
    contrast: float
    motion: float
    motion_direction: Optional[float] = None
    color: Tuple[float, float, float]
    scene_cut: bool
    scene_index: int
    confidence: float = Field(ge=0.0, le=1.0)


class ReconstructedFrame(CamelModel):
    index: int
    t: float
    #: Base64 RGB bytes at the clip's grid resolution.
    pixels: str
    confidence: float = Field(ge=0.0, le=1.0)


class FidelityMetrics(CamelModel):
    ssim: float
    psnr: float
    luminance_correlation: float
    color_correlation: float
    motion_correlation: float
    scene_cut_f1: float
    layout_correlation: float
    composite: float


class FrameFidelity(CamelModel):
    index: int
    t: float
    ssim: float
    psnr: float
    layout_correlation: float


class ReconstructionReport(CamelModel):
    summary: str
    title: str
    findings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


class SessionVersion(CamelModel):
    version: int
    created_at: str
    fidelity: float
    ssim: float
    delta: str


class SessionSummary(CamelModel):
    id: str
    title: str
    created_at: str
    duration_seconds: float
    stimulus_id: str
    stimulus_title: str
    source: StimulusSource
    fidelity: float
    ssim: float
    frame_count: int
    bookmarked: bool = False
    version: int = 1


class SessionRecord(SessionSummary):
    clip: StimulusClip
    metrics: FidelityMetrics
    frame_fidelity: List[FrameFidelity] = Field(default_factory=list)
    scenes: List[SceneEstimate] = Field(default_factory=list)
    report: ReconstructionReport
    latents: List["LatentPointRef"] = Field(default_factory=list)
    history: List[SessionVersion] = Field(default_factory=list)


class LatentPointRef(CamelModel):
    """Latent sample stored with a session record."""

    t: float
    vector: List[float]
    confidence: float


SessionRecord.model_rebuild()
