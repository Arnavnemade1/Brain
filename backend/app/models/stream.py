"""WebSocket event and command models — mirrors ``frontend/src/types/stream.ts``."""

from __future__ import annotations

from typing import List, Optional

from pydantic import Field
from typing_extensions import Literal

from .base import CamelModel
from .memory import (
    FidelityMetrics,
    FrameFidelity,
    ReconstructedFrame,
    ReconstructionReport,
    ReconstructionStatus,
    SceneEstimate,
    StimulusClip,
    VisualFeatures,
)
from .neural import LatentPoint, NeuralFrame
from .pipeline import PipelineStageId, PipelineStageState


class SessionOpenedEvent(CamelModel):
    type: Literal["session.opened"] = "session.opened"
    session_id: str
    sample_rate: int
    channels: List[str]
    clip: StimulusClip


class SyncEvent(CamelModel):
    type: Literal["sync"] = "sync"
    offset_seconds: float
    drift_ppm: float
    residual_ms: float
    markers_matched: int
    markers_expected: int


class FrameEvent(CamelModel):
    type: Literal["frame"] = "frame"
    frame: NeuralFrame


class PipelineEvent(CamelModel):
    type: Literal["pipeline"] = "pipeline"
    stages: List[PipelineStageState]


class LatentEvent(CamelModel):
    type: Literal["latent"] = "latent"
    point: LatentPoint


class ReconstructionFrameEvent(CamelModel):
    type: Literal["reconstruction.frame"] = "reconstruction.frame"
    frame: ReconstructedFrame
    #: Reference frame at the same grid resolution, base64 RGB.
    reference: str
    scene: SceneEstimate
    truth: VisualFeatures
    fidelity: FrameFidelity


class ProgressEvent(CamelModel):
    type: Literal["progress"] = "progress"
    status: ReconstructionStatus
    progress: float
    fidelity: float


class MetricsEvent(CamelModel):
    type: Literal["metrics"] = "metrics"
    metrics: FidelityMetrics


class ReportEvent(CamelModel):
    type: Literal["report"] = "report"
    report: ReconstructionReport


class SessionClosedEvent(CamelModel):
    type: Literal["session.closed"] = "session.closed"
    session_id: str
    reason: Literal["complete", "stopped", "error"]
    message: Optional[str] = None


class StreamErrorEvent(CamelModel):
    type: Literal["error"] = "error"
    stage: Optional[PipelineStageId] = None
    message: str
    recoverable: bool = True


class StreamCommand(CamelModel):
    """Inbound control message. Unknown fields are ignored by design."""

    type: Literal["start", "pause", "resume", "stop", "seek", "speed"]
    speed: Optional[float] = None
    t: Optional[float] = None
