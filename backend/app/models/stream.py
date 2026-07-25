"""WebSocket event and command models — mirrors ``frontend/src/types/stream.ts``.

Events are a discriminated union on ``type``. Each event serialises through
``CamelModel.wire()`` before being pushed to the socket.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import Field
from typing_extensions import Literal

from .base import CamelModel
from .memory import (
    MemoryFragment,
    MemoryNarrative,
    ReconstructionStatus,
    SceneParameters,
    SessionSource,
)
from .neural import NeuralFrame
from .pipeline import PipelineStageId, PipelineStageState


class SessionOpenedEvent(CamelModel):
    type: Literal["session.opened"] = "session.opened"
    session_id: str
    sample_rate: int
    channels: List[str]
    duration: Optional[float] = None
    source: SessionSource


class FrameEvent(CamelModel):
    type: Literal["frame"] = "frame"
    frame: NeuralFrame


class PipelineEvent(CamelModel):
    type: Literal["pipeline"] = "pipeline"
    stages: List[PipelineStageState]


class SceneEvent(CamelModel):
    type: Literal["scene"] = "scene"
    scene: SceneParameters


class SceneUpdateEvent(CamelModel):
    type: Literal["scene.update"] = "scene.update"
    region_confidence: Dict[str, float] = Field(default_factory=dict)
    resolved_regions: List[str] = Field(default_factory=list)
    patch: Dict[str, object] = Field(default_factory=dict)


class FragmentEvent(CamelModel):
    type: Literal["fragment"] = "fragment"
    fragment: MemoryFragment


class ReconstructionEvent(CamelModel):
    type: Literal["reconstruction"] = "reconstruction"
    status: ReconstructionStatus
    progress: float
    confidence: float
    coherence: float


class NarrativeEvent(CamelModel):
    type: Literal["narrative"] = "narrative"
    narrative: MemoryNarrative


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

    type: Literal["start", "pause", "resume", "stop", "seek", "speed", "focus"]
    speed: Optional[float] = None
    t: Optional[float] = None
    region_id: Optional[str] = None
