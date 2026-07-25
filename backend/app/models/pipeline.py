"""Pipeline stage models — mirrors ``frontend/src/types/pipeline.ts``."""

from __future__ import annotations

from typing import List

from pydantic import Field
from typing_extensions import Literal

from .base import CamelModel

PipelineStageId = Literal[
    "stimulus",
    "acquisition",
    "synchronization",
    "artifact_removal",
    "encoder",
    "latent",
    "scene",
    "frames",
    "refinement",
    "replay",
]

StageStatus = Literal["idle", "active", "complete", "degraded", "error"]

#: Execution order of the Memory Reconstruction Engine.
STAGE_ORDER: List[str] = [
    "stimulus",
    "acquisition",
    "synchronization",
    "artifact_removal",
    "encoder",
    "latent",
    "scene",
    "frames",
    "refinement",
    "replay",
]


class PipelineStageState(CamelModel):
    id: PipelineStageId
    status: StageStatus = "idle"
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_ms: float = 0.0
    throughput: float = 0.0
    message: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
