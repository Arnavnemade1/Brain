"""Reconstruction session lifecycle."""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import Field

from ..config import get_settings
from ..models.base import CamelModel
from ..models.memory import StimulusClip
from ..services.sessions import available_clips, condition_choices, prepare
from ..simulation.visual_response import CONDITIONS
from ..stimulus.clips import CLIPS
from ..utils.ids import session_id as new_session_id
from ..websocket.manager import SessionBusy, get_manager

log = logging.getLogger("mindscape.api")

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class SessionRequest(CamelModel):
    clip: str = Field(default="daybreak")
    condition: str = Field(default="research")
    speed: float = Field(default=2.0, ge=0.25, le=16.0)


class SessionCreated(CamelModel):
    session_id: str
    sample_rate: int
    channels: List[str]
    clip: StimulusClip
    condition: str
    stream_url: str
    window_seconds: float
    hop_seconds: float


class ConditionInfo(CamelModel):
    id: str
    label: str
    description: str
    artifact_level: float
    response_gain: float


@router.get("/clips", response_model=List[StimulusClip])
async def list_clips() -> List[StimulusClip]:
    """Reference clips available as stimulus."""
    return available_clips()


@router.get("/conditions", response_model=List[ConditionInfo])
async def list_conditions() -> List[ConditionInfo]:
    """Recording-quality presets for the simulated acquisition."""
    return [ConditionInfo.model_validate(entry) for entry in condition_choices()]


@router.post("", response_model=SessionCreated)
async def create_session(request: SessionRequest) -> SessionCreated:
    """Start a reconstruction session against a reference clip."""
    if request.clip not in CLIPS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown clip {request.clip!r}. Available: {', '.join(CLIPS)}",
        )
    if request.condition not in CONDITIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown condition {request.condition!r}. "
                f"Available: {', '.join(CONDITIONS)}"
            ),
        )

    settings = get_settings()
    sid = new_session_id()

    runner = prepare(
        request.clip, request.condition, speed=request.speed, session_id=sid
    )

    try:
        await get_manager().register(sid, runner)
    except SessionBusy as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return SessionCreated(
        session_id=sid,
        sample_rate=runner.sample_rate,
        channels=list(runner.recording.channels),
        clip=runner.clip,
        condition=request.condition,
        stream_url=f"/ws/session/{sid}",
        window_seconds=settings.window_seconds,
        hop_seconds=round(settings.hop_seconds, 4),
    )


@router.get("/active")
async def list_active() -> dict:
    manager = get_manager()
    return {"count": manager.count, "sessionIds": manager.active_ids}


@router.delete("/{session_id}")
async def stop_session(session_id: str) -> dict:
    runner = get_manager().get(session_id)
    if runner is None:
        raise HTTPException(status_code=404, detail=f"No active session {session_id}")

    runner.stop()
    await get_manager().finish(session_id, persist=True)
    return {"stopped": session_id}
