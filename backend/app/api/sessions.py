"""Session lifecycle endpoints."""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import Field

from ..config import get_settings
from ..models.base import CamelModel
from ..services.ingest import IngestError, load
from ..simulation.generator import generate_session, profile_choices
from ..simulation.profiles import PROFILES
from ..utils.ids import session_id as new_session_id
from ..websocket.manager import SessionBusy, get_manager

log = logging.getLogger("mindscape.api")

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class SimulationRequest(CamelModel):
    profile: str = Field(default="coastal_recall")
    duration: float = Field(default=120.0, ge=8.0, le=900.0)
    speed: float = Field(default=1.0, ge=0.25, le=8.0)
    title: Optional[str] = None


class SessionCreated(CamelModel):
    session_id: str
    sample_rate: int
    channels: List[str]
    duration: float
    source: str
    #: Relative WebSocket path the client should connect to.
    stream_url: str
    window_seconds: float
    hop_seconds: float


class ProfileInfo(CamelModel):
    id: str
    name: str
    description: str
    tags: List[str]
    #: 0–1 recording quality of the notional session, for the launch UI.
    artifact_level: float


@router.get("/profiles", response_model=List[ProfileInfo])
async def list_profiles() -> List[ProfileInfo]:
    """Simulation profiles available for a synthetic session."""
    return [
        ProfileInfo(
            id=profile.id,
            name=profile.name,
            description=profile.description,
            tags=list(profile.tags),
            artifact_level=profile.artifact_level,
        )
        for profile in PROFILES.values()
    ]


@router.post("/simulate", response_model=SessionCreated)
async def create_simulated(request: SimulationRequest) -> SessionCreated:
    """Start a session backed by a synthetic recording."""
    if request.profile not in PROFILES:
        available = ", ".join(PROFILES)
        raise HTTPException(
            status_code=400,
            detail=f"Unknown profile {request.profile!r}. Available: {available}",
        )

    settings = get_settings()
    sid = new_session_id()

    recording = generate_session(
        request.profile,
        request.duration,
        settings.sample_rate,
        seed=abs(hash(sid)) % (2**31),
    )

    profile = PROFILES[request.profile]

    try:
        await get_manager().create(
            sid,
            recording,
            source="simulated",
            title=request.title or profile.name,
            speed=request.speed,
        )
    except SessionBusy as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return SessionCreated(
        session_id=sid,
        sample_rate=recording.sample_rate,
        channels=list(recording.channels),
        duration=round(recording.duration, 2),
        source="simulated",
        stream_url=f"/ws/session/{sid}",
        window_seconds=settings.window_seconds,
        hop_seconds=round(settings.hop_seconds, 4),
    )


@router.post("/upload", response_model=SessionCreated)
async def create_from_upload(
    file: UploadFile = File(...),
    speed: float = Form(default=1.0),
    title: Optional[str] = Form(default=None),
) -> SessionCreated:
    """Start a session from an uploaded EEG recording."""
    settings = get_settings()

    payload = await file.read()
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB limit."
            ),
        )
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        recording = load(payload, file.filename or "recording", settings.sample_rate)
    except IngestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected failure reading %s", file.filename)
        raise HTTPException(
            status_code=400, detail=f"Could not read {file.filename}: {exc}"
        ) from exc

    sid = new_session_id()

    try:
        await get_manager().create(
            sid,
            recording,
            source="uploaded",
            title=title or (file.filename or "Uploaded Recording"),
            speed=speed,
        )
    except SessionBusy as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return SessionCreated(
        session_id=sid,
        sample_rate=recording.sample_rate,
        channels=list(recording.channels),
        duration=round(recording.duration, 2),
        source="uploaded",
        stream_url=f"/ws/session/{sid}",
        window_seconds=settings.window_seconds,
        hop_seconds=round(settings.hop_seconds, 4),
    )


@router.get("/active")
async def list_active() -> dict:
    """Sessions currently streaming in this process."""
    manager = get_manager()
    return {"count": manager.count, "sessionIds": manager.active_ids}


@router.delete("/{session_id}")
async def stop_session(session_id: str) -> dict:
    """Stop a live session and persist whatever was reconstructed."""
    runner = get_manager().get(session_id)
    if runner is None:
        raise HTTPException(status_code=404, detail=f"No active session {session_id}")

    runner.stop()
    await get_manager().finish(session_id, persist=True)
    return {"stopped": session_id}


@router.get("/simulation-profiles/choices")
async def profile_choice_list() -> List[dict]:
    """Compact `(id, name, description)` list for select inputs."""
    return [
        {"id": pid, "name": name, "description": description}
        for pid, name, description in profile_choices()
    ]
