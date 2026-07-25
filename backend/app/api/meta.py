"""Metadata endpoints: health, configuration and pipeline description."""

from __future__ import annotations

from typing import List

from fastapi import APIRouter

from ..config import get_settings
from ..models.base import CamelModel
from ..models.pipeline import STAGE_ORDER
from ..processing.bands import BAND_RANGES
from ..processing.montage import CHANNELS, POSITIONS, REGIONS
from ..reconstruction.decoders import get_decoders
from ..services.store import get_store
from ..websocket.manager import get_manager

router = APIRouter(prefix="/api", tags=["meta"])


class ElectrodeInfo(CamelModel):
    label: str
    x: float
    y: float
    region: str


class SystemInfo(CamelModel):
    version: str
    sample_rate: int
    window_seconds: float
    window_overlap: float
    line_frequency: float
    llm_narratives: bool
    stage_order: List[str]
    stored_sessions: int
    active_sessions: int
    #: Whether fitted decoder weights are loaded.
    decoders_fitted: bool
    decoder_trained_on: List[str]


@router.get("/health")
async def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "version": settings.version,
        "llmNarratives": settings.llm_enabled,
    }


@router.get("/system", response_model=SystemInfo)
async def system() -> SystemInfo:
    """Everything the client needs to configure itself at startup."""
    settings = get_settings()
    return SystemInfo(
        version=settings.version,
        sample_rate=settings.sample_rate,
        window_seconds=settings.window_seconds,
        window_overlap=settings.window_overlap,
        line_frequency=settings.line_frequency,
        llm_narratives=settings.llm_enabled,
        stage_order=list(STAGE_ORDER),
        stored_sessions=get_store().count,
        active_sessions=get_manager().count,
        decoders_fitted=get_decoders().fitted,
        decoder_trained_on=list(get_decoders().trained_on),
    )


@router.get("/montage", response_model=List[ElectrodeInfo])
async def montage() -> List[ElectrodeInfo]:
    """Electrode labels and normalised scalp coordinates for the topography map."""
    lookup = {}
    for region, indices in REGIONS.items():
        for index in indices:
            if index < len(CHANNELS):
                lookup[CHANNELS[index]] = region

    return [
        ElectrodeInfo(
            label=label,
            x=POSITIONS[label][0],
            y=POSITIONS[label][1],
            region=lookup.get(label, "central"),
        )
        for label in CHANNELS
    ]


@router.get("/bands")
async def bands() -> dict:
    """Canonical frequency band edges in Hz."""
    return {name: {"low": low, "high": high} for name, (low, high) in BAND_RANGES.items()}
