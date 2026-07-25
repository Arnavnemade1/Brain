"""Session assembly.

Builds a fully-configured :class:`ReconstructionRunner` from a stimulus
selection, so the API, the WebSocket layer and the offline evaluation harness
all construct sessions the same way.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..config import get_settings
from ..models.memory import StimulusClip, VisualFeatures
from ..reconstruction.runner import ReconstructionRunner
from ..simulation.visual_response import CONDITIONS, simulate
from ..stimulus.clips import CLIPS, FPS, GRID_HEIGHT, GRID_WIDTH, get_clip, render_clip
from ..stimulus.features import extract
from ..utils.ids import session_id as new_session_id


def clip_metadata(clip_id: str) -> StimulusClip:
    """Public description of a stimulus clip."""
    spec = get_clip(clip_id)
    return StimulusClip(
        id=spec.id,
        title=spec.title,
        description=spec.description,
        duration_seconds=round(spec.duration, 2),
        fps=FPS,
        grid_width=GRID_WIDTH,
        grid_height=GRID_HEIGHT,
        source="synthetic",
        tags=list(spec.tags),
        scene_count=len(spec.scenes),
    )


def available_clips() -> List[StimulusClip]:
    return [clip_metadata(clip_id) for clip_id in CLIPS]


def prepare(
    clip_id: str,
    condition: str = "research",
    speed: float = 1.0,
    seed: Optional[int] = None,
    session_id: Optional[str] = None,
) -> ReconstructionRunner:
    """Render a clip, synthesise a recording, and build a runner for it."""
    settings = get_settings()

    frames, scene_indices = render_clip(clip_id)
    features: List[VisualFeatures] = extract(frames, FPS, scene_indices)

    sid = session_id or new_session_id()

    recording = simulate(
        frames,
        features,
        FPS,
        condition if condition in CONDITIONS else "research",
        settings.sample_rate,
        seed=seed if seed is not None else abs(hash(sid)) % (2**31),
    )

    return ReconstructionRunner(
        session_id=sid,
        clip=clip_metadata(clip_id),
        frames=frames,
        features=features,
        recording=recording,
        speed=speed,
    )


def condition_choices() -> List[dict]:
    """Recording conditions offered at session launch."""
    return [
        {
            "id": condition.id,
            "label": condition.label,
            "description": condition.description,
            "artifactLevel": condition.artifact_level,
            "responseGain": condition.response_gain,
        }
        for condition in CONDITIONS.values()
    ]
