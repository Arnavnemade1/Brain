"""Live pipeline stage tracking.

Mirrors the twelve stages the client renders as a flow diagram. The runner
updates this as work moves through, and the resulting states are pushed to the
UI so the animation reflects real processing rather than a scripted loop.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..models.pipeline import STAGE_ORDER, PipelineStageState

#: Stage → the unit it counts in `throughput`, used to phrase messages.
_STAGE_UNITS: Dict[str, str] = {
    "ingest": "samples",
    "cleaning": "windows",
    "denoise": "windows",
    "features": "features",
    "frequency": "spectra",
    "temporal": "windows",
    "memory_inference": "inferences",
    "emotion": "inferences",
    "context": "revisions",
    "scene_params": "revisions",
    "engine": "regions",
    "environment": "frames",
}


class PipelineTracker:
    """Mutable state for every stage of one session."""

    def __init__(self) -> None:
        self._states: Dict[str, PipelineStageState] = {
            stage_id: PipelineStageState(id=stage_id, status="idle", message="Awaiting data")
            for stage_id in STAGE_ORDER
        }

    def snapshot(self) -> List[PipelineStageState]:
        """States in execution order, safe to serialise."""
        return [self._states[stage_id].model_copy() for stage_id in STAGE_ORDER]

    def get(self, stage_id: str) -> PipelineStageState:
        return self._states[stage_id]

    def update(
        self,
        stage_id: str,
        *,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        latency_ms: Optional[float] = None,
        throughput_delta: Optional[float] = None,
        message: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> None:
        state = self._states.get(stage_id)
        if state is None:
            return

        if status is not None:
            state.status = status
        if progress is not None:
            state.progress = max(0.0, min(1.0, progress))
        if latency_ms is not None:
            # Smoothed: raw per-window timings jitter enough to make the
            # readout unreadable.
            state.latency_ms = round(state.latency_ms * 0.7 + latency_ms * 0.3, 3)
        if throughput_delta is not None:
            state.throughput += throughput_delta
        if message is not None:
            state.message = message
        if confidence is not None:
            state.confidence = max(0.0, min(1.0, confidence))

    def mark_all(self, status: str, message: str) -> None:
        for state in self._states.values():
            state.status = status
            state.message = message

    def unit_for(self, stage_id: str) -> str:
        return _STAGE_UNITS.get(stage_id, "units")
