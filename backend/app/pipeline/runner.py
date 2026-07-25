"""Session runner — the engine that drives one reconstruction.

Owns the clock, the analyser, the accumulated evidence and the reconstruction
state, and emits a stream of events describing what it finds. The transport is
somebody else's problem: the runner yields events, and the WebSocket layer
forwards them.

The reconstruction *builds over time*. A scene is only committed once enough
evidence has accumulated, and afterwards regions continue to resolve as
further windows either confirm or undermine them.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import AsyncIterator, Dict, List, Optional, Sequence, Union

import numpy as np

from ..config import get_settings
from ..models.memory import (
    ConfidenceSample,
    EmotionSample,
    MemoryFragment,
    SceneParameters,
    SessionRecord,
    SessionSource,
)
from ..models.neural import NeuralFrame
from ..models.stream import (
    FragmentEvent,
    FrameEvent,
    NarrativeEvent,
    PipelineEvent,
    ReconstructionEvent,
    SceneEvent,
    SceneUpdateEvent,
    SessionClosedEvent,
    SessionOpenedEvent,
    StreamErrorEvent,
)
from ..services.ingest import Recording
from ..simulation.generator import SimulatedRecording
from ..utils.ids import seed_from
from .analyzer import AnalysisResult, WindowAnalyzer
from .context import synthesise_scene
from .fragments import build_fragments
from .inference import MemoryAttributes
from .narrative import generate as generate_narrative
from .stages import PipelineTracker

log = logging.getLogger("mindscape.runner")

StreamEvent = Union[
    SessionOpenedEvent,
    FrameEvent,
    PipelineEvent,
    SceneEvent,
    SceneUpdateEvent,
    FragmentEvent,
    ReconstructionEvent,
    NarrativeEvent,
    SessionClosedEvent,
    StreamErrorEvent,
]

SourceRecording = Union[SimulatedRecording, Recording]

#: Windows of evidence required before the scene is first committed. Below
#: this the model has seen too little to name a setting at all.
WARMUP_WINDOWS = 6

#: How often, in windows, the scene is re-examined for newly resolved regions.
REFINE_INTERVAL = 4


class SessionRunner:
    """Drives one neural session from raw signal to explorable reconstruction."""

    def __init__(
        self,
        session_id: str,
        recording: SourceRecording,
        source: SessionSource,
        title: Optional[str] = None,
        speed: float = 1.0,
    ) -> None:
        settings = get_settings()

        self.session_id = session_id
        self.recording = recording
        self.source = source
        self.title = title or "Untitled Session"
        self.speed = float(np.clip(speed, 0.25, 8.0))

        self.sample_rate = recording.sample_rate
        self.window_samples = int(self.sample_rate * settings.window_seconds)
        self.hop_samples = max(1, int(self.window_samples * (1.0 - settings.window_overlap)))
        self.duration = recording.duration

        self.analyzer = WindowAnalyzer(
            session_id, self.sample_rate, list(recording.channels), settings.line_frequency
        )
        self.tracker = PipelineTracker()

        self._position = 0
        self._paused = False
        self._stopped = False
        self._focus_requests: List[str] = []

        self._frames: List[NeuralFrame] = []
        self._attribute_samples: List[MemoryAttributes] = []
        self._attribute_weights: List[float] = []
        self._confidence_trace: List[ConfidenceSample] = []
        self._emotion_trace: List[EmotionSample] = []

        self.scene: Optional[SceneParameters] = None
        self.fragments: List[MemoryFragment] = []
        self._region_confidence: Dict[str, float] = {}
        self._resolved: set = set()
        self._seed = seed_from(session_id, source)

        self.created_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------ control

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        self._stopped = True

    def set_speed(self, speed: float) -> None:
        self.speed = float(np.clip(speed, 0.25, 8.0))

    def seek(self, seconds: float) -> None:
        target = int(np.clip(seconds, 0.0, max(0.0, self.duration)) * self.sample_rate)
        self._position = max(0, min(target, self.recording.total_samples - 1))

    def focus_region(self, region_id: str) -> None:
        """Spend the next windows of evidence resolving one region."""
        if region_id not in self._focus_requests:
            self._focus_requests.append(region_id)

    @property
    def total_windows(self) -> int:
        usable = max(0, self.recording.total_samples - self.window_samples)
        return max(1, usable // self.hop_samples + 1)

    @property
    def progress(self) -> float:
        return float(np.clip(self._position / max(1, self.recording.total_samples), 0.0, 1.0))

    # ------------------------------------------------------------------- stream

    async def run(self) -> AsyncIterator[StreamEvent]:
        """Yield the full event stream for this session."""
        yield SessionOpenedEvent(
            session_id=self.session_id,
            sample_rate=self.sample_rate,
            channels=list(self.recording.channels),
            duration=round(self.duration, 2),
            source=self.source,
        )

        self.tracker.update(
            "ingest", status="active", message=f"Buffering {self.sample_rate} Hz stream"
        )
        yield PipelineEvent(stages=self.tracker.snapshot())

        window_index = 0

        try:
            while not self._stopped:
                if self._paused:
                    await asyncio.sleep(0.1)
                    continue

                if self._position + self.window_samples > self.recording.total_samples:
                    break

                started = time.perf_counter()

                window = self.recording.window(self._position, self.window_samples)
                t = self._position / float(self.sample_rate)

                try:
                    result = self.analyzer.analyze(window, t)
                except Exception as exc:  # noqa: BLE001
                    log.exception("Analysis failed at t=%.2fs", t)
                    yield StreamErrorEvent(
                        stage="features",
                        message=f"Window analysis failed at {t:.1f}s: {exc}",
                        recoverable=True,
                    )
                    self._position += self.hop_samples
                    continue

                self._record(result)
                self._update_tracker(result, window_index)

                yield FrameEvent(frame=result.frame)
                yield PipelineEvent(stages=self.tracker.snapshot())

                # --- commit the scene once there is enough evidence ------------
                if self.scene is None and len(self._frames) >= WARMUP_WINDOWS:
                    self.scene = self._synthesise()
                    self._region_confidence = {
                        region.id: region.confidence for region in self.scene.regions
                    }
                    self.tracker.update(
                        "scene_params",
                        status="complete",
                        progress=1.0,
                        message=f"Committed: {self.scene.biome.replace('_', ' ')}",
                        confidence=self.analyzer.confidence,
                    )
                    yield SceneEvent(scene=self.scene)

                # --- refine as evidence accumulates ---------------------------
                elif self.scene is not None and window_index % REFINE_INTERVAL == 0:
                    update = self._refine()
                    if update is not None:
                        yield update

                yield ReconstructionEvent(
                    status="reconstructing" if self.scene else "processing",
                    progress=round(self.progress, 4),
                    confidence=round(self.analyzer.confidence, 4),
                    coherence=round(self.analyzer.coherence, 4),
                )

                self._position += self.hop_samples
                window_index += 1

                # Pace to wall-clock so the console reads as a live stream.
                elapsed = time.perf_counter() - started
                target = (self.hop_samples / float(self.sample_rate)) / self.speed
                if target > elapsed:
                    await asyncio.sleep(target - elapsed)

            # --- completion -------------------------------------------------
            async for event in self._finalise():
                yield event

        except asyncio.CancelledError:
            yield SessionClosedEvent(
                session_id=self.session_id, reason="stopped", message="Session cancelled"
            )
            raise

        except Exception as exc:  # noqa: BLE001
            log.exception("Session %s failed", self.session_id)
            self.tracker.mark_all("error", "Session failed")
            yield PipelineEvent(stages=self.tracker.snapshot())
            yield SessionClosedEvent(
                session_id=self.session_id, reason="error", message=str(exc)
            )

    # ------------------------------------------------------------------ helpers

    def _record(self, result: AnalysisResult) -> None:
        frame = result.frame
        self._frames.append(frame)
        self._attribute_samples.append(result.attributes)
        self._attribute_weights.append(frame.neural_confidence)

        self._confidence_trace.append(
            ConfidenceSample(
                t=frame.t,
                confidence=round(frame.neural_confidence, 4),
                coherence=round(frame.coherence, 4),
            )
        )
        self._emotion_trace.append(
            EmotionSample(
                t=frame.t,
                emotion=frame.emotion.primary,
                intensity=round(frame.emotion.intensity, 4),
            )
        )

    def _update_tracker(self, result: AnalysisResult, window_index: int) -> None:
        frame = result.frame
        latencies = result.stage_latencies
        quality = frame.signal_quality

        self.tracker.update(
            "ingest",
            status="active",
            progress=self.progress,
            throughput_delta=self.hop_samples,
            message=f"{len(self.recording.channels)} ch @ {self.sample_rate} Hz",
            confidence=1.0,
        )

        dropped = len(quality.dropped_channels)
        self.tracker.update(
            "cleaning",
            status="degraded" if dropped else "active",
            progress=1.0,
            latency_ms=latencies.get("cleaning", 0.0),
            throughput_delta=1,
            message=(
                f"{dropped} channel(s) excluded"
                if dropped
                else f"Re-referenced, 0.5–45 Hz, {len(self.recording.channels)} ch"
            ),
            confidence=quality.score,
        )

        self.tracker.update(
            "denoise",
            status="degraded" if quality.artifact_ratio > 0.02 else "active",
            progress=1.0,
            latency_ms=latencies.get("denoise", 0.0),
            throughput_delta=1,
            message=(
                f"Notch {get_settings().line_frequency:.0f} Hz · "
                f"{quality.artifact_ratio * 100:.1f}% attenuated"
            ),
            confidence=1.0 - quality.artifact_ratio * 6.0,
        )

        self.tracker.update(
            "features",
            status="active",
            progress=1.0,
            latency_ms=latencies.get("features", 0.0),
            throughput_delta=9,
            message=(
                f"Hjorth, entropy, connectivity {result.features.connectivity:.2f}"
            ),
            confidence=quality.score,
        )

        self.tracker.update(
            "frequency",
            status="active",
            progress=1.0,
            latency_ms=latencies.get("frequency", 0.0),
            throughput_delta=1,
            message=f"Welch PSD · peak {frame.spectrum.peak_frequency:.1f} Hz",
            confidence=quality.score,
        )

        self.tracker.update(
            "temporal",
            status="active",
            progress=min(1.0, len(self._frames) / 24.0),
            latency_ms=latencies.get("temporal", 0.0),
            throughput_delta=1,
            message=f"Coherence {frame.coherence:.2f} over {len(self._frames)} windows",
            confidence=frame.coherence,
        )

        self.tracker.update(
            "memory_inference",
            status="active",
            progress=1.0,
            latency_ms=latencies.get("memory_inference", 0.0),
            throughput_delta=1,
            message=(
                f"Familiarity {result.attributes.familiarity:.2f} · "
                f"scale {result.attributes.spatial_scale:.2f}"
            ),
            confidence=frame.neural_confidence,
        )

        self.tracker.update(
            "emotion",
            status="active",
            progress=1.0,
            latency_ms=latencies.get("emotion", 0.0),
            throughput_delta=1,
            message=(
                f"{frame.emotion.primary} · valence {frame.emotion.valence:+.2f} · "
                f"arousal {frame.emotion.arousal:.2f}"
            ),
            confidence=frame.emotion.intensity,
        )

        if len(self._frames) < WARMUP_WINDOWS:
            self.tracker.update(
                "context",
                status="active",
                progress=len(self._frames) / WARMUP_WINDOWS,
                message=(
                    f"Accumulating evidence "
                    f"({len(self._frames)}/{WARMUP_WINDOWS} windows)"
                ),
                confidence=self.analyzer.confidence,
            )
            self.tracker.update("scene_params", status="idle", message="Awaiting context")
            self.tracker.update("engine", status="idle", message="Awaiting parameters")
            self.tracker.update("environment", status="idle", message="Not yet built")
        else:
            self.tracker.update(
                "context",
                status="complete",
                progress=1.0,
                throughput_delta=1 if window_index % REFINE_INTERVAL == 0 else 0,
                message="Setting, era and narrative framing resolved",
                confidence=self.analyzer.confidence,
            )

    def _aggregate_attributes(self) -> MemoryAttributes:
        return MemoryAttributes.aggregate(self._attribute_samples, self._attribute_weights)

    def _synthesise(self) -> SceneParameters:
        latest = self._frames[-1]
        return synthesise_scene(
            self.session_id,
            self._seed,
            self._aggregate_attributes(),
            latest.emotion,
            latest.cognitive,
            self.analyzer.confidence,
            self.analyzer.coherence,
        )

    def _refine(self) -> Optional[SceneUpdateEvent]:
        """Let accumulated evidence resolve regions of the reconstruction.

        Region confidence climbs toward the session's overall confidence,
        faster for regions the user has asked the engine to focus on. Regions
        crossing the threshold are reported so the client can materialise them.
        """
        if self.scene is None:
            return None

        newly_resolved: List[str] = []
        overall = self.analyzer.confidence
        coherence = self.analyzer.coherence

        for region in self.scene.regions:
            current = self._region_confidence.get(region.id, region.confidence)

            # Each region approaches a ceiling set by its own starting evidence
            # and the session's overall confidence, so a weak region never
            # becomes fully resolved just because time passed.
            ceiling = min(1.0, region.confidence * 0.55 + overall * 0.75)
            rate = 0.12 + 0.25 * coherence

            if region.id in self._focus_requests:
                rate *= 2.6
                ceiling = min(1.0, ceiling + 0.18)

            updated = current + (ceiling - current) * rate
            self._region_confidence[region.id] = float(np.clip(updated, 0.0, 1.0))

            if (
                updated >= self.scene.fragment_threshold
                and region.id not in self._resolved
            ):
                self._resolved.add(region.id)
                newly_resolved.append(region.id)

        self._focus_requests.clear()

        resolved_count = len(self._resolved)
        total = len(self.scene.regions)

        self.tracker.update(
            "engine",
            status="active",
            progress=resolved_count / max(1, total),
            throughput_delta=len(newly_resolved),
            message=f"{resolved_count}/{total} regions resolved",
            confidence=overall,
        )
        self.tracker.update(
            "environment",
            status="active",
            progress=self.progress,
            message="Environment live and stabilising",
            confidence=coherence,
        )

        return SceneUpdateEvent(
            region_confidence={
                k: round(v, 4) for k, v in self._region_confidence.items()
            },
            resolved_regions=newly_resolved,
            patch={},
        )

    async def _finalise(self) -> AsyncIterator[StreamEvent]:
        """Extract fragments, write the narrative and close the session."""
        if not self._frames:
            yield SessionClosedEvent(
                session_id=self.session_id,
                reason="error",
                message="No analysable windows in this recording",
            )
            return

        if self.scene is None:
            # Very short recording — commit on whatever evidence exists.
            self.scene = self._synthesise()
            self._region_confidence = {r.id: r.confidence for r in self.scene.regions}
            yield SceneEvent(scene=self.scene)

        # Apply accumulated refinement to the scene itself, so a session
        # reloaded from storage starts where this one finished.
        for region in self.scene.regions:
            region.confidence = round(
                self._region_confidence.get(region.id, region.confidence), 4
            )

        attributes = self._aggregate_attributes()
        latest = self._frames[-1]

        self.fragments = build_fragments(
            self.session_id,
            self._frames,
            attributes,
            self.scene.biome,
            self.scene.regions,
            self._seed,
            self.duration,
        )

        self.tracker.update(
            "engine",
            status="complete",
            progress=1.0,
            message=f"{len(self.fragments)} fragments recovered",
            confidence=self.analyzer.confidence,
        )
        yield PipelineEvent(stages=self.tracker.snapshot())

        for fragment in self.fragments:
            yield FragmentEvent(fragment=fragment)

        narrative = await generate_narrative(
            self.scene,
            attributes,
            latest.emotion,
            latest.cognitive,
            self.analyzer.confidence,
            self.analyzer.coherence,
            self._seed,
        )
        yield NarrativeEvent(narrative=narrative)

        self.tracker.update(
            "environment",
            status="complete",
            progress=1.0,
            message="Reconstruction ready to explore",
            confidence=self.analyzer.coherence,
        )
        yield PipelineEvent(stages=self.tracker.snapshot())

        yield ReconstructionEvent(
            status="stabilised",
            progress=1.0,
            confidence=round(self.analyzer.confidence, 4),
            coherence=round(self.analyzer.coherence, 4),
        )

        self._narrative = narrative
        yield SessionClosedEvent(
            session_id=self.session_id,
            reason="stopped" if self._stopped else "complete",
            message=None,
        )

    # -------------------------------------------------------------- persistence

    def to_record(self) -> Optional[SessionRecord]:
        """Snapshot this session for storage. ``None`` if nothing was built."""
        if self.scene is None or not self._frames:
            return None

        narrative = getattr(self, "_narrative", None)
        if narrative is None:
            from .narrative import generate_local

            narrative = generate_local(
                self.scene,
                self._aggregate_attributes(),
                self._frames[-1].emotion,
                self._frames[-1].cognitive,
                self.analyzer.confidence,
                self.analyzer.coherence,
                self._seed,
            )

        return SessionRecord(
            id=self.session_id,
            title=narrative.title,
            created_at=self.created_at,
            duration_seconds=round(self._frames[-1].t, 2),
            source=self.source,
            biome=self.scene.biome,
            primary_emotion=_dominant_emotion(self._emotion_trace),
            dominant_cognitive_state=_dominant_state(self._frames),
            confidence=round(self.analyzer.confidence, 4),
            coherence=round(self.analyzer.coherence, 4),
            fragment_count=len(self.fragments),
            bookmarked=False,
            version=1,
            scene=self.scene,
            fragments=self.fragments,
            narrative=narrative,
            confidence_trace=_downsample(self._confidence_trace, 160),
            emotion_trace=_downsample(self._emotion_trace, 160),
            history=[],
        )


def _dominant_emotion(trace: Sequence[EmotionSample]) -> str:
    """Most intensity-weighted emotion across the session."""
    if not trace:
        return "calm"
    totals: Dict[str, float] = {}
    for sample in trace:
        totals[sample.emotion] = totals.get(sample.emotion, 0.0) + sample.intensity
    return max(totals, key=lambda k: totals[k])


def _dominant_state(frames: Sequence[NeuralFrame]) -> str:
    if not frames:
        return "relaxed"
    totals: Dict[str, float] = {}
    for frame in frames:
        state = frame.cognitive.state
        totals[state] = totals.get(state, 0.0) + frame.cognitive.confidence
    return max(totals, key=lambda k: totals[k])


def _downsample(samples: List, limit: int) -> List:
    """Evenly thin a trace so stored sessions stay small."""
    if len(samples) <= limit:
        return samples
    step = len(samples) / float(limit)
    return [samples[int(i * step)] for i in range(limit)]
