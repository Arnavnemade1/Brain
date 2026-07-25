"""Memory Reconstruction Engine — session orchestration.

Drives one reconstruction from reference clip to scored replay, emitting the
event stream the console renders. The transport belongs to the WebSocket
layer; this module only yields events.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import AsyncIterator, Dict, List, Optional, Union

import numpy as np

from ..config import get_settings
from ..encoder.pipeline import AnalysisPipeline
from ..evaluation.metrics import evaluate
from ..models.memory import (
    FrameFidelity,
    ReconstructedFrame,
    SceneEstimate,
    SessionRecord,
    StimulusClip,
    VisualFeatures,
)
from ..models.neural import LatentPoint
from ..models.stream import (
    FrameEvent,
    LatentEvent,
    MetricsEvent,
    PipelineEvent,
    ProgressEvent,
    ReconstructionFrameEvent,
    ReportEvent,
    SessionClosedEvent,
    SessionOpenedEvent,
    StreamErrorEvent,
    SyncEvent,
)
from ..pipeline.stages import PipelineTracker
from ..processing.sync import synchronise
from ..simulation.visual_response import StimulusRecording
from ..stimulus.features import extract
from ..utils.ids import seed_from
from .decoders import get_decoders
from .frames import encode_frame, reconstruct_frame
from .refine import TemporalRefiner
from .report import build_report
from .scene import decode_scenes

log = logging.getLogger("mindscape.reconstruction")

StreamEvent = Union[
    SessionOpenedEvent,
    SyncEvent,
    FrameEvent,
    PipelineEvent,
    LatentEvent,
    ReconstructionFrameEvent,
    ProgressEvent,
    MetricsEvent,
    ReportEvent,
    SessionClosedEvent,
    StreamErrorEvent,
]


class ReconstructionRunner:
    """Runs one memory reconstruction session."""

    def __init__(
        self,
        session_id: str,
        clip: StimulusClip,
        frames: np.ndarray,
        features: List[VisualFeatures],
        recording: StimulusRecording,
        speed: float = 1.0,
    ) -> None:
        settings = get_settings()

        self.session_id = session_id
        self.clip = clip
        self.reference_frames = frames
        self.truth = features
        self.recording = recording
        self.speed = float(np.clip(speed, 0.25, 16.0))

        self.sample_rate = recording.sample_rate
        self.window_samples = int(self.sample_rate * settings.window_seconds)
        self.hop_samples = max(
            1, int(self.window_samples * (1.0 - settings.window_overlap))
        )

        self.tracker = PipelineTracker()
        self.decoders = get_decoders()
        self.refiner = TemporalRefiner()

        self._paused = False
        self._stopped = False
        self._position = 0

        self.latents: List[LatentPoint] = []
        self.scenes: List[SceneEstimate] = []
        self.reconstructed: List[np.ndarray] = []
        self.matched_reference: List[np.ndarray] = []
        self.matched_truth: List[VisualFeatures] = []
        self.frame_fidelity: List[FrameFidelity] = []

        self.metrics = None
        self.report = None
        self.sync = None

        self.created_at = datetime.now(timezone.utc).isoformat()
        self._seed = seed_from(session_id, clip.id)

    # ------------------------------------------------------------------ control

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        self._stopped = True

    def set_speed(self, speed: float) -> None:
        self.speed = float(np.clip(speed, 0.25, 16.0))

    def seek(self, seconds: float) -> None:
        target = int(np.clip(seconds, 0.0, self.clip.duration_seconds) * self.sample_rate)
        self._position = max(0, min(target, self.recording.total_samples - 1))

    @property
    def progress(self) -> float:
        total = max(1, self.recording.total_samples)
        return float(np.clip(self._position / total, 0.0, 1.0))

    # ------------------------------------------------------------------- stream

    async def run(self) -> AsyncIterator[StreamEvent]:
        yield SessionOpenedEvent(
            session_id=self.session_id,
            sample_rate=self.sample_rate,
            channels=list(self.recording.channels),
            clip=self.clip,
        )

        self.tracker.update(
            "stimulus",
            status="complete",
            progress=1.0,
            message=f"{self.clip.title} · {len(self.truth)} frames @ {self.clip.fps:g} fps",
            confidence=1.0,
        )
        self.tracker.update(
            "acquisition",
            status="active",
            message=f"{len(self.recording.channels)} ch @ {self.sample_rate} Hz",
            confidence=1.0,
        )
        yield PipelineEvent(stages=self.tracker.snapshot())

        try:
            # --- Synchronization ---------------------------------------------
            started = time.perf_counter()
            sync, clock = synchronise(
                self.recording.marker_samples,
                self.recording.marker_times,
                self.sample_rate,
            )
            self.sync = sync

            self.tracker.update(
                "synchronization",
                status="complete" if sync.usable else "degraded",
                progress=1.0,
                latency_ms=(time.perf_counter() - started) * 1000,
                message=(
                    f"{sync.markers_matched}/{sync.markers_expected} markers · "
                    f"residual {sync.residual_ms:.1f} ms"
                ),
                confidence=1.0 if sync.usable else 0.4,
            )
            yield SyncEvent(
                offset_seconds=sync.offset_seconds,
                drift_ppm=sync.drift_ppm,
                residual_ms=sync.residual_ms,
                markers_matched=sync.markers_matched,
                markers_expected=sync.markers_expected,
            )
            yield PipelineEvent(stages=self.tracker.snapshot())

            if not sync.usable:
                yield StreamErrorEvent(
                    stage="synchronization",
                    message=(
                        "Alignment is poor; responses may be attributed to the wrong "
                        "frames and fidelity will understate what the signal contains."
                    ),
                    recoverable=True,
                )

            # --- Baseline calibration ----------------------------------------
            pipeline = AnalysisPipeline(
                self.session_id, self.sample_rate, self.recording.channels, clock
            )

            stimulus_start = max(0, clock.to_sample(0.0))
            baseline_windows = [
                self.recording.window(start, self.window_samples)
                for start in range(
                    0, max(0, stimulus_start - self.window_samples), self.hop_samples
                )
            ]
            pipeline.calibrate(baseline_windows)

            self.tracker.update(
                "artifact_removal",
                status="active",
                message=f"Baseline from {len(baseline_windows)} pre-stimulus windows",
                confidence=1.0,
            )
            yield PipelineEvent(stages=self.tracker.snapshot())

            # --- Windowed reconstruction --------------------------------------
            self._position = stimulus_start
            duration = self.clip.duration_seconds
            frame_index = 0

            async for event in self._reconstruct(pipeline, duration, frame_index):
                yield event

            # --- Scoring -------------------------------------------------------
            async for event in self._finalise():
                yield event

        except asyncio.CancelledError:
            yield SessionClosedEvent(
                session_id=self.session_id, reason="stopped", message="Session cancelled"
            )
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("Reconstruction %s failed", self.session_id)
            self.tracker.mark_all("error", "Reconstruction failed")
            yield PipelineEvent(stages=self.tracker.snapshot())
            yield SessionClosedEvent(
                session_id=self.session_id, reason="error", message=str(exc)
            )

    async def _reconstruct(
        self, pipeline: AnalysisPipeline, duration: float, frame_index: int
    ) -> AsyncIterator[StreamEvent]:
        """Main window loop: encode, decode, score, emit."""
        while not self._stopped:
            if self._paused:
                await asyncio.sleep(0.08)
                continue

            if self._position + self.window_samples > self.recording.total_samples:
                break

            loop_started = time.perf_counter()
            window = self.recording.window(self._position, self.window_samples)

            try:
                result = pipeline.process(window, self._position)
            except Exception as exc:  # noqa: BLE001
                log.exception("Window analysis failed at sample %d", self._position)
                yield StreamErrorEvent(
                    stage="encoder", message=str(exc), recoverable=True
                )
                self._position += self.hop_samples
                continue

            t = result.latent.t

            # Windows outside the stimulus interval carry no reconstruction.
            if t < 0 or t > duration:
                self._position += self.hop_samples
                continue

            self.latents.append(result.latent)
            self._update_tracker(result, len(self.latents))

            yield FrameEvent(frame=result.frame)
            yield LatentEvent(point=result.latent)

            # Decode the trajectory so far. Scene boundaries depend on context,
            # so the most recent estimate can change as evidence arrives; only
            # the newest is emitted.
            estimates = decode_scenes(self.latents, self.decoders)
            scene = estimates[-1]
            self.scenes = estimates

            raw = reconstruct_frame(result.latent, scene, self.decoders)
            refined = self.refiner.refine(raw, scene)

            reference = self._reference_at(t)

            from ..evaluation.metrics import layout_correlation, psnr, ssim

            fidelity = FrameFidelity(
                index=frame_index,
                t=round(t, 4),
                ssim=round(ssim(refined, reference), 5),
                psnr=round(psnr(refined, reference), 3),
                layout_correlation=round(layout_correlation(refined, reference), 5),
            )

            self.reconstructed.append(refined)
            self.matched_reference.append(reference)
            self.matched_truth.append(self._truth_at(t))
            self.frame_fidelity.append(fidelity)

            yield ReconstructionFrameEvent(
                frame=ReconstructedFrame(
                    index=frame_index,
                    t=round(t, 4),
                    pixels=encode_frame(refined),
                    confidence=scene.confidence,
                ),
                reference=encode_frame(reference),
                scene=scene,
                truth=self.matched_truth[-1],
                fidelity=fidelity,
            )

            running = float(np.mean([f.ssim for f in self.frame_fidelity]))
            yield ProgressEvent(
                status="reconstructing",
                progress=round(self.progress, 4),
                fidelity=round(max(0.0, running), 4),
            )

            yield PipelineEvent(stages=self.tracker.snapshot())

            frame_index += 1
            self._position += self.hop_samples

            # Pace to wall clock so the console reads as a live stream.
            elapsed = time.perf_counter() - loop_started
            target = (self.hop_samples / float(self.sample_rate)) / self.speed
            if target > elapsed:
                await asyncio.sleep(target - elapsed)

    async def _finalise(self) -> AsyncIterator[StreamEvent]:
        if not self.reconstructed:
            yield SessionClosedEvent(
                session_id=self.session_id,
                reason="error",
                message="No windows fell inside the stimulus interval.",
            )
            return

        self.tracker.update(
            "refinement",
            status="complete",
            progress=1.0,
            message=f"{len(self.reconstructed)} frames stabilised",
            confidence=0.8,
        )

        metrics, per_frame = evaluate(
            self.reconstructed,
            self.matched_reference,
            self.scenes,
            self.matched_truth,
            all_truth=self.truth,
        )
        self.metrics = metrics
        self.frame_fidelity = per_frame

        yield MetricsEvent(metrics=metrics)

        self.report = build_report(
            self.clip, metrics, self.scenes, self.matched_truth, self.sync, self.decoders
        )
        yield ReportEvent(report=self.report)

        self.tracker.update(
            "replay",
            status="complete",
            progress=1.0,
            message=f"Composite fidelity {metrics.composite * 100:.0f}%",
            confidence=metrics.composite,
        )
        yield PipelineEvent(stages=self.tracker.snapshot())

        yield ProgressEvent(
            status="complete", progress=1.0, fidelity=round(metrics.composite, 4)
        )
        yield SessionClosedEvent(
            session_id=self.session_id,
            reason="stopped" if self._stopped else "complete",
            message=None,
        )

    # ------------------------------------------------------------------ helpers

    def _reference_at(self, t: float) -> np.ndarray:
        index = int(round(t * self.clip.fps))
        index = max(0, min(index, self.reference_frames.shape[0] - 1))
        return self.reference_frames[index].astype(np.float64)

    def _truth_at(self, t: float) -> VisualFeatures:
        index = int(round(t * self.clip.fps))
        index = max(0, min(index, len(self.truth) - 1))
        return self.truth[index]

    def _update_tracker(self, result, count: int) -> None:
        latencies = result.stage_latencies
        quality = result.frame.signal_quality
        visual = result.frame.visual

        self.tracker.update(
            "acquisition",
            status="active",
            progress=self.progress,
            throughput_delta=self.hop_samples,
            confidence=quality.score,
        )
        self.tracker.update(
            "artifact_removal",
            status="degraded" if quality.artifact_ratio > 0.02 else "active",
            progress=1.0,
            latency_ms=latencies.get("artifact_removal", 0.0),
            throughput_delta=1,
            message=(
                f"{quality.grade} · {quality.artifact_ratio * 100:.1f}% attenuated"
                + (f" · {len(quality.dropped_channels)} dropped" if quality.dropped_channels else "")
            ),
            confidence=quality.score,
        )
        self.tracker.update(
            "encoder",
            status="active",
            progress=1.0,
            latency_ms=latencies.get("encoder", 0.0),
            throughput_delta=7,
            message=(
                f"drive {visual.occipital_drive:.2f} · motion {visual.motion_response:.2f} "
                f"· transient {visual.transient:.2f}"
            ),
            confidence=result.frame.confidence,
        )
        self.tracker.update(
            "latent",
            status="active",
            progress=1.0,
            latency_ms=latencies.get("latent", 0.0),
            throughput_delta=1,
            message=f"{count} points in trajectory",
            confidence=result.latent.confidence,
        )

        scene_count = (self.scenes[-1].scene_index + 1) if self.scenes else 0
        self.tracker.update(
            "scene",
            status="active",
            progress=1.0,
            throughput_delta=1,
            message=f"{scene_count} scene(s) detected",
            confidence=result.latent.confidence,
        )
        self.tracker.update(
            "frames",
            status="active",
            progress=1.0,
            throughput_delta=1,
            message=f"{self.clip.grid_width}×{self.clip.grid_height} grid",
            confidence=result.latent.confidence,
        )
        self.tracker.update(
            "refinement",
            status="active",
            progress=self.progress,
            message="Enforcing temporal continuity",
            confidence=0.7,
        )

    # -------------------------------------------------------------- persistence

    def to_record(self) -> Optional[SessionRecord]:
        if self.metrics is None or self.report is None:
            return None

        from ..models.memory import LatentPointRef

        step = max(1, len(self.latents) // 160)

        return SessionRecord(
            id=self.session_id,
            title=self.report.title,
            created_at=self.created_at,
            duration_seconds=round(self.clip.duration_seconds, 2),
            stimulus_id=self.clip.id,
            stimulus_title=self.clip.title,
            source=self.clip.source,
            fidelity=round(self.metrics.composite, 5),
            ssim=round(self.metrics.ssim, 5),
            frame_count=len(self.reconstructed),
            bookmarked=False,
            version=1,
            clip=self.clip,
            metrics=self.metrics,
            frame_fidelity=self.frame_fidelity[::step],
            scenes=self.scenes[::step],
            report=self.report,
            latents=[
                LatentPointRef(t=p.t, vector=p.vector, confidence=p.confidence)
                for p in self.latents[::step]
            ],
            history=[],
        )
