"""Windowed analysis: cleaned EEG in, latent trajectory out.

Shared by the live session runner and the offline training harness, so the
decoders are fitted on exactly the representation they will later see. A
training/inference mismatch here would silently poison every fidelity number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..models.neural import (
    LatentPoint,
    NeuralFrame,
    SignalQuality,
    Spectrum,
    VisualResponse,
)
from ..processing import quality as quality_module
from ..processing.bands import BAND_ORDER, integrate_bands
from ..processing.filters import clean_window
from ..processing.features import connectivity
from ..processing.spectral import (
    channel_activation,
    decimate_waveform,
    line_noise_level,
    to_display_spectrum,
    welch_psd,
)
from ..processing.sync import StimulusClock
from .encoder import TemporalEncoder
from .latent import LatentProjector


@dataclass
class WindowResult:
    frame: NeuralFrame
    latent: LatentPoint
    response: VisualResponse
    stage_latencies: Dict[str, float]


class AnalysisPipeline:
    """Runs one session's windows through cleaning, encoding and projection."""

    def __init__(
        self,
        session_id: str,
        sample_rate: int,
        channels: List[str],
        clock: StimulusClock,
        line_frequency: float = 60.0,
    ) -> None:
        self.session_id = session_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.clock = clock
        self.line_frequency = line_frequency

        self.encoder = TemporalEncoder(sample_rate, channels)
        self.projector = LatentProjector()
        self._index = 0

    def calibrate(self, baseline_windows: List[np.ndarray]) -> None:
        """Measure pre-stimulus resting levels.

        Each window is cleaned first, so the baseline is measured through the
        same processing the stimulus windows will pass through.
        """
        cleaned = [
            clean_window(window, self.sample_rate, self.line_frequency)[0]
            for window in baseline_windows
        ]
        self.encoder.fit_baseline(cleaned)

    def process(self, window: np.ndarray, sample_index: int) -> WindowResult:
        """Analyse one window, timestamped on the stimulus clock."""
        import time

        latencies: Dict[str, float] = {}

        started = time.perf_counter()
        cleaned, dropped, artifact_ratio = clean_window(
            window, self.sample_rate, self.line_frequency
        )
        latencies["artifact_removal"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        freqs, psd = welch_psd(cleaned, self.sample_rate)
        bands = integrate_bands(freqs, psd)
        spectrum = to_display_spectrum(freqs, psd)

        coupling = connectivity(cleaned)
        signal_quality = quality_module.assess(
            cleaned,
            self.channels,
            dropped,
            artifact_ratio,
            line_noise_level(freqs, psd, self.line_frequency),
            coupling,
            freqs,
            psd,
        )
        latencies["acquisition"] = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        response = self.encoder.encode(cleaned)
        latencies["encoder"] = (time.perf_counter() - started) * 1000

        # Timestamp at the *centre* of the window on the stimulus clock. Using
        # the leading edge would bias every estimate half a window early
        # against the frame that actually produced the response.
        centre_sample = sample_index + window.shape[1] // 2
        t = self.clock.to_stimulus(centre_sample)

        # Confidence gates everything downstream: a window that is mostly
        # artifact should not be allowed to assert a scene.
        confidence = float(
            np.clip(signal_quality.score * (0.55 + 0.45 * (1.0 - artifact_ratio * 8)), 0, 1)
        )

        started = time.perf_counter()
        latent = self.projector.project(response, confidence, t)
        latencies["latent"] = (time.perf_counter() - started) * 1000

        activation = channel_activation(freqs, psd)
        topography = {
            name: round(float(activation[i]), 4)
            for i, name in enumerate(self.channels)
            if i < len(activation)
        }

        frame = NeuralFrame(
            session_id=self.session_id,
            index=self._index,
            t=round(t, 4),
            band_powers={band: round(bands.get(band, 0.0), 4) for band in BAND_ORDER},
            spectrum=spectrum,
            topography=topography,
            waveform=decimate_waveform(cleaned),
            signal_quality=signal_quality,
            visual=response,
            confidence=round(confidence, 4),
        )

        self._index += 1
        return WindowResult(
            frame=frame, latent=latent, response=response, stage_latencies=latencies
        )
