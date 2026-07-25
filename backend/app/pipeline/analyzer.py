"""Window analyser — raw EEG in, one :class:`NeuralFrame` out.

Owns the rolling state that single-window analysis cannot see: confidence
accumulation, temporal coherence, and the band trajectories the context stage
reads to decide *when* the interpretation has stabilised.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from ..models.neural import NeuralFrame
from ..processing import montage, quality
from ..processing.bands import BAND_ORDER, integrate_bands, spectral_edge
from ..processing.features import (
    WindowFeatures,
    connectivity,
    frontal_alpha_asymmetry,
    hjorth,
    posterior_alpha_ratio,
    spectral_entropy,
    zero_crossing_rate,
)
from ..processing.filters import clean_window
from ..processing.spectral import (
    channel_activation,
    decimate_waveform,
    line_noise_level,
    to_display_spectrum,
    welch_psd,
)
from .inference import MemoryAttributes, infer_cognitive, infer_emotion, infer_memory_attributes


class AnalysisResult:
    """A frame plus the intermediate products later stages need."""

    __slots__ = ("frame", "features", "attributes", "stage_latencies")

    def __init__(
        self,
        frame: NeuralFrame,
        features: WindowFeatures,
        attributes: MemoryAttributes,
        stage_latencies: Dict[str, float],
    ) -> None:
        self.frame = frame
        self.features = features
        self.attributes = attributes
        self.stage_latencies = stage_latencies


class WindowAnalyzer:
    """Stateful analyser for one session's stream of windows."""

    #: Number of recent windows retained for temporal analysis.
    HISTORY = 24

    def __init__(
        self,
        session_id: str,
        sample_rate: int,
        channels: List[str],
        line_frequency: float = 60.0,
    ) -> None:
        self.session_id = session_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.line_frequency = line_frequency

        self._index = 0
        self._band_history: Deque[Dict[str, float]] = deque(maxlen=self.HISTORY)
        self._quality_history: Deque[float] = deque(maxlen=self.HISTORY)
        self._confidence = 0.0
        self._coherence = 0.0

        # Region indices resolved against whatever montage the recording uses,
        # so uploaded files with partial coverage still produce valid features.
        self._frontal_left = self._resolve(montage.FRONTAL_LEFT)
        self._frontal_right = self._resolve(montage.FRONTAL_RIGHT)
        self._posterior = self._resolve(montage.POSTERIOR)
        self._anterior = self._resolve(montage.FRONTAL)

    def _resolve(self, canonical_indices: List[int]) -> List[int]:
        """Map canonical montage indices onto this recording's channel list."""
        names = {montage.CHANNELS[i] for i in canonical_indices if i < len(montage.CHANNELS)}
        resolved = [i for i, name in enumerate(self.channels) if name in names]
        if resolved:
            return resolved
        # No canonical coverage — fall back to a positional approximation so
        # the feature is still defined rather than silently zero.
        span = max(1, len(self.channels) // 3)
        return list(range(min(span, len(self.channels))))

    # ------------------------------------------------------------------ public

    def analyze(self, window: np.ndarray, t: float) -> AnalysisResult:
        """Run the full per-window chain and return a frame."""
        latencies: Dict[str, float] = {}

        with _timed(latencies, "cleaning"):
            cleaned, dropped, artifact_ratio = clean_window(
                window, self.sample_rate, self.line_frequency
            )

        with _timed(latencies, "frequency"):
            freqs, psd = welch_psd(cleaned, self.sample_rate)
            bands = integrate_bands(freqs, psd)
            spectrum = to_display_spectrum(freqs, psd)

        with _timed(latencies, "features"):
            features = self._extract_features(cleaned, freqs, psd)

        with _timed(latencies, "denoise"):
            noise = line_noise_level(freqs, psd, self.line_frequency)
            signal_quality = quality.assess(
                cleaned,
                self.channels,
                dropped,
                artifact_ratio,
                noise,
                features.connectivity,
                freqs,
                psd,
            )

        with _timed(latencies, "temporal"):
            coherence = self._update_coherence(bands, signal_quality.score)

        with _timed(latencies, "emotion"):
            emotion = infer_emotion(bands, features)

        cognitive = infer_cognitive(bands, features)

        with _timed(latencies, "memory_inference"):
            attributes = infer_memory_attributes(bands, features, cognitive, emotion)

        neural_confidence = self._update_confidence(signal_quality.score, cognitive.confidence)

        activation = channel_activation(freqs, psd)
        topography = {
            name: round(float(activation[i]), 4)
            for i, name in enumerate(self.channels)
            if i < len(activation)
        }

        frame = NeuralFrame(
            session_id=self.session_id,
            index=self._index,
            t=round(t, 3),
            band_powers={band: round(bands.get(band, 0.0), 4) for band in BAND_ORDER},
            spectrum=spectrum,
            topography=topography,
            waveform=decimate_waveform(cleaned),
            signal_quality=signal_quality,
            cognitive=cognitive,
            emotion=emotion,
            neural_confidence=round(neural_confidence, 4),
            coherence=round(coherence, 4),
        )

        self._index += 1
        return AnalysisResult(frame, features, attributes, latencies)

    @property
    def band_history(self) -> List[Dict[str, float]]:
        return list(self._band_history)

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def coherence(self) -> float:
        return self._coherence

    # ---------------------------------------------------------------- internal

    def _extract_features(
        self, cleaned: np.ndarray, freqs: np.ndarray, psd: np.ndarray
    ) -> WindowFeatures:
        activity, mobility, complexity = hjorth(cleaned)
        return WindowFeatures(
            activity=float(activity),
            mobility=float(mobility),
            complexity=float(complexity),
            spectral_entropy=spectral_entropy(psd),
            connectivity=connectivity(cleaned),
            frontal_asymmetry=frontal_alpha_asymmetry(
                freqs, psd, self._frontal_left, self._frontal_right
            ),
            posterior_ratio=posterior_alpha_ratio(
                freqs, psd, self._posterior, self._anterior
            ),
            zero_crossing=zero_crossing_rate(cleaned, self.sample_rate),
            spectral_edge=spectral_edge(freqs, psd),
        )

    def _update_coherence(self, bands: Dict[str, float], quality_score: float) -> float:
        """Temporal stability of the band profile across recent windows.

        A memory trace that keeps re-presenting the same spectral signature is
        far better evidence than one strong window, so coherence — not raw
        power — is what gates region resolution in the reconstruction.
        """
        self._band_history.append(dict(bands))
        self._quality_history.append(quality_score)

        if len(self._band_history) < 3:
            self._coherence = 0.15 * quality_score
            return self._coherence

        matrix = np.array(
            [[h.get(band, 0.0) for band in BAND_ORDER] for h in self._band_history]
        )

        # Mean absolute change between consecutive windows, inverted: small
        # drift means a stable interpretation.
        drift = float(np.mean(np.abs(np.diff(matrix, axis=0))))
        stability = float(np.exp(-drift * 22.0))

        # Weight by how much of the recent history was usable signal.
        usable = float(np.mean(self._quality_history))

        target = float(np.clip(stability * usable, 0.0, 1.0))
        # Smooth so a single noisy window cannot collapse the reconstruction.
        self._coherence += (target - self._coherence) * 0.28
        return float(np.clip(self._coherence, 0.0, 1.0))

    def _update_confidence(self, quality_score: float, state_confidence: float) -> float:
        """Accumulating confidence in the reconstruction.

        Rises with usable, decisively-classified windows and decays slowly
        when evidence degrades, so the environment stabilises over a session
        instead of flickering window to window.
        """
        evidence = quality_score * (0.45 + 0.55 * state_confidence)

        # Asymmetric: gaining confidence is slower than losing it, which keeps
        # the reconstruction honest about a recording that turns bad.
        rate = 0.075 if evidence > self._confidence else 0.16
        self._confidence += (evidence - self._confidence) * rate
        return float(np.clip(self._confidence, 0.0, 1.0))


class _timed:
    """Context manager recording elapsed milliseconds into ``sink[key]``."""

    __slots__ = ("sink", "key", "start")

    def __init__(self, sink: Dict[str, float], key: str) -> None:
        self.sink = sink
        self.key = key
        self.start = 0.0

    def __enter__(self) -> "_timed":
        import time

        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        import time

        self.sink[self.key] = (time.perf_counter() - self.start) * 1000.0
