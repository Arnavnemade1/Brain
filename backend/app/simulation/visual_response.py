"""Stimulus-locked EEG synthesis.

Generates scalp EEG *from* the reference video by modelling how the visual
system responds to it. This is what makes fidelity measurement meaningful: the
information the encoder later recovers genuinely travelled through the
waveform, and nothing is passed between the two out of band.

The response model is deliberately conservative. Each mapping below is a
coarse but real property of visual EEG, and — importantly — each is *lossy*.
Colour barely survives, fine spatial detail does not survive at all. The
resulting fidelity ceiling is the honest one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..models.memory import VisualFeatures
from ..processing import montage
from ..processing.bands import BAND_RANGES
from ..stimulus.features import luminance_of
from ..utils.linalg import safe_matmul


@dataclass
class RecordingConditions:
    """How good this notional recording session was."""

    id: str
    label: str
    description: str
    #: 0–1 blink/muscle/contact artifact load.
    artifact_level: float
    #: 0–1 electrode impedance noise.
    impedance: float
    #: Multiplier on the visual response amplitude — attention, arousal, fit.
    response_gain: float
    #: Clock drift between stimulus and recording hardware, parts per million.
    drift_ppm: float


CONDITIONS = {
    "clinical": RecordingConditions(
        id="clinical",
        label="Clinical-grade",
        description="Shielded room, gel electrodes, alert and attentive subject.",
        artifact_level=0.10,
        impedance=0.10,
        response_gain=1.25,
        drift_ppm=18.0,
    ),
    "research": RecordingConditions(
        id="research",
        label="Research lab",
        description="Standard lab conditions with typical blink and movement artifact.",
        artifact_level=0.28,
        impedance=0.24,
        response_gain=1.0,
        drift_ppm=64.0,
    ),
    "consumer": RecordingConditions(
        id="consumer",
        label="Consumer headset",
        description="Dry electrodes, sparse posterior coverage, ambient interference.",
        artifact_level=0.55,
        impedance=0.52,
        response_gain=0.68,
        drift_ppm=180.0,
    ),
    "degraded": RecordingConditions(
        id="degraded",
        label="Degraded",
        description="Poor contact and an inattentive subject. Near the recovery floor.",
        artifact_level=0.78,
        impedance=0.70,
        response_gain=0.42,
        drift_ppm=320.0,
    ),
}


@dataclass
class StimulusRecording:
    """Synthesised EEG plus the markers needed to align it."""

    data: np.ndarray  # (channels, samples), microvolts
    sample_rate: int
    channels: List[str]
    #: Sample indices of recorded event markers, on the *recording* clock.
    marker_samples: List[int]
    #: Stimulus times those markers correspond to, on the *stimulus* clock.
    marker_times: List[float]
    #: True offset and drift, for validating the synchronizer.
    true_offset: float
    true_drift_ppm: float
    condition: str

    @property
    def duration(self) -> float:
        return self.data.shape[1] / float(self.sample_rate)

    @property
    def total_samples(self) -> int:
        return int(self.data.shape[1])

    def window(self, start_sample: int, length: int) -> np.ndarray:
        if start_sample >= self.data.shape[1]:
            return np.zeros((self.data.shape[0], length))
        chunk = self.data[:, start_sample : start_sample + length]
        if chunk.shape[1] < length:
            pad = np.zeros((chunk.shape[0], length - chunk.shape[1]))
            chunk = np.concatenate([chunk, pad], axis=1)
        return chunk


class VisualEEGSimulator:
    """Synthesises EEG from a reference clip's per-frame visual features."""

    def __init__(self, sample_rate: int = 256, seed: Optional[int] = None) -> None:
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)
        self.channels = list(montage.CHANNELS)
        self.n_channels = len(self.channels)

    # ------------------------------------------------------------------ public

    def generate(
        self,
        frames: np.ndarray,
        features: List[VisualFeatures],
        fps: float,
        condition: str = "research",
    ) -> StimulusRecording:
        conditions = CONDITIONS.get(condition, CONDITIONS["research"])

        duration = len(features) / fps
        # The recording starts slightly before playback and runs slightly past.
        lead_in = 1.5
        total_samples = int((duration + lead_in * 2) * self.sample_rate)

        # Clock relationship the synchronizer will have to recover.
        true_offset = lead_in + float(self.rng.uniform(-0.35, 0.35))
        true_drift = conditions.drift_ppm * float(self.rng.uniform(0.6, 1.4))

        data = np.zeros((self.n_channels, total_samples), dtype=np.float64)

        # Continuous background: posterior alpha plus 1/f.
        self._add_background(data)

        # Stimulus-driven responses, written at drifted sample positions.
        self._add_visual_response(
            data, frames, features, fps, true_offset, true_drift, conditions
        )

        self._apply_volume_conduction(data)
        self._add_artifacts(data, conditions)

        # Scale to a plausible scalp range.
        peak = float(np.percentile(np.abs(data), 99.5)) or 1.0
        data = data / peak * 42.0

        marker_times, marker_samples = self._markers(
            features, fps, true_offset, true_drift
        )

        return StimulusRecording(
            data=data,
            sample_rate=self.sample_rate,
            channels=self.channels,
            marker_samples=marker_samples,
            marker_times=marker_times,
            true_offset=true_offset,
            true_drift_ppm=true_drift,
            condition=conditions.id,
        )

    # ---------------------------------------------------------------- internal

    def _stimulus_to_sample(self, t: float, offset: float, drift_ppm: float) -> int:
        """Map a stimulus time onto the recording clock."""
        recorded = offset + t * (1.0 + drift_ppm * 1e-6)
        return int(round(recorded * self.sample_rate))

    def _markers(
        self,
        features: List[VisualFeatures],
        fps: float,
        offset: float,
        drift_ppm: float,
    ) -> Tuple[List[float], List[int]]:
        """Event markers at playback start and each scene transition.

        Real acquisition writes a trigger to the EEG stream at these points.
        Jitter is added because trigger hardware is never perfect, which is
        what the synchronizer's residual error will reflect.
        """
        times: List[float] = []
        samples: List[int] = []

        for feature in features:
            if not feature.scene_cut:
                continue
            times.append(feature.t)
            jitter = float(self.rng.normal(0.0, 0.004)) * self.sample_rate
            samples.append(
                self._stimulus_to_sample(feature.t, offset, drift_ppm) + int(jitter)
            )

        return times, samples

    def _add_background(self, data: np.ndarray) -> None:
        """Ongoing alpha rhythm and 1/f background."""
        n = data.shape[1]
        t = np.arange(n) / self.sample_rate

        for channel in range(self.n_channels):
            # Posterior alpha is strongest occipitally and is the dominant
            # resting rhythm; it is also what visual input suppresses.
            posterior_weight = 0.25
            if channel in montage.OCCIPITAL:
                posterior_weight = 1.0
            elif channel in montage.OCCIPITO_PARIETAL:
                posterior_weight = 0.8
            elif channel in montage.PARIETAL:
                posterior_weight = 0.6

            alpha_freq = 10.0 + self.rng.normal(0, 0.4)
            phase = self.rng.uniform(0, 2 * np.pi)
            # Slow amplitude modulation, as real alpha waxes and wanes.
            envelope = 0.7 + 0.3 * np.sin(2 * np.pi * 0.11 * t + self.rng.uniform(0, 6))
            data[channel] += (
                posterior_weight * 1.6 * envelope * np.sin(2 * np.pi * alpha_freq * t + phase)
            )

            data[channel] += self._pink(n) * 0.9
            data[channel] += self.rng.normal(0, 0.12, n)

    def _pink(self, n: int) -> np.ndarray:
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
        phase = self.rng.uniform(0, 2 * np.pi, len(freqs))
        signal = np.fft.irfft(freqs**-1.0 * np.exp(1j * phase), n=n)
        sigma = float(np.std(signal)) or 1.0
        return signal / sigma

    def _band_noise(self, n: int, low: float, high: float) -> np.ndarray:
        """Band-limited noise, built in the frequency domain."""
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        spectrum = np.zeros(freqs.shape, dtype=complex)
        mask = (freqs >= low) & (freqs <= high)
        count = int(mask.sum())
        if count == 0:
            return np.zeros(n)
        magnitude = np.abs(self.rng.normal(1.0, 0.3, count))
        phase = self.rng.uniform(0, 2 * np.pi, count)
        spectrum[mask] = magnitude * np.exp(1j * phase)
        signal = np.fft.irfft(spectrum, n=n)
        sigma = float(np.std(signal)) or 1.0
        return signal / sigma

    def _add_visual_response(
        self,
        data: np.ndarray,
        frames: np.ndarray,
        features: List[VisualFeatures],
        fps: float,
        offset: float,
        drift_ppm: float,
        conditions: RecordingConditions,
    ) -> None:
        """Write stimulus-driven activity into the recording.

        Five separate response channels, each carrying a different visual
        property and each degrading differently:

        * **Luminance drive** — broadband occipital amplitude follows overall
          brightness. The single most recoverable property.
        * **Alpha suppression** — posterior alpha is attenuated in proportion
          to visual engagement (contrast and detail).
        * **Motion response** — beta/gamma over lateral occipito-temporal
          sites scales with motion energy.
        * **Evoked transient** — an abrupt scene change produces a large,
          time-locked deflection roughly 100–170 ms later.
        * **Field layout** — left/right and upper/lower luminance imbalance
          appears as contralateral amplitude asymmetry. This is the *only*
          spatial information that reaches the scalp, and it is coarse.
        """
        n = data.shape[1]
        gain = conditions.response_gain
        luma = luminance_of(frames)

        # Per-sample interpolated feature envelopes on the recording clock.
        stimulus_times = np.array([f.t for f in features])
        sample_times = (np.arange(n) / self.sample_rate - offset) / (
            1.0 + drift_ppm * 1e-6
        )

        def envelope(values: np.ndarray) -> np.ndarray:
            interpolated = np.interp(
                sample_times, stimulus_times, values, left=0.0, right=0.0
            )
            # Zero outside the stimulus window.
            inside = (sample_times >= stimulus_times[0]) & (
                sample_times <= stimulus_times[-1]
            )
            return interpolated * inside

        luminance = envelope(np.array([f.luminance for f in features]))
        contrast = envelope(np.array([f.contrast for f in features]))
        motion = envelope(np.array([f.motion for f in features]))
        detail = envelope(np.array([f.edge_density for f in features]))

        # Coarse spatial layout: left/right and top/bottom luminance balance.
        h, w = luma.shape[1], luma.shape[2]
        left = luma[:, :, : w // 2].mean(axis=(1, 2))
        right = luma[:, :, w // 2 :].mean(axis=(1, 2))
        top = luma[:, : h // 2, :].mean(axis=(1, 2))
        bottom = luma[:, h // 2 :, :].mean(axis=(1, 2))

        denominator = np.maximum(left + right, 1e-6)
        horizontal = envelope((right - left) / denominator)
        denominator = np.maximum(top + bottom, 1e-6)
        vertical = envelope((top - bottom) / denominator)

        low_gamma = BAND_RANGES["gamma"]
        beta = BAND_RANGES["beta"]

        for channel in range(self.n_channels):
            occipital = channel in montage.OCCIPITAL
            occipito_parietal = channel in montage.OCCIPITO_PARIETAL
            motion_area = channel in montage.MOTION_SENSITIVE
            parietal = channel in montage.PARIETAL

            visual_weight = (
                1.0 if occipital else 0.72 if occipito_parietal else 0.42 if parietal else 0.12
            )

            # --- Luminance drive ---------------------------------------------
            # Broadband response scaling with brightness, strongest occipitally.
            drive = self._band_noise(n, 4.0, 30.0)
            data[channel] += gain * visual_weight * 2.6 * drive * luminance

            # --- Alpha suppression -------------------------------------------
            # Visual engagement attenuates ongoing alpha. Implemented as a
            # negative-going multiplier on the alpha already in the signal.
            engagement = np.clip(contrast * 1.8 + detail * 0.9, 0.0, 1.0)
            alpha_component = self._band_noise(n, *BAND_RANGES["alpha"])
            data[channel] -= gain * visual_weight * 1.9 * alpha_component * engagement

            # --- Motion response ---------------------------------------------
            if motion_area or occipital:
                weight = 1.0 if motion_area else 0.45
                motion_signal = self._band_noise(n, beta[0], low_gamma[1])
                data[channel] += gain * weight * 2.2 * motion_signal * motion

            # --- Spatial layout ----------------------------------------------
            # Contralateral: right-field brightness drives left-hemisphere
            # response. Weak by construction — this is the honest limit on
            # spatial recovery from scalp EEG.
            if channel in montage.POSTERIOR_LEFT:
                data[channel] += gain * 1.1 * drive * np.clip(horizontal, 0, None)
            elif channel in montage.POSTERIOR_RIGHT:
                data[channel] += gain * 1.1 * drive * np.clip(-horizontal, 0, None)

            if channel in montage.POSTERIOR_DORSAL:
                data[channel] += gain * 0.85 * drive * np.clip(-vertical, 0, None)
            elif channel in montage.POSTERIOR_VENTRAL:
                data[channel] += gain * 0.85 * drive * np.clip(vertical, 0, None)

        # --- Evoked transients at scene changes ------------------------------
        for feature in features:
            if not feature.scene_cut:
                continue
            centre = self._stimulus_to_sample(feature.t, offset, drift_ppm)
            self._stamp_vep(data, centre, feature, gain)

    def _stamp_vep(
        self, data: np.ndarray, onset: int, feature: VisualFeatures, gain: float
    ) -> None:
        """Write a visual evoked potential following a scene transition.

        A P1/N1/P2 complex: positive around 100 ms, negative around 170 ms,
        positive around 250 ms, largest at the occipital pole. Amplitude
        scales with the magnitude of the visual change, which is what makes
        scene boundaries the most reliably recovered property in the system.
        """
        n = data.shape[1]
        length = int(0.42 * self.sample_rate)
        if onset < 0 or onset >= n:
            return

        t = np.arange(length) / self.sample_rate

        def component(latency: float, width: float, amplitude: float) -> np.ndarray:
            return amplitude * np.exp(-(((t - latency) / width) ** 2))

        waveform = (
            component(0.100, 0.022, 1.0)
            + component(0.170, 0.030, -1.9)
            + component(0.255, 0.045, 0.85)
        )

        # Scale with how large a visual change this was.
        magnitude = 0.5 + feature.luminance * 1.2 + feature.contrast * 1.5

        start = max(0, onset)
        end = min(n, onset + length)
        segment = waveform[: end - start]

        for channel in range(self.n_channels):
            if channel in montage.OCCIPITAL:
                weight = 1.0
            elif channel in montage.OCCIPITO_PARIETAL:
                weight = 0.75
            elif channel in montage.PARIETAL:
                weight = 0.45
            else:
                weight = 0.12

            jitter = 1.0 + float(self.rng.normal(0, 0.08))
            data[channel, start:end] += gain * weight * magnitude * 5.5 * segment * jitter

    def _apply_volume_conduction(self, data: np.ndarray) -> None:
        """Spatially smear sources across neighbouring electrodes."""
        positions = np.array(
            [montage.POSITIONS.get(name, (0.0, 0.0)) for name in self.channels],
            dtype=np.float64,
        )
        deltas = positions[:, None, :] - positions[None, :, :]
        distance = np.sqrt(np.sum(deltas**2, axis=-1))

        weights = np.exp(-((distance / 0.5) ** 2))
        weights /= weights.sum(axis=1, keepdims=True)

        k = 0.45
        mixing = (1.0 - k) * np.eye(self.n_channels) + k * weights
        data[:] = safe_matmul(mixing, data, context="volume conduction")

    def _add_artifacts(self, data: np.ndarray, conditions: RecordingConditions) -> None:
        """Blinks, muscle bursts, mains hum, impedance noise, contact loss."""
        n = data.shape[1]
        duration = n / self.sample_rate
        level = conditions.artifact_level

        # Blinks — frontal, large, slow.
        for _ in range(int(self.rng.poisson(duration * 0.30 * (0.7 + 2.0 * level)))):
            centre = int(self.rng.uniform(0, n))
            width = int(self.rng.uniform(0.12, 0.30) * self.sample_rate)
            shape = -np.exp(-((np.linspace(-1, 2.5, width)) ** 2) * 3.0)
            amplitude = self.rng.uniform(6.0, 13.0) * (0.8 + level)
            self._stamp(data, montage.FRONTAL, centre, shape * amplitude, decay=0.5)

        # Muscle — temporal, brief, broadband.
        for _ in range(int(self.rng.poisson(duration * 1.2 * level))):
            centre = int(self.rng.uniform(0, n))
            width = int(self.rng.uniform(0.08, 0.22) * self.sample_rate)
            burst = self.rng.normal(0, 1.0, width) * np.hanning(width)
            amplitude = self.rng.uniform(2.5, 6.0) * (0.5 + level)
            self._stamp(data, montage.CENTRAL, centre, burst * amplitude, decay=0.4)

        # Mains hum.
        t = np.arange(n) / self.sample_rate
        data += 0.20 * (0.4 + level) * np.sin(
            2 * np.pi * 60.0 * t + self.rng.uniform(0, 2 * np.pi)
        )

        # Per-electrode impedance noise.
        poor = self.rng.random(self.n_channels) < (0.08 + conditions.impedance * 0.5)
        for channel in np.flatnonzero(poor):
            severity = self.rng.uniform(0.3, 1.5) * (0.4 + conditions.impedance)
            data[channel] += self.rng.normal(0, severity, n)

        # Sustained contact loss.
        for _ in range(int(self.rng.poisson(duration * 0.05 * level * 4.0))):
            channel = int(self.rng.integers(0, self.n_channels))
            length = int(self.rng.uniform(1.0, 4.0) * self.sample_rate)
            start = int(self.rng.uniform(0, max(1, n - length)))
            span = min(length, n - start)
            if span <= 8:
                continue
            drift = np.cumsum(self.rng.normal(0, 1.0, span))
            drift -= np.linspace(drift[0], drift[-1], span)
            sigma = float(np.std(drift)) or 1.0
            scale = self.rng.uniform(5.0, 14.0) * (0.4 + level)
            data[channel, start : start + span] += (
                drift / sigma * scale * np.hanning(span)
            )

    def _stamp(
        self,
        data: np.ndarray,
        channels: List[int],
        centre: int,
        payload: np.ndarray,
        decay: float,
    ) -> None:
        n = data.shape[1]
        half = len(payload) // 2
        start = max(0, centre - half)
        end = min(n, start + len(payload))
        if end <= start:
            return
        segment = payload[: end - start]

        primary = set(channels)
        for channel in range(data.shape[0]):
            weight = 1.0 if channel in primary else decay * 0.3
            data[channel, start:end] += segment * weight


def simulate(
    frames: np.ndarray,
    features: List[VisualFeatures],
    fps: float,
    condition: str = "research",
    sample_rate: int = 256,
    seed: Optional[int] = None,
) -> StimulusRecording:
    """Convenience wrapper used by the session service."""
    return VisualEEGSimulator(sample_rate=sample_rate, seed=seed).generate(
        frames, features, fps, condition
    )
