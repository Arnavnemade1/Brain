"""Synthetic EEG generator.

Produces multi-channel signal whose spectral composition follows a
:class:`~app.simulation.profiles.MemoryProfile`. The output is deliberately
imperfect — 1/f background, blink and muscle artifacts, mains hum, occasional
electrode dropout — so the cleaning and quality stages have real work to do.

Nothing about the profile is exposed downstream: the pipeline recovers state
from the waveform alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..processing import montage
from ..processing.bands import BAND_ORDER, BAND_RANGES
from ..utils.linalg import safe_matmul
from .profiles import MemoryProfile, get_profile

#: Per-region gain for each band, reflecting where rhythms actually dominate.
#: Rows are bands, columns are montage regions.
_REGION_GAIN = {
    "delta": {"frontal": 1.25, "central": 1.00, "temporal": 0.85, "parietal": 0.80, "occipital": 0.70},
    "theta": {"frontal": 1.35, "central": 1.05, "temporal": 1.10, "parietal": 0.85, "occipital": 0.75},
    "alpha": {"frontal": 0.55, "central": 0.80, "temporal": 0.90, "parietal": 1.35, "occipital": 1.60},
    "beta":  {"frontal": 1.20, "central": 1.25, "temporal": 1.00, "parietal": 0.90, "occipital": 0.70},
    "gamma": {"frontal": 1.10, "central": 1.15, "temporal": 1.05, "parietal": 0.95, "occipital": 0.85},
}


def _region_of(channel_index: int) -> str:
    for name, members in montage.REGIONS.items():
        if channel_index in members:
            return name
    return "central"


@dataclass
class SimulatedRecording:
    """A generated session, sliceable by the streaming runner."""

    data: np.ndarray  # (channels, samples)
    sample_rate: int
    channels: List[str]
    profile_id: str
    duration: float

    def window(self, start_sample: int, length: int) -> np.ndarray:
        end = start_sample + length
        if start_sample >= self.data.shape[1]:
            return np.zeros((self.data.shape[0], length))
        chunk = self.data[:, start_sample:end]
        if chunk.shape[1] < length:  # pad the final partial window
            pad = np.zeros((chunk.shape[0], length - chunk.shape[1]))
            chunk = np.concatenate([chunk, pad], axis=1)
        return chunk

    @property
    def total_samples(self) -> int:
        return int(self.data.shape[1])


class EEGGenerator:
    """Synthesises EEG matching a memory profile's spectral trajectory."""

    def __init__(self, sample_rate: int = 256, seed: Optional[int] = None) -> None:
        self.sample_rate = sample_rate
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- public

    def generate(
        self,
        profile_id: str,
        duration: float,
        channels: Optional[List[str]] = None,
    ) -> SimulatedRecording:
        profile = get_profile(profile_id)
        chans = channels or montage.CHANNELS
        n_channels = len(chans)
        n_samples = int(duration * self.sample_rate)

        envelopes = self._band_envelopes(profile, n_samples)
        data = np.zeros((n_channels, n_samples))

        for ch in range(n_channels):
            region = _region_of(ch)
            data[ch] = self._channel_signal(envelopes, region, n_samples)

        data = self._apply_coupling(data, chans, profile.global_coupling)
        data = self._apply_asymmetry(data, profile, n_samples)
        data = self._add_background(data)
        data = self._add_artifacts(data, profile, n_samples)

        # Scale to a plausible ±80 µV scalp range.
        peak = float(np.max(np.abs(data))) or 1.0
        data = data / peak * 68.0

        return SimulatedRecording(
            data=data,
            sample_rate=self.sample_rate,
            channels=list(chans),
            profile_id=profile.id,
            duration=duration,
        )

    # --------------------------------------------------------------- internal

    def _band_envelopes(self, profile: MemoryProfile, n_samples: int) -> dict:
        """Smooth per-band amplitude envelopes following the phase script."""
        phases = profile.normalised_phases()
        boundaries = np.cumsum([p.weight for p in phases]) * n_samples

        envelopes = {band: np.zeros(n_samples) for band in BAND_ORDER}
        t = np.arange(n_samples)

        start = 0
        for phase, end_f in zip(phases, boundaries):
            end = int(end_f)
            if end <= start:
                continue
            total = sum(phase.bands.values()) or 1.0
            for band in BAND_ORDER:
                envelopes[band][start:end] = phase.bands.get(band, 0.0) / total
            start = end

        if start < n_samples:  # fill any rounding remainder
            last = phases[-1]
            total = sum(last.bands.values()) or 1.0
            for band in BAND_ORDER:
                envelopes[band][start:] = last.bands.get(band, 0.0) / total

        # Smooth phase transitions over ~4 s so the temporal stage sees drift
        # rather than step changes, then add slow spontaneous fluctuation.
        kernel_len = max(3, int(self.sample_rate * 4.0))
        kernel = np.hanning(kernel_len)
        kernel /= kernel.sum()

        for band in BAND_ORDER:
            smoothed = np.convolve(envelopes[band], kernel, mode="same")
            wobble_hz = 0.05 + 0.04 * BAND_ORDER.index(band)
            wobble = 1.0 + 0.16 * np.sin(
                2 * np.pi * wobble_hz * t / self.sample_rate
                + self.rng.uniform(0, 2 * np.pi)
            )
            envelopes[band] = np.maximum(smoothed * wobble, 0.0)

        return envelopes

    def _channel_signal(self, envelopes: dict, region: str, n_samples: int) -> np.ndarray:
        """Sum of band-limited noise, weighted by envelope and region gain."""
        out = np.zeros(n_samples)
        for band in BAND_ORDER:
            low, high = BAND_RANGES[band]
            gain = _REGION_GAIN[band].get(region, 1.0)
            component = self._band_limited_noise(low, high, n_samples)
            # Amplitude scales as sqrt of power, hence the square root here.
            out += component * np.sqrt(envelopes[band]) * gain
        return out

    def _band_limited_noise(self, low: float, high: float, n: int) -> np.ndarray:
        """Noise confined to a band, built in the frequency domain.

        Shaping the spectrum directly (rather than filtering white noise) gives
        exact band edges and avoids filter roll-off bleeding between bands.
        """
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        spectrum = np.zeros(freqs.shape, dtype=complex)

        mask = (freqs >= low) & (freqs <= high)
        count = int(mask.sum())
        if count == 0:
            return np.zeros(n)

        magnitude = self.rng.normal(1.0, 0.28, count)
        # Within-band 1/f tilt keeps low edges slightly hotter, as in real EEG.
        tilt = (freqs[mask] / max(low, 0.5)) ** -0.35
        phase = self.rng.uniform(0, 2 * np.pi, count)
        spectrum[mask] = np.abs(magnitude) * tilt * np.exp(1j * phase)

        signal = np.fft.irfft(spectrum, n=n)
        sigma = float(np.std(signal))
        return signal / sigma if sigma > 1e-12 else signal

    def _apply_coupling(
        self, data: np.ndarray, channels: List[str], coupling: float
    ) -> np.ndarray:
        """Volume-conduct each source across neighbouring electrodes.

        Mixing in a single *global* mean would be undone exactly by the common
        average reference in the cleaning stage, leaving the recovered
        connectivity identical for every profile. Real volume conduction is
        spatially local — an electrode sees its neighbours far more than the
        far side of the scalp — so the mixing kernel is distance-weighted.
        That structure survives re-referencing and is what the connectivity
        feature actually measures.
        """
        k = float(np.clip(coupling, 0.0, 0.95))
        if k <= 0.01 or data.shape[0] < 2:
            return data

        positions = np.array(
            [montage.POSITIONS.get(name, (0.0, 0.0)) for name in channels], dtype=np.float64
        )
        deltas = positions[:, None, :] - positions[None, :, :]
        distance = np.sqrt(np.sum(deltas**2, axis=-1))

        # Gaussian falloff; ~0.55 units is roughly one inter-electrode step.
        weights = np.exp(-((distance / 0.55) ** 2))
        weights /= weights.sum(axis=1, keepdims=True)

        mixing = (1.0 - k) * np.eye(data.shape[0]) + k * weights
        return safe_matmul(mixing, data, context="volume conduction")

    def _apply_asymmetry(
        self, data: np.ndarray, profile: MemoryProfile, n_samples: int
    ) -> np.ndarray:
        """Impose frontal alpha asymmetry tracking the profile's valence.

        This is what the emotion stage later recovers as valence — the link is
        genuine, mediated entirely through the signal.
        """
        phases = profile.normalised_phases()
        valence = np.zeros(n_samples)
        boundaries = np.cumsum([p.weight for p in phases]) * n_samples

        start = 0
        for phase, end_f in zip(phases, boundaries):
            end = int(end_f)
            if end > start:
                valence[start:end] = phase.valence
            start = end
        if start < n_samples:
            valence[start:] = phases[-1].valence

        kernel = np.hanning(max(3, int(self.sample_rate * 3.0)))
        kernel /= kernel.sum()
        valence = np.convolve(valence, kernel, mode="same")

        # Positive valence → relatively less left-frontal alpha (more left
        # activation), so attenuate left and boost right. The modulation depth
        # has to clear the blink contamination that lands on these same
        # frontal leads, or the effect is unrecoverable in practice.
        left_gain = 1.0 - 0.38 * valence
        right_gain = 1.0 + 0.38 * valence

        out = data.copy()
        for idx in montage.FRONTAL_LEFT:
            if idx < out.shape[0]:
                out[idx] *= left_gain
        for idx in montage.FRONTAL_RIGHT:
            if idx < out.shape[0]:
                out[idx] *= right_gain
        return out

    def _add_background(self, data: np.ndarray) -> np.ndarray:
        """1/f pink background and a small uncorrelated white floor.

        Kept deliberately low. Broadband noise contributes power in proportion
        to band *width*, so it inflates beta (17 Hz wide) and gamma far more
        than alpha (5 Hz) — enough background and every recording analyses as
        beta-dominant regardless of its actual rhythms.
        """
        n = data.shape[1]
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
        shape = freqs ** -1.0

        out = data.copy()
        for ch in range(data.shape[0]):
            phase = self.rng.uniform(0, 2 * np.pi, len(freqs))
            pink = np.fft.irfft(shape * np.exp(1j * phase), n=n)
            sigma = float(np.std(pink)) or 1.0
            out[ch] += 0.16 * pink / sigma
            out[ch] += self.rng.normal(0, 0.025, n)
        return out

    def _add_artifacts(
        self, data: np.ndarray, profile: MemoryProfile, n_samples: int
    ) -> np.ndarray:
        """Blinks, muscle bursts, mains hum and occasional electrode dropout."""
        out = data.copy()
        level = float(np.clip(profile.artifact_level, 0.0, 1.0))
        duration = n_samples / self.sample_rate

        # --- Eye blinks: large slow frontal deflections -----------------------
        # Spontaneous blink rate is roughly 15–20/min at rest and climbs under
        # load, so ~0.3/s baseline scaled by the recording's artifact level.
        blink_rate = 0.30 * (0.7 + 2.0 * level)
        for _ in range(int(self.rng.poisson(duration * blink_rate))):
            centre = int(self.rng.uniform(0, n_samples))
            width = int(self.rng.uniform(0.12, 0.30) * self.sample_rate)
            shape = self._blink_shape(width)
            amplitude = self.rng.uniform(4.0, 9.0) * (0.8 + level)
            self._stamp(out, montage.FRONTAL, centre, shape * amplitude, decay=0.55)

        # --- Muscle bursts: brief broadband temporal activity -----------------
        for _ in range(int(self.rng.poisson(duration * 1.1 * level))):
            centre = int(self.rng.uniform(0, n_samples))
            width = int(self.rng.uniform(0.08, 0.25) * self.sample_rate)
            burst = self.rng.normal(0, 1.0, width) * np.hanning(width)
            amplitude = self.rng.uniform(2.5, 6.0) * (0.5 + level)
            self._stamp(out, montage.TEMPORAL, centre, burst * amplitude, decay=0.4)

        # --- Per-electrode impedance noise -----------------------------------
        # A couple of leads in any real montage sit at higher impedance and
        # carry a raised high-frequency floor. This is what separates an
        # excellent recording from a merely usable one.
        poor_leads = self.rng.random(out.shape[0]) < (0.10 + level * 0.45)
        for ch in np.flatnonzero(poor_leads):
            severity = self.rng.uniform(0.3, 1.4) * (0.4 + level)
            out[ch] += self.rng.normal(0, severity, n_samples)

        # --- Mains hum: low-level, present on every channel -------------------
        t = np.arange(n_samples) / self.sample_rate
        hum_amplitude = 0.14 * (0.4 + level)
        hum = hum_amplitude * np.sin(2 * np.pi * 60.0 * t + self.rng.uniform(0, 2 * np.pi))
        out += hum

        # --- Contact instability: sustained poor-electrode stretches ----------
        # Discrete blinks alone never push a window past a few percent artifact.
        # What actually degrades an ambulatory recording is a lead losing solid
        # contact for seconds at a time, producing large drifting excursions.
        # These stretches are what separate a usable session from a poor one.
        for _ in range(int(self.rng.poisson(duration * 0.10 * level * 4.0))):
            ch = int(self.rng.integers(0, out.shape[0]))
            length = int(self.rng.uniform(1.0, 4.5) * self.sample_rate)
            start = int(self.rng.uniform(0, max(1, n_samples - length)))
            span = min(length, n_samples - start)
            if span <= 8:
                continue

            # Random-walk drift, tapered so it fades in and out of contact.
            drift = np.cumsum(self.rng.normal(0, 1.0, span))
            drift -= np.linspace(drift[0], drift[-1], span)
            taper = np.hanning(span)
            scale = self.rng.uniform(6.0, 16.0) * (0.4 + level)
            sigma = float(np.std(drift)) or 1.0
            out[ch, start : start + span] += drift / sigma * scale * taper

        # --- Electrode dropout: one lead goes flat for a stretch --------------
        if self.rng.random() < level * 0.9 and out.shape[0] > 4:
            ch = int(self.rng.integers(0, out.shape[0]))
            start = int(self.rng.uniform(0, max(1, n_samples * 0.7)))
            length = int(self.rng.uniform(1.5, 5.0) * self.sample_rate)
            out[ch, start : start + length] *= 0.01

        return out

    def _blink_shape(self, width: int) -> np.ndarray:
        """Asymmetric blink waveform: fast down-stroke, slower recovery."""
        t = np.linspace(-1.0, 2.5, max(4, width))
        return -np.exp(-(t**2) * 3.0) + 0.35 * np.exp(-((t - 1.2) ** 2) * 2.0)

    def _stamp(
        self,
        data: np.ndarray,
        channels: List[int],
        centre: int,
        payload: np.ndarray,
        decay: float,
    ) -> None:
        """Add ``payload`` at ``centre`` on ``channels``, bleeding onto others."""
        n = data.shape[1]
        half = len(payload) // 2
        start = max(0, centre - half)
        end = min(n, start + len(payload))
        if end <= start:
            return
        segment = payload[: end - start]

        primary = set(channels)
        for ch in range(data.shape[0]):
            weight = 1.0 if ch in primary else decay * 0.35
            if weight <= 0.0:
                continue
            data[ch, start:end] += segment * weight


def generate_session(
    profile_id: str,
    duration: float,
    sample_rate: int = 256,
    seed: Optional[int] = None,
) -> SimulatedRecording:
    """Convenience wrapper used by the session service."""
    return EEGGenerator(sample_rate=sample_rate, seed=seed).generate(profile_id, duration)


def profile_choices() -> List[Tuple[str, str, str]]:
    """``(id, name, description)`` for the session-launch UI."""
    from .profiles import PROFILES

    return [(p.id, p.name, p.description) for p in PROFILES.values()]
