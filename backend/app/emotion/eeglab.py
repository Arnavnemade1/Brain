"""Minimal EEGLAB `.set` / `.fdt` reader.

DENS ships as EEGLAB: a MATLAB struct (`.set`) holding metadata and a raw
float32 binary (`.fdt`) holding the signal. Reading it directly rather than
via MNE keeps the dependency surface small and, more importantly, keeps the
memory behaviour explicit — these recordings are ~350 MB each and we only ever
need the windows spanning a stimulus trial.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy import io as sio

log = logging.getLogger("mindscape.eeglab")


@dataclass
class EEGLABRecording:
    """Memory-mapped access to an EEGLAB recording."""

    path: Path
    sampling_rate: float
    channel_names: List[str]
    n_channels: int
    n_samples: int
    #: (channels, 3) electrode coordinates: X anterior+, Y left+, Z superior+.
    #: NaN rows are non-EEG sensors (ECG, EMG) and must be excluded.
    positions: np.ndarray

    _data: np.memmap

    #: Channel names that are never scalp EEG.
    NON_EEG = {"ECG", "EKG", "EMG", "EOG", "VEOG", "HEOG", "RESP", "GSR", "TRIG", "STATUS"}

    @property
    def eeg_channels(self) -> List[int]:
        """Indices of real scalp electrodes.

        Prefers coordinates when the `.set` carries them, since that also
        excludes physiological channels. Some BIDS exports keep positions in a
        separate `electrodes.tsv` instead, leaving every coordinate NaN — in
        that case fall back to names, rather than returning nothing and
        silently producing zero-channel epochs downstream.
        """
        located = [
            i for i in range(self.n_channels) if np.all(np.isfinite(self.positions[i]))
        ]
        if located:
            return located

        return [
            i
            for i, name in enumerate(self.channel_names)
            if name.strip().upper() not in self.NON_EEG
        ]

    @property
    def duration(self) -> float:
        return self.n_samples / self.sampling_rate

    def window(self, start: int, length: int) -> np.ndarray:
        """``(channels, length)`` in microvolts, zero-padded at the edges."""
        out = np.zeros((self.n_channels, max(0, length)), dtype=np.float32)
        if length <= 0:
            return out
        lo = max(0, start)
        hi = min(self.n_samples, start + length)
        if hi <= lo:
            return out
        chunk = np.asarray(self._data[:, lo:hi], dtype=np.float32)
        out[:, lo - start : lo - start + chunk.shape[1]] = chunk
        return out

    def close(self) -> None:
        data = getattr(self, "_data", None)
        if data is not None and hasattr(data, "_mmap"):
            data._mmap.close()  # noqa: SLF001

    def __enter__(self) -> "EEGLABRecording":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _unwrap(value):
    """Peel the nested 1x1 object arrays scipy produces for MATLAB structs."""
    while isinstance(value, np.ndarray) and value.size == 1 and value.dtype == object:
        value = value.item()
    while isinstance(value, np.ndarray) and value.size == 1 and value.ndim > 0:
        value = value.reshape(-1)[0]
    return value


def _scalar(value, default: float = 0.0) -> float:
    value = _unwrap(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read(set_path: Path) -> EEGLABRecording:
    """Open an EEGLAB recording.

    Only the `.set` header is parsed eagerly; the signal itself is memory
    mapped from the companion `.fdt`.
    """
    set_path = Path(set_path)
    raw = sio.loadmat(set_path, squeeze_me=False, struct_as_record=False)

    if "EEG" in raw:
        eeg = _unwrap(raw["EEG"])
    else:
        # Some exports store the struct fields at the top level.
        eeg = type("EEG", (), {k: v for k, v in raw.items() if not k.startswith("__")})

    sampling_rate = _scalar(getattr(eeg, "srate", 1000.0), 1000.0)
    n_channels = int(_scalar(getattr(eeg, "nbchan", 0)))
    n_samples = int(_scalar(getattr(eeg, "pnts", 0)))

    # Channel labels live in a struct array.
    names: List[str] = []
    chanlocs = getattr(eeg, "chanlocs", None)
    if chanlocs is not None:
        locations = np.atleast_1d(chanlocs).reshape(-1)
        for entry in locations:
            label = getattr(entry, "labels", None)
            label = _unwrap(label)
            names.append(str(label).strip() if label is not None else "")
    if len(names) != n_channels:
        names = [f"Ch{i + 1}" for i in range(n_channels)]

    # Electrode coordinates. Regions are resolved geometrically rather than by
    # label because this is a 128-channel EGI net (E1..E128), whose names carry
    # no anatomical meaning on their own.
    positions = np.full((n_channels, 3), np.nan)
    if chanlocs is not None:
        locations = np.atleast_1d(chanlocs).reshape(-1)
        for i, entry in enumerate(locations[:n_channels]):
            for j, axis in enumerate(("X", "Y", "Z")):
                raw_value = _unwrap(getattr(entry, axis, None))
                try:
                    if raw_value is not None and np.size(raw_value):
                        positions[i, j] = float(raw_value)
                except (TypeError, ValueError):
                    pass

    # `data` is normally the .fdt filename; occasionally the array is inline.
    data_field = _unwrap(getattr(eeg, "data", None))

    if isinstance(data_field, np.ndarray) and data_field.dtype != object and data_field.ndim == 2:
        array = np.ascontiguousarray(data_field.astype(np.float32))
        n_channels = array.shape[0]
        n_samples = array.shape[1]
        memmap = array  # type: ignore[assignment]
    else:
        filename = str(data_field).strip()
        binary = set_path.parent / filename
        if not binary.exists():
            binary = set_path.with_suffix(".fdt")
        if not binary.exists():
            raise FileNotFoundError(f"missing .fdt for {set_path.name}")

        expected = binary.stat().st_size // 4
        if n_channels <= 0:
            raise ValueError(f"{set_path.name}: unknown channel count")
        if n_samples <= 0 or n_samples * n_channels != expected:
            # Trust the file size over a possibly stale header.
            n_samples = expected // n_channels

        # EEGLAB writes channels-fastest (Fortran order): all channels for
        # sample 0, then all channels for sample 1, and so on.
        memmap = np.memmap(
            binary, dtype="<f4", mode="r", shape=(n_channels, n_samples), order="F"
        )

    log.info(
        "%s: %d ch @ %.0f Hz, %.1f min",
        set_path.name,
        n_channels,
        sampling_rate,
        n_samples / sampling_rate / 60,
    )

    return EEGLABRecording(
        path=set_path,
        sampling_rate=sampling_rate,
        channel_names=names[:n_channels],
        n_channels=n_channels,
        n_samples=n_samples,
        positions=positions,
        _data=memmap,
    )
