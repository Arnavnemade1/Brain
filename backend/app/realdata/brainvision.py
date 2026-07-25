"""Minimal BrainVision reader.

ds004018 ships as BrainVision: a text header (`.vhdr`), a marker file
(`.vmrk`) and raw binary (`.eeg`). The format is simple enough that a direct
reader is preferable to pulling in MNE — it keeps the dependency surface small
and, more usefully, makes the memory behaviour explicit. These recordings are
~550 MB each and we only ever need short windows around stimulus onsets, so
the reader memory-maps the binary and slices it rather than loading whole
recordings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_DTYPES: Dict[str, np.dtype] = {
    "IEEE_FLOAT_32": np.dtype("<f4"),
    "INT_16": np.dtype("<i2"),
    "UINT_16": np.dtype("<u2"),
}


@dataclass
class BrainVisionHeader:
    data_file: str
    marker_file: str
    n_channels: int
    sampling_rate: float
    dtype: np.dtype
    multiplexed: bool
    channel_names: List[str]
    #: Per-channel scale factor to microvolts.
    resolutions: np.ndarray


def read_header(path: Path) -> BrainVisionHeader:
    """Parse a `.vhdr` file."""
    # BrainVision headers are latin-1 in practice (µV in channel units).
    text = path.read_text(encoding="utf-8", errors="replace")

    def field(name: str, default: Optional[str] = None) -> str:
        match = re.search(rf"^{name}=(.*)$", text, re.MULTILINE)
        if match is None:
            if default is None:
                raise ValueError(f"{path.name}: missing {name}")
            return default
        return match.group(1).strip()

    binary_format = field("BinaryFormat", "IEEE_FLOAT_32")
    if binary_format not in _DTYPES:
        raise ValueError(f"{path.name}: unsupported BinaryFormat {binary_format}")

    n_channels = int(field("NumberOfChannels"))
    # Sampling interval is in microseconds.
    sampling_rate = 1e6 / float(field("SamplingInterval"))

    # Channel entries must be read from [Channel Infos] alone. A [Coordinates]
    # section uses the same `Ch<n>=` key for electrode positions, so scanning
    # the whole file matches twice as many lines as there are channels and
    # silently yields positions where names should be.
    section = re.search(
        r"^\[Channel Infos\]\s*$(.*?)(?=^\[|\Z)", text, re.MULTILINE | re.DOTALL
    )
    channel_block = section.group(1) if section else ""

    names_by_index: Dict[int, str] = {}
    resolutions = np.ones(n_channels, dtype=np.float64)

    for match in re.finditer(
        r"^Ch(\d+)=([^,]*),([^,]*),([^,]*)", channel_block, re.MULTILINE
    ):
        index = int(match.group(1)) - 1
        if not 0 <= index < n_channels:
            continue
        names_by_index[index] = match.group(2).strip()
        raw_resolution = match.group(4).strip()
        if raw_resolution:
            try:
                value = float(raw_resolution)
                if value > 0:
                    resolutions[index] = value
            except ValueError:
                pass

    names = [names_by_index.get(i, f"Ch{i + 1}") for i in range(n_channels)]

    return BrainVisionHeader(
        data_file=field("DataFile"),
        marker_file=field("MarkerFile", ""),
        n_channels=n_channels,
        sampling_rate=sampling_rate,
        dtype=_DTYPES[binary_format],
        multiplexed=field("DataOrientation", "MULTIPLEXED").upper() == "MULTIPLEXED",
        channel_names=names,
        resolutions=resolutions,
    )


class BrainVisionRecording:
    """Memory-mapped access to a BrainVision recording."""

    def __init__(self, vhdr_path: Path) -> None:
        self.path = Path(vhdr_path)
        self.header = read_header(self.path)

        binary = self.path.parent / self.header.data_file
        if not binary.exists():
            raise FileNotFoundError(f"missing binary {binary}")

        itemsize = self.header.dtype.itemsize
        total = binary.stat().st_size // itemsize
        self.n_samples = total // self.header.n_channels

        # Memory-map rather than read: these files are hundreds of megabytes
        # and we only touch short windows around each stimulus onset.
        self._raw = np.memmap(
            binary,
            dtype=self.header.dtype,
            mode="r",
            shape=(self.n_samples, self.header.n_channels)
            if self.header.multiplexed
            else (self.header.n_channels, self.n_samples),
        )

    @property
    def sampling_rate(self) -> float:
        return self.header.sampling_rate

    @property
    def channel_names(self) -> List[str]:
        return self.header.channel_names

    @property
    def duration(self) -> float:
        return self.n_samples / self.header.sampling_rate

    def window(self, start: int, length: int) -> np.ndarray:
        """``(channels, length)`` in microvolts, zero-padded at the edges."""
        out = np.zeros((self.header.n_channels, length), dtype=np.float32)
        if length <= 0:
            return out

        lo = max(0, start)
        hi = min(self.n_samples, start + length)
        if hi <= lo:
            return out

        if self.header.multiplexed:
            chunk = np.asarray(self._raw[lo:hi, :]).T
        else:
            chunk = np.asarray(self._raw[:, lo:hi])

        # Scale by the per-channel resolution regardless of storage type.
        # The BrainVision spec allows float files to be pre-scaled (resolution
        # 1), but this recording stores raw amplifier units in float32 with an
        # explicit 0.0488281 µV/unit resolution — the standard BrainAmp value.
        # Skipping the scale because the dtype is float leaves amplitudes
        # twenty times too large.
        chunk = chunk.astype(np.float32) * self.header.resolutions[:, None].astype(
            np.float32
        )

        out[:, lo - start : lo - start + chunk.shape[1]] = chunk
        return out

    def close(self) -> None:
        raw = getattr(self, "_raw", None)
        if raw is not None and hasattr(raw, "_mmap"):
            raw._mmap.close()  # noqa: SLF001
        self._raw = None  # type: ignore[assignment]

    def __enter__(self) -> "BrainVisionRecording":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
