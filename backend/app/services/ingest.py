"""EEG recording ingestion.

Supports CSV/TSV and EDF. EDF is parsed with a built-in reader so uploads work
out of the box; when MNE is installed it is used instead, which additionally
brings BDF, FIF, GDF and correct handling of unusual headers.

Every path returns the same shape: ``(channels, samples)`` float64 microvolts,
a sample rate, and canonical channel labels.
"""

from __future__ import annotations

import io
import logging
import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..processing.montage import normalise_channel_names

log = logging.getLogger("mindscape.ingest")


class IngestError(ValueError):
    """Raised when a recording cannot be read or is unusable."""


@dataclass
class Recording:
    data: np.ndarray  # (channels, samples), microvolts
    sample_rate: int
    channels: List[str]
    source_name: str

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


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def load(payload: bytes, filename: str, fallback_rate: int = 256) -> Recording:
    """Read an uploaded recording into a :class:`Recording`."""
    name = filename.lower()

    if name.endswith((".edf", ".bdf")):
        recording = _load_with_mne(payload, filename) or _load_edf(payload, filename)
    elif name.endswith((".fif", ".fif.gz", ".gdf", ".set", ".vhdr")):
        recording = _load_with_mne(payload, filename)
        if recording is None:
            raise IngestError(
                f"{filename} requires MNE. Install it with "
                "`pip install -r requirements-optional.txt`, or upload CSV or EDF."
            )
    elif name.endswith((".csv", ".tsv", ".txt")):
        recording = _load_delimited(payload, filename, fallback_rate)
    else:
        raise IngestError(
            f"Unsupported file type: {filename}. Accepted formats are CSV, TSV and EDF."
        )

    return _validate(recording)


def _validate(recording: Recording) -> Recording:
    if recording.data.ndim != 2 or recording.data.shape[0] == 0:
        raise IngestError("Recording contains no channels.")
    if recording.data.shape[1] < recording.sample_rate:
        raise IngestError("Recording is shorter than one second.")
    if not np.isfinite(recording.data).all():
        # Interpolating would silently invent signal; zeroing is honest and the
        # quality stage will flag the affected channels.
        bad = int(np.sum(~np.isfinite(recording.data)))
        log.warning("Zeroing %d non-finite sample(s) in %s", bad, recording.source_name)
        recording.data = np.nan_to_num(recording.data, nan=0.0, posinf=0.0, neginf=0.0)
    return recording


# --------------------------------------------------------------------------- #
# CSV / TSV                                                                   #
# --------------------------------------------------------------------------- #


def _load_delimited(payload: bytes, filename: str, fallback_rate: int) -> Recording:
    """Read delimited text: one column per channel, one row per sample."""
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = payload.decode("latin-1")

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise IngestError("File is empty.")

    delimiter = "\t" if "\t" in lines[0] else ","

    # An optional `# sample_rate: 256` comment header.
    sample_rate = fallback_rate
    while lines and lines[0].lstrip().startswith("#"):
        comment = lines.pop(0).lstrip("#").strip().lower()
        if "sample_rate" in comment or "sfreq" in comment:
            for token in comment.replace(":", " ").replace("=", " ").split():
                try:
                    sample_rate = int(float(token))
                    break
                except ValueError:
                    continue

    if not lines:
        raise IngestError("File contains no data rows.")

    first = lines[0].split(delimiter)
    has_header = any(
        cell.strip() and not _is_number(cell) for cell in first
    )

    if has_header:
        channels = [c.strip() for c in first]
        rows = lines[1:]
    else:
        channels = [f"CH{i + 1}" for i in range(len(first))]
        rows = lines

    if not rows:
        raise IngestError("File contains a header but no samples.")

    values: List[List[float]] = []
    for line_number, row in enumerate(rows, start=2):
        cells = row.split(delimiter)
        if len(cells) != len(channels):
            continue  # ragged row — skip rather than fail the whole upload
        try:
            values.append([float(c) if c.strip() else 0.0 for c in cells])
        except ValueError:
            if line_number <= 3:
                raise IngestError(
                    f"Could not parse numeric data on line {line_number} of {filename}."
                )
            continue

    if not values:
        raise IngestError("No numeric rows could be parsed.")

    matrix = np.array(values, dtype=np.float64).T  # → (channels, samples)

    # Drop a leading time column if present: monotonically increasing and
    # spanning a plausible duration rather than oscillating like EEG.
    if matrix.shape[0] > 1 and _looks_like_time_column(matrix[0]):
        inferred = _rate_from_time_column(matrix[0])
        if inferred:
            sample_rate = inferred
        matrix = matrix[1:]
        channels = channels[1:]

    return Recording(
        data=matrix,
        sample_rate=sample_rate,
        channels=normalise_channel_names(channels),
        source_name=filename,
    )


def _is_number(cell: str) -> bool:
    try:
        float(cell.strip())
        return True
    except ValueError:
        return False


def _looks_like_time_column(column: np.ndarray) -> bool:
    if column.size < 3:
        return False
    diffs = np.diff(column)
    return bool(np.all(diffs > 0) and np.std(diffs) < np.mean(diffs) * 0.1)


def _rate_from_time_column(column: np.ndarray) -> Optional[int]:
    if column.size < 2:
        return None
    step = float(np.median(np.diff(column)))
    if step <= 0:
        return None
    rate = 1.0 / step
    # Times may be in seconds or milliseconds.
    if rate < 1.0:
        rate *= 1000.0
    if 1.0 <= rate <= 20000.0:
        return int(round(rate))
    return None


# --------------------------------------------------------------------------- #
# EDF                                                                         #
# --------------------------------------------------------------------------- #


def _load_edf(payload: bytes, filename: str) -> Recording:
    """Minimal EDF reader covering the standard 16-bit integer layout.

    Enough for the common case so uploads work without optional dependencies.
    Anything unusual should go through MNE.
    """
    if len(payload) < 256:
        raise IngestError("File is too short to be a valid EDF.")

    stream = io.BytesIO(payload)

    def ascii_field(size: int) -> str:
        return stream.read(size).decode("ascii", errors="replace").strip()

    stream.read(8)  # version
    ascii_field(80)  # patient
    ascii_field(80)  # recording
    ascii_field(8)  # start date
    ascii_field(8)  # start time

    try:
        header_bytes = int(ascii_field(8))
        stream.read(44)  # reserved
        record_count = int(ascii_field(8))
        record_duration = float(ascii_field(8))
        signal_count = int(ascii_field(4))
    except ValueError as exc:
        raise IngestError(f"Malformed EDF header in {filename}: {exc}") from exc

    if signal_count <= 0:
        raise IngestError("EDF declares no signals.")
    if record_duration <= 0:
        raise IngestError("EDF declares a non-positive record duration.")

    labels = [ascii_field(16) for _ in range(signal_count)]
    for _ in range(signal_count):
        ascii_field(80)  # transducer

    physical_dimensions = [ascii_field(8) for _ in range(signal_count)]
    physical_min = [_safe_float(ascii_field(8)) for _ in range(signal_count)]
    physical_max = [_safe_float(ascii_field(8)) for _ in range(signal_count)]
    digital_min = [_safe_float(ascii_field(8)) for _ in range(signal_count)]
    digital_max = [_safe_float(ascii_field(8)) for _ in range(signal_count)]

    for _ in range(signal_count):
        ascii_field(80)  # prefiltering

    samples_per_record = []
    for _ in range(signal_count):
        try:
            samples_per_record.append(int(ascii_field(8)))
        except ValueError:
            raise IngestError("Malformed EDF signal header.")

    # Annotation channels carry text, not signal.
    keep = [
        i
        for i, label in enumerate(labels)
        if "annotation" not in label.lower() and samples_per_record[i] > 0
    ]
    if not keep:
        raise IngestError("EDF contains no signal channels.")

    # Channels may be sampled at different rates; MindScape needs one montage
    # rate, so keep the most common and drop the rest.
    rates = [samples_per_record[i] for i in keep]
    modal_rate = max(set(rates), key=rates.count)
    dropped = [labels[i] for i in keep if samples_per_record[i] != modal_rate]
    if dropped:
        log.warning(
            "Dropping %d EDF channel(s) sampled at a different rate: %s",
            len(dropped),
            ", ".join(dropped[:4]),
        )
    keep = [i for i in keep if samples_per_record[i] == modal_rate]

    record_size = sum(samples_per_record)
    stream.seek(header_bytes)
    body = stream.read()

    expected = record_count * record_size * 2
    if record_count <= 0 or len(body) < expected:
        # Some writers leave the record count as -1 for streamed files.
        record_count = len(body) // (record_size * 2)
        if record_count <= 0:
            raise IngestError("EDF contains no readable data records.")

    raw = np.frombuffer(
        body[: record_count * record_size * 2], dtype="<i2"
    ).reshape(record_count, record_size)

    offsets = np.cumsum([0] + samples_per_record)
    channels_data = []

    for i in keep:
        block = raw[:, offsets[i] : offsets[i + 1]].reshape(-1).astype(np.float64)

        # Digital → physical scaling.
        d_span = digital_max[i] - digital_min[i]
        p_span = physical_max[i] - physical_min[i]
        if d_span and p_span:
            block = (block - digital_min[i]) * (p_span / d_span) + physical_min[i]

        # Normalise millivolts and volts to microvolts.
        unit = physical_dimensions[i].lower()
        if unit == "mv":
            block *= 1000.0
        elif unit == "v":
            block *= 1_000_000.0

        channels_data.append(block)

    sample_rate = int(round(modal_rate / record_duration))

    return Recording(
        data=np.array(channels_data),
        sample_rate=max(1, sample_rate),
        channels=normalise_channel_names([labels[i] for i in keep]),
        source_name=filename,
    )


def _safe_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return 0.0


# --------------------------------------------------------------------------- #
# MNE (optional)                                                              #
# --------------------------------------------------------------------------- #


def _load_with_mne(payload: bytes, filename: str) -> Optional[Recording]:
    """Read via MNE when it is installed. ``None`` if unavailable."""
    try:
        import mne  # type: ignore
    except ImportError:
        return None

    import tempfile
    from pathlib import Path

    suffix = Path(filename).suffix or ".edf"

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)

        try:
            raw = mne.io.read_raw(str(temp_path), preload=True, verbose="ERROR")
            raw.pick("eeg")
            if len(raw.ch_names) == 0:
                return None

            # MNE returns volts; MindScape works in microvolts.
            data = raw.get_data() * 1_000_000.0

            return Recording(
                data=np.asarray(data, dtype=np.float64),
                sample_rate=int(round(raw.info["sfreq"])),
                channels=normalise_channel_names(list(raw.ch_names)),
                source_name=filename,
            )
        finally:
            temp_path.unlink(missing_ok=True)

    except Exception as exc:  # noqa: BLE001
        log.warning("MNE could not read %s (%s); trying built-in reader", filename, exc)
        return None
