"""Minimal EDF+ reader.

The motor recordings ship as EDF+, which neither of the existing readers
(BrainVision, EEGLAB) handles. Written here rather than pulled in as a
dependency for the same reason as the others: the format is simple, and owning
the parse means a decoding result can never be blamed on an opaque library
step.

EDF is a fixed-width ASCII header followed by interleaved int16 data records.
EDF+ adds an "EDF Annotations" signal carrying time-stamped annotation lists
(TALs) as raw bytes in the same record stream, which is where the task labels
live in this dataset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("mindscape.edf")

#: Label of the EDF+ signal carrying annotations rather than samples.
ANNOTATION_LABEL = "EDF Annotations"

_HEADER_BYTES = 256
_SIGNAL_HEADER_BYTES = 256


@dataclass
class Annotation:
    """One time-stamped event from the EDF+ annotation stream."""

    onset: float
    duration: float
    label: str


@dataclass
class EdfRecording:
    """Continuous EEG plus the annotations describing what was asked for."""

    #: (channels, samples) in the file's physical units, here microvolts.
    data: np.ndarray
    channel_names: List[str]
    sampling_rate: float
    annotations: List[Annotation]
    subject: str
    run: str

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration(self) -> float:
        return self.n_samples / self.sampling_rate


def _ascii(blob: bytes) -> str:
    return blob.decode("ascii", errors="replace").strip()


def _parse_tals(blob: bytes) -> List[Annotation]:
    """Parse one record's worth of EDF+ annotation bytes.

    A TAL is ``+onset[\\x15duration]\\x14label\\x14`` and records are null
    padded. The first TAL of each record only timestamps the record itself and
    carries an empty label, so it drops out naturally.
    """
    out: List[Annotation] = []
    for chunk in blob.split(b"\x00"):
        if not chunk:
            continue
        parts = chunk.split(b"\x14")
        if len(parts) < 2:
            continue

        timing = parts[0]
        if b"\x15" in timing:
            onset_bytes, duration_bytes = timing.split(b"\x15", 1)
        else:
            onset_bytes, duration_bytes = timing, b"0"

        try:
            onset = float(onset_bytes)
            duration = float(duration_bytes) if duration_bytes.strip() else 0.0
        except ValueError:
            continue

        for label in parts[1:]:
            text = label.decode("ascii", errors="replace").strip()
            if text:
                out.append(Annotation(onset=onset, duration=duration, label=text))
    return out


def read_edf(path: Path) -> EdfRecording:
    """Read one EDF+ file into memory."""
    raw = path.read_bytes()

    n_signals = int(_ascii(raw[252:256]))
    n_records = int(_ascii(raw[236:244]))
    record_duration = float(_ascii(raw[244:252]))
    header_bytes = int(_ascii(raw[184:192]))

    # The signal header is a sequence of blocks, each holding one field for
    # *every* signal before the next field begins. Spelling the layout out as
    # (name, width) and deriving the offsets removes the off-by-one-field
    # class of bug entirely — computing them by hand read the physical
    # dimension ("uV") where the physical minimum was meant to be.
    layout: Tuple[Tuple[str, int], ...] = (
        ("label", 16),
        ("transducer", 80),
        ("physical_dim", 8),
        ("physical_min", 8),
        ("physical_max", 8),
        ("digital_min", 8),
        ("digital_max", 8),
        ("prefilter", 80),
        ("samples_per_record", 8),
        ("reserved", 32),
    )
    starts: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    for name, width in layout:
        starts[name] = (cursor, width)
        cursor += width * n_signals

    def field(name: str, index: int) -> bytes:
        offset, width = starts[name]
        base = _HEADER_BYTES + offset + width * index
        return raw[base : base + width]

    def numbers(name: str) -> np.ndarray:
        return np.array([float(_ascii(field(name, i))) for i in range(n_signals)])

    labels = [_ascii(field("label", i)) for i in range(n_signals)]
    physical_min = numbers("physical_min")
    physical_max = numbers("physical_max")
    digital_min = numbers("digital_min")
    digital_max = numbers("digital_max")
    samples_per_record = numbers("samples_per_record").astype(int)

    record_size = int(samples_per_record.sum())
    body = np.frombuffer(
        raw, dtype="<i2", count=n_records * record_size, offset=header_bytes
    ).reshape(n_records, record_size)

    offsets = np.concatenate([[0], np.cumsum(samples_per_record)])

    signal_indices = [i for i, name in enumerate(labels) if name != ANNOTATION_LABEL]
    annotation_indices = [i for i, name in enumerate(labels) if name == ANNOTATION_LABEL]

    # --- samples -----------------------------------------------------------
    rates = samples_per_record[signal_indices] / record_duration
    if len(set(rates.tolist())) != 1:
        raise ValueError(f"{path.name}: mixed sampling rates {sorted(set(rates))}")
    sampling_rate = float(rates[0])

    per_record = int(samples_per_record[signal_indices[0]])
    data = np.empty((len(signal_indices), n_records * per_record), dtype=np.float32)

    for row, index in enumerate(signal_indices):
        block = body[:, offsets[index] : offsets[index + 1]].astype(np.float32)

        # EDF stores integers; the header carries the affine map back to
        # physical units. Skipping this leaves the data in ADC counts, which
        # is the same class of bug that made the BrainVision amplitudes 20x
        # too large.
        span = digital_max[index] - digital_min[index]
        scale = (physical_max[index] - physical_min[index]) / span if span else 1.0
        data[row] = (
            (block - digital_min[index]) * scale + physical_min[index]
        ).reshape(-1)

    # --- annotations -------------------------------------------------------
    annotations: List[Annotation] = []
    for index in annotation_indices:
        blob = body[:, offsets[index] : offsets[index + 1]].astype("<i2").tobytes()
        # Each record's annotation bytes are contiguous; parsing the whole
        # stream at once is equivalent and simpler than per-record slicing.
        annotations.extend(_parse_tals(blob))

    annotations.sort(key=lambda a: a.onset)

    stem = path.stem
    subject, run = (stem[:4], stem[4:]) if len(stem) >= 5 else (stem, "")

    log.info(
        "%s: %d channels x %d samples (%.0f Hz, %.0f s), %d annotations",
        path.name,
        data.shape[0],
        data.shape[1],
        sampling_rate,
        data.shape[1] / sampling_rate,
        len(annotations),
    )

    return EdfRecording(
        data=data,
        channel_names=[labels[i].strip(".") for i in signal_indices],
        sampling_rate=sampling_rate,
        annotations=annotations,
        subject=subject,
        run=run,
    )
