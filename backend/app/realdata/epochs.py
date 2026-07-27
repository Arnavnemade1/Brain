"""Stimulus-locked epoch extraction from the real EEG recordings.

Turns a 36-minute continuous recording into trial-aligned epochs: one per
image presentation, time-locked to onset. Everything downstream — decoding,
identification, reconstruction — operates on these.

Preprocessing is deliberately conservative. Every step is one that would be
uncontroversial in an ERP paper, because the point of using real data is to
avoid the accusation that the result is a processing artifact.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal

from .brainvision import BrainVisionRecording

log = logging.getLogger("mindscape.realdata")

#: Epoch window relative to stimulus onset, in seconds.
EPOCH_START = -0.100
EPOCH_END = 0.600

#: Epochs are stored at this rate. Visual ERP components of interest (P1, N1,
#: P2a) are all well below 100 Hz, so 250 Hz is generous and keeps the cache
#: an order of magnitude smaller than the 1000 Hz source.
TARGET_RATE = 250.0

#: Baseline correction interval, relative to onset.
BASELINE = (-0.100, 0.0)

#: Band-pass applied to the continuous recording before epoching.
BAND = (0.1, 40.0)

#: Trials whose peak-to-peak amplitude exceeds this (µV) are rejected as
#: artifact. 150 µV is a standard, permissive threshold for RSVP data.
REJECT_PEAK_TO_PEAK = 150.0


@dataclass
class EpochSet:
    """Trial-aligned EEG plus the labels describing what was shown."""

    #: (trials, channels, times) in microvolts.
    data: np.ndarray
    times: np.ndarray
    channel_names: List[str]
    #: 1-based stimulus id, 1..200.
    stimulus: np.ndarray
    #: Presentation onset in seconds from the start of the recording. Needed
    #: to lay a reconstruction out on a real time axis rather than assuming a
    #: uniform 5 Hz — the streams have breaks between blocks.
    onset_s: np.ndarray
    #: Filename of the image shown on each trial.
    stimulus_name: List[str]
    #: animate / inanimate.
    level_a: List[str]
    #: 10 categories (mammal, tools, …).
    level_b: List[str]
    #: 50 concepts (monkey, jeans, …).
    level_c: List[str]
    subject: str
    sampling_rate: float

    @property
    def n_trials(self) -> int:
        return int(self.data.shape[0])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            # float16 is ample: after baseline correction the signal spans a
            # few tens of microvolts, far inside half-precision resolution,
            # and it halves an already-large cache.
            data=self.data.astype(np.float16),
            times=self.times,
            channel_names=np.array(self.channel_names),
            stimulus=self.stimulus,
            onset_s=self.onset_s,
            stimulus_name=np.array(self.stimulus_name),
            level_a=np.array(self.level_a),
            level_b=np.array(self.level_b),
            level_c=np.array(self.level_c),
            subject=self.subject,
            sampling_rate=self.sampling_rate,
        )

    @classmethod
    def load(cls, path: Path) -> "EpochSet":
        blob = np.load(path, allow_pickle=False)
        return cls(
            data=blob["data"].astype(np.float32),
            times=blob["times"],
            channel_names=[str(c) for c in blob["channel_names"]],
            stimulus=blob["stimulus"],
            # Caches written before onsets were recorded fall back to the
            # nominal 5 Hz spacing rather than failing to load.
            onset_s=(
                blob["onset_s"]
                if "onset_s" in blob.files
                else np.arange(blob["stimulus"].size, dtype=np.float64) * 0.2
            ),
            stimulus_name=[str(s) for s in blob["stimulus_name"]],
            level_a=[str(s) for s in blob["level_a"]],
            level_b=[str(s) for s in blob["level_b"]],
            level_c=[str(s) for s in blob["level_c"]],
            subject=str(blob["subject"]),
            sampling_rate=float(blob["sampling_rate"]),
        )


def read_events(path: Path) -> List[dict]:
    with open(path) as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _preprocess(recording: BrainVisionRecording) -> np.ndarray:
    """Band-pass and re-reference the whole recording.

    Filtering must happen on the continuous data, before epoching: applying a
    filter to short epochs produces severe edge artifacts across exactly the
    interval the decoder reads.
    """
    n_channels = recording.header.n_channels
    data = recording.window(0, recording.n_samples).astype(np.float32)

    nyquist = recording.sampling_rate / 2.0
    low = BAND[0] / nyquist
    high = min(BAND[1] / nyquist, 0.99)

    sos = sp_signal.butter(4, [low, high], btype="band", output="sos")
    data = sp_signal.sosfiltfilt(sos, data, axis=-1).astype(np.float32)

    # Common average reference. The recording is Cz-referenced, which puts a
    # site-specific bias into every channel; average referencing removes it
    # and is the convention for this kind of multivariate decoding.
    data -= data.mean(axis=0, keepdims=True)

    log.info(
        "preprocessed %d channels x %d samples (%.0f s)",
        n_channels,
        data.shape[1],
        recording.duration,
    )
    return data


def extract(
    vhdr_path: Path,
    events_path: Path,
    subject: str,
    max_trials: Optional[int] = None,
) -> EpochSet:
    """Extract stimulus-locked epochs from one recording."""
    events = read_events(events_path)

    with BrainVisionRecording(vhdr_path) as recording:
        rate = recording.sampling_rate
        continuous = _preprocess(recording)
        channel_names = list(recording.channel_names)
        n_samples = recording.n_samples

    start_offset = int(round(EPOCH_START * rate))
    end_offset = int(round(EPOCH_END * rate))
    length = end_offset - start_offset

    decimation = max(1, int(round(rate / TARGET_RATE)))

    kept: List[np.ndarray] = []
    stimulus: List[int] = []
    onsets: List[float] = []
    names: List[str] = []
    a: List[str] = []
    b: List[str] = []
    c: List[str] = []
    rejected = 0

    baseline_lo = int(round((BASELINE[0] - EPOCH_START) * rate))
    baseline_hi = int(round((BASELINE[1] - EPOCH_START) * rate))

    for row in events:
        # Targets are the colour-change detection task, not object trials.
        if row.get("istarget", "0") != "0":
            continue
        name = row.get("stimulusname", "")
        if not name.endswith(".png"):
            continue

        try:
            onset = int(float(row["onset"]))
            number = int(row["stimulusnumber"])
        except (KeyError, ValueError):
            continue

        begin = onset + start_offset
        if begin < 0 or begin + length > n_samples:
            continue

        epoch = continuous[:, begin : begin + length]

        # Baseline correction: subtract the pre-stimulus mean per channel, so
        # each trial is expressed as a deflection from its own starting level
        # rather than carrying slow drift into the decoder.
        epoch = epoch - epoch[:, baseline_lo:baseline_hi].mean(axis=1, keepdims=True)

        peak_to_peak = float(np.max(epoch) - np.min(epoch))
        if peak_to_peak > REJECT_PEAK_TO_PEAK:
            rejected += 1
            continue

        kept.append(epoch[:, ::decimation].astype(np.float32))
        stimulus.append(number)
        onsets.append(onset / rate)
        names.append(name)
        a.append(row.get("levelA", ""))
        b.append(row.get("levelB", ""))
        c.append(row.get("levelC", ""))

        if max_trials is not None and len(kept) >= max_trials:
            break

    if not kept:
        raise RuntimeError(f"{subject}: no usable trials extracted")

    data = np.stack(kept)
    times = (np.arange(data.shape[2]) * decimation / rate) + EPOCH_START

    log.info(
        "%s: %d trials kept, %d rejected (%.1f%%), %d unique stimuli",
        subject,
        len(kept),
        rejected,
        100.0 * rejected / max(1, rejected + len(kept)),
        len(set(stimulus)),
    )

    return EpochSet(
        data=data,
        times=times,
        channel_names=channel_names,
        stimulus=np.array(stimulus, dtype=np.int16),
        onset_s=np.array(onsets, dtype=np.float64),
        stimulus_name=names,
        level_a=a,
        level_b=b,
        level_c=c,
        subject=subject,
        sampling_rate=rate / decimation,
    )


def cache_path(root: Path, subject: str, run: str = "01") -> Path:
    return root / "derivatives" / "epochs" / f"{subject}_run-{run}_epochs.npz"


def load_or_extract(
    root: Path, subject: str, run: str = "01", force: bool = False
) -> EpochSet:
    """Epochs for one subject, extracting and caching on first use."""
    cached = cache_path(root, subject, run)
    if cached.exists() and not force:
        return EpochSet.load(cached)

    eeg_dir = root / subject / "eeg"
    vhdr = eeg_dir / f"{subject}_task-rsvp_run-{run}_eeg.vhdr"
    events = eeg_dir / f"{subject}_task-rsvp_run-{run}_events.tsv"

    if not vhdr.exists() or not events.exists():
        raise FileNotFoundError(
            f"{subject} run-{run} not downloaded. Run tools/fetch_dataset.py."
        )

    epochs = extract(vhdr, events, subject)
    epochs.save(cached)
    return epochs


def available_subjects(root: Path) -> List[str]:
    """Subjects with either a cached epoch file or a downloaded recording."""
    found = set()

    cache_dir = root / "derivatives" / "epochs"
    if cache_dir.exists():
        for path in cache_dir.glob("sub-*_epochs.npz"):
            found.add(path.name.split("_")[0])

    for path in sorted(root.glob("sub-*")):
        vhdr = path / "eeg" / f"{path.name}_task-rsvp_run-01_eeg.vhdr"
        binary = path / "eeg" / f"{path.name}_task-rsvp_run-01_eeg.eeg"
        if vhdr.exists() and binary.exists() and binary.stat().st_size > 0:
            found.add(path.name)

    return sorted(found)
