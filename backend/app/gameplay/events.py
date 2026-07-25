"""Gameplay events and EEG epochs around them.

Each event is a moment in the activity: the player fired, collected a reward,
or crashed. The reconstruction task is to recover that sequence from EEG
alone, which is a genuinely different proposition from the emotion work —
these are among the largest and most replicated effects in the literature.

* **Reward and error feedback** produce the feedback-related negativity and
  reward positivity: a frontocentral deflection roughly 250-350 ms after an
  outcome, with a well-documented amplitude difference between good and bad
  outcomes.
* **Button presses** are preceded by motor preparation and accompanied by
  mu/beta desynchronisation over sensorimotor cortex.

Neither is subtle, which is why this reconstructs where emotion did not.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("mindscape.gameplay")

#: Event epoch, relative to onset. Covers motor preparation before an action
#: and the full feedback complex after an outcome.
EPOCH_START = -0.30
EPOCH_END = 0.80

#: Epochs are stored at this rate; feedback components are well under 30 Hz.
TARGET_RATE = 125.0

BAND = (0.5, 30.0)

#: Trials whose peak-to-peak exceeds this are rejected. Gameplay involves
#: real movement, so this is deliberately permissive.
REJECT_PEAK_TO_PEAK = 250.0

#: The outcome categories the reconstruction recovers.
REWARD_EVENTS = {"COLLECT_STAR", "MISSILE_HIT_ENEMY", "COLLECT_AMMO"}
ERROR_EVENTS = {"PLAYER_CRASH_WALL", "PLAYER_CRASH_ENEMY"}
ACTION_EVENTS = {"SHOOT_BUTTON"}

#: Coarse labels used for reconstruction and display.
CATEGORIES = ["reward", "crash", "shoot"]


def categorise(trial_type: str) -> Optional[str]:
    if trial_type in REWARD_EVENTS:
        return "reward"
    if trial_type in ERROR_EVENTS:
        return "crash"
    if trial_type in ACTION_EVENTS:
        return "shoot"
    return None


@dataclass
class GameEvent:
    onset: float
    trial_type: str
    category: str


@dataclass
class EpochSet:
    """Event-locked EEG plus what was happening."""

    data: np.ndarray  # (events, channels, times)
    times: np.ndarray
    channel_names: List[str]
    onsets: np.ndarray
    categories: List[str]
    trial_types: List[str]
    subject: str
    sampling_rate: float
    duration: float

    @property
    def n_events(self) -> int:
        return int(self.data.shape[0])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            data=self.data.astype(np.float16),
            times=self.times,
            channel_names=np.array(self.channel_names),
            onsets=self.onsets,
            categories=np.array(self.categories),
            trial_types=np.array(self.trial_types),
            subject=self.subject,
            sampling_rate=self.sampling_rate,
            duration=self.duration,
        )

    @classmethod
    def load(cls, path: Path) -> "EpochSet":
        blob = np.load(path, allow_pickle=False)
        return cls(
            data=blob["data"].astype(np.float32),
            times=blob["times"],
            channel_names=[str(c) for c in blob["channel_names"]],
            onsets=blob["onsets"],
            categories=[str(c) for c in blob["categories"]],
            trial_types=[str(c) for c in blob["trial_types"]],
            subject=str(blob["subject"]),
            sampling_rate=float(blob["sampling_rate"]),
            duration=float(blob["duration"]),
        )


def read_events(path: Path) -> List[GameEvent]:
    """Every categorised gameplay event, in order."""
    with open(path) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    events: List[GameEvent] = []
    for row in rows:
        trial_type = (row.get("trial_type") or "").strip()
        category = categorise(trial_type)
        if category is None:
            continue
        try:
            onset = float(row["onset"])
        except (KeyError, ValueError):
            continue
        events.append(GameEvent(onset=onset, trial_type=trial_type, category=category))

    return sorted(events, key=lambda e: e.onset)


def extract(set_path: Path, events_path: Path, subject: str) -> EpochSet:
    """Event-locked epochs from one gameplay recording."""
    from scipy import signal as sp_signal

    from ..emotion.eeglab import read

    events = read_events(events_path)
    if not events:
        raise RuntimeError(f"{subject}: no gameplay events")

    with read(set_path) as recording:
        rate = recording.sampling_rate
        channels = recording.eeg_channels
        names = [recording.channel_names[i] for i in channels]

        continuous = recording.window(0, recording.n_samples)[channels]
        duration = recording.duration

    nyquist = rate / 2.0
    sos = sp_signal.butter(
        4, [BAND[0] / nyquist, min(BAND[1] / nyquist, 0.99)], btype="band", output="sos"
    )
    continuous = sp_signal.sosfiltfilt(sos, continuous, axis=-1).astype(np.float32)
    continuous -= continuous.mean(axis=0, keepdims=True)

    start_offset = int(round(EPOCH_START * rate))
    end_offset = int(round(EPOCH_END * rate))
    length = end_offset - start_offset
    decimation = max(1, int(round(rate / TARGET_RATE)))

    # Baseline is the pre-event interval, so each epoch is a deflection from
    # its own starting level rather than carrying slow drift.
    baseline_hi = int(round(-EPOCH_START * rate))

    kept: List[np.ndarray] = []
    onsets: List[float] = []
    categories: List[str] = []
    trial_types: List[str] = []
    rejected = 0

    for event in events:
        begin = int(round(event.onset * rate)) + start_offset
        if begin < 0 or begin + length > continuous.shape[1]:
            continue

        epoch = continuous[:, begin : begin + length]
        epoch = epoch - epoch[:, :baseline_hi].mean(axis=1, keepdims=True)

        if float(np.max(epoch) - np.min(epoch)) > REJECT_PEAK_TO_PEAK:
            rejected += 1
            continue

        kept.append(epoch[:, ::decimation].astype(np.float32))
        onsets.append(event.onset)
        categories.append(event.category)
        trial_types.append(event.trial_type)

    if not kept:
        raise RuntimeError(f"{subject}: no usable epochs")

    data = np.stack(kept)
    times = (np.arange(data.shape[2]) * decimation / rate) + EPOCH_START

    counts: Dict[str, int] = {}
    for category in categories:
        counts[category] = counts.get(category, 0) + 1

    log.info(
        "%s: %d epochs kept, %d rejected (%.0f%%) — %s",
        subject,
        len(kept),
        rejected,
        100.0 * rejected / max(1, rejected + len(kept)),
        ", ".join(f"{k} {v}" for k, v in sorted(counts.items())),
    )

    return EpochSet(
        data=data,
        times=times,
        channel_names=names,
        onsets=np.array(onsets),
        categories=categories,
        trial_types=trial_types,
        subject=subject,
        sampling_rate=rate / decimation,
        duration=duration,
    )


DATA_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data" / "ds003517"
TASK = "ContinuousVideoGamePlay"
RUN = "02"


def paths_for(root: Path, subject: str) -> Tuple[Path, Path]:
    stem = f"{subject}_task-{TASK}_run-{RUN}"
    directory = root / subject / "eeg"
    return directory / f"{stem}_eeg.set", directory / f"{stem}_events.tsv"


def available_subjects(root: Path = DATA_ROOT) -> List[str]:
    if not root.exists():
        return []
    found = []
    for path in sorted(root.glob("sub-*")):
        set_path, events_path = paths_for(root, path.name)
        binary = set_path.with_suffix(".fdt")
        if set_path.exists() and events_path.exists() and binary.exists() and binary.stat().st_size:
            found.append(path.name)
    return found


def load_or_extract(root: Path, subject: str, force: bool = False) -> EpochSet:
    cached = root / "derivatives" / "gameplay_epochs" / f"{subject}_epochs.npz"
    if cached.exists() and not force:
        return EpochSet.load(cached)

    set_path, events_path = paths_for(root, subject)
    if not set_path.exists():
        raise FileNotFoundError(f"{subject} not downloaded")

    epochs = extract(set_path, events_path, subject)
    epochs.save(cached)
    return epochs
