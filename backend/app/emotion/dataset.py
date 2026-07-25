"""Assembling the DENS feature dataset.

Extracts and caches the affective feature sequence for every trial of every
downloaded subject. The cache matters: each recording is a 350 MB memory-map
and re-extracting on every experiment would dominate the runtime.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .decoder import TrialFeatures
from .eeglab import read
from .features import FEATURE_NAMES, extract_sequence, resolve_regions
from .trials import available_subjects, load_baseline, load_trials

log = logging.getLogger("mindscape.emotion.dataset")

DATA_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data" / "ds003751"


def cache_path(root: Path, subject: str) -> Path:
    return root / "derivatives" / "emotion_features" / f"{subject}_features.npz"


def extract_subject(root: Path, subject: str) -> List[TrialFeatures]:
    """Feature sequences for every trial of one subject."""
    trials = load_trials(root, subject)
    if not trials:
        return []

    set_path = root / subject / "eeg" / f"{subject}_task-Emotion_eeg.set"
    out: List[TrialFeatures] = []

    baseline_window = load_baseline(root, subject)

    with read(set_path) as recording:
        regions = resolve_regions(recording.positions, recording.eeg_channels)

        # Resting-baseline feature vector. Every trial is expressed as a
        # deviation from it, so a feature means "how far from this person's
        # own resting state" rather than an absolute level that is dominated
        # by their anatomy and electrode fit.
        baseline_vector = None
        if baseline_window is not None:
            b_onset, b_duration = baseline_window
            _, b_features = extract_sequence(recording, b_onset, b_duration, regions)
            if b_features.size:
                baseline_vector = b_features.mean(axis=0)
                log.info("%s: baseline from %.0f s of rest", subject, b_duration)

        for trial in trials:
            times, features = extract_sequence(
                recording, trial.onset, trial.duration, regions
            )
            if not features.size:
                continue
            if baseline_vector is not None:
                features = features - baseline_vector
            out.append(
                TrialFeatures(
                    subject=subject,
                    trial_index=trial.index,
                    stimulus=trial.stimulus,
                    times=times,
                    features=features,
                    valence=trial.valence,
                    arousal=trial.arousal,
                    quadrant=trial.derived_quadrant,
                    emotion=trial.emotion,
                    click_times=list(trial.click_times),
                    duration=trial.duration,
                )
            )

    log.info("%s: %d trials with features", subject, len(out))
    return out


def save(trials: List[TrialFeatures], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.array([t.features for t in trials], dtype=object),
        times=np.array([t.times for t in trials], dtype=object),
        meta=np.array(
            [
                (
                    t.subject,
                    t.trial_index,
                    t.stimulus,
                    t.valence,
                    t.arousal,
                    t.quadrant,
                    t.emotion,
                    ";".join(str(c) for c in t.click_times),
                    t.duration,
                )
                for t in trials
            ],
            dtype=object,
        ),
        allow_pickle=True,
    )


def load(path: Path) -> List[TrialFeatures]:
    blob = np.load(path, allow_pickle=True)
    out: List[TrialFeatures] = []
    for features, times, meta in zip(blob["features"], blob["times"], blob["meta"]):
        clicks = [float(c) for c in str(meta[7]).split(";") if c]
        out.append(
            TrialFeatures(
                subject=str(meta[0]),
                trial_index=int(meta[1]),
                stimulus=str(meta[2]),
                times=np.asarray(times, dtype=float),
                features=np.asarray(features, dtype=float),
                valence=float(meta[3]),
                arousal=float(meta[4]),
                quadrant=str(meta[5]),
                emotion=str(meta[6]),
                click_times=clicks,
                duration=float(meta[8]),
            )
        )
    return out


def load_all(
    root: Path = DATA_ROOT, subjects: Optional[List[str]] = None, force: bool = False
) -> List[TrialFeatures]:
    """Every trial from every available subject, extracting and caching."""
    names = subjects or available_subjects(root)
    everything: List[TrialFeatures] = []

    for subject in names:
        cached = cache_path(root, subject)
        if cached.exists() and not force:
            everything.extend(load(cached))
            continue
        trials = extract_subject(root, subject)
        if trials:
            save(trials, cached)
            everything.extend(trials)

    log.info(
        "%d trials from %d subject(s), %d feature windows",
        len(everything),
        len({t.subject for t in everything}),
        sum(t.features.shape[0] for t in everything),
    )
    return everything
