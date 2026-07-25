"""Trial extraction from DENS.

One trial is a person watching one emotionally evocative video clip, with EEG
recorded continuously throughout. Attached to each trial:

* self-reported **valence, arousal, dominance** on 1-9 scales;
* an affective **quadrant** label (HVHA / HVLA / LVHA / LVLA);
* a named **emotion category**;
* the **timestamp of a mouse click** marking the moment the emotion was felt.

That click is what makes per-frame emotion decoding checkable rather than
merely plausible: a decoder that only reproduced the trial-average rating
would show no structure around it, and one that is genuinely tracking the
felt experience should.
"""

from __future__ import annotations

import ast
import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("mindscape.emotion.trials")

#: Event codes in the DENS events file.
STIMULUS_ONSET = "stm"
CLICK = "clic"
BASELINE_START = "base"
BASELINE_END = "bend"

#: Ratings are on a 1-9 scale; the midpoint splits high from low.
RATING_MIDPOINT = 5.0


@dataclass
class Trial:
    """One video viewing."""

    subject: str
    index: int
    stimulus: str
    #: Seconds into the recording where the clip started.
    onset: float
    duration: float

    valence: float
    arousal: float
    dominance: float
    liking: float
    familiarity: float
    relevance: float

    #: HVHA / HVLA / LVHA / LVLA, or "" when the subject did not report one.
    quadrant: str
    emotion: str
    #: Seconds into the clip where the subject marked feeling the emotion.
    click_times: List[float] = field(default_factory=list)

    @property
    def high_valence(self) -> bool:
        return self.valence >= RATING_MIDPOINT

    @property
    def high_arousal(self) -> bool:
        return self.arousal >= RATING_MIDPOINT

    @property
    def derived_quadrant(self) -> str:
        """Quadrant from the numeric ratings.

        Used in preference to the self-reported label, which subjects left
        blank on a good fraction of trials.
        """
        return ("HV" if self.high_valence else "LV") + ("HA" if self.high_arousal else "LA")


def _parse_float(value: str, default: float = float("nan")) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _parse_clicks(value: str) -> List[float]:
    """Click timestamps, stored as a stringified Python list."""
    text = (value or "").strip()
    if not text or text in {"n/a", "[]", "{}"}:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [float(m) for m in re.findall(r"\d+\.?\d*", text)]
    if isinstance(parsed, (int, float)):
        return [float(parsed)]
    if isinstance(parsed, (list, tuple)):
        return [float(v) for v in parsed if isinstance(v, (int, float))]
    return []


def _parse_emotion(value: str) -> str:
    """First named emotion from the nested dict the task wrote."""
    text = (value or "").strip()
    if not text or text in {"{}", "n/a"}:
        return ""
    match = re.search(r"u?['\"]([A-Za-z][A-Za-z /-]{2,}?):", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"u?['\"]([A-Za-z][A-Za-z /-]{2,})['\"]", text)
    return match.group(1).strip() if match else ""


def _normalise_stimulus(name: str) -> str:
    return re.sub(r"\.(mp4|m4v|avi|mov)$", "", (name or "").strip(), flags=re.I)


def load_trials(root: Path, subject: str) -> List[Trial]:
    """Every video-viewing trial for one subject."""
    events_path = root / subject / "eeg" / f"{subject}_task-emotion_events.tsv"
    beh_path = root / subject / "beh" / f"{subject}_task-Emotion_beh.tsv"

    if not events_path.exists():
        raise FileNotFoundError(f"missing events for {subject}")

    with open(events_path) as handle:
        events = list(csv.DictReader(handle, delimiter="\t"))

    ratings: List[dict] = []
    if beh_path.exists():
        with open(beh_path) as handle:
            ratings = list(csv.DictReader(handle, delimiter="\t"))

    # Stimulus onsets in order. Onsets in this file are milliseconds.
    stimuli = [e for e in events if e.get("trial_type") == STIMULUS_ONSET]
    clicks = [e for e in events if e.get("trial_type") == CLICK]

    trials: List[Trial] = []

    for index, event in enumerate(stimuli):
        onset_ms = _parse_float(event.get("onset", ""), 0.0)
        onset = onset_ms / 1000.0

        # A trial runs until the next stimulus or the first rating prompt
        # after it, whichever comes first.
        following = [
            _parse_float(e.get("onset", ""), 0.0) / 1000.0
            for e in events
            if _parse_float(e.get("onset", ""), 0.0) / 1000.0 > onset
            and e.get("trial_type") in {"vlnc", STIMULUS_ONSET}
        ]
        end = min(following) if following else onset + 60.0
        duration = max(1.0, end - onset)

        row = ratings[index] if index < len(ratings) else {}

        label = _normalise_stimulus(event.get("label", ""))
        stimulus = _normalise_stimulus(row.get("stimuliName", "")) or label

        # Clicks belonging to this trial, expressed relative to clip start.
        within = [
            _parse_float(c.get("onset", ""), 0.0) / 1000.0 - onset
            for c in clicks
            if onset <= _parse_float(c.get("onset", ""), 0.0) / 1000.0 <= end
        ]
        if not within:
            within = [t for t in _parse_clicks(row.get("MouseClick", "")) if 0 <= t <= duration]

        trials.append(
            Trial(
                subject=subject,
                index=index,
                stimulus=stimulus,
                onset=onset,
                duration=duration,
                valence=_parse_float(row.get("valence", "")),
                arousal=_parse_float(row.get("arousal", "")),
                dominance=_parse_float(row.get("dominance", "")),
                liking=_parse_float(row.get("liking", "")),
                familiarity=_parse_float(row.get("familiarity", "")),
                relevance=_parse_float(row.get("relevance", "")),
                quadrant=(row.get("Quadrant", "") or "").strip(),
                emotion=_parse_emotion(row.get("emotionCateg", "")),
                click_times=sorted(within),
            )
        )

    usable = [t for t in trials if np.isfinite(t.valence) and np.isfinite(t.arousal)]
    log.info(
        "%s: %d trials (%d rated), %d with click markers",
        subject,
        len(trials),
        len(usable),
        sum(1 for t in usable if t.click_times),
    )
    return usable


def load_baseline(root: Path, subject: str) -> Optional[Tuple[float, float]]:
    """The resting-baseline window, as ``(onset, duration)`` in seconds.

    DENS records a rest period before the clips begin. Expressing affective
    features relative to it is what makes them comparable across people:
    absolute EEG power differs more between two subjects' skulls than between
    two emotions in the same subject, so an un-baselined cross-subject model
    mostly learns who is wearing the cap.
    """
    events_path = root / subject / "eeg" / f"{subject}_task-emotion_events.tsv"
    if not events_path.exists():
        return None

    with open(events_path) as handle:
        events = list(csv.DictReader(handle, delimiter="\t"))

    starts = [e for e in events if e.get("trial_type") == BASELINE_START]
    ends = [e for e in events if e.get("trial_type") == BASELINE_END]
    if not starts:
        return None

    onset = _parse_float(starts[0].get("onset", ""), 0.0) / 1000.0
    if ends:
        end = _parse_float(ends[0].get("onset", ""), 0.0) / 1000.0
        duration = max(2.0, end - onset)
    else:
        duration = 20.0
    return onset, duration


def available_subjects(root: Path) -> List[str]:
    """Subjects whose signal and behaviour files are both present."""
    if not root.exists():
        return []
    found = []
    for path in sorted(root.glob("sub-*")):
        binary = path / "eeg" / f"{path.name}_task-Emotion_eeg.fdt"
        header = path / "eeg" / f"{path.name}_task-Emotion_eeg.set"
        beh = path / "beh" / f"{path.name}_task-Emotion_beh.tsv"
        if header.exists() and binary.exists() and binary.stat().st_size > 0 and beh.exists():
            found.append(path.name)
    return found
