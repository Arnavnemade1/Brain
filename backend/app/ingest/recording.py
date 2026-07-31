"""One representation for a recording, whatever it arrived as.

Three readers already exist in this codebase — BrainVision, EEGLAB and EDF+ —
written independently against three datasets, each returning its own shape.
That was the right order to build them in and the wrong shape to keep: a
question asked across datasets has to be asked of one type, not three.

This wraps them without replacing them. The readers stay where they are and
keep their format-specific detail; :func:`load` dispatches on extension and
returns the common form. Nothing here re-parses bytes.

What the common form deliberately carries:

* **Microvolts, always.** Each reader already applies its format's own scaling
  (EDF's digital-to-physical affine, BrainVision's per-channel resolution), so
  the invariant is that a value here is a microvolt.
* **Lazy samples.** Two of the three readers memory-map their files precisely
  because recordings run to gigabytes, and an eager ``data`` array would throw
  that away. :class:`Recording` holds a window accessor and materialises only
  when asked, so building an inventory across hundreds of files reads headers
  and nothing else.
* **Its own sampling rate, unresampled.** Resampling is a decision that belongs
  to the analysis, not the loader — a 160 Hz recording and a 1000 Hz one differ
  in what they can resolve, and hiding that behind a uniform rate would let a
  gamma-band result be computed from a recording that cannot represent gamma.
  :meth:`Recording.usable_band` reports the honest ceiling instead.
* **Provenance.** Dataset, subject and source path travel with the data, so a
  pooled result can always be traced back to which recordings produced it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .montage import Projection, project

log = logging.getLogger("mindscape.ingest")


@dataclass
class Event:
    """One annotation, in seconds from the start of the recording."""

    onset: float
    duration: float
    label: str


@dataclass
class Recording:
    """Continuous EEG in a form every dataset can be expressed in."""

    channel_names: List[str]
    sampling_rate: float
    n_samples: int
    #: (start, length) -> (channels, length) float32 microvolts.
    reader: Callable[[int, int], np.ndarray]

    events: List[Event] = field(default_factory=list)
    dataset: str = ""
    subject: str = ""
    source: Optional[Path] = None

    _materialised: Optional[np.ndarray] = None

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)

    @property
    def duration(self) -> float:
        return self.n_samples / self.sampling_rate

    def window(self, start: int, length: int) -> np.ndarray:
        """Samples in microvolts, without loading the rest of the file."""
        return self.reader(start, length)

    @property
    def data(self) -> np.ndarray:
        """The whole recording. Materialised once, then cached.

        Deliberately a property rather than a field: reaching for it is a
        decision to hold the entire recording in memory, and it should read
        like one at the call site.
        """
        if self._materialised is None:
            self._materialised = self.reader(0, self.n_samples)
        return self._materialised

    def usable_band(self, margin: float = 0.8) -> float:
        """Highest frequency this recording can honestly support, in Hz.

        Nyquist with a margin, because anti-alias filters roll off before the
        theoretical limit. A 160 Hz recording tops out around 64 Hz, so asking
        it for 70 Hz gamma returns filter artefact rather than signal.
        """
        return self.sampling_rate / 2.0 * margin

    def projection(self) -> Projection:
        return project(self.channel_names)

    def to_standard(self, start: int = 0, length: Optional[int] = None):
        """Re-express a window as the 19-channel 10-20 space.

        Positions this montage lacks come back as NaN rows rather than zeros or
        interpolations. Zeros would read as a real, silent electrode and would
        quietly pull any average toward zero; NaN forces a caller to decide
        what to do about a missing site.
        """
        span = self.n_samples if length is None else length
        chunk = self.window(start, span)
        mapping = self.projection()
        out = np.full((len(mapping.indices), span), np.nan, dtype=np.float32)
        for row, index in enumerate(mapping.indices):
            if index is not None:
                out[row] = chunk[index]
        return out, mapping


def _from_brainvision(path: Path, dataset: str, subject: str) -> Recording:
    from ..realdata.brainvision import BrainVisionRecording

    native = BrainVisionRecording(path)
    return Recording(
        channel_names=list(native.channel_names),
        sampling_rate=float(native.sampling_rate),
        n_samples=int(native.n_samples),
        reader=native.window,
        events=[],  # markers live in a sidecar .vmrk, read by the epoching layer
        dataset=dataset,
        subject=subject,
        source=path,
    )


def _from_eeglab(path: Path, dataset: str, subject: str) -> Recording:
    from ..emotion import eeglab

    native = eeglab.read(path)
    return Recording(
        channel_names=list(native.channel_names),
        sampling_rate=float(native.sampling_rate),
        n_samples=int(native.n_samples),
        reader=native.window,
        events=[],  # DENS keeps its events in BIDS sidecars, not the .set
        dataset=dataset,
        subject=subject,
        source=path,
    )


def _from_edf(path: Path, dataset: str, subject: str) -> Recording:
    from ..motor import edf

    native = edf.read_edf(path)
    # EDF is read eagerly by its own reader; wrap the array in the same
    # accessor shape so callers cannot tell the difference.
    samples = np.asarray(native.data, dtype=np.float32)

    def read(start: int, length: int) -> np.ndarray:
        out = np.zeros((samples.shape[0], length), dtype=np.float32)
        lo, hi = max(0, start), min(samples.shape[1], start + length)
        if hi > lo:
            out[:, lo - start : lo - start + (hi - lo)] = samples[:, lo:hi]
        return out

    return Recording(
        channel_names=list(native.channel_names),
        sampling_rate=float(native.sampling_rate),
        n_samples=int(samples.shape[1]),
        reader=read,
        events=[
            Event(onset=a.onset, duration=a.duration, label=a.label)
            for a in native.annotations
        ],
        dataset=dataset,
        subject=subject,
        source=path,
    )


#: Extension to reader. Keyed by suffix because BIDS names files by modality
#: and task, not by format, so the extension is the only reliable signal.
READERS = {
    ".vhdr": _from_brainvision,
    ".set": _from_eeglab,
    ".edf": _from_edf,
    ".bdf": _from_edf,
}


def load(path: Path, dataset: str = "", subject: str = "") -> Recording:
    """Read any supported recording into the common form."""
    path = Path(path)
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(
            f"no reader for {path.suffix!r}; supported: {', '.join(sorted(READERS))}"
        )
    return reader(path, dataset or path.parent.name, subject or path.stem)


def summarise(recording: Recording) -> Dict[str, object]:
    """A one-line description, for the manifest."""
    mapping = recording.projection()
    return {
        "dataset": recording.dataset,
        "subject": recording.subject,
        "channels": recording.n_channels,
        "samplingRate": recording.sampling_rate,
        "durationSeconds": round(recording.duration, 1),
        "usableBandHz": round(recording.usable_band(), 1),
        "standardCoverage": round(mapping.coverage, 3),
        "matched": mapping.matched,
        "missing": mapping.missing,
        "events": len(recording.events),
    }
