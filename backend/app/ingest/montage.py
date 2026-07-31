"""A common electrode space, so recordings from different labs can be compared.

The obstacle to combining EEG datasets is not file format — that is a morning's
work — it is that the recordings do not measure the same places. This project
already holds a 63-channel BrainVision cap, a 129-channel EGI net whose sensors
are named E1..E128 and carry no anatomical meaning, and a 64-channel BCI2000
montage. Nothing lines up.

The honest common denominator is the **international 10-20 system**: nineteen
positions defined by fractions of head circumference rather than by hardware,
present in essentially every montage since 1958, and the set most consumer
headsets sample from. Projecting onto it throws away resolution, and that is
the point — a shared 19-channel space in which a result can be tested for
replication is worth more than a high-resolution one in which it cannot.

Two things this deliberately does **not** do:

* **Interpolate missing positions.** A cap without T7 does not get a T7
  invented from its neighbours. Interpolation would let a channel that was
  never recorded carry weight in a decoder, and the resulting number would be
  partly about the interpolation kernel. Missing is reported as missing.
* **Rescale across datasets.** Amplitudes differ with amplifier gain and
  reference; the fix is per-recording standardisation at feature time, which
  the decoders already do, not a fudge factor here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

#: The nineteen 10-20 positions, in a fixed order so feature vectors from
#: different datasets are directly comparable index by index.
STANDARD_1020: Tuple[str, ...] = (
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
)

#: Older nomenclature and common spelling variants. T3/T4/T5/T6 were renamed
#: T7/T8/P7/P8 in 1991 and both are still in circulation; a dataset using the
#: old names would otherwise silently lose four channels.
ALIASES: Dict[str, str] = {
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
    "FP1": "Fp1", "FP2": "Fp2",
    "FPZ": "Fpz", "FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "OZ": "Oz",
}

#: EGI's geodesic nets label sensors E1..E128 with no anatomical meaning. This
#: maps the 10-20 equivalents for the 128-channel net, which is what DENS uses.
#: Taken from the manufacturer's published equivalence table.
EGI_128: Dict[str, str] = {
    "E22": "Fp1", "E9": "Fp2",
    "E33": "F7", "E24": "F3", "E11": "Fz", "E124": "F4", "E122": "F8",
    "E45": "T7", "E36": "C3", "Cz": "Cz", "E104": "C4", "E108": "T8",
    "E58": "P7", "E52": "P3", "E62": "Pz", "E92": "P4", "E96": "P8",
    "E70": "O1", "E83": "O2",
}


def canonical(name: str) -> str:
    """Normalise one channel label to 10-20 naming where possible."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", str(name)).strip()
    if not cleaned:
        return ""

    # BrainVision and BCI2000 both pad labels with dots ("Fc5.", "C3..").
    upper = cleaned.upper()
    if upper in ALIASES:
        return ALIASES[upper]

    # Title-case the letters, keep the digits: "FC5" -> "Fc5", "c3" -> "C3".
    match = re.match(r"^([A-Za-z]+)(\d*)$", cleaned)
    if not match:
        return cleaned
    letters, digits = match.groups()
    if letters.upper() == "E" and digits:
        return cleaned.upper()  # EGI sensor, resolved separately
    return letters.capitalize() + digits


@dataclass
class Projection:
    """How one recording's channels map onto the shared space."""

    #: Index into the recording's channels for each of STANDARD_1020, or None.
    indices: List[Optional[int]]
    matched: List[str]
    missing: List[str]

    @property
    def coverage(self) -> float:
        return len(self.matched) / len(STANDARD_1020)

    def describe(self) -> str:
        return (
            f"{len(self.matched)}/{len(STANDARD_1020)} 10-20 positions "
            f"({self.coverage * 100:.0f}%)"
            + (f", missing {', '.join(self.missing)}" if self.missing else "")
        )


def project(channel_names: Sequence[str]) -> Projection:
    """Locate the 10-20 positions within a recording's channel list."""
    lookup: Dict[str, int] = {}
    for index, raw in enumerate(channel_names):
        name = canonical(raw)
        if name.startswith("E") and name in EGI_128:
            name = EGI_128[name]
        # First occurrence wins; duplicates in a montage are a labelling error
        # and taking the later one would silently pick the wrong sensor.
        lookup.setdefault(name, index)

    indices: List[Optional[int]] = []
    matched: List[str] = []
    missing: List[str] = []
    for position in STANDARD_1020:
        index = lookup.get(position)
        indices.append(index)
        (matched if index is not None else missing).append(position)

    return Projection(indices=indices, matched=matched, missing=missing)
