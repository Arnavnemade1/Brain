"""10–20 electrode montage: labels, normalised scalp coordinates, region groups.

Coordinates are unit-circle positions (nasion up, +x right) that the client
projects directly onto the topography map.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

CHANNELS: List[str] = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T3",
    "C3",
    "Cz",
    "C4",
    "T4",
    "T5",
    "P3",
    "Pz",
    "P4",
    "T6",
    "O1",
    "O2",
]

#: Normalised scalp positions, x ∈ [-1, 1] (left→right), y ∈ [-1, 1] (back→front).
POSITIONS: Dict[str, Tuple[float, float]] = {
    "Fp1": (-0.28, 0.92),
    "Fp2": (0.28, 0.92),
    "F7": (-0.76, 0.55),
    "F3": (-0.40, 0.58),
    "Fz": (0.00, 0.60),
    "F4": (0.40, 0.58),
    "F8": (0.76, 0.55),
    "T3": (-0.96, 0.00),
    "C3": (-0.48, 0.00),
    "Cz": (0.00, 0.00),
    "C4": (0.48, 0.00),
    "T4": (0.96, 0.00),
    "T5": (-0.76, -0.55),
    "P3": (-0.40, -0.58),
    "Pz": (0.00, -0.60),
    "P4": (0.40, -0.58),
    "T6": (0.76, -0.55),
    "O1": (-0.28, -0.92),
    "O2": (0.28, -0.92),
}

INDEX: Dict[str, int] = {name: i for i, name in enumerate(CHANNELS)}


def indices(*labels: str) -> List[int]:
    return [INDEX[label] for label in labels if label in INDEX]


#: Region groups used by the feature and inference stages.
FRONTAL_LEFT = indices("Fp1", "F7", "F3")
FRONTAL_RIGHT = indices("Fp2", "F8", "F4")
FRONTAL = FRONTAL_LEFT + indices("Fz") + FRONTAL_RIGHT
CENTRAL = indices("C3", "Cz", "C4", "T3", "T4")
PARIETAL = indices("P3", "Pz", "P4", "T5", "T6")
OCCIPITAL = indices("O1", "O2")
POSTERIOR = PARIETAL + OCCIPITAL
TEMPORAL = indices("T3", "T4", "T5", "T6")

#: Lobe membership, for the region-activation readout in the console.
REGIONS: Dict[str, List[int]] = {
    "frontal": FRONTAL,
    "central": CENTRAL,
    "temporal": TEMPORAL,
    "parietal": PARIETAL,
    "occipital": OCCIPITAL,
}


def normalise_channel_names(names: List[str]) -> List[str]:
    """Map arbitrary uploaded channel labels onto the canonical montage.

    Uploaded recordings use inconsistent conventions (``EEG Fp1-REF``, ``FP1``,
    ``T7`` for ``T3``). Anything unrecognised keeps its original label and is
    simply excluded from region groups.
    """
    aliases = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
    resolved: List[str] = []

    for raw in names:
        token = raw.strip().upper()
        for prefix in ("EEG ", "EEG-"):
            if token.startswith(prefix):
                token = token[len(prefix) :]
        token = token.split("-")[0].strip()

        canonical = next((c for c in CHANNELS if c.upper() == token), None)
        if canonical is None and token in aliases:
            canonical = aliases[token]
        resolved.append(canonical if canonical else raw.strip())

    return resolved
