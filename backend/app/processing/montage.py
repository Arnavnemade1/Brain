"""Electrode montage for visual reconstruction.

Weighted toward posterior coverage. The visual hierarchy sits occipital and
occipito-temporal, and that is where essentially all recoverable information
about visual input appears at the scalp. Frontal sites are retained mostly as
an artifact reference — blinks and eye movement are largest there, which makes
them useful for detecting contamination even though they carry little visual
signal themselves.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

CHANNELS: List[str] = [
    "Fp1",
    "Fp2",
    "F3",
    "Fz",
    "F4",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "PO7",
    "PO3",
    "POz",
    "PO4",
    "PO8",
    "O1",
    "Oz",
    "O2",
]

#: Normalised scalp positions, x ∈ [-1, 1] (left→right), y ∈ [-1, 1] (back→front).
POSITIONS: Dict[str, Tuple[float, float]] = {
    "Fp1": (-0.28, 0.92),
    "Fp2": (0.28, 0.92),
    "F3": (-0.40, 0.58),
    "Fz": (0.00, 0.60),
    "F4": (0.40, 0.58),
    "T7": (-0.96, 0.00),
    "C3": (-0.48, 0.00),
    "Cz": (0.00, 0.00),
    "C4": (0.48, 0.00),
    "T8": (0.96, 0.00),
    "P7": (-0.78, -0.50),
    "P3": (-0.42, -0.56),
    "Pz": (0.00, -0.58),
    "P4": (0.42, -0.56),
    "P8": (0.78, -0.50),
    "PO7": (-0.60, -0.74),
    "PO3": (-0.30, -0.78),
    "POz": (0.00, -0.80),
    "PO4": (0.30, -0.78),
    "PO8": (0.60, -0.74),
    "O1": (-0.26, -0.94),
    "Oz": (0.00, -0.97),
    "O2": (0.26, -0.94),
}

INDEX: Dict[str, int] = {name: i for i, name in enumerate(CHANNELS)}


def indices(*labels: str) -> List[int]:
    return [INDEX[label] for label in labels if label in INDEX]


# --- Functional groups ------------------------------------------------------

#: Primary visual cortex projection. Luminance response is largest here.
OCCIPITAL = indices("O1", "Oz", "O2")

#: Extended occipital pole, including parieto-occipital sites.
OCCIPITO_PARIETAL = indices("PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2")

#: Lateral occipito-temporal — the motion-sensitive region (human MT/V5).
MOTION_SENSITIVE = indices("PO7", "PO8", "P7", "P8")

#: Parietal, where visual alpha desynchronisation is measured.
PARIETAL = indices("P7", "P3", "Pz", "P4", "P8")

#: Posterior as a whole.
POSTERIOR = indices(
    "P7", "P3", "Pz", "P4", "P8", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"
)

#: Frontal — artifact reference, minimal visual content.
FRONTAL = indices("Fp1", "Fp2", "F3", "Fz", "F4")
CENTRAL = indices("C3", "Cz", "C4", "T7", "T8")

#: Hemisphere splits. Contralateral organisation of the visual field means the
#: left/right difference carries coarse horizontal layout.
POSTERIOR_LEFT = indices("P7", "P3", "PO7", "PO3", "O1")
POSTERIOR_RIGHT = indices("P8", "P4", "PO8", "PO4", "O2")

#: Dorsal/ventral split. The upper visual field projects below the calcarine
#: sulcus and vice versa, so this difference carries coarse vertical layout.
POSTERIOR_DORSAL = indices("P7", "P3", "Pz", "P4", "P8")
POSTERIOR_VENTRAL = indices("O1", "Oz", "O2")

REGIONS: Dict[str, List[int]] = {
    "frontal": FRONTAL,
    "central": CENTRAL,
    "parietal": PARIETAL,
    "occipito-parietal": indices("PO7", "PO3", "POz", "PO4", "PO8"),
    "occipital": OCCIPITAL,
}


def normalise_channel_names(names: List[str]) -> List[str]:
    """Map arbitrary uploaded channel labels onto the canonical montage."""
    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
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
