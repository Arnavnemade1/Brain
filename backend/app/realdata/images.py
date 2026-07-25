"""Stimulus images and their visual embeddings.

Two representations of each of the 200 objects:

* a **semantic** label hierarchy parsed from the filename — animate/inanimate,
  one of eight categories, one of 52 concepts;
* a **visual** embedding of hand-computed image statistics.

The visual embedding is deliberately hand-computed rather than taken from a
pretrained network. A CNN embedding would be a stronger target, but it would
also make it impossible to say *which* image property the EEG carried: every
result would be "the network's layer-N features", which is not an answer.
These features are individually nameable, so a decoding result can be
attributed to colour, or luminance layout, or edge density specifically.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("mindscape.images")

#: Grid used for the coarse luminance layout feature.
LAYOUT_GRID = (4, 4)

#: Orientation bands for the edge-energy feature.
ORIENTATIONS = 4

_NAME_PATTERN = re.compile(
    r"^stim(?P<number>\d+)_(?P<level_a>[^_]+)_(?P<level_b>[^_]+)_"
    r"(?P<level_c>.+)_(?P<exemplar>\d+)\.png$"
)


@dataclass
class Stimulus:
    number: int
    filename: str
    level_a: str
    level_b: str
    level_c: str
    exemplar: int

    @property
    def label(self) -> str:
        return self.level_c.replace("_", " ")


def parse_name(filename: str) -> Optional[Stimulus]:
    match = _NAME_PATTERN.match(filename)
    if match is None:
        return None
    return Stimulus(
        number=int(match.group("number")),
        filename=filename,
        level_a=match.group("level_a"),
        level_b=match.group("level_b"),
        level_c=match.group("level_c"),
        exemplar=int(match.group("exemplar")),
    )


def load_catalogue(stimuli_dir: Path) -> Dict[int, Stimulus]:
    """Every object stimulus, keyed by its 1-based number."""
    catalogue: Dict[int, Stimulus] = {}
    for path in sorted(stimuli_dir.glob("stim*.png")):
        stimulus = parse_name(path.name)
        # Targets are the fixation task, not objects; they have a different
        # naming shape and are excluded by the pattern.
        if stimulus is not None and stimulus.level_a in ("animate", "inanimate"):
            catalogue[stimulus.number] = stimulus
    return catalogue


# --------------------------------------------------------------------------- #
# Visual features                                                             #
# --------------------------------------------------------------------------- #

FEATURE_NAMES: List[str] = (
    ["mean_r", "mean_g", "mean_b", "saturation", "luminance", "contrast"]
    + [f"layout_{i}" for i in range(LAYOUT_GRID[0] * LAYOUT_GRID[1])]
    + [f"edge_{i}" for i in range(ORIENTATIONS)]
    + ["edge_density", "coverage", "aspect", "central_mass", "spatial_freq"]
)


def _to_array(path: Path, size: int = 96) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as handle:
        # Stimuli are PNGs on a plain background, sometimes with alpha.
        image = handle.convert("RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def embed_image(path: Path) -> np.ndarray:
    """Visual embedding of one stimulus, in :data:`FEATURE_NAMES` order."""
    rgb = _to_array(path)
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

    features: List[float] = [
        float(rgb[..., 0].mean()),
        float(rgb[..., 1].mean()),
        float(rgb[..., 2].mean()),
        float((rgb.max(axis=-1) - rgb.min(axis=-1)).mean()),
        float(luma.mean()),
        float(luma.std()),
    ]

    # Coarse luminance layout.
    h, w = luma.shape
    rows = np.linspace(0, h, LAYOUT_GRID[0] + 1).astype(int)
    cols = np.linspace(0, w, LAYOUT_GRID[1] + 1).astype(int)
    for r in range(LAYOUT_GRID[0]):
        for c in range(LAYOUT_GRID[1]):
            block = luma[rows[r] : rows[r + 1], cols[c] : cols[c + 1]]
            features.append(float(block.mean()) if block.size else 0.0)

    # Oriented edge energy, from simple directional differences.
    gy, gx = np.gradient(luma)
    magnitude = np.hypot(gx, gy)
    angle = np.arctan2(gy, gx) % np.pi
    for i in range(ORIENTATIONS):
        lo = i * np.pi / ORIENTATIONS
        hi = (i + 1) * np.pi / ORIENTATIONS
        mask = (angle >= lo) & (angle < hi)
        features.append(float(magnitude[mask].sum() / magnitude.size) if mask.any() else 0.0)

    features.append(float(magnitude.mean()))

    # Object coverage: these are objects on a light background, so darker-
    # than-background pixels approximate the silhouette.
    background = float(np.median(luma))
    silhouette = luma < background - 0.06
    features.append(float(silhouette.mean()))

    if silhouette.any():
        ys, xs = np.nonzero(silhouette)
        height = ys.max() - ys.min() + 1
        width = xs.max() - xs.min() + 1
        features.append(float(width / max(height, 1)))
        centre = np.hypot(ys.mean() - h / 2, xs.mean() - w / 2) / (h / 2)
        features.append(float(centre))
    else:
        features.extend([1.0, 0.0])

    # Dominant spatial frequency from the radially-averaged power spectrum.
    spectrum = np.abs(np.fft.rfft2(luma - luma.mean()))
    fy = np.fft.fftfreq(spectrum.shape[0])[:, None]
    fx = np.fft.rfftfreq(luma.shape[1])[None, :]
    radius = np.hypot(fy, fx)
    weight = spectrum.sum()
    features.append(float((spectrum * radius).sum() / weight) if weight > 0 else 0.0)

    return np.array(features, dtype=np.float32)


@dataclass
class StimulusEmbeddings:
    """Visual and semantic representations of every stimulus."""

    numbers: np.ndarray
    #: (stimuli, features) standardised visual embedding.
    visual: np.ndarray
    catalogue: Dict[int, Stimulus]

    #: Semantic label arrays, aligned with `numbers`.
    level_a: List[str]
    level_b: List[str]
    level_c: List[str]

    def index_of(self, number: int) -> int:
        matches = np.nonzero(self.numbers == number)[0]
        if matches.size == 0:
            raise KeyError(f"stimulus {number} not in catalogue")
        return int(matches[0])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            numbers=self.numbers,
            visual=self.visual,
            filenames=np.array([self.catalogue[n].filename for n in self.numbers]),
            level_a=np.array(self.level_a),
            level_b=np.array(self.level_b),
            level_c=np.array(self.level_c),
        )

    @classmethod
    def load(cls, path: Path) -> "StimulusEmbeddings":
        blob = np.load(path, allow_pickle=False)
        numbers = blob["numbers"]
        catalogue = {}
        for number, filename in zip(numbers, blob["filenames"]):
            stimulus = parse_name(str(filename))
            if stimulus is not None:
                catalogue[int(number)] = stimulus
        return cls(
            numbers=numbers,
            visual=blob["visual"],
            catalogue=catalogue,
            level_a=[str(s) for s in blob["level_a"]],
            level_b=[str(s) for s in blob["level_b"]],
            level_c=[str(s) for s in blob["level_c"]],
        )


def build_embeddings(stimuli_dir: Path) -> StimulusEmbeddings:
    catalogue = load_catalogue(stimuli_dir)
    if not catalogue:
        raise FileNotFoundError(f"no stimuli found in {stimuli_dir}")

    numbers = np.array(sorted(catalogue), dtype=np.int16)
    vectors = np.stack([embed_image(stimuli_dir / catalogue[n].filename) for n in numbers])

    # Standardise so no single feature dominates a distance computation
    # purely because of its units.
    mean = vectors.mean(axis=0, keepdims=True)
    scale = vectors.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    visual = (vectors - mean) / scale

    log.info("embedded %d stimuli into %d features", len(numbers), visual.shape[1])

    return StimulusEmbeddings(
        numbers=numbers,
        visual=visual.astype(np.float32),
        catalogue=catalogue,
        level_a=[catalogue[n].level_a for n in numbers],
        level_b=[catalogue[n].level_b for n in numbers],
        level_c=[catalogue[n].level_c for n in numbers],
    )


def load_or_build(root: Path, force: bool = False) -> StimulusEmbeddings:
    cached = root / "derivatives" / "stimuli_embeddings.npz"
    if cached.exists() and not force:
        return StimulusEmbeddings.load(cached)
    embeddings = build_embeddings(root / "stimuli")
    embeddings.save(cached)
    return embeddings


def thumbnail(path: Path, size: int = 64) -> np.ndarray:
    """Small RGB array of a stimulus, for display in the replay UI."""
    return _to_array(path, size=size)
