"""Procedural reference clips.

Every clip is generated deterministically from a seed, so the reference video
is byte-identical on every run. That matters more than realism: fidelity is
measured *against* these frames, and a stimulus that varied between runs would
make scores incomparable.

Clips are built to exercise the properties EEG can plausibly carry — luminance
swings, motion, scene cuts, spatial layout — and to differ from each other in
which of those dominate, so the evaluation shows where reconstruction succeeds
and where it does not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

#: Reconstruction grid. Small on purpose — see the `frames` pipeline stage.
#: Scalp EEG carries spatial *gist*; a larger grid would be mostly invention.
GRID_WIDTH = 32
GRID_HEIGHT = 18

#: Frames per second of the reference. Also the reconstruction rate.
FPS = 10.0


@dataclass
class Scene:
    """One continuous shot within a clip."""

    duration: float
    #: Renders a frame given time-within-scene and a normalised progress 0–1.
    render: Callable[[np.ndarray, float, float], None]
    label: str


@dataclass
class ClipSpec:
    id: str
    title: str
    description: str
    tags: List[str]
    scenes: List[Scene] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(scene.duration for scene in self.scenes)


# --------------------------------------------------------------------------- #
# Rendering primitives                                                        #
# --------------------------------------------------------------------------- #

_Y, _X = np.mgrid[0:GRID_HEIGHT, 0:GRID_WIDTH]
_NX = _X / (GRID_WIDTH - 1)
_NY = _Y / (GRID_HEIGHT - 1)


def _fill(frame: np.ndarray, color: Tuple[float, float, float]) -> None:
    frame[..., 0] = color[0]
    frame[..., 1] = color[1]
    frame[..., 2] = color[2]


def _disc(
    frame: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    color: Tuple[float, float, float],
    softness: float = 0.06,
) -> None:
    """Anti-aliased filled circle in normalised coordinates."""
    aspect = GRID_WIDTH / GRID_HEIGHT
    distance = np.sqrt(((_NX - cx) * aspect) ** 2 + (_NY - cy) ** 2)
    mask = np.clip((radius - distance) / max(softness, 1e-4), 0.0, 1.0)
    for channel in range(3):
        frame[..., channel] = frame[..., channel] * (1 - mask) + color[channel] * mask


def _bar(
    frame: np.ndarray,
    x0: float,
    x1: float,
    color: Tuple[float, float, float],
    softness: float = 0.02,
) -> None:
    """Vertical band between two normalised x positions."""
    left = np.clip((_NX - x0) / max(softness, 1e-4), 0.0, 1.0)
    right = np.clip((x1 - _NX) / max(softness, 1e-4), 0.0, 1.0)
    mask = left * right
    for channel in range(3):
        frame[..., channel] = frame[..., channel] * (1 - mask) + color[channel] * mask


def _gradient(
    frame: np.ndarray,
    top: Tuple[float, float, float],
    bottom: Tuple[float, float, float],
) -> None:
    for channel in range(3):
        frame[..., channel] = top[channel] + (bottom[channel] - top[channel]) * _NY


def _horizon(
    frame: np.ndarray,
    height: float,
    sky: Tuple[float, float, float],
    ground: Tuple[float, float, float],
) -> None:
    mask = np.clip((_NY - height) * 12.0, 0.0, 1.0)
    for channel in range(3):
        frame[..., channel] = sky[channel] * (1 - mask) + ground[channel] * mask


# --------------------------------------------------------------------------- #
# Scene renderers                                                             #
# --------------------------------------------------------------------------- #


def _scene_sunrise(frame: np.ndarray, t: float, progress: float) -> None:
    """Slow global luminance ramp — the easiest property to recover."""
    sky_top = (0.04 + 0.28 * progress, 0.05 + 0.30 * progress, 0.12 + 0.42 * progress)
    sky_low = (0.10 + 0.65 * progress, 0.08 + 0.42 * progress, 0.14 + 0.24 * progress)
    _gradient(frame, sky_top, sky_low)
    _horizon(frame, 0.72, sky_low, (0.05 + 0.12 * progress, 0.06 + 0.13 * progress, 0.08))
    sun_y = 0.78 - progress * 0.34
    _disc(frame, 0.62, sun_y, 0.06 + progress * 0.03, (1.0, 0.86 + 0.1 * progress, 0.62))


def _scene_traverse(frame: np.ndarray, t: float, progress: float) -> None:
    """Strong lateral motion — exercises motion-sensitive response."""
    _gradient(frame, (0.16, 0.19, 0.26), (0.08, 0.09, 0.13))
    _horizon(frame, 0.66, (0.14, 0.16, 0.22), (0.07, 0.08, 0.10))
    # Columns sweeping right to left at constant speed.
    for i in range(7):
        x = ((i / 7.0) - t * 0.22) % 1.2 - 0.1
        _bar(frame, x, x + 0.045, (0.55, 0.58, 0.66))
    _disc(frame, 0.5, 0.3, 0.05, (0.85, 0.88, 0.95))


def _scene_pulse(frame: np.ndarray, t: float, progress: float) -> None:
    """Rhythmic full-field flashes — strong, periodic evoked responses."""
    phase = 0.5 + 0.5 * math.sin(t * math.pi * 1.6)
    level = 0.08 + phase * 0.72
    _fill(frame, (level * 0.92, level * 0.96, level))
    _disc(frame, 0.5, 0.5, 0.16 + phase * 0.08, (level * 0.4, level * 0.5, level * 0.75))


def _scene_interior(frame: np.ndarray, t: float, progress: float) -> None:
    """Static, detailed, low motion — tests spatial layout recovery."""
    _gradient(frame, (0.20, 0.18, 0.16), (0.10, 0.09, 0.08))
    # A bright window on the left third.
    _bar(frame, 0.08, 0.30, (0.78, 0.80, 0.74))
    _bar(frame, 0.17, 0.21, (0.20, 0.19, 0.17), softness=0.01)
    # Darker mass on the right.
    _bar(frame, 0.62, 0.95, (0.13, 0.12, 0.12))
    _disc(frame, 0.78, 0.42, 0.055, (0.62, 0.55, 0.38))


def _scene_dark(frame: np.ndarray, t: float, progress: float) -> None:
    """Near-black with a faint moving point — near the noise floor."""
    _fill(frame, (0.03, 0.035, 0.05))
    x = 0.5 + 0.32 * math.sin(t * 0.55)
    y = 0.5 + 0.16 * math.cos(t * 0.4)
    _disc(frame, x, y, 0.035, (0.30, 0.34, 0.42), softness=0.09)


def _scene_field(frame: np.ndarray, t: float, progress: float) -> None:
    """Bright, open, slow drift — high luminance and low detail."""
    _gradient(frame, (0.42, 0.58, 0.78), (0.66, 0.74, 0.82))
    _horizon(frame, 0.58, (0.66, 0.74, 0.82), (0.34, 0.46, 0.24))
    for i in range(4):
        x = ((i / 4.0) + t * 0.045) % 1.15 - 0.08
        _disc(frame, x, 0.30, 0.05, (0.92, 0.93, 0.94), softness=0.1)


def _scene_corridor(frame: np.ndarray, t: float, progress: float) -> None:
    """Forward motion with expanding structure — radial optic flow."""
    _fill(frame, (0.06, 0.07, 0.09))
    for i in range(5):
        phase = ((i / 5.0) + t * 0.3) % 1.0
        size = 0.06 + phase * 0.5
        level = 0.16 + phase * 0.5
        _disc(frame, 0.5, 0.5, size, (level * 0.7, level * 0.78, level), softness=0.03)
        _disc(frame, 0.5, 0.5, size * 0.82, (0.06, 0.07, 0.09), softness=0.03)


def _scene_flare(frame: np.ndarray, t: float, progress: float) -> None:
    """A single bright object against dark — high contrast, low complexity."""
    _fill(frame, (0.05, 0.05, 0.07))
    radius = 0.10 + 0.06 * math.sin(t * 1.1)
    _disc(frame, 0.5, 0.48, radius, (0.95, 0.82, 0.45), softness=0.14)
    _disc(frame, 0.5, 0.48, radius * 0.5, (1.0, 0.95, 0.8), softness=0.05)


# --------------------------------------------------------------------------- #
# Clip library                                                                #
# --------------------------------------------------------------------------- #

CLIPS: Dict[str, ClipSpec] = {}


def _register(spec: ClipSpec) -> None:
    CLIPS[spec.id] = spec


_register(
    ClipSpec(
        id="daybreak",
        title="Daybreak",
        description=(
            "A slow sunrise, a bright open field, then a dim interior. Dominated by "
            "large luminance changes — the property EEG recovers most reliably."
        ),
        tags=["luminance", "slow", "3 scenes"],
        scenes=[
            Scene(8.0, _scene_sunrise, "sunrise"),
            Scene(7.0, _scene_field, "open field"),
            Scene(7.0, _scene_interior, "interior"),
        ],
    )
)

_register(
    ClipSpec(
        id="transit",
        title="Transit",
        description=(
            "Lateral movement past structures, then forward motion down a corridor. "
            "Built to test motion recovery rather than brightness."
        ),
        tags=["motion", "optic flow", "3 scenes"],
        scenes=[
            Scene(7.0, _scene_traverse, "lateral pass"),
            Scene(8.0, _scene_corridor, "forward motion"),
            Scene(6.0, _scene_traverse, "lateral pass"),
        ],
    )
)

_register(
    ClipSpec(
        id="signal_test",
        title="Signal Test",
        description=(
            "Rhythmic full-field flashes alternating with darkness. A deliberately "
            "easy case: strong periodic evoked responses with unambiguous boundaries."
        ),
        tags=["evoked response", "high contrast", "4 scenes"],
        scenes=[
            Scene(5.0, _scene_pulse, "flash sequence"),
            Scene(5.0, _scene_dark, "dark"),
            Scene(5.0, _scene_pulse, "flash sequence"),
            Scene(5.0, _scene_flare, "single source"),
        ],
    )
)

_register(
    ClipSpec(
        id="low_light",
        title="Low Light",
        description=(
            "Extended near-darkness with faint movement. A deliberately hard case — "
            "the visual response barely clears the noise floor."
        ),
        tags=["hard case", "low SNR", "2 scenes"],
        scenes=[
            Scene(11.0, _scene_dark, "near dark"),
            Scene(9.0, _scene_flare, "faint source"),
        ],
    )
)

_register(
    ClipSpec(
        id="cut_sequence",
        title="Cut Sequence",
        description=(
            "Six short shots with abrupt transitions between very different scenes. "
            "Tests scene-boundary detection above everything else."
        ),
        tags=["scene cuts", "fast", "6 scenes"],
        scenes=[
            Scene(3.5, _scene_field, "bright field"),
            Scene(3.5, _scene_dark, "dark"),
            Scene(3.5, _scene_interior, "interior"),
            Scene(3.5, _scene_flare, "bright source"),
            Scene(3.5, _scene_traverse, "motion"),
            Scene(3.5, _scene_sunrise, "sky"),
        ],
    )
)


def _scene_overcast(frame: np.ndarray, t: float, progress: float) -> None:
    """Flat mid-grey with minimal structure — mid-luminance anchor."""
    level = 0.38 + 0.05 * math.sin(t * 0.3)
    _gradient(frame, (level, level * 1.02, level * 1.08), (level * 0.88, level * 0.9, level * 0.94))
    _horizon(frame, 0.70, (level, level, level * 1.05), (level * 0.6, level * 0.62, level * 0.58))


def _scene_glare(frame: np.ndarray, t: float, progress: float) -> None:
    """Very bright, near-saturated — the top of the luminance range."""
    level = 0.72 + 0.16 * math.sin(t * 0.5)
    _fill(frame, (level, level * 0.99, level * 0.95))
    _disc(frame, 0.42, 0.44, 0.2, (min(1.0, level + 0.2),) * 3, softness=0.2)


def _scene_shapes(frame: np.ndarray, t: float, progress: float) -> None:
    """Multiple moving objects at mid brightness — mixed motion and layout."""
    _gradient(frame, (0.24, 0.26, 0.32), (0.16, 0.17, 0.20))
    for i in range(3):
        x = 0.5 + 0.34 * math.sin(t * (0.5 + i * 0.22) + i * 2.1)
        y = 0.5 + 0.22 * math.cos(t * (0.4 + i * 0.18) + i)
        shade = 0.45 + i * 0.18
        _disc(frame, x, y, 0.07 + i * 0.02, (shade, shade * 0.96, shade * 0.88))


def _scene_split(frame: np.ndarray, t: float, progress: float) -> None:
    """Strong left/right asymmetry — targets the lateral layout channel."""
    _fill(frame, (0.10, 0.11, 0.14))
    edge = 0.5 + 0.3 * math.sin(t * 0.35)
    _bar(frame, 0.0, edge, (0.62, 0.64, 0.66))


_register(
    ClipSpec(
        id="grey_day",
        title="Grey Day",
        description=(
            "Flat overcast light with a bright glare sequence. Anchors the middle "
            "and top of the luminance range, where most material sits."
        ),
        tags=["mid luminance", "low motion", "3 scenes"],
        scenes=[
            Scene(7.0, _scene_overcast, "overcast"),
            Scene(6.0, _scene_glare, "glare"),
            Scene(7.0, _scene_overcast, "overcast"),
        ],
    )
)

_register(
    ClipSpec(
        id="objects",
        title="Objects in Motion",
        description=(
            "Several objects moving independently against a mid-grey field, then a "
            "sweeping left/right split. Exercises motion and horizontal layout together."
        ),
        tags=["motion", "layout", "3 scenes"],
        scenes=[
            Scene(8.0, _scene_shapes, "moving objects"),
            Scene(6.0, _scene_split, "field split"),
            Scene(6.0, _scene_shapes, "moving objects"),
        ],
    )
)

_register(
    ClipSpec(
        id="range_sweep",
        title="Range Sweep",
        description=(
            "Steps through dark, mid and bright scenes in sequence. Built to span "
            "the full luminance range within a single recording."
        ),
        tags=["luminance range", "calibration", "5 scenes"],
        scenes=[
            Scene(4.5, _scene_dark, "dark"),
            Scene(4.5, _scene_interior, "dim interior"),
            Scene(4.5, _scene_overcast, "mid grey"),
            Scene(4.5, _scene_field, "bright field"),
            Scene(4.5, _scene_glare, "glare"),
        ],
    )
)


def get_clip(clip_id: str) -> ClipSpec:
    return CLIPS.get(clip_id, CLIPS["daybreak"])


def render_clip(clip_id: str) -> Tuple[np.ndarray, List[int]]:
    """Render a clip to ``(frames, scene_index_per_frame)``.

    ``frames`` is ``(n, GRID_HEIGHT, GRID_WIDTH, 3)`` float32 in 0–1.
    """
    spec = get_clip(clip_id)
    frames: List[np.ndarray] = []
    scene_indices: List[int] = []

    for scene_index, scene in enumerate(spec.scenes):
        count = max(1, int(round(scene.duration * FPS)))
        for i in range(count):
            local_t = i / FPS
            progress = i / max(1, count - 1)
            frame = np.zeros((GRID_HEIGHT, GRID_WIDTH, 3), dtype=np.float32)
            scene.render(frame, local_t, progress)
            np.clip(frame, 0.0, 1.0, out=frame)
            frames.append(frame)
            scene_indices.append(scene_index)

    return np.stack(frames), scene_indices
