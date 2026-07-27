"""A real-world terrain renderer.

The existing background loop draws layered silhouettes — five stacked curves,
convincing enough as wallpaper because nothing in it has to be consistent.
This is a different thing: an actual heightfield, viewed through a camera, ray
marched until the ground is hit and then shaded. Terrain rendered this way has
real parallax, real occlusion, and a horizon that behaves, because the geometry
exists rather than being suggested.

Everything is numpy over flat ray arrays. No GPU, no dependency beyond what is
already here.

What makes rendered terrain read as a place rather than as a grey lump, in
rough order of how much each one matters:

1. **Atmospheric perspective.** Distance fades toward the sky colour. Without
   it there is no depth cue at all and the mountains look like cardboard.
2. **Ridged noise.** Ordinary fractal noise makes rolling hills; folding it
   about zero makes ridgelines and valleys, which is what mountains have.
3. **Materials by slope and altitude.** Snow settles where it is high and
   flat, rock shows where it is steep, grass fills the rest.
4. **Shadows.** A short march toward the sun. Cheap, and the single thing
   that makes the relief legible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Noise                                                                       #
# --------------------------------------------------------------------------- #


def _hash2(ix: np.ndarray, iz: np.ndarray) -> np.ndarray:
    """Deterministic pseudo-random value in 0-1 for integer lattice points."""
    n = (ix.astype(np.int64) * 374761393 + iz.astype(np.int64) * 668265263) & 0x7FFFFFFF
    n = (n ^ (n >> 13)) * 1274126177 & 0x7FFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFFFF).astype(np.float32) / float(0xFFFFFF)


def _value_noise(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Smoothly interpolated lattice noise in 0-1."""
    ix = np.floor(x).astype(np.int64)
    iz = np.floor(z).astype(np.int64)
    fx = (x - ix).astype(np.float32)
    fz = (z - iz).astype(np.float32)

    # Smoothstep, so the lattice does not show up as visible creases.
    ux = fx * fx * (3.0 - 2.0 * fx)
    uz = fz * fz * (3.0 - 2.0 * fz)

    a = _hash2(ix, iz)
    b = _hash2(ix + 1, iz)
    c = _hash2(ix, iz + 1)
    d = _hash2(ix + 1, iz + 1)

    return (a * (1 - ux) + b * ux) * (1 - uz) + (c * (1 - ux) + d * ux) * uz


#: Relief retained where the elevation envelope is at its lowest. Zero gives
#: mirror-flat lowlands, which look like a rendering failure rather than like
#: a plain.
BASE_RELIEF = 0.22


@dataclass
class Terrain:
    """A fractal heightfield."""

    #: Vertical scale in world units.
    amplitude: float = 260.0
    #: Horizontal scale: larger means broader landforms.
    wavelength: float = 900.0
    octaves: int = 9
    #: How ridged the landforms are. 0 gives rolling hills, 1 sharp ridgelines.
    ridged: float = 0.85
    seed: float = 0.0

    def height(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Ground height at world coordinates.

        A **ridged multifractal**, not a plain sum of octaves. The difference
        is the ``weight`` term: each octave is attenuated by how high the
        previous one stood, so detail accumulates along ridgelines and is
        suppressed in valleys. Summing octaves at uniform strength — which the
        first version did — gives ground that is equally rough everywhere, and
        equally rough everywhere reads as noise rather than as landscape.

        A low-frequency envelope on top decides where there are mountains at
        all, so the world has lowlands and ranges instead of one endless
        massif.
        """
        px = (x + self.seed * 137.0).astype(np.float32)
        pz = (z + self.seed * 311.0).astype(np.float32)

        # Where the land rises at all. Broad, and biased so that a good share
        # of the world stays low enough to hold water.
        envelope = _value_noise(px / (self.wavelength * 7.0), pz / (self.wavelength * 7.0))
        envelope = np.clip((envelope - 0.16) / 0.68, 0.0, 1.0)
        envelope = envelope * envelope * (3.0 - 2.0 * envelope)
        # A floor, so the lowlands still undulate instead of going billiard
        # flat wherever the envelope happens to bottom out.
        envelope = BASE_RELIEF + (1.0 - BASE_RELIEF) * envelope

        total = np.zeros_like(px)
        normaliser = 0.0
        amplitude = 1.0
        frequency = 1.0 / self.wavelength
        weight = np.ones_like(px)

        angle = 0.7
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        qx, qz = px, pz

        for _ in range(self.octaves):
            sample = _value_noise(qx * frequency, qz * frequency)
            ridge = 1.0 - np.abs(sample * 2.0 - 1.0)
            ridge = ridge * ridge
            ridge = sample * (1 - self.ridged) + ridge * self.ridged

            total += ridge * amplitude * weight
            normaliser += amplitude

            # Carry the ridge forward: the next octave bites hardest where
            # this one is already high. The constant term keeps fine detail
            # alive in the valleys instead of sanding them smooth.
            weight = np.clip(0.38 + ridge * 1.25, 0.0, 1.0)

            qx, qz = qx * cos_a - qz * sin_a, qx * sin_a + qz * cos_a
            frequency *= 2.03
            amplitude *= 0.5

        return (total / max(normaliser, 1e-6)) * self.amplitude * envelope

    def summit(self, extent: float = 5000.0, samples: int = 64) -> float:
        """Highest ground within view, sampled.

        Fractal noise never attains its nominal amplitude, so anything keyed
        to `amplitude` — the snow line above all — lands out of reach.
        """
        axis = np.linspace(-extent, extent, samples, dtype=np.float32)
        gx, gz = np.meshgrid(axis, axis)
        return float(self.height(gx.ravel(), gz.ravel()).max())


@dataclass
class Camera:
    position: Tuple[float, float, float] = (0.0, 190.0, 0.0)
    yaw: float = 0.0
    pitch: float = -0.06
    #: Vertical field of view in radians.
    fov: float = 1.05

    def grounded(self, terrain: "Terrain", clearance: float = 26.0) -> "Camera":
        """Same camera, lifted to sit just above the ground beneath it.

        A viewpoint floating at a fixed altitude ends up either buried in a
        hillside or high above everything, and neither gives a foreground.
        Standing on the terrain is what puts near geometry in the lower frame.
        """
        x = np.array([self.position[0]], dtype=np.float32)
        z = np.array([self.position[2]], dtype=np.float32)
        ground = float(terrain.height(x, z)[0])
        return Camera(
            position=(self.position[0], ground + clearance, self.position[2]),
            yaw=self.yaw,
            pitch=self.pitch,
            fov=self.fov,
        )


@dataclass
class Sky:
    #: Sun direction, normalised on use. Kept well above the horizon: at a
    #: low elevation every upward-facing surface receives the same grazing
    #: light and the relief disappears into flat colour.
    sun: Tuple[float, float, float] = (-0.48, 0.46, 0.75)
    zenith: Tuple[float, float, float] = (0.16, 0.30, 0.52)
    horizon: Tuple[float, float, float] = (0.62, 0.68, 0.74)
    sun_tint: Tuple[float, float, float] = (1.00, 0.86, 0.66)
    #: 0 is night, 1 is full day. Scales everything the sun contributes.
    daylight: float = 1.0
    haze: float = 1.0


@dataclass
class Water:
    level: float = 96.0
    colour: Tuple[float, float, float] = (0.10, 0.20, 0.26)
    enabled: bool = True


def _normalise(vector: Sequence[float]) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _rays(width: int, height: int, camera: Camera) -> np.ndarray:
    """Unit direction for every pixel, shaped (pixels, 3)."""
    aspect = width / height
    half = math.tan(camera.fov * 0.5)

    sx = (np.arange(width, dtype=np.float32) + 0.5) / width * 2.0 - 1.0
    sy = 1.0 - (np.arange(height, dtype=np.float32) + 0.5) / height * 2.0
    gx, gy = np.meshgrid(sx * half * aspect, sy * half)

    direction = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size, np.float32)], axis=1)

    cp, sp = math.cos(camera.pitch), math.sin(camera.pitch)
    cy, sy_ = math.cos(camera.yaw), math.sin(camera.yaw)
    pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    yaw = np.array([[cy, 0, sy_], [0, 1, 0], [-sy_, 0, cy]], dtype=np.float32)

    direction = direction @ pitch.T @ yaw.T
    return direction / np.linalg.norm(direction, axis=1, keepdims=True)


def _march(
    origin: np.ndarray,
    direction: np.ndarray,
    terrain: Terrain,
    far: float,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Distance along each ray to the ground, and whether it hit.

    Step size grows geometrically with distance, because terrain detail
    shrinks in screen space as it recedes and uniform steps spend the whole
    budget resolving far geometry a pixel wide.

    The growth rate is **derived from the range and the step count** rather
    than fixed. A hardcoded rate silently under-reaches: at 1.4% a ray needs
    over four hundred steps to cross twenty kilometres, so with a budget of
    two hundred it simply stops early and the distance beyond is rendered as
    empty sky, with nothing to indicate anything was missed.
    """
    near = 1.0
    floor = 1.6
    # Solve for the geometric rate that just reaches `far` in `steps`, after
    # the near region where the constant floor dominates.
    knee = floor / 0.014
    remaining = max(steps - int(knee / floor), 12)
    growth = max((far / max(knee, 1.0)) ** (1.0 / remaining) - 1.0, 0.010)

    n = direction.shape[0]
    distance = np.full(n, near, dtype=np.float32)
    hit = np.zeros(n, dtype=bool)
    hit_at = np.full(n, far, dtype=np.float32)

    # Rays pointing up above the highest ground can never hit; excluding them
    # early keeps the march off most of the sky.
    live = ~((direction[:, 1] > 0) & (origin[1] > terrain.amplitude))

    for _ in range(steps):
        if not live.any():
            break
        idx = np.nonzero(live)[0]
        d = distance[idx]
        point = origin[None, :] + direction[idx] * d[:, None]
        ground = terrain.height(point[:, 0], point[:, 2])
        below = point[:, 1] < ground

        newly = idx[below]
        hit[newly] = True
        hit_at[newly] = d[below]
        live[newly] = False

        # Advance proportionally to distance, with a floor so near geometry
        # is not skipped.
        distance[idx] = d + np.maximum(d * growth, floor)
        live[idx[distance[idx] > far]] = False

    return hit_at, hit


def _refine(
    origin: np.ndarray,
    direction: np.ndarray,
    terrain: Terrain,
    distance: np.ndarray,
    hit: np.ndarray,
    iterations: int = 10,
) -> np.ndarray:
    """Bisect between the last miss and the first hit.

    The coarse march overshoots into the ground by up to a step, which shows
    up as a stair-stepped silhouette along every ridgeline.
    """
    idx = np.nonzero(hit)[0]
    if idx.size == 0:
        return distance

    # Widen the bracket generously: the caller's march may have used a larger
    # growth rate than this assumes, and a bracket that does not contain the
    # surface converges to the wrong side of it.
    step_back = np.maximum(distance[idx] * 0.06, 2.0)
    near = np.maximum(distance[idx] - step_back, 0.5)
    far = distance[idx]

    for _ in range(iterations):
        mid = 0.5 * (near + far)
        point = origin[None, :] + direction[idx] * mid[:, None]
        inside = point[:, 1] < terrain.height(point[:, 0], point[:, 2])
        far = np.where(inside, mid, far)
        near = np.where(inside, near, mid)

    distance = distance.copy()
    distance[idx] = far
    return distance


def _shadow(
    point: np.ndarray, sun: np.ndarray, terrain: Terrain, steps: int = 18
) -> np.ndarray:
    """Fraction of sunlight reaching each point, 0 shadowed to 1 lit."""
    lit = np.ones(point.shape[0], dtype=np.float32)
    distance = np.full(point.shape[0], 6.0, dtype=np.float32)

    for _ in range(steps):
        probe = point + sun[None, :] * distance[:, None]
        ground = terrain.height(probe[:, 0], probe[:, 2])
        gap = probe[:, 1] - ground
        # Soft shadows: how close the ray passes to the ground, not just
        # whether it is blocked. A binary test gives hard, wrong-looking edges.
        lit = np.minimum(lit, np.clip(gap / (distance * 0.09 + 1.0), 0.0, 1.0))
        distance *= 1.55

    return np.clip(lit, 0.0, 1.0)


def _materials(
    height: np.ndarray,
    slope: np.ndarray,
    summit: float,
    water_level: float,
    x: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Surface colour from altitude and steepness.

    Albedos are deliberately dark. Real ground reflects 10-30% of the light
    that falls on it; picking colours that look right on screen *before*
    lighting is applied gives a landscape the pallor of wet cardboard once it
    is lit.

    ``summit`` is the terrain's actual highest point rather than its nominal
    amplitude. Fractal noise never reaches its own maximum, so normalising by
    the amplitude puts the snow line above anything that exists.
    """
    rock = np.array([0.30, 0.27, 0.24], np.float32)
    dark_rock = np.array([0.16, 0.145, 0.14], np.float32)
    grass = np.array([0.115, 0.170, 0.085], np.float32)
    dry_grass = np.array([0.235, 0.215, 0.120], np.float32)
    sand = np.array([0.42, 0.37, 0.27], np.float32)
    snow = np.array([0.94, 0.95, 0.98], np.float32)

    altitude = np.clip(
        (height - water_level) / max(summit - water_level, 1e-6), 0, 1
    )[:, None]
    steep = np.clip((slope - 0.30) / 0.30, 0, 1)[:, None]

    # A slow drift between lush and dry vegetation, so the lower slopes are
    # not one flat colour across the whole valley.
    patch = _value_noise(x / 620.0, z / 620.0)[:, None]
    ground_cover = grass[None, :] * (1 - patch * 0.75) + dry_grass[None, :] * (patch * 0.75)

    colour = ground_cover * (1 - altitude) + rock[None, :] * altitude
    # Shoreline.
    shore = np.clip(1.0 - altitude / 0.045, 0, 1)
    colour = colour * (1 - shore) + sand[None, :] * shore

    # Snow above the line, and only where it is not too steep to settle.
    snow_line = np.clip((altitude - 0.46) / 0.26, 0, 1)
    settles = np.clip(1.0 - steep * 1.3, 0, 1)
    colour = colour * (1 - snow_line * settles) + snow[None, :] * snow_line * settles

    # Exposed rock on steep faces, whatever the altitude.
    colour = colour * (1 - steep) + dark_rock[None, :] * steep
    return colour


def _sky_colour(direction: np.ndarray, sky: Sky) -> np.ndarray:
    sun = _normalise(sky.sun)
    up = np.clip(direction[:, 1], 0.0, 1.0)[:, None]

    zenith = np.asarray(sky.zenith, np.float32)[None, :]
    horizon = np.asarray(sky.horizon, np.float32)[None, :]
    colour = horizon * (1 - up**0.55) + zenith * up**0.55

    # Sun disc and the glow around it.
    alignment = np.clip(direction @ sun, 0.0, 1.0)[:, None]
    tint = np.asarray(sky.sun_tint, np.float32)[None, :]
    colour = colour + tint * (alignment**700) * 1.1 * sky.daylight
    colour = colour + tint * (alignment**18) * 0.13 * sky.daylight
    return colour


def render(
    width: int,
    height: int,
    terrain: Terrain,
    camera: Camera,
    sky: Sky,
    water: Water,
    steps: int = 150,
    far: float = 9000.0,
    supersample: int = 1,
    summit: Optional[float] = None,
) -> np.ndarray:
    """Render one frame as float RGB in 0-1."""
    w, h = width * supersample, height * supersample

    if summit is None:
        summit = terrain.summit()

    direction = _rays(w, h, camera)
    origin = np.asarray(camera.position, np.float32)
    sun = _normalise(sky.sun)

    distance, hit = _march(origin, direction, terrain, far, steps)
    distance = _refine(origin, direction, terrain, distance, hit)

    image = _sky_colour(direction, sky)

    if hit.any():
        idx = np.nonzero(hit)[0]
        point = origin[None, :] + direction[idx] * distance[idx][:, None]

        # Normal from the height gradient. The sample spacing grows with
        # distance so that far-away normals are not read from noise finer than
        # a pixel, which would sparkle.
        epsilon = np.maximum(distance[idx] * 0.0016, 0.7)
        hx = terrain.height(point[:, 0] + epsilon, point[:, 2])
        hx0 = terrain.height(point[:, 0] - epsilon, point[:, 2])
        hz = terrain.height(point[:, 0], point[:, 2] + epsilon)
        hz0 = terrain.height(point[:, 0], point[:, 2] - epsilon)

        normal = np.stack(
            [-(hx - hx0), 2.0 * epsilon, -(hz - hz0)], axis=1
        ).astype(np.float32)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)

        slope = 1.0 - normal[:, 1]
        ground = terrain.height(point[:, 0], point[:, 2])
        colour = _materials(
            ground, slope, summit, water.level, point[:, 0], point[:, 2]
        )

        lambert = np.clip(normal @ sun, 0.0, 1.0)[:, None]
        shade = _shadow(point, sun, terrain)[:, None]
        # Sky light from above, which is what keeps shadowed faces from going
        # black rather than merely dark.
        # Ambient kept low relative to the sun. Lifting it is the easy way to
        # avoid black shadows and the reliable way to kill all the modelling.
        ambient = np.asarray(sky.horizon, np.float32)[None, :] * 0.22
        sun_light = np.asarray(sky.sun_tint, np.float32)[None, :] * 1.75 * sky.daylight

        lit = colour * (ambient + sun_light * lambert * shade)

        if water.enabled:
            depth = np.clip((water.level - ground) / 26.0, 0, 1)[:, None]
            surface = np.asarray(water.colour, np.float32)[None, :]
            # A flat sheen toward the sky where the water is deep.
            sheen = np.asarray(sky.horizon, np.float32)[None, :] * 0.30
            lit = lit * (1 - depth) + (surface + sheen) * depth

        # Atmospheric perspective. Exponential, keyed to distance, toward the
        # sky in that direction so the fade matches what is behind it.
        fog = 1.0 - np.exp(-(distance[idx][:, None] / 3800.0) ** 1.15 * sky.haze)
        lit = lit * (1 - fog) + _sky_colour(direction[idx], sky) * fog

        image[idx] = lit

    image = image.reshape(h, w, 3)
    if supersample > 1:
        image = image.reshape(height, supersample, width, supersample, 3).mean(axis=(1, 3))

    # Filmic-ish curve, so bright sky rolls off instead of clipping flat.
    image = image / (image + 1.05)
    image = np.clip(image * 1.42, 0.0, 1.0) ** (1.0 / 2.2)

    # Tone mapping compresses all three channels alike, which drains colour
    # along with the highlights. A little saturation put back afterwards is
    # what keeps the vegetation green rather than grey.
    luma = (image @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
    image = np.clip(luma + (image - luma) * 1.35, 0.0, 1.0)
    return image.astype(np.float32)


def sea_level(terrain: Terrain, quantile: float = 0.14, extent: float = 8000.0) -> float:
    """A water height that floods the low ground and nothing else.

    Choosing this by hand is how the camera keeps ending up underwater: the
    height a fractal actually produces depends on every parameter at once, so
    a literal that looked right for one terrain drowns the next.
    """
    axis = np.linspace(-extent, extent, 72, dtype=np.float32)
    gx, gz = np.meshgrid(axis, axis)
    return float(np.quantile(terrain.height(gx.ravel(), gz.ravel()), quantile))


def find_viewpoint(
    terrain: Terrain,
    extent: float = 4000.0,
    samples: int = 48,
    quantile: float = 0.72,
    clearance: float = 55.0,
) -> Camera:
    """Pick somewhere worth standing, and something worth looking at.

    Dropping a camera at the origin lands it wherever the noise happens to be
    — which in practice is the bottom of a pit as often as a ridge, and a view
    from a pit is a wall. This puts the camera on high ground and turns it
    toward the highest peak in range, so the frame has a foreground shoulder,
    a middle distance and a summit.
    """
    axis = np.linspace(-extent, extent, samples, dtype=np.float32)
    gx, gz = np.meshgrid(axis, axis)
    flat_x, flat_z = gx.ravel(), gz.ravel()
    heights = terrain.height(flat_x, flat_z)

    # Stand on high-but-not-highest ground: the summit itself looks down on
    # everything and flattens the composition.
    target = float(np.quantile(heights, quantile))
    stand = int(np.argmin(np.abs(heights - target)))
    position = (float(flat_x[stand]), 0.0, float(flat_z[stand]))

    # Look at the tallest peak that is far enough away to sit in the distance.
    offset_x = flat_x - position[0]
    offset_z = flat_z - position[2]
    distance = np.hypot(offset_x, offset_z)
    eligible = distance > extent * 0.25
    if eligible.any():
        peak = int(np.argmax(np.where(eligible, heights, -np.inf)))
        yaw = math.atan2(float(offset_x[peak]), float(offset_z[peak]))
    else:
        yaw = 0.0

    camera = Camera(position=position, yaw=yaw, pitch=0.02, fov=1.0)
    return camera.grounded(terrain, clearance=clearance)


def compose(
    terrain: Terrain,
    distance: float = 5200.0,
    clearance: float = 60.0,
    fov: float = 0.78,
    extent: float = 9000.0,
) -> Tuple[Camera, float]:
    """A camera standing back from the highest peak, and a water level.

    Returns both together because they are not independent: the viewpoint has
    to be above the water for the near ground to be land.
    """
    axis = np.linspace(-extent, extent, 90, dtype=np.float32)
    gx, gz = np.meshgrid(axis, axis)
    flat_x, flat_z = gx.ravel(), gz.ravel()
    heights = terrain.height(flat_x, flat_z)

    peak = int(np.argmax(heights))
    px, pz = float(flat_x[peak]), float(flat_z[peak])

    bearing = math.atan2(px, pz)
    stand = (px - math.sin(bearing) * distance, 0.0, pz - math.cos(bearing) * distance)

    camera = Camera(position=stand, yaw=bearing, pitch=0.03, fov=fov).grounded(
        terrain, clearance=clearance
    )

    water = sea_level(terrain, extent=extent)
    # Keep the shoreline below the viewer, whatever the fractal produced.
    water = min(water, camera.position[1] - clearance * 0.6)
    return camera, water
