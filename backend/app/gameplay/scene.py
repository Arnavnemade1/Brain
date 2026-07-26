"""Render the memory as a scene.

Draws the game world the player was inside — corridor, ship, enemies, HUD —
twice: once from the actual record, once from what the EEG recovered. Played
back at video rate, side by side, that is the memory video.

The rule this module follows, and the reason it can be trusted: **nothing is
drawn that was not decoded.** Where the signal supports a property it is
rendered; where it does not, the reconstruction shows uncertainty rather than
a plausible invention. A generated dream-city would look far better here and
would be almost entirely fabrication.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PANEL_W, PANEL_H = 520, 300

VOID = (8, 10, 18)
CORRIDOR = (26, 34, 52)
WALL = (52, 68, 100)
SHIP = (120, 200, 255)
ENEMY = (232, 104, 96)
REWARD = (96, 214, 156)
MUZZLE = (255, 226, 140)
INK = (238, 242, 249)
INK_FAINT = (96, 106, 128)


@dataclass
class SceneState:
    """Everything needed to draw one frame of the world."""

    t: float
    #: Vertical position of the ship in the corridor, 0-1.
    ship_y: float = 0.5
    #: Corridor opening, 0-1, as a fraction of panel height.
    aperture: float = 0.62
    #: 0-1 recent firing activity.
    firing: float = 0.0
    #: 0-1 recent reward activity.
    reward: float = 0.0
    #: 0-1 recent damage.
    damage: float = 0.0
    #: Enemies as (distance 0-1, angle radians).
    enemies: Sequence[Tuple[float, float]] = ()
    #: 0-1 how much the renderer trusts this frame.
    confidence: float = 1.0
    #: Scroll offset so the corridor moves.
    scroll: float = 0.0


def draw_scene(draw, ox: int, oy: int, state: SceneState, reconstructed: bool, font) -> None:
    """Draw one world panel at ``(ox, oy)``."""
    w, h = PANEL_W, PANEL_H

    draw.rectangle([ox, oy, ox + w, oy + h], fill=VOID)

    # --- Corridor ---------------------------------------------------------
    aperture = max(0.15, min(0.95, state.aperture))
    top = oy + h * (0.5 - aperture / 2)
    bottom = oy + h * (0.5 + aperture / 2)

    draw.rectangle([ox, top, ox + w, bottom], fill=CORRIDOR)

    # Scrolling wall texture, so motion is visible frame to frame.
    step = 46
    offset = int(state.scroll * step) % step
    for x in range(ox - step, ox + w + step, step):
        sx = x + offset
        if ox <= sx <= ox + w:
            draw.line([sx, oy, sx, top], fill=WALL, width=2)
            draw.line([sx, bottom, sx, oy + h], fill=WALL, width=2)

    draw.line([ox, top, ox + w, top], fill=WALL, width=3)
    draw.line([ox, bottom, ox + w, bottom], fill=WALL, width=3)

    # --- Ship -------------------------------------------------------------
    ship_x = ox + w * 0.22
    ship_y = top + (bottom - top) * max(0.0, min(1.0, state.ship_y))

    if state.damage > 0.05:
        # Damage flashes the hull and shakes the frame.
        shake = state.damage * 5
        ship_x += math.sin(state.t * 40) * shake
        ship_y += math.cos(state.t * 37) * shake

    hull = ENEMY if state.damage > 0.3 else SHIP
    draw.polygon(
        [
            (ship_x + 18, ship_y),
            (ship_x - 12, ship_y - 11),
            (ship_x - 5, ship_y),
            (ship_x - 12, ship_y + 11),
        ],
        fill=hull,
    )

    # Engine trail.
    for i in range(3):
        alpha = 1 - i / 3
        draw.line(
            [ship_x - 12 - i * 9, ship_y, ship_x - 18 - i * 9, ship_y],
            fill=tuple(int(c * alpha * 0.7) for c in SHIP),
            width=3,
        )

    # --- Firing -----------------------------------------------------------
    if state.firing > 0.05:
        reach = ox + w * (0.3 + 0.7 * state.firing)
        draw.line([ship_x + 18, ship_y, reach, ship_y], fill=MUZZLE, width=2)
        flare = 5 + state.firing * 7
        draw.ellipse(
            [ship_x + 14, ship_y - flare, ship_x + 14 + flare * 2, ship_y + flare],
            fill=MUZZLE,
        )

    # --- Enemies ----------------------------------------------------------
    for distance, angle in state.enemies:
        distance = max(0.05, min(1.0, distance))
        ex = ox + w * (0.35 + 0.6 * distance)
        ey = oy + h / 2 + math.sin(angle) * (h * 0.32)
        size = 6 + (1 - distance) * 10
        draw.ellipse([ex - size, ey - size, ex + size, ey + size], outline=ENEMY, width=2)
        draw.ellipse(
            [ex - size / 2, ey - size / 2, ex + size / 2, ey + size / 2], fill=ENEMY
        )

    # --- Reward -----------------------------------------------------------
    if state.reward > 0.05:
        for i in range(3):
            rx = ox + w * (0.5 + 0.16 * i)
            ry = oy + h / 2 + math.sin(state.t * 3 + i) * 40
            size = 4 + state.reward * 7
            draw.ellipse([rx - size, ry - size, rx + size, ry + size], fill=REWARD)

    # --- Uncertainty ------------------------------------------------------
    # The reconstruction dims as confidence falls, so a viewer can see where
    # the signal was weak rather than being shown a confident-looking guess.
    if reconstructed and state.confidence < 0.95:
        # Scanlines drawn *over* the finished scene. An earlier version
        # repainted the panel with the background colour first, which erased
        # the very reconstruction it was meant to qualify and rendered the
        # panel empty. Uncertainty must veil the picture, never replace it.
        gap = max(2, int(2 + (1 - state.confidence) * 6))
        for y in range(int(oy) + 1, int(oy + h), gap):
            draw.line([ox + 1, y, ox + w - 1, y], fill=VOID, width=1)

    draw.rectangle([ox, oy, ox + w, oy + h], outline=(40, 48, 66), width=1)


def state_from_events(
    t: float,
    onsets: np.ndarray,
    categories: Sequence[str],
    rng_seed: int = 0,
    decay: float = 0.9,
) -> SceneState:
    """Scene state at time ``t`` from a known event stream."""
    categories = np.asarray(categories)

    def recent(kind: str) -> float:
        mask = categories == kind
        if not mask.any():
            return 0.0
        past = onsets[mask][onsets[mask] <= t]
        if past.size == 0:
            return 0.0
        age = t - past[-1]
        return float(np.exp(-age / decay))

    firing = recent("shoot")
    reward = recent("reward")
    damage = recent("crash")

    # The world reacts to what is happening: the corridor tightens under
    # damage, and enemies crowd in when the player is engaging them.
    aperture = 0.70 - 0.22 * damage
    ship_y = 0.5 + 0.16 * math.sin(t * 0.7) - 0.18 * damage

    enemies: List[Tuple[float, float]] = []
    threat = max(firing, damage)
    if threat > 0.1:
        count = 1 + int(threat * 2)
        for i in range(count):
            enemies.append((0.35 + 0.5 * (1 - threat), math.sin(t * 1.3 + i * 2.1)))

    return SceneState(
        t=t,
        ship_y=ship_y,
        aperture=aperture,
        firing=firing,
        reward=reward,
        damage=damage,
        enemies=enemies,
        confidence=1.0,
        scroll=t * 2.2,
    )


def state_from_decoded(
    t: float,
    times: np.ndarray,
    probabilities: np.ndarray,
    classes: Sequence[str],
    decay: float = 0.9,
) -> SceneState:
    """Scene state at time ``t`` from decoded posteriors alone.

    Driven by probability rather than a hard decision, so a moment the
    decoder is unsure about renders as a faint, hesitant scene instead of a
    confident wrong one.
    """
    if times.size == 0:
        return SceneState(t=t)

    index = int(np.clip(np.searchsorted(times, t) - 1, 0, times.size - 1))
    lookup = {name: i for i, name in enumerate(classes)}

    def probability(name: str) -> float:
        column = lookup.get(name)
        if column is None:
            return 0.0
        # Decay from the most recent step so events persist briefly on
        # screen rather than flickering for a single frame.
        window = probabilities[max(0, index - 6) : index + 1, column]
        if window.size == 0:
            return 0.0
        ages = (t - times[max(0, index - 6) : index + 1]) / decay
        weights = np.exp(-np.clip(ages, 0, 10))
        return float(np.max(window * weights))

    firing = probability("shoot")
    reward = probability("reward")
    damage = probability("crash")
    idle = probability("idle")

    confidence = float(np.clip(max(firing, reward, damage, idle), 0.25, 1.0))

    aperture = 0.70 - 0.22 * damage
    ship_y = 0.5 + 0.16 * math.sin(t * 0.7) - 0.18 * damage

    enemies: List[Tuple[float, float]] = []
    threat = max(firing, damage)
    if threat > 0.35:
        count = 1 + int(threat * 2)
        for i in range(count):
            enemies.append((0.35 + 0.5 * (1 - threat), math.sin(t * 1.3 + i * 2.1)))

    return SceneState(
        t=t,
        ship_y=ship_y,
        aperture=aperture,
        firing=firing,
        reward=reward,
        damage=damage,
        enemies=enemies,
        confidence=confidence,
        scroll=t * 2.2,
    )
