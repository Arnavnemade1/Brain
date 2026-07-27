"""Render a terrain view.

    backend/.venv/bin/python -m tools.render_terrain --out /tmp/view.png
    backend/.venv/bin/python -m tools.render_terrain --seed 7 --width 1600

Every parameter that used to be a literal buried in the renderer is exposed
here, because the ones that matter — where the water sits, how high the camera
stands, how far the march reaches — interact, and fixing any of them by hand
is what put the camera underwater three times while this was being built.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from app.scene.terrain import Sky, Terrain, Water, compose, render

DOCS = Path(__file__).resolve().parent.parent.parent / "docs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DOCS / "terrain.png"))
    parser.add_argument("--width", type=int, default=1120)
    parser.add_argument("--height", type=int, default=620)
    parser.add_argument("--seed", type=float, default=3.0)
    parser.add_argument("--amplitude", type=float, default=1150.0)
    parser.add_argument("--wavelength", type=float, default=1500.0)
    parser.add_argument("--octaves", type=int, default=10)
    parser.add_argument("--ridged", type=float, default=0.95)
    parser.add_argument("--distance", type=float, default=5600.0)
    parser.add_argument("--clearance", type=float, default=110.0)
    parser.add_argument("--fov", type=float, default=0.75)
    parser.add_argument("--pitch", type=float, default=-0.05)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--far", type=float, default=22000.0)
    parser.add_argument("--supersample", type=int, default=1)
    args = parser.parse_args()

    terrain = Terrain(
        amplitude=args.amplitude,
        wavelength=args.wavelength,
        octaves=args.octaves,
        ridged=args.ridged,
        seed=args.seed,
    )

    camera, water_level = compose(
        terrain, distance=args.distance, clearance=args.clearance, fov=args.fov
    )
    camera.pitch = args.pitch

    print(
        f"camera at y={camera.position[1]:.0f}, water at {water_level:.0f}, "
        f"summit {terrain.summit(extent=9000):.0f}"
    )

    started = time.perf_counter()
    image = render(
        args.width,
        args.height,
        terrain,
        camera,
        Sky(),
        Water(level=water_level),
        steps=args.steps,
        far=args.far,
        supersample=args.supersample,
    )
    print(f"rendered {args.width}x{args.height} in {time.perf_counter() - started:.1f}s")

    import imageio.v2 as imageio

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out, (np.clip(image, 0, 1) * 255).astype(np.uint8))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
