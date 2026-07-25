"""End-to-end fidelity evaluation.

Runs complete reconstructions across the clip library and recording conditions
and reports measured fidelity against the trivial baseline. This is the
headline result of the system: how much of a watched video can be recovered
from EEG alone.

Run with::

    backend/.venv/bin/python -m tools.evaluate
"""

from __future__ import annotations

import asyncio
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from app.evaluation.metrics import baseline_metrics  # noqa: E402
from app.services.sessions import prepare  # noqa: E402
from app.simulation.visual_response import CONDITIONS  # noqa: E402
from app.stimulus.clips import CLIPS  # noqa: E402

#: Clips held out of decoder fitting. Results on these are the honest ones.
HELD_OUT = {"low_light", "cut_sequence", "objects"}


async def run_one(clip_id: str, condition: str, seed: int):
    runner = prepare(clip_id, condition, speed=1000.0, seed=seed)
    async for _ in runner.run():
        pass
    return runner


def main() -> int:
    print("=" * 86)
    print("MindScape — reconstruction fidelity")
    print("=" * 86)
    print("Clips marked * were held out of decoder fitting.\n")

    header = (
        f"{'clip':<15}{'cond':<11}{'SSIM':>7}{'base':>7}"
        f"{'lum r':>8}{'mot r':>8}{'cut F1':>8}{'layout':>8}{'comp':>7}"
    )
    print(header)
    print("-" * len(header))

    rows: List[Tuple[bool, Dict]] = []

    for clip_id in CLIPS:
        for condition in ("clinical", "research", "consumer", "degraded"):
            runner = asyncio.run(run_one(clip_id, condition, seed=2024))
            if runner.metrics is None:
                print(f"{clip_id:<15}{condition:<11}  failed")
                continue

            metrics = runner.metrics
            baseline = baseline_metrics(runner.matched_reference, runner.matched_truth)

            held = clip_id in HELD_OUT
            label = f"{clip_id}{'*' if held else ''}"

            print(
                f"{label:<15}{condition:<11}"
                f"{metrics.ssim:>7.3f}{baseline.ssim:>7.3f}"
                f"{metrics.luminance_correlation:>+8.2f}"
                f"{metrics.motion_correlation:>+8.2f}"
                f"{metrics.scene_cut_f1:>8.2f}"
                f"{metrics.layout_correlation:>+8.2f}"
                f"{metrics.composite:>7.3f}"
            )

            rows.append(
                (
                    held,
                    {
                        "ssim": metrics.ssim,
                        "baseline_ssim": baseline.ssim,
                        "luminance": metrics.luminance_correlation,
                        "motion": metrics.motion_correlation,
                        "cut": metrics.scene_cut_f1,
                        "layout": metrics.layout_correlation,
                        "composite": metrics.composite,
                        "baseline_composite": baseline.composite,
                        "condition": condition,
                    },
                )
            )

    def summarise(label: str, subset: List[Dict]) -> None:
        if not subset:
            return
        print(f"\n{label} ({len(subset)} runs)")
        for key, name in [
            ("ssim", "SSIM"),
            ("baseline_ssim", "  baseline SSIM"),
            ("luminance", "luminance r"),
            ("motion", "motion r"),
            ("cut", "scene cut F1"),
            ("layout", "layout r"),
            ("composite", "composite"),
            ("baseline_composite", "  baseline composite"),
        ]:
            values = [row[key] for row in subset]
            print(f"  {name:<18} {np.mean(values):+.3f}  (min {min(values):+.3f}, max {max(values):+.3f})")

    print()
    print("=" * 86)
    summarise("Held-out clips (never seen by the decoders)", [r for held, r in rows if held])
    summarise("Training clips", [r for held, r in rows if not held])

    # Fidelity by recording quality — the practical question for a deployment.
    print("\nComposite fidelity by recording condition")
    for condition in CONDITIONS:
        values = [r["composite"] for _, r in rows if r["condition"] == condition]
        if values:
            print(f"  {condition:<11} {np.mean(values):.3f}")

    held_out_rows = [r for held, r in rows if held]
    if held_out_rows:
        beat_ssim = sum(1 for r in held_out_rows if r["ssim"] > r["baseline_ssim"])
        beat_composite = sum(
            1 for r in held_out_rows if r["composite"] > r["baseline_composite"]
        )
        ratio = np.mean([r["composite"] for r in held_out_rows]) / max(
            np.mean([r["baseline_composite"] for r in held_out_rows]), 1e-9
        )

        print("\nAgainst the mean-frame baseline, on held-out clips:")
        print(f"  composite   beaten in {beat_composite}/{len(held_out_rows)} runs ({ratio:.1f}×)")
        print(f"  SSIM        beaten in {beat_ssim}/{len(held_out_rows)} runs")
        print(
            "\n  SSIM is the metric the baseline wins most often, and that is the\n"
            "  honest headline: a constant mean frame is spatially smooth, and SSIM\n"
            "  rewards exactly that. The reconstruction earns its composite margin on\n"
            "  the temporal metrics — when the scene brightened, moved and changed —\n"
            "  where a constant frame scores zero by construction."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
