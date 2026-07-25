"""Fit the valence and arousal readouts against the simulator's ground truth.

The affect model maps spectral and feature measures onto the valence–arousal
plane. Rather than hand-tuning those coefficients, this script regresses the
scripted values from ``simulation/profiles.py`` onto candidate predictors and
reports the fit, so ``inference.infer_emotion`` can use measured coefficients.

Run with::

    backend/.venv/bin/python -m tools.fit_affect
"""

from __future__ import annotations

import sys
import warnings
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from app.pipeline.analyzer import WindowAnalyzer  # noqa: E402
from app.pipeline.normalisation import z  # noqa: E402
from app.simulation.generator import generate_session  # noqa: E402
from app.simulation.profiles import PROFILES  # noqa: E402
from tools.calibrate import phase_at  # noqa: E402

SAMPLE_RATE = 256
WINDOW = 512
HOP = 256
DURATION = 90.0
SEEDS = (3, 17, 41, 59)


def collect() -> Dict[str, np.ndarray]:
    rows: Dict[str, List[float]] = {
        "target_valence": [],
        "target_arousal": [],
        "fast_share": [],
        "alpha": [],
        "theta": [],
        "delta": [],
        "beta": [],
        "gamma": [],
        "asymmetry": [],
        "z_mobility": [],
        "z_edge": [],
        "z_entropy": [],
        "z_zcr": [],
    }

    for pid, profile in PROFILES.items():
        for seed in SEEDS:
            rec = generate_session(pid, DURATION, SAMPLE_RATE, seed=seed)
            analyzer = WindowAnalyzer(f"fit-{pid}", SAMPLE_RATE, rec.channels, 60.0)

            for start in range(0, rec.total_samples - WINDOW, HOP):
                t = start / SAMPLE_RATE
                result = analyzer.analyze(rec.window(start, WINDOW), t)
                frame = result.frame
                phase = phase_at(profile, t, DURATION)
                bands = frame.band_powers
                features = result.features

                total = sum(bands.values()) or 1.0
                rows["target_valence"].append(phase.valence)
                rows["target_arousal"].append(phase.arousal)
                rows["fast_share"].append((bands["beta"] + bands["gamma"]) / total)
                for band in ("delta", "theta", "alpha", "beta", "gamma"):
                    rows[band].append(bands[band])
                rows["asymmetry"].append(features.frontal_asymmetry)
                rows["z_mobility"].append(z("mobility", features.mobility))
                rows["z_edge"].append(z("spectral_edge", features.spectral_edge))
                rows["z_entropy"].append(z("spectral_entropy", features.spectral_entropy))
                rows["z_zcr"].append(z("zero_crossing", features.zero_crossing))

    return {k: np.array(v) for k, v in rows.items()}


def fit(target: np.ndarray, predictors: Dict[str, np.ndarray]) -> None:
    names = list(predictors)
    design = np.column_stack([predictors[n] for n in names] + [np.ones(len(target))])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)

    prediction = design @ coefficients
    residual = target - prediction
    r2 = 1.0 - float(np.sum(residual**2) / max(np.sum((target - target.mean()) ** 2), 1e-12))
    r = float(np.corrcoef(target, prediction)[0, 1])

    for name, c in zip(names, coefficients[:-1]):
        marginal = float(np.corrcoef(target, predictors[name])[0, 1])
        print(f"    {name:<14} coef {c:+7.3f}   (marginal r {marginal:+.3f})")
    print(f"    {'intercept':<14} coef {coefficients[-1]:+7.3f}")
    print(f"    → R² {r2:.3f}   r {r:+.3f}")


def main() -> int:
    print("Collecting windows…")
    data = collect()
    n = len(data["target_arousal"])
    print(f"{n} windows across {len(PROFILES)} profiles × {len(SEEDS)} seeds\n")

    print("AROUSAL")
    fit(
        data["target_arousal"],
        {
            "fast_share": data["fast_share"],
            "z_mobility": data["z_mobility"],
            "z_edge": data["z_edge"],
            "z_entropy": data["z_entropy"],
        },
    )

    print("\nAROUSAL (fast_share only)")
    fit(data["target_arousal"], {"fast_share": data["fast_share"]})

    print("\nVALENCE")
    fit(
        data["target_valence"],
        {
            "asymmetry": data["asymmetry"],
            "alpha": data["alpha"],
            "theta": data["theta"],
            "beta": data["beta"],
        },
    )

    print("\nVALENCE (asymmetry only)")
    fit(data["target_valence"], {"asymmetry": data["asymmetry"]})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
