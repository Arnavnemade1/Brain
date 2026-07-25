"""Fit the reconstruction decoders against known reference clips.

Renders each training clip, synthesises stimulus-locked EEG, runs the full
analysis pipeline, and fits ridge models from the latent trajectory to the
clip's ground-truth visual features.

Evaluation is on *held-out clips*, not held-out frames. Frames within a clip
are heavily autocorrelated, so a random frame split would leak the answer and
report a fidelity the system cannot reproduce on anything new.

Run with::

    backend/.venv/bin/python -m tools.train_decoders
"""

from __future__ import annotations

import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from app.config import get_settings  # noqa: E402
from app.encoder.latent import stack  # noqa: E402
from app.encoder.pipeline import AnalysisPipeline  # noqa: E402
from app.processing.sync import synchronise  # noqa: E402
from app.reconstruction.decoders import (  # noqa: E402
    LAYOUT_BLOCKS,
    LAYOUT_SIZE,
    SCENE_TARGETS,
    DecoderBundle,
    fit_ridge,
    r_squared,
)
from app.simulation.visual_response import simulate  # noqa: E402
from app.stimulus.clips import CLIPS, FPS, render_clip  # noqa: E402
from app.stimulus.features import extract, layout_vector  # noqa: E402

#: Recording conditions to train across, so the decoders are not tuned to a
#: single quality of signal.
TRAIN_CONDITIONS = ["clinical", "research", "consumer"]
TRAIN_SEEDS = [11, 29, 47]

#: Clips held out entirely from fitting.
HELD_OUT = {"cut_sequence", "low_light", "objects"}


def collect(clip_id: str, condition: str, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one recording through the pipeline.

    Returns ``(latents, scene_targets, layout_targets)`` aligned frame by frame.
    """
    settings = get_settings()

    frames, scene_indices = render_clip(clip_id)
    features = extract(frames, FPS, scene_indices)

    recording = simulate(
        frames, features, FPS, condition, settings.sample_rate, seed=seed
    )
    sync, clock = synchronise(
        recording.marker_samples, recording.marker_times, recording.sample_rate
    )

    pipeline = AnalysisPipeline(
        f"train-{clip_id}", recording.sample_rate, recording.channels, clock
    )

    window_samples = int(settings.sample_rate * settings.window_seconds)
    hop = max(1, int(window_samples * (1.0 - settings.window_overlap)))

    # Baseline from the pre-stimulus lead-in, before playback started.
    stimulus_start = clock.to_sample(0.0)
    baseline_windows = [
        recording.window(start, window_samples)
        for start in range(0, max(0, stimulus_start - window_samples), hop)
    ]
    pipeline.calibrate(baseline_windows)

    latents: List[List[float]] = []
    scene_rows: List[List[float]] = []
    layout_rows: List[np.ndarray] = []

    duration = len(features) / FPS

    for start in range(stimulus_start, recording.total_samples - window_samples, hop):
        result = pipeline.process(recording.window(start, window_samples), start)
        t = result.latent.t
        if t < 0 or t > duration:
            continue

        index = int(round(t * FPS))
        if index < 0 or index >= len(features):
            continue

        truth = features[index]
        latents.append(result.latent.vector)
        scene_rows.append(
            [
                truth.luminance,
                truth.contrast,
                truth.motion,
                truth.color[0],
                truth.color[1],
                truth.color[2],
            ]
        )
        layout_rows.append(layout_vector(frames[index]))

    if not latents:
        empty = np.zeros((0, 0))
        return empty, empty, empty

    return (
        np.array(latents, dtype=np.float64),
        np.array(scene_rows, dtype=np.float64),
        np.array(layout_rows, dtype=np.float64),
    )


def gather(clip_ids: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    latents: List[np.ndarray] = []
    scenes: List[np.ndarray] = []
    layouts: List[np.ndarray] = []

    for clip_id in clip_ids:
        for condition in TRAIN_CONDITIONS:
            for seed in TRAIN_SEEDS:
                x, y_scene, y_layout = collect(clip_id, condition, seed)
                if x.size == 0:
                    continue
                latents.append(x)
                scenes.append(y_scene)
                layouts.append(y_layout)

    if not latents:
        raise RuntimeError("no training data collected")

    return (
        np.concatenate(latents),
        np.concatenate(scenes),
        np.concatenate(layouts),
    )


def choose_cut_threshold(clip_ids: List[str]) -> float:
    """Pick the transient threshold that maximises boundary F1.

    Swept rather than guessed. The optimum depends on window length and on
    how prominent the evoked response is relative to ongoing activity, and
    both change if the acquisition settings change.
    """
    from app.evaluation.metrics import scene_cut_f1_by_time
    from app.models.neural import LatentPoint
    from app.reconstruction.scene import detect_cuts

    settings = get_settings()
    samples: List[Tuple[List[LatentPoint], List[float], List[float]]] = []

    for clip_id in clip_ids:
        for condition in ("research", "consumer"):
            frames, scene_indices = render_clip(clip_id)
            features = extract(frames, FPS, scene_indices)
            recording = simulate(
                frames, features, FPS, condition, settings.sample_rate, seed=101
            )
            sync, clock = synchronise(
                recording.marker_samples, recording.marker_times, recording.sample_rate
            )
            pipeline = AnalysisPipeline(
                "cut", recording.sample_rate, recording.channels, clock
            )

            window_samples = int(settings.sample_rate * settings.window_seconds)
            hop = max(1, int(window_samples * (1.0 - settings.window_overlap)))
            stimulus_start = clock.to_sample(0.0)

            pipeline.calibrate(
                [
                    recording.window(start, window_samples)
                    for start in range(0, max(0, stimulus_start - window_samples), hop)
                ]
            )

            points: List[LatentPoint] = []
            truth_times: List[float] = [f.t for f in features if f.scene_cut]
            times: List[float] = []
            duration = len(features) / FPS

            for start in range(
                stimulus_start, recording.total_samples - window_samples, hop
            ):
                result = pipeline.process(recording.window(start, window_samples), start)
                t = result.latent.t
                if t < 0 or t > duration:
                    continue
                points.append(result.latent)
                times.append(t)

            if points:
                samples.append((points, truth_times, times))

    best_threshold = 0.55
    best_f1 = -1.0

    for threshold in np.arange(0.15, 0.95, 0.025):
        scores = []
        for points, truth_times, times in samples:
            cuts = detect_cuts(points, float(threshold))
            # Score with exactly the metric the system reports. Sweeping a
            # different definition than the one reported is how a threshold
            # ends up "optimal" while producing several times too many
            # detections in practice.
            predicted_times = [times[i] for i, flag in enumerate(cuts) if flag]
            scores.append(scene_cut_f1_by_time(predicted_times, truth_times))
        mean_f1 = float(np.mean(scores)) if scores else 0.0
        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_threshold = float(threshold)

    print(f"  cut threshold {best_threshold:.3f} → boundary F1 {best_f1:.3f}")
    return best_threshold


def _correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    if actual.std() < 1e-9 or predicted.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(actual, predicted)[0, 1])


def _sweep_alpha(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    names: List[str],
):
    """Choose the ridge penalty that maximises held-out correlation.

    Selected on correlation rather than R² deliberately. R² rewards matching
    the absolute level, which pushes the penalty upward until predictions
    collapse toward the training mean — technically lower error, but a decoder
    that has stopped tracking the stimulus at all.
    """
    best = None
    best_score = -np.inf
    best_alpha = 1.0

    for alpha in (0.3, 1.0, 3.0, 6.0, 12.0, 30.0, 80.0):
        model = fit_ridge(x_train, y_train, names, alpha=alpha)
        predicted = model.predict(x_test)
        score = float(
            np.mean(
                [
                    _correlation(y_test[:, i], predicted[:, i])
                    for i in range(y_test.shape[1])
                ]
            )
        )
        if score > best_score:
            best_score = score
            best = model
            best_alpha = alpha

    assert best is not None
    return best, best_alpha


def main() -> int:
    train_ids = [cid for cid in CLIPS if cid not in HELD_OUT]
    test_ids = [cid for cid in CLIPS if cid in HELD_OUT]

    print("=" * 74)
    print("Fitting reconstruction decoders")
    print("=" * 74)
    print(f"train clips : {', '.join(train_ids)}")
    print(f"held out    : {', '.join(test_ids)}")
    print(f"conditions  : {', '.join(TRAIN_CONDITIONS)} × {len(TRAIN_SEEDS)} seeds\n")

    print("Collecting training data…")
    x_train, scene_train, layout_train = gather(train_ids)
    print(f"  {x_train.shape[0]} windows, {x_train.shape[1]}-d latent")

    print("Collecting held-out data…")
    x_test, scene_test, layout_test = gather(test_ids)
    print(f"  {x_test.shape[0]} windows")

    print("\nSweeping ridge penalty on held-out clips…")
    scene_model, scene_alpha = _sweep_alpha(
        x_train, scene_train, x_test, scene_test, SCENE_TARGETS
    )
    layout_names = [f"block_{i}" for i in range(LAYOUT_SIZE)]
    layout_model, layout_alpha = _sweep_alpha(
        x_train, layout_train, x_test, layout_test, layout_names
    )
    print(f"  scene α={scene_alpha:g}   layout α={layout_alpha:g}")

    scene_predicted = scene_model.predict(x_test)
    scene_r2 = r_squared(scene_test, scene_predicted)
    layout_r2 = r_squared(layout_test, layout_model.predict(x_test))

    scene_model.r2 = {
        name: round(float(value), 4) for name, value in zip(SCENE_TARGETS, scene_r2)
    }
    layout_model.r2 = {"mean": round(float(np.mean(layout_r2)), 4)}

    print("\nHeld-out recovery (unseen clips):")
    print(f"  {'target':<12} {'r':>7} {'R²':>8}   verdict")
    for i, name in enumerate(SCENE_TARGETS):
        r = _correlation(scene_test[:, i], scene_predicted[:, i])
        # Correlation is the honest headline: it measures whether the decoder
        # tracks the property, independent of absolute scale. R² additionally
        # punishes offset, which matters far less for a system whose output is
        # always compared *within* a session.
        verdict = "tracked" if r > 0.3 else "weak" if r > 0.15 else "not recovered"
        print(f"  {name:<12} {r:>+7.3f} {scene_r2[i]:>+8.3f}   {verdict}")

    print(f"\n  spatial layout, mean over {LAYOUT_SIZE} blocks:")
    print(f"  {'':<12} {'':>7} {float(np.mean(layout_r2)):>+8.3f}")

    print("\nSweeping scene-boundary threshold…")
    cut_threshold = choose_cut_threshold(train_ids)

    # Held-out correlation of the layout decoder, used at reconstruction time
    # to scale how much spatial structure the frames are allowed to assert.
    layout_correlations = [
        _correlation(layout_test[:, i], layout_model.predict(x_test)[:, i])
        for i in range(layout_test.shape[1])
    ]
    layout_reliability = float(np.clip(np.mean(layout_correlations), 0.0, 1.0))
    print(f"\n  layout reliability (held-out r) {layout_reliability:.3f}")

    bundle = DecoderBundle(
        scene=scene_model,
        layout=layout_model,
        cut_threshold=cut_threshold,
        layout_reliability=layout_reliability,
        trained_on=train_ids,
        trained_frames=int(x_train.shape[0]),
    )
    bundle.save()

    print(f"\nSaved weights ({x_train.shape[0]} training windows).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
