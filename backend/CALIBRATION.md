# Reconstruction fidelity

Measured recovery of a watched video from EEG alone. Nothing is passed between
the simulator and the reconstruction engine: the encoder sees only the
synthesised waveform, and every number below is scored against the reference
clip the subject "watched".

Reproduce:

```bash
backend/.venv/bin/python -m tools.train_decoders   # fit decoders
backend/.venv/bin/python -m tools.evaluate         # score reconstructions
```

## Headline result

40 runs · 8 clips × 4 recording conditions · 32×18 reconstruction grid

| | Reconstruction | Mean-frame baseline |
| --- | --- | --- |
| **Composite fidelity** | **0.489** | 0.187 |
| SSIM | 0.439 | 0.508 |
| Luminance tracking | r = +0.32 | 0 |
| Motion tracking | r = +0.57 | 0 |
| Scene boundary F1 | 0.69 | 0 |
| Spatial layout | r = +0.44 | — |

On clips **never seen during decoder fitting**, the reconstruction beats the
baseline composite in **10 of 12 runs, by 2.3×**.

### Fidelity by recording quality

| Condition | Composite |
| --- | --- |
| Research lab | 0.523 |
| Clinical-grade | 0.497 |
| Consumer headset | 0.482 |
| Degraded | 0.404 |

Fidelity degrades gracefully rather than collapsing, and the ordering is
sensible — but note the spread is narrower than the artifact levels suggest,
because baselining absorbs a good deal of the difference.

## The honest headline: SSIM loses to a constant image

The reconstruction scores **below** the mean-frame baseline on SSIM (0.439 vs
0.508), and beats it on that metric in only 5 of 12 held-out runs.

This is not a bug, and it is the single most important thing to understand
about the system. A constant average frame is spatially smooth and structurally
self-consistent, which is exactly what SSIM rewards. Scalp EEG carries almost
no spatial information — roughly left/right and upper/lower field balance —
so a reconstruction that honestly renders *only* what it recovered cannot beat
a smooth constant on a spatial metric.

What the reconstruction does recover is **temporal**: when the scene got
brighter, when it moved, and where it cut. The baseline scores exactly zero on
all three by construction. That is where the 2.3× composite margin comes from.

Anyone reporting an impressive-looking SSIM for EEG video reconstruction should
be asked what a constant frame scores on the same material.

## What is and is not recoverable

Held-out correlation between decoded and true values, on unseen clips:

| Property | r | Verdict |
| --- | --- | --- |
| Motion energy | +0.62 | tracked |
| Luminance | +0.54 | tracked |
| Colour | +0.54 | tracked, but see below |
| Contrast | +0.39 | tracked |
| Scene boundaries | F1 0.78 | tracked |
| Spatial layout | R² −0.18 | **not recovered** |

- **Scene boundaries are the most reliable recovery.** An abrupt visual change
  produces a large time-locked evoked response that survives heavy noise.
- **Spatial layout is not recovered.** This is the physical ceiling, not a
  decoder deficiency. The frame reconstruction scales its spatial variation by
  the *measured* layout reliability, so where layout is unrecoverable the
  output correctly collapses toward a uniform field at the decoded brightness.
- **Colour's r = +0.54 is misleading.** In this clip library colour correlates
  with luminance (scenes are largely neutral), so the colour decoder is
  substantially riding the luminance signal. Chromatic information is weakly
  represented at the scalp; treat this number with suspicion.
- **Absolute brightness is uncertain.** The encoder measures response relative
  to a pre-stimulus baseline, so it tracks how brightness *changed* far better
  than what it was. Held-out R² for luminance is negative (−0.08) while its
  correlation is +0.54 — the shape is right, the level is not.

## Synchronization

Offset is recovered to within 0.4–3.1 ms. **Clock drift is not estimable** at
these clip lengths: with three markers over 20 s and ~5 ms trigger jitter, the
standard error on the slope is around ±230 ppm against a true drift of tens of
ppm. The system detects this and fits offset only rather than reporting a
number dominated by its own uncertainty.

Worst-case misalignment across a whole clip is then 0.7–5.8 ms — negligible
against a 2 s analysis window, so the simpler model is provably sufficient here.

## Calibration decisions worth knowing

Several of these were bugs found by measurement, not design choices:

- **Scene-boundary threshold is swept on the metric the system reports.** An
  earlier version swept a window-level F1 while reporting a time-matched one;
  the "optimal" threshold produced 11 detections for 3 real cuts. Fixing the
  mismatch moved F1 from 0.54 to 0.69.

- **Boundary F1 is scored against every reference frame.** Scoring against
  ground truth sampled at reconstruction times drops most real cuts — they are
  instantaneous events on a 10 fps timeline while reconstructions arrive about
  once a second — and scored 0.29 against a truth series that had itself lost
  the answer.

- **Ridge penalty is selected on held-out correlation, not R².** Selecting on
  R² rewards matching the absolute level, which drives the penalty up until
  predictions collapse toward the training mean: lower error, no tracking.

- **Held-out split is by clip, not by frame.** Frames within a clip are heavily
  autocorrelated, so a random frame split leaks the answer.

- **Frame reconstruction asserts spatial structure only in proportion to
  measured layout reliability.** Asserting full decoded contrast regardless
  made reconstructions score *below* a flat mean frame: confidently wrong
  detail is worse than honest smoothness.

- **Signal quality grade thresholds are calibrated** to the composite score's
  actual distribution (~0.75–0.99), not an even split of 0–1.

## Limitations of this evaluation

- The EEG is **synthetic**, generated by a forward model of visual response
  (occipital luminance drive, alpha suppression, motion-sensitive activity,
  evoked transients, field-balance asymmetry). Real EEG carries additional
  noise, individual variability and non-stationarity this does not capture.
  These numbers are an **upper bound**.
- The clip library is procedural and small. Real video has far richer spatial
  statistics, which would likely lower layout and colour recovery further.
- Decoders are linear by choice. A nonlinear model might extract more, but with
  a 16-dimensional latent over a handful of recoverable quantities, it would
  mostly fit noise — and would make every result harder to attribute.
