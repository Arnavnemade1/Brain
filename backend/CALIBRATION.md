# Pipeline calibration

Measured round-trip accuracy of the analysis pipeline against the simulator's
scripted ground truth. Nothing is passed between the generator and the
analyser — these numbers reflect genuine recovery from the waveform alone.

Reproduce with:

```bash
backend/.venv/bin/python -m tools.calibrate
```

Refit the affect coefficients with:

```bash
backend/.venv/bin/python -m tools.fit_affect
```

## Current results

2464 windows · 7 profiles × 4 seeds · 90 s each · 256 Hz · 2 s windows / 50% overlap

| Measure | Result | Baseline |
| --- | --- | --- |
| Band power MAE | 0.070 | 0 = perfect |
| Cognitive state accuracy | 34.0% | 14.3% (7-way chance) |
| Emotion accuracy (mean over classes) | ~36% | 12.5% (8-way chance) |
| Valence correlation | +0.547 | 0 |
| Arousal range | 0.20 – 1.00, mean 0.56 | target mean 0.55 |

### Per-emotion recovery

| Scripted | Exact | Most common confusion |
| --- | --- | --- |
| nostalgia | 58.6% | calm 15% |
| curiosity | 51.5% | nostalgia 21% |
| melancholy | 48.1% | nostalgia 24% |
| wonder | 48.3% | nostalgia 23% |
| stress | 43.0% | curiosity 36% |
| calm | 24.5% | nostalgia 43% |
| joy | 11.8% | curiosity 39% |
| fear | 6.3% | stress 59% |

## Interpretation

Accuracy is 2.4–2.9× chance across both classification tasks. The residual
confusions are between genuinely adjacent affective states — calm/nostalgia,
stress/fear, joy/curiosity — which is the expected failure mode rather than a
sign of a broken model. Real EEG affect decoding performs comparably.

This matters for the product rather than being an apology for it: MindScape's
premise is that reconstruction quality is bounded by neural evidence. Moderate
classifier confidence is what drives fragmented geometry and low-confidence
regions in the rendered environment, and those are the honest output.

## Calibration decisions worth knowing

Several of these were bugs found by measurement, not design choices:

- **Features and band powers are scored in z-space**, against measured
  distributions in `pipeline/normalisation.py`. Earlier versions scored
  against hand-written raw targets, several of which fell outside the
  feature's actual range and contributed nothing. Relative band power does
  not average 0.2 per band — gamma averages 0.074 — so scoring against a
  nominal even spectrum handed a permanent bonus to any state weighting
  gamma negatively.

- **Affect coefficients are fitted**, not chosen (`tools/fit_affect.py`).

- **Valence uses frontal alpha asymmetry alone.** Regression finds raw alpha
  power predicts scripted valence slightly better, but only because relaxed
  alpha-rich phases happen to carry positive valence in the profile library.
  Fitting that would encode a quirk of the test data, not a real relationship.

- **Arousal variance is restored** by 1/r after fitting. A least-squares fit
  shrinks toward the mean, and a readout compressed into the middle third can
  never reach the `calm` or `fear` anchors.

- **Frontal band power uses a median across channels.** Blink artifacts land
  on the same frontal electrodes the asymmetry measure depends on; one
  contaminated lead dominates a three-channel mean. This change alone moved
  valence recovery from r=+0.297 to r=+0.547.

- **Signal quality grade thresholds are calibrated** to the composite score's
  actual distribution (~0.75–0.99), not an even split of 0–1, which would
  grade every session "excellent".

- **Memory attributes are logistic-squashed sums of z-scores.** Written as
  raw band values with additive constants, every attribute saturated above
  0.8 (familiarity spanned only 0.67–0.98) and the biome selector collapsed
  onto whichever setting expected the highest values. Attribute spread across
  profiles roughly doubled after the change.

- **Spatial scale deliberately excludes theta**, which is the familiarity
  marker. Sharing the term made the two attributes contradict each other for
  interior memories — theta-rich *and* small-scale — so no interior setting
  could ever be selected.

- **Biomes are matched by attribute *pattern*, then sampled.** Inferred
  attributes cluster near the middle of the range while biome profiles span
  the full axis, so nearest-neighbour scoring in raw units only ever returned
  mid-range settings. Cosine similarity on standardised vectors fixes the
  scale mismatch; seeded softmax sampling over the top candidates then
  reflects that leading scores are usually within a few percent of each other.
  Argmax presented that ambiguity as a determination.

  Current spread: 8 distinct biomes over 42 runs (7 profiles × 6 seeds), with
  per-profile variation staying inside semantically coherent groups — coastal
  recall yields ocean/beach/forest, ruins wander yields ruins/monumental/
  mountains. Selection remains deterministic for a given recording and seed.
