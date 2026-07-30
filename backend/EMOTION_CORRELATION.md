# Correlating EEG with emotion, as widely as the data allows

A deliberately wide search — 98 predictors against 3 targets, 294 correlations
— and a set of controls sized to match. The short answer is that nothing
survives, and one previously reported result does not survive either.

```bash
cd backend
.venv/bin/python -m tools.emotion_correlate          # the wide search
.venv/bin/python -m tools.decode_emotion             # now with a within-subject control
```

**Dataset.** ds003751 (DENS). 9 subjects, 95 trials, 23 feature windows per
trial. Valence and arousal self-rated 1–9, both spanning the full range.

---

## Why run this at all

[EMOTION.md](EMOTION.md) already reported that cross-subject emotion decoding
fails on this dataset. But it tested by **classifying trials into four
affective quadrants**, and binning a 1–9 rating into two levels throws away
most of its information. A continuous correlation is the more sensitive test,
so a negative classification result does not settle the question.

It also searched narrowly. This searches wide: every one of the 14 features
reduced seven ways within each trial — level, variability, linear trend, early
mean, late mean, early-to-late shift, and range — because an emotion might be
carried by how a signal *moves* through a clip rather than by its average.

## What guards a search this wide

294 correlations at p < 0.05 yield about 15 hits from noise alone. Reporting
the largest as a finding is the standard way emotion-EEG results fail to
replicate, so four guards run alongside:

1. **Correlation within subject, then pooled by Fisher z.** Absolute EEG power
   differs more between two skulls than between any two states in one person,
   so a correlation across pooled subjects would largely measure who was
   wearing the net.
2. **A permutation null that respects that.** 2,000 shuffles of the ratings
   *within* subject, identical draws for every test so no feature gets a lucky
   null.
3. **Benjamini-Hochberg across the whole family**, counting every test run.
4. **A trial-order control.** With ten trials per person, a feature that merely
   drifts across a session can correlate with anything else that drifts. Every
   correlation is recomputed with trial index partialled out.

---

## Result 1: nothing survives correction

| | |
| --- | --- |
| Tests run | 294 |
| Survive FDR at q < 0.05 | **0** |
| Hits at uncorrected p < 0.05 | 19 |
| Expected by chance | 15 |

Nineteen against fifteen expected. That is the whole result.

The strongest single association is `beta_alpha_ratio` late in the trial
against arousal: **r = −0.461** (partial r = −0.536 with trial order removed),
p = 0.001 uncorrected, **q = 0.147**.

It is worth naming because it is interesting and because it is *not* a result.
Two things stand out about it:

- **The direction is backwards from convention.** Higher self-reported arousal
  goes with a *lower* beta/alpha ratio and *more* posterior alpha. The standard
  reading — the one `app/scene/cognitive.py` uses to drive colour and travel
  speed — has beta/alpha rising with arousal.
- **Every subject agrees on that sign.** 9 of 9, and a jackknife moves the
  pooled r by at most ±0.067, so no single subject carries it.

## Result 2: the unanimity is not exceptional either

Nine of nine looks compelling until it is counted properly. Across all 196
predictor × target combinations tested for sign:

| | |
| --- | --- |
| Combinations showing unanimous sign | **1** |
| Expected by chance (2 × 0.5⁹ × 196) | **0.8** |

Observing one is exactly what 196 coin-flip sign tests produce. And no feature
*family* shows three or more unanimous aggregates — if this were a mechanism,
`beta_alpha_ratio::mean`, `posterior_alpha::late` and `alpha_suppression::late`
should agree too. They do not.

So the sign consistency is not evidence. It is the one unanimous result a
search this wide was always going to contain.

## Result 3: no combination works either

Cross-validated ridge over all 98 predictors, correlated within subject and
pooled:

| Target | Regime | Real | Shuffled |
| --- | --- | --- | --- |
| arousal | leave-one-subject-out | 0.177 | 0.120 |
| valence | leave-one-subject-out | −0.190 | −0.106 |

Barely separable from its own control for arousal, and worse than the control
for valence.

---

## Result 4: the within-subject figure had no control, and fails once given one

This is the substantive change to the record.

`EMOTION.md` reported that arousal is weakly readable inside a calibrated
subject — r = +0.19, 58.8% balanced accuracy, against a stated chance of 50%.
That was the one affective reading the project claimed survived.

**It had no shuffled control.** `within_subject()` took no `shuffle` argument;
only the cross-subject path was controlled, and "chance = 50%" was an
assumption. Adding the control:

| Within-subject arousal | Real | Shuffled control |
| --- | --- | --- |
| Correlation | +0.188 | **+0.220** |
| Balanced accuracy | 58.8% | **63.1%** |

| Within-subject valence | Real | Shuffled control |
| --- | --- | --- |
| Correlation | +0.064 | −0.215 |
| Balanced accuracy | 42.8% | 33.1% |

**The shuffled control beats the real model on arousal**, on both metrics —
the same failure the cross-subject quadrant test showed (38.9% against 34.7%).

Note also that the control scores *above* the assumed 50%. Leave-one-out over
ten trials is structurally biased: dropping one sample pulls the training mean
away from the held-out value, which inflates the metric on its own. The real
baseline for this regime is around 63%, not 50%, and 58.8% sits below it.

The same artifact shows in the multivariate model, where within-subject
leave-one-out returned r = −0.536 real against −0.606 shuffled — both large and
negative, which is a property of the design rather than of either model.

---

## What this means

**There is no emotion decoding in this dataset, in any regime tested.** Four
independent guards agree: FDR, the uncorrected hit count, the sign-unanimity
count, and the shuffled controls in both regimes.

For the product, the consequence is narrow but real. `app/scene/cognitive.py`
binds `frontal_asymmetry` to the warmth of the light and `beta_alpha_ratio` to
colour and travel speed, both labelled *interpretation* rather than
measurement. That labelling was already right, and is now the only thing
holding those two channels up: their affective reading has no support here, and
the one directional hint in the data points the opposite way to the convention
they encode. The band powers themselves remain exact — they are readings, not
predictions — which is why the memory-of-place render is unaffected.

### What would change the answer

- **More subjects.** 9 subjects × ~10 trials is 95 points against 98
  predictors. DENS ships 40 subjects; the local copy has 9. A search this wide
  needs several times the data before a moderate effect could clear FDR.
- **More trials per subject.** Ten is too few for the within-subject regime to
  behave, as the shuffled control demonstrates.
- **Continuous ratings rather than one per clip.** A single retrospective
  rating for a whole clip cannot be matched to a moment; DENS has click
  markers, and the per-moment test around them also fails (effect size −0.12).
- **A different measurement.** As everywhere else in this project, this
  dominates the rest.
