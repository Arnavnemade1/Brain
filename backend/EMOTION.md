# Per-frame emotion from continuous EEG

Video output, and a negative result worth reading carefully.

**Dataset.** ds003751 — Mishra, Asif, Tiwary & Srinivasan (2023), *Dataset on
Emotion with Naturalistic Stimuli (DENS)*. Forty subjects watched emotionally
evocative video clips with EEG recorded continuously on a 129-channel EGI net
at 250 Hz. Nine subjects, 95 trials and 2,185 feature windows were used here.

Reproduce:

```bash
cd backend
.venv/bin/python -m tools.fetch_emotion --subjects 10   # ~1.2 GB
.venv/bin/python -m tools.decode_emotion               # validation + controls
.venv/bin/python -m tools.render_memory --subject sub-mit003   # writes an .mp4
```

---

## What works: the video

`tools/render_memory.py` writes a real `.mp4` — 960×540, 25 fps, one frame per
moment of the viewing. Each frame carries the decoded valence and arousal at
that instant, the trajectory so far drawn on the affective circumplex, the
nearest named emotion, the affective quadrant, a colour-coded timeline of the
whole clip, and a marker at the moment the subject clicked to say they felt
something.

This part is solid. It is a genuine per-frame emotional reconstruction rendered
as a playable video file.

## What does not work: the decoding underneath it

| Cross-subject (leave-one-subject-out) | Result | Chance |
| --- | --- | --- |
| Valence, balanced accuracy | 49.2% | 50% |
| Arousal, balanced accuracy | 47.9% | 50% |
| Valence correlation | r = −0.20 | 0 |
| Quadrant, 4-way | 34.7% | 25% |
| — **shuffled control, quadrant** | **38.9%** | 25% |

**The shuffled control scores higher than the real model.** That is the whole
story: there is no cross-subject emotion decoding here. A model fitted on eight
people does not read the ninth.

Within-subject, with the model calibrated on that person's own other trials:

| Within-subject (leave-one-trial-out) | Result | Chance |
| --- | --- | --- |
| Arousal correlation | **r = +0.19** | 0 |
| Arousal, balanced accuracy | **58.8%** | 50% |
| Valence correlation | r = +0.06 | 0 |
| Valence, balanced accuracy | 42.8% | 50% |

So: **arousal is weakly readable within a calibrated subject. Valence is not
readable at all.** The rendered video uses within-subject calibration for that
reason, and its footer says so on every frame.

### The per-moment test also fails

DENS subjects clicked at the moment they felt the emotion. Nothing in fitting
ever sees those clicks, so they are an independent check on whether the decoder
tracks anything *within* a trial. Decoded emotional intensity near a click was
0.295 against 0.314 elsewhere in the same trials — a difference of −0.02,
effect size −0.12, in the wrong direction, over 143 clicks.

The within-trial variation the video displays is therefore **not validated**.
It is what the model outputs moment to moment, and it should be read as the
decoder's fluctuation rather than as the subject's changing feeling.

---

## Two measurement errors found on the way

Both inflated results before they were caught, and both are the kind that
quietly survive into published numbers:

1. **Wrong chance level.** Valence classes split 62/38, so a decoder that
   always guesses the majority scores 62%. The first run reported "61.1%
   accuracy vs 50% chance" and looked like a positive result. Against the
   correct majority baseline it was performing *below* trivial. Every accuracy
   here is now balanced accuracy, which sits at 50% for any constant guess.

2. **No baseline normalisation.** DENS records ~20 s of rest before the clips.
   Absolute EEG power differs more between two people's skulls than between
   two emotions in the same person, so features are now expressed relative to
   each subject's own resting state. This was a genuine methodological fix; it
   did not rescue the cross-subject result.

One thing checked and cleared: the frontal-asymmetry sign convention. The
negative valence correlation looked like it might be an inverted left/right
axis, but E22/Fp1 sits at Y=+3.08 and E9/Fp2 at Y=−3.08, confirming Y+ = left
and the formula correct. The negative correlation is noise, not a bug.

---

## Why this is honest rather than tuned

With nine subjects and 95 trials, cross-subject affective decoding is
underpowered. Published emotion-recognition accuracies mostly come from DEAP
(32 subjects, 40 trials each) or SEED (15 subjects, many more trials), and the
strong ones are usually **within-subject** — a different and much easier
question than the one tested here.

It would have been easy to keep adjusting the feature set, the regulariser and
the window length until something crossed significance on 95 trials. That is
p-hacking, and the result would not replicate. The decoder was fitted once with
literature-standard features, validated properly, and reported as it came out.

---

## What would make this work

- **More subjects.** Nine is far too few for cross-subject transfer. DEAP-scale
  data (32+) is the minimum for the cross-subject claim.
- **Per-subject calibration.** Already the stronger path here, and the
  realistic one: a headset worn for a bowling night would calibrate on its
  wearer first.
- **Continuous emotion annotation.** DENS labels a whole clip with one rating.
  Per-frame decoding cannot be properly supervised — or properly scored —
  without continuously-annotated affect, which datasets like RECOLA provide.
- **Physiological channels.** DENS also records ECG and EMG. Heart rate and
  facial muscle activity carry more arousal and valence information than scalp
  EEG does, and both are already in these files, unused.
