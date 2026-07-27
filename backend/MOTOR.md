# Reconstructing what the body was doing

The earlier reconstructions ask what someone was *looking at*. This one asks
what they were *doing* — and it is the question scalp EEG is best equipped to
answer.

**Dataset.** PhysioNet EEG Motor Movement/Imagery Database (eegmmidb), Schalk
et al. 2004. 64-channel BCI2000 at 160 Hz. Fourteen subjects here, recorded
while they physically opened and closed their fists and feet on cue.

Only the **executed movement** runs are used. The database also contains
matched motor *imagery* runs, which are the ones most BCI papers use; the
question here is what the body actually did, so imagined movement is the wrong
condition and was never downloaded.

Reproduce:

```bash
cd backend
.venv/bin/python -m tools.fetch_motor --subjects 14        # ~210 MB
.venv/bin/python -m tools.decode_motor --subjects 14       # the tables below
.venv/bin/python -m tools.render_motor_video --subject S002
```

![a frame from the reconstruction](../docs/motor_video_frame.png)

---

## Why this works better than reading vision

Moving a limb suppresses the mu (8-13 Hz) and beta (13-30 Hz) rhythms over the
**opposite** sensorimotor strip — event-related desynchronization. Feet, whose
cortical representation sits on the midline between the hemispheres, suppress
it over the vertex instead.

That is a large effect with a clear spatial signature at exactly the scale
scalp electrodes can resolve. It does not require recovering fine detail, which
is what made the object work hard.

---

## Results

Fourteen subjects, leave-one-run-out: the model is trained on a person's other
runs and never on the one it decodes. Every figure is **balanced** across
actions — rest is 54% of every recording, so a decoder answering "rest" and
nothing else would score 54% raw.

| Per 1-second window | Real | Shuffled | Chance | Range |
| --- | --- | --- | --- | --- |
| Exact action, five ways | **30.0%** | 18.5% | 20% | 19-46% |
| Right limbs, action may be wrong | **53.6%** | 24.0% | — | 38-83% |
| Right about moving at all | **66.7%** | 30.3% | — | 54-86% |

Broken into the questions that are actually separable:

| Sub-question | Real | Chance | Range |
| --- | --- | --- | --- |
| Moving vs rest | **72.6%** | 50% | 63-87% |
| Fists vs feet | **74.8%** | 50% | 55-99% |
| Left vs right fist | **60.6%** | 50% | 52-79% |
| Which of four limbs | 27.0% | 25% | 8-48% |

**The gradient is the finding.** How well a distinction decodes tracks how far
apart the body parts sit on the cortex. Hands versus feet — opposite ends of
the motor homunculus — is easy. Left hand versus right hand, a few centimetres
apart across the midline, is barely better than a coin. Which of four limbs,
which requires both distinctions at once, is close to chance.

### Grading the errors

Scoring only exact matches hides that structure, the same way it did for the
object reconstruction. So errors are graded: answering *both fists* when the
truth was *right fist* means the decoder saw hand movement and could not
lateralise it, which is a different quality of error from answering *both
feet*. That is the gap between the 30.0% exact and the 53.6% right-limbs row.

---

## Read the spread, not the mean

The range column is not noise. Some people's sensorimotor rhythms are simply
much more readable than others': S004 reaches 44.6% exact while S011 sits at
18.5%, below the 20% chance line. This split is well documented and it is why
a single-subject number from this dataset means very little on its own.

**The shipped video is S002, one of the stronger subjects**, and every frame
of it says so along with the group mean and range.

---

## Two things that did not work

**Common spatial patterns.** The standard front end for this kind of decoding,
implemented and fitted per fold on training data only, one-versus-rest. It came
out +0.6 points on moving-versus-rest, +1.5 on left-versus-right fist, and
**-3.9** on fists-versus-feet, and 2.5 points worse on the five-class problem.
It did not earn its complexity here and the module was deleted; features are
the plain per-channel log band powers, which have the side benefit of staying
individually nameable.

**Constraining the decoder to the actions a run could contain.** Each run only
ever contains two of the four movements, so restricting the posterior to those
lifts raw accuracy from 32.2% to 41.2%. But chance rises from 20% to 33% at the
same time, so the effect size *falls* — 1.25x chance against 1.61x. The
constraint mostly hands the model the easy fists-versus-feet call. The
unconstrained decoder is the honest one, and the confusions it makes are
informative rather than embarrassing.

---

## What this is not

These are cued, isolated limb movements in a lab, not cooking or sport. That
gap is not an oversight: vigorous whole-body movement floods scalp EEG with
muscle and motion artifact, which is why public datasets pairing EEG with
free real-world activity — and with video of it — barely exist. The honest
version of "reconstruct someone playing a sport" needs either a measurement
that tolerates movement, or a dataset that does not yet exist publicly.

## What would raise the ceiling

- **More electrodes over sensorimotor cortex, or a Laplacian montage.** The
  left/right distinction is spatially fine and 64 channels spread over the
  whole scalp under-samples exactly where it matters.
- **Per-subject band tuning.** Peak mu frequency varies by several Hz between
  people; fixed 8-13 Hz costs the subjects whose peak sits outside it.
- **Riemannian tangent-space features.** The natural next thing to try after
  CSP failed, and it works on covariances that are already being computed.
