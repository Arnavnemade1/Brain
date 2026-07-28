# A memory of place, from brain state

A continuous EEG recording of someone having an experience, rendered as
terrain you travel through. Light, air, colour, ground and the speed of
travel all follow the recording moment to moment.

![a frame](../docs/memory_of_place_frame.png)

```bash
cd backend
.venv/bin/python -m tools.render_memory_of_place --seconds 60
```

---

## Measured, not inferred

This is the distinction the whole thing rests on, and it is what separates
this from the earlier reconstructions.

When this project decoded *what someone was looking at*, there was an
inference gap that had to be quantified every time: object identity 7.6%
correct, visual properties r = 0.47 to 0.69, cross-subject emotion a
documented failure. Each of those numbers is an error bar on a guess about
the world.

**Band power is not a guess.** Alpha power over occipital cortex *is* alpha
power over occipital cortex. Total power is the amplitude that was there. The
five rhythms, the signal intensity and the way they move over time are direct
readings of the recording, so binding a landscape to them is a faithful
rendering of the signal rather than a claim about the world. There is no
accuracy figure here because there is nothing being predicted.

## The bindings

| Channel | Status | Drives |
| --- | --- | --- |
| signal intensity (total power) | **measured** | how vivid the scene is |
| gamma power | **measured** | sharpness of detail |
| posterior alpha | interpretation | how far the world recedes into haze |
| alpha suppression | interpretation | clarity of the air |
| beta / alpha | interpretation | colour, and speed of travel |
| frontal theta | interpretation | how bare the ground is |
| theta / beta | interpretation | how the light fades |
| frontal asymmetry | interpretation | warmth of the light |

Rows marked *interpretation* are places where a cognitive meaning is being
assumed on top of a measurement — that posterior alpha indexes visual
disengagement, that beta over alpha indexes arousal, and so on. These are
standard readings and each is contested at the edges.

Frontal asymmetry gets the **smallest influence of anything here**, on
purpose: it is the one reading this project has already tested and found
wanting. Cross-subject emotion decoding on this very dataset failed, with a
shuffled control scoring *higher* than the real model. See
[`EMOTION.md`](EMOTION.md).

Every channel that moves the picture is drawn on the frame, colour-coded by
status, so nothing can shape the scene without being visible.

## What this is not

The subject was watching video clips in a laboratory. **Nobody in this dataset
was on a mountain.** The terrain is a generated heightfield fixed before any
EEG is read, and it never varies — not with the signal, not between frames.

So: the *place* is not decoded. The *state* is. What you are watching is a
real brain-state trajectory given a landscape to move through, not a
recovered memory of a landscape.

To make this a genuine reconstruction of somewhere, the recording has to come
from someone actually there — see "What a real recording needs" below.

## Standardisation is within subject

Absolute EEG power differs more between two people's skulls than between any
two states in the same person. Features are z-scored within subject before
anything is rendered; without that the scene would be showing head size and
electrode impedance.

## Why the renderer is split in two

Marching rays into a heightfield costs about a minute a frame. Re-shading a
prepared view costs 0.02 s. Geometry is computed once per camera keyframe —
44 of them along the path — and every frame in between is a re-shade.

That is what makes an hour-long idea into a ten-minute render, and it also
enforces the honesty constraint mechanically: geometry is built before any
brain state is read, so it *cannot* vary with the signal even by accident.

## What a real recording needs

For this to reconstruct a place rather than illustrate a state:

- **EEG recorded on location**, wearing the headset while actually there.
- **A calibration block** — a few minutes of eyes-open and eyes-closed rest
  gives the per-subject baseline that every figure here is expressed against.
- **Synchronised video** from the wearer's point of view. Without ground
  truth there is nothing to score, which is the exact wall that stopped the
  naturalistic-video dataset ([`REALLIFE.md`](REALLIFE.md)) from supporting a
  picture at all.
- **Expect motion artifact.** Walking on uneven ground floods scalp EEG with
  muscle activity. The three controls in [`MOTOR.md`](MOTOR.md) — frequency
  profile, muscle-site decoding, and the spatial check — are what would tell
  you whether a result is cortical or just your neck.
