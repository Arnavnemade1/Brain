# An environment reconstructed from EEG

Video: `tools/render_environment_video.py`. Two landscapes side by side —
one graded from the photograph that was on screen, one from EEG alone.

![a frame](../docs/environment_frame.png)

```bash
cd backend
.venv/bin/python -m tools.render_environment_video --seconds 60
```

---

## The measurement this rests on

Object identification on this dataset is 7.6% exact. That is far too weak to
build a world from. But identification is the wrong thing to ask of EEG.

Regressing EEG onto the **visual statistics** of what was on screen — held
out, averaged over repetitions, correlated across the 200 stimuli — gives:

| Property | r | shuffled |
| --- | --- | --- |
| edge density | **0.69** | 0.07 |
| mean red | 0.64 | −0.02 |
| saturation | 0.64 | 0.05 |
| luminance | 0.62 | −0.01 |
| contrast | 0.58 | 0.01 |
| coverage | 0.53 | 0.05 |
| mean blue | 0.47 | 0.04 |

Shuffled controls stay under |r| = 0.13 throughout.

**Scalp EEG carries how a scene looked far better than what was in it.** A
landscape is exactly the kind of output that low-level visual statistics can
honestly drive, which is why this works where object reconstruction does not.

## Is it cortical, or is it the eyes?

This measurement had no artifact control until now, which was a gap: visual
stimuli evoke blinks and saccades, the corneo-retinal dipole is an order of
magnitude larger than cortical signal, and it lands hardest on frontal
electrodes. Brightness drives pupil and lid behaviour directly — so "luminance
decodes at r = 0.62" is precisely the result an ocular artifact would produce.

Fitting the same regression on electrode groups separately:

| Group | channels | luminance | edge density |
| --- | --- | --- | --- |
| **occipital** | 8 | **0.561** | **0.663** |
| parietal | 9 | 0.521 | 0.604 |
| temporal (muscle) | 8 | 0.409 | 0.498 |
| central | 7 | 0.373 | 0.397 |
| **frontal pole (ocular)** | 7 | **0.302** | **0.329** |

**Occipital carries roughly double what the ocular sites do**, with a clean
posterior-to-anterior gradient. That is what visual decoding looks like and the
opposite of EOG contamination. Every group scores above zero, which is expected
— common-average referencing spreads a strong focal source everywhere — but the
ordering is unambiguous.

A spectral control was also run and is **not** informative here. The motor
controls could lean on frequency because mu/beta desynchronisation is an
oscillation, so band-splitting separates it from broadband muscle. A visual
evoked response is a transient, and a transient is broadband by construction:
luminance decodes at 0.52, 0.52, 0.49 and 0.52 across four bands from 0.5 to
40 Hz. Flat is what genuine ERP decoding looks like, so the test settles
nothing. Reported because it was run.

## How few electrodes does this need?

If eight occipital sites carry most of it, a research cap is not required to
reproduce this.

| Montage | channels | luminance | edge density |
| --- | --- | --- | --- |
| all | 63 | 0.602 | 0.687 |
| 10-20 standard | 18 | 0.566 | 0.636 |
| occipital | 8 | 0.561 | **0.663** |
| occipital | 4 | 0.542 | 0.629 |
| occipital | 2 | 0.450 | 0.489 |

Two things fall out, and the second is the useful one.

**Four occipital electrodes retain about 92% of the full result** (0.629 against
0.687 on edge density). The marginal value of channels 5 through 63 is small.

**Eight occipital electrodes beat the nineteen-channel 10-20 montage** — 0.663
against 0.636 — despite having fewer than half as many. The 10-20 spreads
sensors over the whole head, including frontal and temporal sites that this
table shows carry little; the occipital eight are all signal. **Placement
dominates count.** For anyone planning a recording, that is worth more than any
other number on this page: buy position, not channels.

## What is decoded and what is not

| Decoded from EEG | Fixed prior |
| --- | --- |
| brightness | the landform |
| haze in the distance | the camera |
| colour saturation | sun position, and every shadow |
| warmth of the light | the material palette |
| coolness of the sky | |
| how bare the ground is | |
| how high the water stands | |

The terrain is generated before any brain activity is read and never changes.
Only appearance moves. A property must clear r = 0.35 held-out before it is
allowed to drive anything; everything else is pinned at its prior value, and
the render states which is which.

**Mean per-moment agreement between the two panels: r = 0.38.**

## How to read it honestly

The person was looking at photographs of isolated objects, not at a valley.
The left panel is not what they saw — it is what their photograph's statistics
imply about a scene, and the right panel is the same thing derived from their
brain activity. Where the two agree, the decode worked.

This is a reconstruction of **appearance**, not of place. Nothing here
recovers geometry, and no amount of modelling would: the landform is one
generated heightfield out of infinitely many, and EEG says nothing about which.

## Why the renderer is split in two

Marching rays into a heightfield costs about a minute a frame. Re-shading a
prepared view costs 0.02 s — 3,500 frames a minute. Separating geometry from
shading is what makes a video possible at all, and it enforces the honest
constraint at the same time: geometry is computed once, before any decoding,
so it *cannot* vary with the signal even by accident.
