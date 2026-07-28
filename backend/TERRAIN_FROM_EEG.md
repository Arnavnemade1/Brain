# Can the terrain stop being fixed?

In every reconstruction so far the heightfield is chosen before any EEG is
read. Only *appearance* is decoded — light, haze, colour, ground cover. The
landform itself is prior, and both `ENVIRONMENT.md` and `MEMORY_OF_PLACE.md`
say so on every frame.

This asks whether the shape could come from the signal instead.

```bash
cd backend
.venv/bin/python -m tools.spatial_limit --subjects 3
```

**Short answer: yes, at roughly nine or ten degrees of freedom.** Enough for
broad landform — where the ground rises and falls across a scene. Nowhere near
enough for ridgelines, and detail would still have to be prior.

---

## Method

Reduce each stimulus to a coarse luminance grid, regress EEG onto every cell
of that grid held out, and correlate the decoded grid against the true one
across the 200 stimuli. A cell must clear **r = 0.35** — the same bar the
appearance work uses, so the two are directly comparable.

Three subjects from ds004018, repetition-balanced folds, ridge regression.

## Per-cell correlation barely degrades with resolution

| grid | cells | mean r | best | worst | shuffled | above 0.35 |
| --- | --- | --- | --- | --- | --- | --- |
| 1×1 | 1 | 0.606 | 0.606 | 0.606 | 0.011 | 100% |
| 2×2 | 4 | 0.560 | 0.629 | 0.489 | 0.031 | 100% |
| 4×4 | 16 | 0.527 | 0.643 | 0.369 | 0.043 | 100% |
| 8×8 | 64 | 0.480 | 0.615 | 0.293 | 0.053 | 94% |
| 12×12 | 144 | 0.458 | 0.619 | 0.253 | 0.055 | 87% |

Taken at face value this says 144 height values are recoverable, which would
be a remarkable claim. It is the wrong reading, and the rest of this document
is why.

## The cells are not independent

These stimuli are objects photographed on a plain background. Most cells are
therefore driven by one underlying factor — how much of the object covers them
— which is close to overall coverage and luminance, and both of those decode
well on their own. A hundred and forty-four correlated numbers is not a
hundred and forty-four degrees of freedom.

Two checks separate real layout from one global number wearing a grid costume.

**Strip each image's own mean brightness.** What survives is layout alone.

| | mean r | above 0.35 |
| --- | --- | --- |
| cells, image mean removed | **0.467** | 92% |

So it is *not* purely global brightness. Real spatial structure is there.

**Count the principal components that survive.** This is the honest number.

| component | variance | r | |
| --- | --- | --- | --- |
| 1 | 32.6% | 0.620 | recovered |
| 2 | 14.4% | 0.638 | recovered |
| 3 | 7.8% | 0.446 | recovered |
| 4 | 6.8% | 0.394 | recovered |
| 5 | 5.6% | 0.435 | recovered |
| 6 | 4.0% | 0.502 | recovered |
| 7 | 3.6% | 0.465 | recovered |
| 8 | 2.5% | 0.391 | recovered |
| 9 | 2.0% | 0.225 | — |
| 10 | 1.7% | 0.380 | recovered |
| 11 | 1.6% | 0.335 | — |
| 12 | 1.4% | 0.239 | — |

**Nine of twelve clear the bar**, covering about 80% of the grid's variance.
That is the count that can honestly drive geometry.

---

## What this licenses

A heightfield built from **nine or ten decoded coefficients** over a smooth
spatial basis. In practice: a low-order surface — broad rises, a basin, a
ridge running one way rather than another — with everything finer supplied by
the fractal, at a declared amplitude, exactly as appearance already works.

That is a real improvement on a fixed landform. The shape would vary with the
recording, and the variation would be earned rather than decorative.

## What it does not license

**Ridgelines, peaks, valleys as such.** Ten coefficients over a scene is
roughly a 3×3 surface. Anything sharper is the fractal, and the fractal is
invention.

**A claim that this is the terrain they saw.** The measurement comes from
people viewing *objects on plain backgrounds*. The luminance layout of a
photograph of a crab is not a landscape, and nothing here shows that the
spatial structure of an actual outdoor scene decodes the same way. It is a
plausible transfer and it is untested.

**Any of it without ground truth.** Every number above exists because
ds004018 ships the images. A recording made on a mountain with no synchronised
video has nothing to score against, and the same pipeline would produce a
confident-looking heightfield with no way to know if it were meaningless.

---

## The honest next step

Decoded landform is worth building, but it should not ship as "the terrain is
now decoded". It should ship the way appearance did:

1. Decode the components, hold the bar at r = 0.35, and drop any that miss.
2. Render the decoded surface **beside the true one**, as
   `render_environment_video.py` already does for appearance, so the agreement
   is visible rather than asserted.
3. State the split on the frame: this many coefficients from EEG, everything
   finer from the fractal.

Until a dataset exists with EEG recorded against real outdoor scenes, that is
as far as this goes — and the fixed-terrain caveat in `MEMORY_OF_PLACE.md`
stays true for anything recorded outside a lab.
