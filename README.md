# MindScape

**A memory reconstruction platform.** MindScape recovers a watched video from
the EEG recorded while it played, and scores the result against the original.

The objective is measurable reconstruction fidelity — not artistic
visualisation. Memories rather than dreams, precisely because a reference event
exists: every claim the system makes can be checked against what was actually
shown.

---

## What it actually recovers

Honest numbers, on clips never seen during decoder fitting:

| | Reconstruction | Mean-frame baseline |
| --- | --- | --- |
| **Composite fidelity** | **0.489** | 0.187 |
| SSIM | 0.439 | **0.508** |
| Luminance tracking | r = +0.32 | 0 |
| Motion tracking | r = +0.57 | 0 |
| Scene boundary F1 | 0.69 | 0 |

**The reconstruction loses to a constant average frame on SSIM.** That is the
most important number here, and it is not a defect. Scalp EEG carries almost no
spatial information, so an honest reconstruction cannot beat a spatially smooth
constant on a spatial metric. What it does recover is *temporal* — when the
scene brightened, when it moved, where it cut — which the baseline scores zero
on by construction. Hence the 2.3× composite margin.

Full methodology and every calibration decision: [`backend/CALIBRATION.md`](backend/CALIBRATION.md).

> The EEG here is **synthetic**, produced by a forward model of visual response.
> Reported fidelity is an upper bound; real recordings carry noise and
> individual variability the model does not capture.

---

## The pipeline

```
Video → EEG Recording → Signal Synchronization → Artifact Removal
      → Temporal Neural Encoder → Neural Memory Latent Space
      → Scene Reconstruction → Frame Reconstruction → Video Refinement
      → Memory Replay
```

Each stage is a real operation, visible live in the console with its own
latency and throughput:

- **Signal Synchronization** — recording and playback clocks run independently.
  Offset is recovered to within 0.4–3.1 ms from event markers. Drift is
  reported *only* when the marker geometry can support the estimate.
- **Artifact Removal** — common-average referencing, zero-phase 0.5–45 Hz
  band-pass, mains notch, and robust MAD artifact compression that preserves
  temporal continuity rather than excising segments.
- **Temporal Neural Encoder** — seven visual-response measures: occipital
  luminance drive, alpha suppression, motion response, evoked transients,
  detail response, and lateral/vertical field balance.
- **Neural Memory Latent Space** — a 16-dimensional trajectory carrying the
  current measurement plus temporal context. Nothing downstream reads raw signal.
- **Frame Reconstruction** — a 32×18 grid. Small by construction: spatial
  variation is scaled by *measured* layout reliability, so where layout is
  unrecoverable the output collapses toward a uniform field rather than
  asserting structure it does not have.
- **Memory Replay** — reconstruction and reference side by side, with per-frame
  and aggregate fidelity.

---

## Running it

Requires Python 3.9+ and Node 20+.

```bash
make install
```

Fit the decoders (required — without weights the system falls back to an
untrained mapping and says so):

```bash
cd backend && .venv/bin/python -m tools.train_decoders
```

Run both services:

```bash
make dev
```

Then open <http://localhost:5173>.

| Target | Purpose |
| --- | --- |
| `make backend` | FastAPI on :8000 |
| `make frontend` | Vite on :5173 |
| `make check` | TypeScript project check plus backend import smoke test |
| `make build` | Production frontend build |

### Verification harnesses

```bash
cd backend
.venv/bin/python -m tools.train_decoders   # fit decoders, report held-out recovery
.venv/bin/python -m tools.evaluate         # score reconstructions vs baseline
```

---

## Architecture

```
backend/app/
  stimulus/        reference clips and ground-truth visual features
  simulation/      stimulus-locked EEG synthesis (forward visual model)
  processing/      filtering, spectral estimation, quality, synchronization
  encoder/         temporal encoder, latent projection, analysis pipeline
  reconstruction/  fitted decoders, scene/frame decoding, refinement, runner
  evaluation/      SSIM, PSNR, correlations, scene-boundary F1, baselines
  api/ websocket/  REST and streaming transport
  services/        session assembly and persistence

frontend/src/
  components/replay/    the Memory Replay surface and fidelity dashboard
  components/session/   pipeline flow, telemetry, launcher
  components/viz/       spectrum, topography, waveform
  components/landing/   marketing sections
  stores/ services/     Zustand state, REST and WebSocket clients
  types/                the contract mirrored by the Pydantic models
```

The TypeScript types in `frontend/src/types` and the Pydantic models in
`backend/app/models` are deliberate mirrors — the backend serialises camelCase
so no mapping layer is needed between them.

---

## Design notes

**Why memories and not dreams.** Dreams cannot be validated. A watched video
gives a reference against which reconstruction quality is an objective
measurement rather than a matter of taste.

**Why the reconstruction is deliberately low-resolution.** Predicting per-pixel
detail from scalp EEG would be predicting noise, and the metrics would
correctly punish it. The resolution is set by what the signal supports.

**Why every metric is shown against a baseline and a ceiling.** An SSIM of
0.44 is uninterpretable alone. Against a trivial baseline and the practical
information ceiling, the same number becomes a readable result.

---

## Configuration

Everything has a working default; copy `.env.example` to `.env` only to
override. Sample rate, window length and overlap are the settings most worth
changing — they alter what the encoder can resolve, so re-run the training
harness after touching them.
