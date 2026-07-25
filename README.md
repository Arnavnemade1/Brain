# MindScape

**A neural memory reconstruction platform.** MindScape transforms scalp EEG brainwave signals into explorable digital environments, recovering watched experiences and scoring the result against the original reference stimulus.

> **"Entering Someone Else's Reality"** — What if memories, dreams, and emotions could become places you could physically enter and explore? Read our full vision & architecture: [`docs/VISION_AND_INSPIRATION.md`](docs/VISION_AND_INSPIRATION.md).

---


## Real human EEG

The headline capability runs on **recordings of actual people** viewing 200
objects — not simulation. From held-out brain activity the system ranks all 200
candidates and returns the best matches.

| | Result | Chance |
| --- | --- | --- |
| Exact object recovered (200-way) | **6.1%** | 0.5% |
| Correct object in top 5 | **17.8%** | 2.5% |
| Animate vs inanimate | **67.4%** | 50% |
| Category (10-way) | **32.4%** | 10% |

Twelve times chance for an exact hit — which also means the top candidate is
wrong 94% of the time. The errors are the informative part: when the exact
object is not first, the candidates beating it are usually from the same
category. **EEG carries what kind of thing was seen far more reliably than
which particular one.**

Controls: shuffled labels collapse to chance (0.88%), pre-stimulus decoding
sits at 49.9%, peak at +284 ms, and a strict time split matches the
cross-validated numbers.

Full methodology, the dataset choice, and why generation was rejected in
favour of retrieval: [`backend/REAL_DATA.md`](backend/REAL_DATA.md).

---

## Simulated end-to-end pipeline

The video-reconstruction pipeline below runs on synthetic EEG from a forward
model of visual response, which lets it be evaluated frame by frame against a
known reference.

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

> The EEG in *this* section is **synthetic**, produced by a forward model of
> visual response. Reported fidelity is an upper bound. The real-EEG results
> above are measured on human recordings.

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

# Real human EEG
.venv/bin/python -m tools.fetch_dataset --subjects 4   # ~2.4 GB from OpenNeuro
.venv/bin/python -m tools.decode_real                  # accuracy + leakage controls
.venv/bin/python -m tools.reconstruct_real --trials 8  # retrieval + contact sheet

# Simulated pipeline
.venv/bin/python -m tools.train_decoders   # fit decoders, report held-out recovery
.venv/bin/python -m tools.evaluate         # score reconstructions vs baseline
```

---

## Architecture

```
backend/app/
  realdata/        real EEG: BrainVision reader, epoching, decoding, retrieval
  stimulus/        reference clips and ground-truth visual features
  simulation/      stimulus-locked EEG synthesis (forward visual model)
  processing/      filtering, spectral estimation, quality, synchronization
  encoder/         temporal encoder, latent projection, analysis pipeline
  reconstruction/  fitted decoders, scene/frame decoding, refinement, runner
  evaluation/      SSIM, PSNR, correlations, scene-boundary F1, baselines
  api/ websocket/  REST and streaming transport
  services/        session assembly and persistence

frontend/src/
  pages/RealSubjects    what a person saw beside what their EEG recovered
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
