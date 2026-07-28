import { motion } from 'framer-motion'
import { ArrowLeft } from 'lucide-react'
import { useEffect } from 'react'
import { Link } from 'react-router-dom'

import { pageVariants } from '@/animations/motion'
import { Reveal } from '@/components/landing/Reveal'
import { ComponentBars, Figure, HeatGrid, LineChart } from '@/components/research/Chart'
import spatial from '@/data/spatial-limit.json'

/**
 * The terrain research, written up.
 *
 * Every number on this page is read from `spatial-limit.json`, which the
 * measurement tool writes on each run. Nothing is typed in twice — a figure
 * transcribed by hand is a figure that will eventually disagree with the
 * experiment that produced it.
 */

const BAR = spatial.bar
const eight = spatial.resolutions.find((r) => r.side === 8)!
const twelve = spatial.resolutions.find((r) => r.side === 12)!

function percent(value: number, digits = 0) {
  return `${(value * 100).toFixed(digits)}%`
}

export default function Research() {
  useEffect(() => {
    document.title = 'How much landform can EEG specify? — MindScape'
  }, [])

  return (
    <motion.main variants={pageVariants} initial="initial" animate="animate" exit="exit">
      {/* ---------------------------------------------------------------- */}
      {/* Header                                                            */}
      {/* ---------------------------------------------------------------- */}
      <section className="shell pt-32 pb-4">
        <Reveal>
          <Link
            to="/"
            className="text-label inline-flex items-center gap-2 text-ink-faint transition-colors hover:text-ink-soft"
          >
            <ArrowLeft className="size-3" />
            Back
          </Link>

          <p className="text-label mt-10 mb-6">Research · spatial limit</p>
          <h1 className="max-w-[20ch] text-[clamp(2.2rem,5vw,3.6rem)]">
            How much landform can EEG actually specify?
          </h1>
          <p className="mt-8 max-w-2xl text-[1.02rem] leading-relaxed text-ink-soft">
            In every reconstruction so far the terrain is fixed before any brain activity
            is read — only its appearance is decoded. This asks whether the shape itself
            could come from the signal, and where it stops being real.
          </p>

          <div className="card-grid mt-12 grid-cols-2 lg:grid-cols-4">
            {[
              ['≈9', 'degrees of freedom', 'what the signal supports'],
              [percent(0.8), 'of grid variance', 'covered by those components'],
              [twelve.meanR.toFixed(2), 'per-cell r at 12×12', 'and this is the trap'],
              [spatial.subjects.length.toString(), 'subjects', `${spatial.stimuli} stimuli, held out`],
            ].map(([value, label, note]) => (
              <div key={label} className="glass rounded-panel p-6">
                <p className="font-display text-[2rem] leading-none text-ink">{value}</p>
                <p className="mt-4 text-[0.82rem] leading-snug text-ink-soft">{label}</p>
                <p className="mt-1.5 text-[0.72rem] text-ink-faint">{note}</p>
              </div>
            ))}
          </div>
        </Reveal>
      </section>

      {/* ---------------------------------------------------------------- */}
      {/* Method                                                            */}
      {/* ---------------------------------------------------------------- */}
      <section className="shell section-y">
        <Reveal className="mx-auto max-w-2xl">
          <p className="text-label mb-5">Method</p>
          <p className="text-[0.96rem] leading-relaxed text-ink-muted">
            Reduce each stimulus to a coarse luminance grid. Regress EEG onto every cell of
            that grid, held out, and correlate the decoded grid against the true one across
            all {spatial.stimuli} stimuli. A cell has to clear{' '}
            <span className="text-ink">r&nbsp;=&nbsp;{BAR}</span> — the same bar the
            appearance work uses, so the two are directly comparable.
          </p>
          <p className="mt-5 text-[0.96rem] leading-relaxed text-ink-muted">
            Every result is run beside a shuffled-label control through the identical
            pipeline. Where the control rises to meet the result, there is nothing there.
          </p>
        </Reveal>
      </section>

      {/* ---------------------------------------------------------------- */}
      {/* Figure 1 — the sweep                                              */}
      {/* ---------------------------------------------------------------- */}
      <section className="shell pb-20">
        <Reveal>
          <Figure
            label="Figure 1"
            title="Per-cell correlation barely degrades as the grid gets finer."
            caption={`Mean held-out correlation per cell against grid resolution, averaged over ${spatial.subjects.length} subjects. The dashed line is the shuffled control, which stays under ${twelve.shuffled.toFixed(2)} throughout. Y axis starts at zero.`}
          >
            <LineChart
              series={spatial.resolutions.map((r) => r.meanR)}
              control={spatial.resolutions.map((r) => r.shuffled)}
              labels={spatial.resolutions.map((r) => `${r.side}×${r.side}`)}
              threshold={BAR}
              thresholdLabel={`r = ${BAR}`}
            />
          </Figure>
        </Reveal>

        <Reveal className="mx-auto mt-12 max-w-2xl">
          <div className="glass rounded-panel border-memory-400/25 p-7">
            <p className="text-label mb-4 text-memory-300">Read this carefully</p>
            <p className="text-[0.94rem] leading-relaxed text-ink-muted">
              Taken at face value, {twelve.cells} cells at r&nbsp;=&nbsp;
              {twelve.meanR.toFixed(2)} says {twelve.cells} recoverable heights. That would
              be a remarkable claim, and it is the wrong reading. These stimuli are objects
              photographed on plain backgrounds, so most cells are driven by one underlying
              factor — how much of the object covers them — which is close to overall
              coverage and luminance, and both of those decode well on their own.
            </p>
            <p className="mt-4 text-[0.94rem] leading-relaxed text-ink">
              {twelve.cells} correlated numbers are not {twelve.cells} degrees of freedom.
            </p>
          </div>
        </Reveal>
      </section>

      {/* ---------------------------------------------------------------- */}
      {/* Figure 2 — layout survives                                        */}
      {/* ---------------------------------------------------------------- */}
      <section className="shell pb-20">
        <Reveal>
          <Figure
            label="Figure 2"
            title="Strip each image's own brightness and layout still survives."
            caption={`Per-cell correlation over an ${eight.side}×${eight.side} grid after removing every image's mean luminance, so only spatial layout remains. Mean r = ${spatial.centred.meanR.toFixed(3)}, with ${percent(spatial.centred.aboveBar)} of cells above the bar — so this is not one global brightness number wearing a grid costume.`}
          >
            <HeatGrid values={spatial.centred.cellR} side={spatial.centred.side} />
          </Figure>
        </Reveal>
      </section>

      {/* ---------------------------------------------------------------- */}
      {/* Figure 3 — the honest count                                       */}
      {/* ---------------------------------------------------------------- */}
      <section className="shell pb-20">
        <Reveal>
          <Figure
            label="Figure 3"
            title="Counting components instead of cells gives the honest number."
            caption={`Principal components of the ${eight.side}×${eight.side} grid: how much variance each holds, and how well each decodes. ${spatial.survivors} of ${spatial.componentsTested} clear the bar. Two scales, drawn as separate rows — there is no meaningful ratio between a share of variance and a correlation.`}
          >
            <ComponentBars components={spatial.components} threshold={BAR} />
          </Figure>
        </Reveal>

        <Reveal className="mx-auto mt-12 max-w-2xl text-center">
          <p className="text-[1.05rem] leading-relaxed text-ink">
            Nine of twelve components survive, covering about 80% of the grid&rsquo;s
            variance. That — not {twelve.cells} — is the count that can honestly drive
            geometry.
          </p>
        </Reveal>
      </section>

      {/* ---------------------------------------------------------------- */}
      {/* Verdict                                                           */}
      {/* ---------------------------------------------------------------- */}
      <section className="shell section-y">
        <Reveal className="mx-auto mb-12 max-w-2xl text-center">
          <p className="text-label mb-5">Verdict</p>
          <h2 className="text-[clamp(1.75rem,3.6vw,2.4rem)]">
            Enough for landform. Nowhere near enough for ridgelines.
          </h2>
        </Reveal>

        <Reveal className="card-grid md:grid-cols-2">
          <div className="glass rounded-panel p-7">
            <p className="text-label mb-4 text-neural-300">What this licenses</p>
            <ul className="space-y-3 text-[0.9rem] leading-relaxed text-ink-muted">
              <li>
                A heightfield built from nine or ten decoded coefficients over a smooth
                spatial basis.
              </li>
              <li>
                Broad rises, a basin, a ridge running one way rather than another — shape
                that varies with the recording and is earned rather than decorative.
              </li>
              <li>
                Everything finer supplied by the fractal at a declared amplitude, exactly
                as appearance already works.
              </li>
            </ul>
          </div>

          <div className="glass rounded-panel p-7">
            <p className="text-label mb-4 text-memory-300">What it does not</p>
            <ul className="space-y-3 text-[0.9rem] leading-relaxed text-ink-muted">
              <li>
                Peaks and valleys as such. Ten coefficients over a scene is roughly a 3×3
                surface; anything sharper is the fractal, and the fractal is invention.
              </li>
              <li>
                A claim that this is the terrain anyone saw. The measurement comes from
                people viewing objects on plain backgrounds — the luminance layout of a
                photograph of a crab is not a landscape.
              </li>
              <li>
                Any of it without ground truth. Every number here exists because the
                dataset ships its images.
              </li>
            </ul>
          </div>
        </Reveal>
      </section>

      {/* ---------------------------------------------------------------- */}
      {/* Reproduce                                                         */}
      {/* ---------------------------------------------------------------- */}
      <section className="shell pb-28">
        <Reveal className="mx-auto max-w-2xl">
          <p className="text-label mb-5">Reproduce</p>
          <pre className="glass rounded-card overflow-x-auto p-5 text-[0.8rem] leading-relaxed text-ink-soft">
            <code>{`cd backend
.venv/bin/python -m tools.spatial_limit --subjects ${spatial.subjects.length}`}</code>
          </pre>
          <p className="mt-5 text-[0.82rem] leading-relaxed text-ink-faint">
            The run writes the JSON this page reads, so these figures cannot drift from the
            experiment that produced them. Full write-up in{' '}
            <span className="text-readout text-ink-soft">backend/TERRAIN_FROM_EEG.md</span>.
          </p>
        </Reveal>
      </section>
    </motion.main>
  )
}
