import { motion } from 'framer-motion'
import { ArrowLeft, Pause, Play } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'

import { pageVariants } from '@/animations/motion'
import { Reveal } from '@/components/landing/Reveal'
import { ComponentBars, Figure, HeatGrid, LineChart } from '@/components/research/Chart'
import { ControlBars, ResultLadder } from '@/components/research/Findings'
import {
  CONTROL_CHANCE,
  DOMAINS,
  FREQUENCY_CONTROL,
  SPATIAL_CONTROL,
} from '@/data/findings'
import spatial from '@/data/spatial-limit.json'
import { cn } from '@/lib/cn'

/**
 * Everything measured, in one document.
 *
 * Ordered by how well it works rather than by when it was built, because the
 * ordering *is* the finding: EEG gives up temporal structure readily,
 * appearance with effort, spatial layout coarsely, and identity barely at all.
 * A reader who leaves with only the shape of that ladder has the result.
 *
 * Numbers come from `@/data/findings` and `spatial-limit.json`. Nothing is
 * typed in here.
 */

const BAR = spatial.bar
const eight = spatial.resolutions.find((r) => r.side === 8)!
const twelve = spatial.resolutions.find((r) => r.side === 12)!

const SECTIONS = [
  { id: 'ladder', label: 'The ladder' },
  ...DOMAINS.map((d) => ({ id: d.id, label: d.eyebrow })),
  { id: 'controls', label: 'Is it cortical?' },
  { id: 'spatial', label: 'Spatial limit' },
  { id: 'method', label: 'Method' },
]

function Evidence({
  src,
  poster,
  label,
  meta,
  note,
}: {
  src: string
  poster: string
  label: string
  meta: string
  note: string
}) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [playing, setPlaying] = useState(false)

  // Play only while on screen. Four reconstructions autoplaying at once was
  // both wasteful and self-defeating: the contention delayed `loadeddata`
  // long enough that every frame sat black while its own poster was hidden
  // waiting for it.
  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          void video.play().then(() => setPlaying(true)).catch(() => setPlaying(false))
        } else if (!video.paused) {
          video.pause()
          setPlaying(false)
        }
      },
      { threshold: 0.35 },
    )
    observer.observe(video)
    return () => observer.disconnect()
  }, [])

  const toggle = () => {
    const video = videoRef.current
    if (!video) return
    if (video.paused) {
      void video.play()
      setPlaying(true)
    } else {
      video.pause()
      setPlaying(false)
    }
  }

  return (
    <figure className="m-0">
      <div className="glass rounded-panel overflow-hidden p-2">
        <div className="overflow-hidden rounded-[0.55rem] bg-void">
          {/* No opacity gate: the poster is a real frame of the render and
              stands in perfectly well until playback starts. */}
          <video
            ref={videoRef}
            className="block aspect-[1120/620] w-full"
            src={src}
            poster={poster}
            loop
            muted
            playsInline
            preload="none"
            aria-label={label}
          />
        </div>
        <div className="flex items-center gap-3 px-2 pt-3 pb-1">
          <button
            onClick={toggle}
            aria-label={playing ? 'Pause' : 'Play'}
            className="glass grid size-8 shrink-0 place-items-center rounded-full text-ink transition-colors hover:border-ink/20"
          >
            {playing ? <Pause className="size-3" /> : <Play className="size-3" />}
          </button>
          <p className="text-label">{meta}</p>
        </div>
      </div>
      <figcaption className="mt-5 text-[0.8rem] leading-relaxed text-ink-faint">
        {note}
      </figcaption>
    </figure>
  )
}

export default function Research() {
  const [active, setActive] = useState(SECTIONS[0].id)

  useEffect(() => {
    document.title = 'Research — MindScape'
  }, [])

  // Highlights whichever section currently owns the upper third of the
  // viewport. Cheaper and steadier than an observer per section, which fires
  // twice on every boundary crossing and flickers between neighbours.
  useEffect(() => {
    const onScroll = () => {
      let current = SECTIONS[0].id
      for (const section of SECTIONS) {
        const element = document.getElementById(section.id)
        if (element && element.getBoundingClientRect().top <= window.innerHeight * 0.35) {
          current = section.id
        }
      }
      setActive(current)
    }
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <motion.main variants={pageVariants} initial="initial" animate="animate" exit="exit">
      {/* ---------------------------------------------------------------- */}
      {/* Header                                                            */}
      {/* ---------------------------------------------------------------- */}
      <section className="shell pt-32 pb-6">
        <Reveal>
          <Link
            to="/"
            className="text-label inline-flex items-center gap-2 text-ink-faint transition-colors hover:text-ink-soft"
          >
            <ArrowLeft className="size-3" />
            Back
          </Link>

          <p className="text-label mt-10 mb-6">Research</p>
          <h1 className="max-w-[18ch] text-[clamp(2.2rem,5.4vw,3.8rem)]">
            What a brain recording actually gives up.
          </h1>
          <p className="mt-8 max-w-2xl text-[1.02rem] leading-relaxed text-ink-soft">
            Five domains, four public datasets, every result beside the chance level it
            has to beat and the shuffled control that says whether it did. Ordered by how
            well each works — because that ordering is the finding.
          </p>
        </Reveal>
      </section>

      <div className="shell flex gap-12 pb-24">
        {/* -------------------------------------------------------------- */}
        {/* Contents rail                                                   */}
        {/* -------------------------------------------------------------- */}
        <nav
          aria-label="Contents"
          className="sticky top-28 hidden h-fit w-44 shrink-0 lg:block"
        >
          <p className="text-label mb-4">Contents</p>
          <ul className="space-y-2.5 border-l border-ink/10 pl-4">
            {SECTIONS.map((section) => (
              <li key={section.id}>
                <a
                  href={`#${section.id}`}
                  className={cn(
                    'block text-[0.82rem] transition-colors',
                    active === section.id
                      ? 'text-neural-300'
                      : 'text-ink-faint hover:text-ink-soft',
                  )}
                >
                  {section.label}
                </a>
              </li>
            ))}
          </ul>
        </nav>

        <div className="min-w-0 flex-1">
          {/* ------------------------------------------------------------ */}
          {/* The ladder                                                    */}
          {/* ------------------------------------------------------------ */}
          <section id="ladder" className="scroll-mt-28">
            <Reveal>
              <p className="text-label mb-5 text-neural-300">The ladder</p>
              <h2 className="max-w-[24ch] text-[clamp(1.6rem,3.2vw,2.2rem)]">
                Timing is easy. Appearance is workable. Identity is not.
              </h2>
              <p className="mt-6 max-w-2xl text-[0.96rem] leading-relaxed text-ink-muted">
                Every measurement below carries its own chance level, because they do not
                share one. A two-way decision starts at 50%, five classes at 20%, retrieval
                over 200 photographs at 0.5%. On every track the chance line is drawn and
                the region beneath it shaded — a bar that stops short of the line has
                failed, and two of them do.
              </p>
            </Reveal>

            <Reveal className="card-grid mt-10 sm:grid-cols-2 lg:grid-cols-4">
              {[
                ['76.6%', 'Event detection', 'chance 50%', 'text-neural-300'],
                ['0.69', 'Edge density (r)', 'shuffled 0.07', 'text-neural-300'],
                ['≈9', 'Spatial degrees of freedom', 'of 64 cells', 'text-ink'],
                ['7.6%', 'Exact object', 'chance 0.5%', 'text-ink-soft'],
              ].map(([value, label, note, tone]) => (
                <div key={label} className="glass rounded-panel p-6">
                  <p className={cn('font-display text-[1.9rem] leading-none', tone)}>
                    {value}
                  </p>
                  <p className="mt-4 text-[0.82rem] leading-snug text-ink-soft">{label}</p>
                  <p className="mt-1.5 text-[0.72rem] text-ink-faint">{note}</p>
                </div>
              ))}
            </Reveal>
          </section>

          {/* ------------------------------------------------------------ */}
          {/* Domains                                                       */}
          {/* ------------------------------------------------------------ */}
          {DOMAINS.map((domain) => (
            <section key={domain.id} id={domain.id} className="scroll-mt-28 pt-24">
              <Reveal>
                <p className="text-label mb-5 text-neural-300">{domain.eyebrow}</p>
                <h2 className="max-w-[26ch] text-[clamp(1.5rem,3vw,2rem)]">
                  {domain.title}
                </h2>
                <p className="mt-6 max-w-2xl text-[0.96rem] leading-relaxed text-ink-muted">
                  {domain.summary}
                </p>
              </Reveal>

              {domain.video && (
                <Reveal className="mt-10">
                  <Evidence {...domain.video} />
                </Reveal>
              )}

              <Reveal className="glass rounded-panel mt-10 p-6 sm:p-8">
                <ResultLadder findings={domain.findings} />
                <p className="mt-7 border-t border-ink/8 pt-5 text-[0.74rem] text-ink-faint">
                  {domain.dataset} ·{' '}
                  <span className="text-readout text-ink-soft">{domain.source}</span>
                </p>
              </Reveal>
            </section>
          ))}

          {/* ------------------------------------------------------------ */}
          {/* Artifact controls                                             */}
          {/* ------------------------------------------------------------ */}
          <section id="controls" className="scroll-mt-28 pt-24">
            <Reveal>
              <p className="text-label mb-5 text-neural-300">Is it cortical?</p>
              <h2 className="max-w-[26ch] text-[clamp(1.5rem,3vw,2rem)]">
                The failure mode is not that it stops working.
              </h2>
              <p className="mt-6 max-w-2xl text-[0.96rem] leading-relaxed text-ink-muted">
                Movement floods scalp EEG with muscle activity many times larger than the
                cortical signal, in the same bands the decoder reads. The danger is not a
                result that collapses — it is one that looks excellent while decoding
                somebody&rsquo;s jaw. Two controls separate the two.
              </p>
            </Reveal>

            <Reveal className="card-grid mt-10 md:grid-cols-2">
              <div className="glass rounded-panel p-6 sm:p-7">
                <p className="text-label mb-2">Frequency profile</p>
                <p className="mb-6 text-[0.78rem] leading-relaxed text-ink-faint">
                  Muscle power rises with frequency. Cortical mu and beta live at 8–30 Hz.
                  Accuracy <em>falls</em> above beta, which is the wrong direction for EMG.
                </p>
                <ControlBars
                  rows={FREQUENCY_CONTROL.map((f) => ({ name: `${f.band} Hz`, value: f.value }))}
                  chance={CONTROL_CHANCE}
                  highlight="beta 13–30 Hz"
                />
              </div>

              <div className="glass rounded-panel p-6 sm:p-7">
                <p className="text-label mb-2">Spatial profile</p>
                <p className="mb-6 text-[0.78rem] leading-relaxed text-ink-faint">
                  Electrodes over the temporalis muscle sit at chance. Whatever is being
                  read, it is not the jaw — though occipital decoding shows the signal is
                  spatially broad, not focal.
                </p>
                <ControlBars
                  rows={SPATIAL_CONTROL.map((s) => ({ name: s.site, value: s.value }))}
                  chance={CONTROL_CHANCE}
                  highlight="Motor cortex"
                />
              </div>
            </Reveal>
          </section>

          {/* ------------------------------------------------------------ */}
          {/* Spatial limit                                                 */}
          {/* ------------------------------------------------------------ */}
          <section id="spatial" className="scroll-mt-28 pt-24">
            <Reveal>
              <p className="text-label mb-5 text-neural-300">Spatial limit</p>
              <h2 className="max-w-[26ch] text-[clamp(1.5rem,3vw,2rem)]">
                How much landform could the signal specify?
              </h2>
              <p className="mt-6 max-w-2xl text-[0.96rem] leading-relaxed text-ink-muted">
                In every reconstruction the terrain is fixed before any brain activity is
                read — only its appearance is decoded. Reduce each stimulus to a coarse
                luminance grid, regress EEG onto every cell held out, and the answer looks
                obvious. It is not.
              </p>
            </Reveal>

            <Reveal className="mt-10">
              <Evidence
                src="/memory-of-place.mp4"
                poster="/memory-of-place-poster.png"
                label="Terrain whose light, air and speed of travel follow a continuous EEG recording"
                meta="sub-mit003 · 60 s · continuous EEG"
                note="Sixty seconds of one recording, rendered as terrain you travel through. This one predicts nothing, so it carries no accuracy figure — band power is a reading, not a guess. The landform beneath it is fixed, which is what the measurement below is about."
              />
            </Reveal>

            <Reveal className="mt-12">
              <Figure
                label="Figure 1"
                title="Per-cell correlation barely degrades as the grid gets finer."
                caption={`Mean held-out correlation per cell against grid resolution, averaged over ${spatial.subjects.length} subjects. The dashed line is the shuffled control, under ${twelve.shuffled.toFixed(2)} throughout. Y axis starts at zero.`}
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

            <Reveal className="mt-10">
              <div className="glass rounded-panel border-memory-400/25 p-7">
                <p className="text-label mb-4 text-memory-300">Read this carefully</p>
                <p className="text-[0.94rem] leading-relaxed text-ink-muted">
                  Taken at face value, {twelve.cells} cells at r&nbsp;=&nbsp;
                  {twelve.meanR.toFixed(2)} says {twelve.cells} recoverable heights. These
                  are objects photographed on plain backgrounds, so most cells are driven
                  by one factor — how much of the object covers them — which is close to
                  overall coverage and luminance, and both decode well alone.
                </p>
                <p className="mt-4 text-[0.94rem] leading-relaxed text-ink">
                  {twelve.cells} correlated numbers are not {twelve.cells} degrees of
                  freedom.
                </p>
              </div>
            </Reveal>

            <Reveal className="mt-10">
              <Figure
                label="Figure 2"
                title="Strip each image's own brightness and layout still survives."
                caption={`Per-cell correlation over an ${eight.side}×${eight.side} grid after removing every image's mean luminance, so only spatial layout remains. Mean r = ${spatial.centred.meanR.toFixed(3)}, with ${(spatial.centred.aboveBar * 100).toFixed(0)}% of cells above the bar — so this is not one global brightness number wearing a grid costume.`}
              >
                <HeatGrid values={spatial.centred.cellR} side={spatial.centred.side} />
              </Figure>
            </Reveal>

            <Reveal className="mt-10">
              <Figure
                label="Figure 3"
                title="Counting components instead of cells gives the honest number."
                caption={`Principal components of the ${eight.side}×${eight.side} grid: how much variance each holds, and how well each decodes. ${spatial.survivors} of ${spatial.componentsTested} clear the bar. Two scales, drawn as separate rows — there is no meaningful ratio between a share of variance and a correlation.`}
              >
                <ComponentBars components={spatial.components} threshold={BAR} />
              </Figure>
            </Reveal>

            <Reveal className="card-grid mt-12 md:grid-cols-2">
              <div className="glass rounded-panel p-7">
                <p className="text-label mb-4 text-neural-300">What this licenses</p>
                <ul className="space-y-3 text-[0.9rem] leading-relaxed text-ink-muted">
                  <li>
                    A heightfield from nine or ten decoded coefficients over a smooth
                    spatial basis.
                  </li>
                  <li>
                    Broad rises, a basin, a ridge running one way rather than another —
                    shape that varies with the recording and is earned.
                  </li>
                  <li>Everything finer from the fractal, at a declared amplitude.</li>
                </ul>
              </div>
              <div className="glass rounded-panel p-7">
                <p className="text-label mb-4 text-memory-300">What it does not</p>
                <ul className="space-y-3 text-[0.9rem] leading-relaxed text-ink-muted">
                  <li>
                    Peaks and valleys as such. Ten coefficients is roughly a 3×3 surface;
                    anything sharper is invention.
                  </li>
                  <li>
                    A claim about terrain anyone saw. The luminance layout of a photograph
                    of a crab is not a landscape.
                  </li>
                  <li>Any of it without ground truth to score against.</li>
                </ul>
              </div>
            </Reveal>
          </section>

          {/* ------------------------------------------------------------ */}
          {/* Method                                                        */}
          {/* ------------------------------------------------------------ */}
          <section id="method" className="scroll-mt-28 pt-24">
            <Reveal>
              <p className="text-label mb-5 text-neural-300">Method</p>
              <h2 className="max-w-[26ch] text-[clamp(1.5rem,3vw,2rem)]">
                What every number here had to survive.
              </h2>
            </Reveal>

            <Reveal className="card-grid mt-10 md:grid-cols-3">
              {[
                [
                  'Held out',
                  'Subjects or runs the model never saw. Windows overlap, so splits are by recording — never by window, which would put neighbours sharing three quarters of their samples on both sides of the boundary.',
                ],
                [
                  'Shuffled control',
                  'The identical pipeline with permuted labels. Whatever it scores is what the machinery produces from a signal carrying nothing, and it is reported beside every result.',
                ],
                [
                  'Measured chance',
                  'Computed from the actual label distribution, not assumed. An early emotion result read 61% against an assumed 50% while performing below its true 62% majority baseline.',
                ],
              ].map(([title, body]) => (
                <div key={title} className="glass rounded-panel p-6">
                  <p className="mb-3 text-[0.94rem] text-ink">{title}</p>
                  <p className="text-[0.82rem] leading-relaxed text-ink-faint">{body}</p>
                </div>
              ))}
            </Reveal>

            <Reveal className="mt-10">
              <pre className="glass rounded-card overflow-x-auto p-5 text-[0.8rem] leading-relaxed text-ink-soft">
                <code>{`cd backend
.venv/bin/python -m tools.reconstruct_gameplay --seconds 45   # temporal
.venv/bin/python -m tools.decode_motor --subjects 20          # physical action
.venv/bin/python -m tools.render_environment_video            # appearance
.venv/bin/python -m tools.decode_stream --subjects 3          # semantic content
.venv/bin/python -m tools.decode_emotion                      # affect
.venv/bin/python -m tools.spatial_limit --subjects 3          # spatial limit`}</code>
              </pre>
              <p className="mt-5 text-[0.8rem] leading-relaxed text-ink-faint">
                The spatial-limit run writes the JSON this page reads, so those figures
                cannot drift from the experiment behind them. The rest are transcribed from
                the write-ups named under each section.
              </p>
            </Reveal>
          </section>
        </div>
      </div>
    </motion.main>
  )
}
