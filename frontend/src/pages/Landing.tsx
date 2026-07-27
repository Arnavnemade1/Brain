import { motion } from 'framer-motion'
import { ArrowRight, ChevronDown } from 'lucide-react'
import { useEffect } from 'react'
import { Link } from 'react-router-dom'

import { pageVariants } from '@/animations/motion'
import { MemoryFilm } from '@/components/landing/MemoryFilm'
import { Reveal } from '@/components/landing/Reveal'
import { PIPELINE_STAGES } from '@/types'

/** Headline results. Every figure is measured; see backend/*.md. */
const RESULTS = [
  {
    value: '76.6%',
    label: 'Event detection recall',
    note: 'from EEG alone, no log',
  },
  { value: '72.6%', label: 'Detection precision', note: 'continuous decoding' },
  { value: '60.7%', label: 'Reward vs error', note: 'chance 50%' },
  { value: '17.8%', label: 'Object in top 5', note: 'chance 2.5%' },
]

const PRINCIPLES = [
  {
    title: 'Measured, not illustrated',
    body: 'Every reconstruction is scored against the reference the subject actually saw or did. Nothing is generated to fill a gap.',
  },
  {
    title: 'Held out, every time',
    body: 'Results come from subjects the model never trained on, with shuffled-label controls run through the identical pipeline.',
  },
  {
    title: 'Absence over invention',
    body: 'Where the signal carries nothing, the output shows nothing. Uncertainty is rendered, not smoothed away.',
  },
]

export default function Landing() {
  useEffect(() => {
    document.title = 'MindScape — Memory Reconstruction'
  }, [])

  return (
    <motion.main variants={pageVariants} initial="initial" animate="animate" exit="exit">
      {/* ---------------------------------------------------------------- */}
      {/* Hero                                                              */}
      {/* ---------------------------------------------------------------- */}
      {/* Full height, so the vista behind it is actually a vista. The copy is
          centred in the sky above the horizon line, which sits at ~62% of the
          frame; the scroll cue below sits over the ridges. */}
      <section className="shell relative flex min-h-[100svh] flex-col justify-center pt-20 pb-32 text-center">
        <Reveal>
          <p className="text-label mb-7">Memory Reconstruction Engine</p>
          <h1 className="mx-auto max-w-[19ch] text-[clamp(2.6rem,6vw,4.4rem)]">
            Reconstruct what someone lived, from brain activity.
          </h1>
          <p className="mx-auto mt-8 max-w-xl text-[1.02rem] leading-relaxed text-ink-soft">
            MindScape decodes EEG recorded during real experience and rebuilds it as video
            — then scores the result against what actually happened.
          </p>

          <div className="mt-10 flex flex-col items-stretch justify-center gap-3 sm:flex-row sm:items-center">
            <Link
              to="/real"
              className="inline-flex h-11 items-center justify-center gap-2 rounded-full bg-ink px-6 text-[0.9rem] font-medium whitespace-nowrap text-abyss transition-opacity hover:opacity-90"
            >
              See the reconstructions
              <ArrowRight className="size-3.5" />
            </Link>
            <a
              href="#film"
              className="glass inline-flex h-11 items-center justify-center rounded-full px-6 text-[0.9rem] whitespace-nowrap text-ink transition-colors hover:border-ink/16"
            >
              Watch the memory video
            </a>
          </div>
        </Reveal>

        <a
          href="#film"
          aria-label="Scroll to the memory video"
          className="absolute inset-x-0 bottom-10 mx-auto w-fit text-ink-faint transition-colors hover:text-ink-soft"
        >
          <ChevronDown className="size-5 animate-bounce" />
        </a>
      </section>

      {/* Everything below the fold sits on a surface. The vista is fixed, so
          without this the cards and tables would scroll over open sky. */}
      <div className="content-surface">
        {/* ---------------------------------------------------------------- */}
        {/* The memory video                                                  */}
        {/* ---------------------------------------------------------------- */}
        <section id="film" className="shell section-y scroll-mt-20">
          <Reveal className="mx-auto mb-12 max-w-2xl text-center">
            <p className="text-label mb-5">The memory video</p>
            <h2 className="text-[clamp(1.75rem,3.6vw,2.6rem)]">
              The world as it was, and as the brain recalled it.
            </h2>
            <p className="mt-6 text-[0.98rem] leading-relaxed text-ink-muted">
              A person playing a game with a headset on. Left is what happened. Right is
              what their EEG alone recovered — no event log, decoded every 250&nbsp;ms.
            </p>
          </Reveal>

          <Reveal>
            <MemoryFilm
              src="/memory-video.mp4"
              poster="/memory-video-poster.png"
              label="Gameplay reconstructed from EEG, side by side with what actually happened"
              meta="sub-005 · 60 s continuous · loops"
              stats={[
                ['Detection', '76.6% recall · 72.6% precision'],
                ['Decoding rate', 'every 250 ms, no event log'],
                ['Validation', 'leave-one-subject-out'],
              ]}
              note="The rhythm of the session — when things happened, the bursts and the lulls — is genuinely recovered. Which particular event occurred at each moment is not, and the right panel dims to show it."
            />
          </Reveal>
        </section>

        {/* ---------------------------------------------------------------- */}
        {/* Real-world imagery                                                */}
        {/* ---------------------------------------------------------------- */}
        <section id="reallife" className="shell section-y scroll-mt-20">
          <Reveal className="mx-auto mb-12 max-w-2xl text-center">
            <p className="text-label mb-5">Real life</p>
            <h2 className="text-[clamp(1.75rem,3.6vw,2.6rem)]">
              The same engine, on photographs of the real world.
            </h2>
            <p className="mt-6 text-[0.98rem] leading-relaxed text-ink-muted">
              Someone viewing real-world objects while EEG records. Left is the photograph
              on screen. The other two panels are what the system retrieved from brain
              activity alone — from a single glimpse, and from eight.
            </p>
          </Reveal>

          <Reveal>
            <MemoryFilm
              src="/reallife-video.mp4"
              poster="/reallife-video-poster.png"
              label="Real-world photographs retrieved from EEG, beside the image actually viewed"
              meta="sub-01 · 200 candidates · loops"
              stats={[
                ['Same object in top 5', '42.4% · chance 10%'],
                ['Exact photo in top 5', '20.1% · chance 2.5%'],
                ['Validation', 'shuffled + strict time split'],
              ]}
              note="What comes back reliably is the kind of thing seen, not which one. Watch the ranked strip: when a hat was on screen the answer is often a hat — a different hat, outlined blue rather than green."
            />
          </Reveal>
        </section>

        {/* ---------------------------------------------------------------- */}
        {/* Physical action                                                   */}
        {/* ---------------------------------------------------------------- */}
        <section id="action" className="shell section-y scroll-mt-20">
          <Reveal className="mx-auto mb-12 max-w-2xl text-center">
            <p className="text-label mb-5">Physical action</p>
            <h2 className="text-[clamp(1.75rem,3.6vw,2.6rem)]">
              Not what they saw. What they did.
            </h2>
            <p className="mt-6 text-[0.98rem] leading-relaxed text-ink-muted">
              Someone opening and closing their fists and feet while EEG records. Left is
              the action they performed, right is the action recovered from their motor
              cortex alone — in real time, decoded every 250&nbsp;ms.
            </p>
          </Reveal>

          <Reveal>
            <MemoryFilm
              src="/motor-video.mp4"
              poster="/motor-video-poster.png"
              label="A figure showing the limb movement recovered from EEG, beside the movement actually performed"
              meta="S002 · 75 s · real time · loops"
              stats={[
                ['Moving vs still', '76.3% · chance 50%'],
                ['Fists vs feet', '75.5% · chance 50%'],
                ['Which fist', '62.0% · chance 50%'],
              ]}
              note="The right figure lights each limb by the decoder's odds, not by its single best guess — so uncertainty is shown rather than resolved. Across 20 subjects the exact action is 36.7% against 20% chance, ranging 24-60% between people."
            />
          </Reveal>
        </section>

        {/* ---------------------------------------------------------------- */}
        {/* Results                                                           */}
        {/* ---------------------------------------------------------------- */}
        <section className="shell section-y">
          <Reveal className="mx-auto mb-12 max-w-2xl text-center">
            <p className="text-label mb-5">Measured</p>
            <h2 className="text-[clamp(1.75rem,3.6vw,2.6rem)]">
              Results on people the model never saw.
            </h2>
          </Reveal>

          <Reveal className="card-grid grid-cols-2 lg:grid-cols-4">
            {RESULTS.map((result) => (
              <div
                key={result.label}
                className="glass rounded-panel flex flex-col justify-between p-6 text-center"
              >
                <p className="font-display text-[2rem] leading-none text-ink">
                  {result.value}
                </p>
                <p className="mt-4 text-[0.82rem] leading-snug text-ink-soft">
                  {result.label}
                </p>
                <p className="mt-1.5 text-[0.72rem] text-ink-faint">{result.note}</p>
              </div>
            ))}
          </Reveal>
        </section>

        {/* ---------------------------------------------------------------- */}
        {/* Pipeline                                                          */}
        {/* ---------------------------------------------------------------- */}
        <section className="shell section-y">
          <Reveal className="mx-auto mb-12 max-w-2xl text-center">
            <p className="text-label mb-5">The engine</p>
            <h2 className="text-[clamp(1.75rem,3.6vw,2.6rem)]">
              Ten stages, signal to scored replay.
            </h2>
          </Reveal>

          <Reveal className="card-grid grid-cols-2 md:grid-cols-5">
            {PIPELINE_STAGES.map((stage, index) => (
              <div key={stage.id} className="glass rounded-card flex flex-col p-5">
                <span className="text-readout mb-3 text-[0.65rem] text-neural-300">
                  {String(index + 1).padStart(2, '0')}
                </span>
                <p className="text-[0.86rem] leading-snug text-ink">{stage.label}</p>
                <p className="mt-2 text-[0.72rem] leading-relaxed text-ink-faint">
                  {stage.summary}
                </p>
              </div>
            ))}
          </Reveal>
        </section>

        {/* ---------------------------------------------------------------- */}
        {/* Principles                                                        */}
        {/* ---------------------------------------------------------------- */}
        <section className="shell section-y">
          <Reveal className="card-grid md:grid-cols-3">
            {PRINCIPLES.map((principle) => (
              <div key={principle.title} className="glass rounded-panel p-7">
                <h3 className="text-[1.02rem]">{principle.title}</h3>
                <p className="mt-3 text-[0.86rem] leading-relaxed text-ink-muted">
                  {principle.body}
                </p>
              </div>
            ))}
          </Reveal>
        </section>

        {/* ---------------------------------------------------------------- */}
        {/* Close                                                             */}
        {/* ---------------------------------------------------------------- */}
        <section className="shell pb-28 text-center">
          <Reveal>
            <h2 className="mx-auto max-w-xl text-[clamp(1.6rem,3.2vw,2.3rem)]">
              Every number here has a control behind it.
            </h2>
            <div className="mt-9 flex justify-center">
              <Link
                to="/real"
                className="inline-flex h-11 items-center justify-center gap-2 rounded-full bg-ink px-6 text-[0.9rem] font-medium whitespace-nowrap text-abyss transition-opacity hover:opacity-90"
              >
                Open the reconstructions
                <ArrowRight className="size-3.5" />
              </Link>
            </div>
            <p className="mx-auto mt-12 max-w-lg text-[0.72rem] leading-relaxed text-ink-faint">
              A research prototype. Reconstructions are inferences from neural signal, not
              recordings of experience.
            </p>
          </Reveal>
        </section>
      </div>
    </motion.main>
  )
}
