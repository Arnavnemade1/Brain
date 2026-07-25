import { motion } from 'framer-motion'
import { Activity, Boxes, Brain, Radio } from 'lucide-react'
import type { ReactNode } from 'react'

import { useReveal } from '@/hooks/useReveal'

import { Section } from './Section'

interface Pillar {
  icon: ReactNode
  title: string
  body: string
  detail: string[]
}

const PILLARS: Pillar[] = [
  {
    icon: <Radio className="size-4" />,
    title: 'Signal processing',
    body: 'Real DSP, not a decorative waveform. Every reading traces back to a measurement.',
    detail: [
      'Common-average referencing and closed-form detrending',
      'Zero-phase 0.5–45 Hz band-pass, 50/60 Hz notch',
      'Robust MAD artifact compression preserving temporal continuity',
      'Welch PSD integrated across five canonical bands',
    ],
  },
  {
    icon: <Activity className="size-4" />,
    title: 'Quality gating',
    body: 'A poor recording produces a visibly poor reconstruction. That is the point.',
    detail: [
      'Channel deviation and noise-floor outlier detection',
      'Composite score calibrated to its measured distribution',
      'Confidence accumulates slowly, decays quickly',
      'Coherence tracks spectral stability across windows',
    ],
  },
  {
    icon: <Brain className="size-4" />,
    title: 'Inference',
    body: 'Transparent weighted scorers over named neural measures — no black boxes.',
    detail: [
      'Seven-way cognitive state, eight-way affect distribution',
      'Valence from frontal alpha asymmetry, arousal from fast-band share',
      'Coefficients fitted against ground truth, not hand-picked',
      'Episodic attributes: familiarity, scale, temporal depth, vividness',
    ],
  },
  {
    icon: <Boxes className="size-4" />,
    title: 'Reconstruction',
    body: 'Procedural worlds built from a seeded, reproducible parameter set.',
    detail: [
      'Biomes matched on attribute pattern, then sampled by confidence',
      'Region-level confidence drives geometry completeness',
      'Emotion maps to palette, lighting, weather and soundscape',
      'The same recording always rebuilds the same world',
    ],
  },
]

const STACK = [
  {
    group: 'Interface',
    items: ['React 19', 'TypeScript', 'Vite', 'TailwindCSS', 'Framer Motion', 'Zustand'],
  },
  {
    group: 'Rendering',
    items: ['Three.js', 'React Three Fiber', 'Drei', 'Postprocessing'],
  },
  {
    group: 'Services',
    items: ['Python', 'FastAPI', 'WebSockets', 'NumPy', 'SciPy', 'MNE (optional)'],
  },
]

function PillarCard({ pillar, index }: { pillar: Pillar; index: number }) {
  const reveal = useReveal({ delay: (index % 2) * 0.08 })

  return (
    <motion.article
      ref={reveal.ref as React.RefObject<HTMLElement>}
      {...reveal.props}
      className="glass rounded-panel p-7 transition-colors duration-500 hover:border-ink/16"
    >
      <div className="mb-5 flex items-center gap-3">
        <span className="glass-faint grid size-9 place-items-center rounded-xl text-neural-300">
          {pillar.icon}
        </span>
        <h3 className="text-[1.02rem]">{pillar.title}</h3>
      </div>

      <p className="mb-5 text-sm leading-relaxed text-ink-muted">{pillar.body}</p>

      <ul className="space-y-2">
        {pillar.detail.map((line) => (
          <li key={line} className="flex gap-2.5 text-[0.82rem] leading-relaxed text-ink-faint">
            <span className="mt-[0.5em] size-1 shrink-0 rounded-full bg-neural-400/50" />
            {line}
          </li>
        ))}
      </ul>
    </motion.article>
  )
}

export function Technology() {
  const stack = useReveal()

  return (
    <Section
      id="technology"
      label="Technology"
      title="Built so the science and the spectacle answer to the same numbers."
      lede="The environment is downstream of the signal chain. Change what the EEG says and the world changes with it — there is no decorative layer sitting on top."
    >
      <div className="grid gap-5 md:grid-cols-2">
        {PILLARS.map((pillar, index) => (
          <PillarCard key={pillar.title} pillar={pillar} index={index} />
        ))}
      </div>

      <motion.div
        ref={stack.ref as React.RefObject<HTMLDivElement>}
        {...stack.props}
        className="mt-6 grid gap-5 sm:grid-cols-3"
      >
        {STACK.map((column) => (
          <div key={column.group} className="glass-faint rounded-card p-5">
            <p className="text-label mb-3.5">{column.group}</p>
            <ul className="flex flex-wrap gap-1.5">
              {column.items.map((item) => (
                <li
                  key={item}
                  className="rounded-md border border-ink/8 px-2 py-1 text-[0.72rem] text-ink-muted"
                >
                  {item}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </motion.div>
    </Section>
  )
}
