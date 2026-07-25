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
    title: 'Synchronization',
    body: 'A response attributed to the wrong frame corrupts every downstream number.',
    detail: [
      'Marker alignment against playback and scene-transition triggers',
      'Offset recovered to within 0.4-3.1 ms',
      'Drift reported only when the marker geometry can support it',
      'Residual scored over every marker, including rejected outliers',
    ],
  },
  {
    icon: <Brain className="size-4" />,
    title: 'Encoding',
    body: 'Seven visual-response measures, then a 16-dimensional memory latent.',
    detail: [
      'Occipital luminance drive and posterior alpha suppression',
      'Motion-sensitive beta/gamma over occipito-temporal sites',
      'Evoked transients marking abrupt visual change',
      'Field-balance asymmetry — the only spatial signal that reaches the scalp',
    ],
  },
  {
    icon: <Boxes className="size-4" />,
    title: 'Fidelity measurement',
    body: 'The deliverable. Scored at native resolution against the reference.',
    detail: [
      'SSIM, PSNR, luminance, motion and colour correlation',
      'Scene-boundary F1 matched on time, against every reference frame',
      'Every metric shown against a trivial-baseline and a practical ceiling',
      'Decoders fitted on held-out clips, never held-out frames',
    ],
  },
]

const STACK = [
  {
    group: 'Interface',
    items: ['React 19', 'TypeScript', 'Vite', 'TailwindCSS', 'Framer Motion', 'Zustand'],
  },
  {
    group: 'Reconstruction',
    items: ['Ridge decoders', 'Latent trajectory', 'Temporal refinement', 'SSIM / PSNR'],
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
      title="Built so every claim has a number behind it."
      lede="The reconstruction is downstream of the signal chain, and the fidelity score is downstream of the reconstruction. There is no decorative layer anywhere in it."
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
