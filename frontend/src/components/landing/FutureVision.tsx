import { motion } from 'framer-motion'

import { useReveal } from '@/hooks/useReveal'

import { Section } from './Section'

interface Horizon {
  phase: string
  title: string
  body: string
  items: string[]
}

const HORIZONS: Horizon[] = [
  {
    phase: 'Near',
    title: 'Subject-specific decoders',
    body: 'Models fitted to one person rather than an averaged population.',
    items: [
      'Per-subject calibration sessions and baselines',
      'Nonlinear decoders once the data supports them',
      'Higher electrode counts for real spatial recovery',
      'Recorded rather than simulated EEG',
    ],
  },
  {
    phase: 'Mid',
    title: 'Semantic reconstruction',
    body: 'Recovering what was in the scene, not only how it behaved over time.',
    items: [
      'Object and category decoding from evoked components',
      'Generative priors constrained by measured fidelity',
      'Cross-session accumulation over repeated viewings',
      'Reconstruction from recalled rather than viewed material',
    ],
  },
  {
    phase: 'Far',
    title: 'Higher-bandwidth acquisition',
    body: 'The spatial ceiling lifts only when the measurement does.',
    items: [
      'fMRI and MEG for genuine spatial resolution',
      'Intracranial recording where clinically available',
      'Real-time decoding from live acquisition',
      'Brain-computer interface integration',
    ],
  },
]

function HorizonCard({ horizon, index }: { horizon: Horizon; index: number }) {
  const reveal = useReveal({ delay: index * 0.1 })

  return (
    <motion.article
      ref={reveal.ref as React.RefObject<HTMLElement>}
      {...reveal.props}
      className="glass rounded-panel relative overflow-hidden p-7"
    >
      {/* Each card is progressively dimmer — further out, less certain. */}
      <div
        className="pointer-events-none absolute inset-x-0 top-0 h-px"
        style={{
          background: `linear-gradient(90deg, transparent, color-mix(in oklab, var(--color-neural-300) ${70 - index * 22}%, transparent), transparent)`,
        }}
      />

      <p className="text-label mb-4">{horizon.phase} term</p>
      <h3 className="text-[1.05rem] leading-snug">{horizon.title}</h3>
      <p className="mt-3 text-sm leading-relaxed text-ink-muted">{horizon.body}</p>

      <ul className="mt-6 space-y-2.5 border-t border-ink/8 pt-5">
        {horizon.items.map((item) => (
          <li
            key={item}
            className="text-[0.82rem] leading-relaxed text-ink-faint"
            style={{ opacity: 1 - index * 0.14 }}
          >
            {item}
          </li>
        ))}
      </ul>
    </motion.article>
  )
}

export function FutureVision() {
  return (
    <Section
      id="future"
      label="Future vision"
      title="What raises the ceiling."
      lede="Fidelity here is bounded by what scalp EEG carries, not by the decoder. Everything above the acquisition layer is built to take a richer signal without being rewritten."
    >
      <div className="grid gap-5 lg:grid-cols-3">
        {HORIZONS.map((horizon, index) => (
          <HorizonCard key={horizon.phase} horizon={horizon} index={index} />
        ))}
      </div>
    </Section>
  )
}
