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
    title: 'Personal neural profiles',
    body: 'Models that learn one person rather than an averaged population.',
    items: [
      'Per-user baselines replacing population norms',
      'Adaptive reconstruction weighted by prior sessions',
      'Long-term memory graphs across a whole archive',
      'Natural-language search over reconstructed memories',
    ],
  },
  {
    phase: 'Mid',
    title: 'Memory as a connected space',
    body: 'Individual reconstructions stop being isolated and start relating to each other.',
    items: [
      'Cross-memory relationship mapping',
      'Reconstruction refinement from accumulated evidence',
      'Emotion-aware detail synthesis',
      'Collaborative exploration of a shared reconstruction',
    ],
  },
  {
    phase: 'Far',
    title: 'Direct neural interfaces',
    body: 'The acquisition layer moves from recordings to live, high-density signal.',
    items: [
      'Real-time consumer and research EEG hardware',
      'Immersive VR exploration of reconstructions',
      'Brain–computer interface integration',
      'AI-assisted recovery of degraded memory traces',
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
      title="What this becomes when the signal gets better."
      lede="The architecture assumes the acquisition layer will improve. Everything above it — inference, context, reconstruction — is built to take a better signal without being rewritten."
    >
      <div className="grid gap-5 lg:grid-cols-3">
        {HORIZONS.map((horizon, index) => (
          <HorizonCard key={horizon.phase} horizon={horizon} index={index} />
        ))}
      </div>
    </Section>
  )
}
