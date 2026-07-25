import { motion, useScroll, useTransform } from 'framer-motion'
import { useRef } from 'react'

import { useReveal } from '@/hooks/useReveal'
import { cn } from '@/lib/cn'
import { PIPELINE_STAGES } from '@/types'
import type { PipelineStageMeta } from '@/types'

import { Section } from './Section'

const BAND_TONE: Record<PipelineStageMeta['band'], string> = {
  stimulus: 'var(--color-memory-300)',
  acquisition: 'var(--color-neural-300)',
  encoding: 'var(--color-cortex-400)',
  reconstruction: 'var(--color-emotion-calm)',
}

const BAND_LABEL: Record<PipelineStageMeta['band'], string> = {
  stimulus: 'Stimulus',
  acquisition: 'Acquisition',
  encoding: 'Encoding',
  reconstruction: 'Reconstruction',
}

function StageRow({ stage, index }: { stage: PipelineStageMeta; index: number }) {
  const tone = BAND_TONE[stage.band]
  const reveal = useReveal({ delay: (index % 4) * 0.06 })

  return (
    <motion.li
      ref={reveal.ref as React.RefObject<HTMLLIElement>}
      {...reveal.props}
      className="group relative grid grid-cols-[auto_1fr] gap-x-6 pb-11 last:pb-0"
    >
      {/* Rail node */}
      <div className="relative flex flex-col items-center">
        <span
          className="relative z-10 grid size-9 shrink-0 place-items-center rounded-full border transition-all duration-500 group-hover:scale-110"
          style={{
            borderColor: `color-mix(in oklab, ${tone} 45%, transparent)`,
            background: `color-mix(in oklab, ${tone} 10%, var(--color-abyss))`,
          }}
        >
          <span
            className="text-readout text-[0.62rem]"
            style={{ color: tone }}
          >
            {String(index + 1).padStart(2, '0')}
          </span>
        </span>
      </div>

      <div className="min-w-0 pt-1">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <h3 className="text-[1.02rem] text-ink">{stage.label}</h3>
          <span
            className="text-[0.6rem] tracking-[0.14em] uppercase"
            style={{ color: `color-mix(in oklab, ${tone} 75%, transparent)` }}
          >
            {BAND_LABEL[stage.band]}
          </span>
        </div>
        <p className="mt-1.5 text-sm text-ink-muted">{stage.summary}</p>
        <p className="mt-3 max-w-2xl text-[0.85rem] leading-relaxed text-ink-faint">
          {stage.detail}
        </p>
      </div>
    </motion.li>
  )
}

export function HowItWorks() {
  const railRef = useRef<HTMLDivElement>(null)
  const note = useReveal()

  const { scrollYProgress } = useScroll({
    target: railRef,
    offset: ['start 72%', 'end 55%'],
  })

  // The connecting line draws itself as the reader descends, so the sequence
  // reads as flow rather than as a list.
  const railHeight = useTransform(scrollYProgress, [0, 1], ['0%', '100%'])

  return (
    <Section
      id="how-it-works"
      label="How memory reconstruction works"
      title="Ten stages between scalp voltage and a scored reconstruction."
      lede={
        <>
          Memories, not dreams: because a reference event exists, every stage can be
          checked against what was actually shown. Nothing is invented to fill a gap —
          uncertainty is carried forward and shows up in the fidelity numbers.
        </>
      }
    >
      <div ref={railRef} className="relative">
        {/* Static rail */}
        <div className="absolute top-2 bottom-2 left-[17px] w-px bg-ink/8" aria-hidden />
        {/* Progress rail */}
        <motion.div
          style={{ height: railHeight }}
          className="absolute top-2 left-[17px] w-px bg-gradient-to-b from-neural-300 via-cortex-400 to-memory-300"
          aria-hidden
        />

        <ol className="relative">
          {PIPELINE_STAGES.map((stage, index) => (
            <StageRow key={stage.id} stage={stage} index={index} />
          ))}
        </ol>
      </div>

      <motion.p
        ref={note.ref as React.RefObject<HTMLParagraphElement>}
        {...note.props}
        className={cn(
          'glass-faint rounded-card mt-14 px-6 py-5',
          'text-[0.85rem] leading-relaxed text-ink-muted',
        )}
      >
        <span className="text-ink">On honesty.</span> On clips never seen during
        fitting, reconstructions beat a mean-frame baseline by 2.3× on composite
        fidelity — but lose to it on SSIM, because a constant image is spatially smooth
        and scalp EEG carries almost no spatial detail. What is recovered is temporal:
        when the scene brightened, when it moved, and where it cut. Both numbers are
        reported.
      </motion.p>
    </Section>
  )
}
