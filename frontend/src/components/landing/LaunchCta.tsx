import { motion } from 'framer-motion'
import { ArrowRight, Upload, Waves } from 'lucide-react'
import { Link } from 'react-router-dom'

import { NeuralField } from '@/components/viz/NeuralField'
import { useReveal } from '@/hooks/useReveal'
import { Button } from '@/ui'

export function LaunchCta() {
  const reveal = useReveal()

  return (
    <section className="relative mx-auto w-[min(76rem,calc(100vw-2.5rem))] pt-16 pb-32">
      <motion.div
        ref={reveal.ref as React.RefObject<HTMLDivElement>}
        {...reveal.props}
        className="glass-strong rounded-panel relative overflow-hidden px-8 py-20 text-center sm:px-16"
      >
        <NeuralField
          className="absolute inset-0 size-full opacity-40"
          density={60}
          activity={0.45}
          color="#6fd8fb"
        />
        <div
          className="pointer-events-none absolute inset-0"
          style={{
            background:
              'radial-gradient(ellipse 60% 60% at 50% 50%, transparent 10%, rgba(8,11,20,0.85) 75%)',
          }}
        />

        <div className="relative">
          <p className="text-label mb-6">Launch experience</p>
          <h2 className="mx-auto max-w-2xl text-[clamp(1.9rem,4.6vw,3.1rem)] leading-[1.06]">
            Run a reconstruction and see what survives.
          </h2>
          <p className="mx-auto mt-6 max-w-lg text-[1.02rem] leading-relaxed text-ink-muted">
            Pick a reference clip and a recording quality. The reconstruction decodes in
            real time and is scored against the original, frame by frame.
          </p>

          <div className="mt-10 flex flex-wrap items-center justify-center gap-3">
            <Link to="/session">
              <Button variant="primary" size="lg" icon={<Waves className="size-4" />}>
                Begin a reconstruction
              </Button>
            </Link>
            <Link to="/library">
              <Button variant="secondary" size="lg" icon={<Upload className="size-4" />}>
                Past reconstructions
              </Button>
            </Link>
          </div>

          <Link
            to="/library"
            className="mt-8 inline-flex items-center gap-1.5 text-sm text-ink-muted transition-colors hover:text-ink"
          >
            Read the fidelity methodology
            <ArrowRight className="size-3.5" />
          </Link>
        </div>
      </motion.div>

      <footer className="mt-14 flex flex-col items-center gap-3 text-center">
        <div className="rule-fade w-full max-w-xl" />
        <p className="max-w-lg text-[0.72rem] leading-relaxed text-ink-faint">
          MindScape is a research prototype running on simulated EEG. Reported fidelity
          is an upper bound; real recordings carry noise and variability the forward
          model does not capture.
        </p>
      </footer>
    </section>
  )
}
