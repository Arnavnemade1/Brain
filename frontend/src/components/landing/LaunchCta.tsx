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
            Start a session and watch a memory assemble itself.
          </h2>
          <p className="mx-auto mt-6 max-w-lg text-[1.02rem] leading-relaxed text-ink-muted">
            Run a simulated neural stream, or upload an EEG recording of your own. The
            reconstruction builds in real time as the signal arrives.
          </p>

          <div className="mt-10 flex flex-wrap items-center justify-center gap-3">
            <Link to="/session">
              <Button variant="primary" size="lg" icon={<Waves className="size-4" />}>
                Begin a neural session
              </Button>
            </Link>
            <Link to="/session?mode=upload">
              <Button variant="secondary" size="lg" icon={<Upload className="size-4" />}>
                Upload a recording
              </Button>
            </Link>
          </div>

          <Link
            to="/library"
            className="mt-8 inline-flex items-center gap-1.5 text-sm text-ink-muted transition-colors hover:text-ink"
          >
            Browse previously reconstructed memories
            <ArrowRight className="size-3.5" />
          </Link>
        </div>
      </motion.div>

      <footer className="mt-14 flex flex-col items-center gap-3 text-center">
        <div className="rule-fade w-full max-w-xl" />
        <p className="max-w-lg text-[0.72rem] leading-relaxed text-ink-faint">
          MindScape is a research prototype. Reconstructions are the model's best fit to
          neural evidence and should not be treated as records of real events.
        </p>
      </footer>
    </section>
  )
}
