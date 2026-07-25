import { motion, useScroll, useTransform } from 'framer-motion'
import { ArrowRight, ChevronDown } from 'lucide-react'
import { useRef } from 'react'
import { Link } from 'react-router-dom'

import { EASE_GLIDE } from '@/animations/motion'
import { NeuralField } from '@/components/viz/NeuralField'
import { useReducedMotion } from '@/hooks/useMotionPreference'
import { Button } from '@/ui'

const HEADLINE = ['Walk through', 'your own', 'memories.']

export function Hero() {
  const ref = useRef<HTMLElement>(null)
  const reduced = useReducedMotion()

  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ['start start', 'end start'],
  })

  // The hero recedes as the page scrolls rather than simply leaving — it
  // should read as depth, not as a slide transition.
  const opacity = useTransform(scrollYProgress, [0, 0.75], [1, 0])
  const scale = useTransform(scrollYProgress, [0, 1], [1, 0.94])
  const blur = useTransform(scrollYProgress, [0, 0.8], ['blur(0px)', 'blur(10px)'])

  return (
    <section
      ref={ref}
      className="relative grid min-h-dvh place-items-center overflow-hidden px-6 pt-28 pb-16"
    >
      <NeuralField
        className="absolute inset-0 size-full opacity-70"
        density={110}
        activity={0.65}
      />

      {/* Vignette keeps the field from competing with the headline. */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            'radial-gradient(ellipse 70% 55% at 50% 45%, transparent 20%, rgba(5,7,13,0.82) 78%)',
        }}
      />

      <motion.div
        style={reduced ? undefined : { opacity, scale, filter: blur }}
        className="relative z-10 mx-auto max-w-4xl text-center"
      >
        <motion.p
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, ease: EASE_GLIDE, delay: 0.15 }}
          className="text-label mb-8"
        >
          Memory Reconstruction Platform
        </motion.p>

        <h1 className="text-[clamp(2.6rem,8vw,5.6rem)] leading-[0.98] font-normal">
          {HEADLINE.map((line, index) => (
            <motion.span
              key={line}
              initial={{ opacity: 0, y: 34, filter: 'blur(12px)' }}
              animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
              transition={{
                duration: 1.1,
                ease: EASE_GLIDE,
                delay: 0.3 + index * 0.14,
              }}
              className="block"
            >
              {index === 2 ? (
                <span className="bg-gradient-to-br from-memory-200 via-memory-300 to-memory-500 bg-clip-text text-transparent">
                  {line}
                </span>
              ) : (
                line
              )}
            </motion.span>
          ))}
        </h1>

        <motion.p
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, ease: EASE_GLIDE, delay: 0.8 }}
          className="mx-auto mt-9 max-w-xl text-[1.05rem] leading-relaxed text-ink-muted"
        >
          MindScape interprets EEG activity and reconstructs the experience behind it as
          an environment you can move through. Not a chart of your brainwaves — a place.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, ease: EASE_GLIDE, delay: 1.0 }}
          className="mt-11 flex flex-wrap items-center justify-center gap-3"
        >
          <Link to="/session">
            <Button variant="primary" size="lg" icon={<ArrowRight className="size-4" />}>
              Begin a neural session
            </Button>
          </Link>
          <a href="#how-it-works">
            <Button variant="ghost" size="lg">
              How it works
            </Button>
          </a>
        </motion.div>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 1.3 }}
          className="mt-10 text-[0.72rem] leading-relaxed text-ink-faint"
        >
          Reconstructions are inferences from neural signal, not recovered recordings.
          <br className="hidden sm:block" />
          Where the evidence runs out, the world stays unfinished.
        </motion.p>
      </motion.div>

      <motion.a
        href="#how-it-works"
        aria-label="Scroll to how it works"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.8, duration: 1 }}
        className="absolute bottom-9 left-1/2 -translate-x-1/2 text-ink-faint transition-colors hover:text-ink-soft"
      >
        <motion.span
          animate={reduced ? undefined : { y: [0, 7, 0] }}
          transition={{ duration: 2.6, repeat: Infinity, ease: 'easeInOut' }}
          className="block"
        >
          <ChevronDown className="size-5" />
        </motion.span>
      </motion.a>
    </section>
  )
}
