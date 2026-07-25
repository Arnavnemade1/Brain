import { motion, useScroll, useSpring } from 'framer-motion'
import { useEffect } from 'react'
import { Link } from 'react-router-dom'

import { pageVariants } from '@/animations/motion'
import { FutureVision } from '@/components/landing/FutureVision'
import { Hero } from '@/components/landing/Hero'
import { HowItWorks } from '@/components/landing/HowItWorks'
import { LaunchCta } from '@/components/landing/LaunchCta'
import { Technology } from '@/components/landing/Technology'

const SECTIONS = [
  { id: 'how-it-works', label: 'Process' },
  { id: 'technology', label: 'Technology' },
  { id: 'future', label: 'Future' },
]

export default function Landing() {
  const { scrollYProgress } = useScroll()
  const progress = useSpring(scrollYProgress, { stiffness: 90, damping: 26, restDelta: 0.001 })

  useEffect(() => {
    document.title = 'MindScape — Memory Reconstruction Platform'
  }, [])

  return (
    <motion.main variants={pageVariants} initial="initial" animate="animate" exit="exit">
      {/* Reading progress — the only persistent chrome on this page. */}
      <motion.div
        style={{ scaleX: progress }}
        className="fixed inset-x-0 top-0 z-50 h-px origin-left bg-gradient-to-r from-neural-400 via-cortex-400 to-memory-300"
        aria-hidden
      />

      {/* Minimal floating header — the landing page should not carry app chrome. */}
      <div className="fixed inset-x-0 top-0 z-40">
        <div className="mx-auto flex w-[min(76rem,calc(100vw-2.5rem))] items-center justify-between py-5">
          <span className="flex items-center gap-2.5">
            <span className="relative grid size-6 place-items-center">
              <span className="absolute inset-0 rounded-full border border-neural-300/40" />
              <span className="size-1.5 rounded-full bg-memory-300 shadow-[0_0_10px] shadow-memory-400" />
            </span>
            <span className="font-display text-[0.95rem] tracking-tight">MindScape</span>
          </span>

          <nav className="hidden items-center gap-7 md:flex" aria-label="Sections">
            {SECTIONS.map((section) => (
              <a
                key={section.id}
                href={`#${section.id}`}
                className="text-[0.8rem] text-ink-muted transition-colors hover:text-ink"
              >
                {section.label}
              </a>
            ))}
          </nav>

          <Link
            to="/session"
            className="glass rounded-pill px-4 py-2 text-[0.8rem] text-ink transition-colors hover:border-ink/20"
          >
            Launch
          </Link>
        </div>
      </div>

      <Hero />
      <HowItWorks />
      <Technology />
      <FutureVision />
      <LaunchCta />
    </motion.main>
  )
}
