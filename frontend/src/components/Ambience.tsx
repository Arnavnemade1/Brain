import { motion } from 'framer-motion'

import { cn } from '@/lib/cn'
import { useReducedMotion } from '@/hooks/useMotionPreference'

interface AmbienceProps {
  /** Hex accents driving the two light pools. */
  primary?: string
  secondary?: string
  intensity?: number
  className?: string
}

/**
 * The ambient light field behind every page.
 *
 * Two slow-drifting radial pools plus a fine grain layer. The grain matters:
 * large dark gradients band badly on 8-bit displays, and a little noise is
 * what keeps them looking like light rather than like stepped bars.
 */
export function Ambience({
  primary = '#16a3d6',
  secondary = '#6247c7',
  intensity = 1,
  className,
}: AmbienceProps) {
  const reduced = useReducedMotion()

  return (
    <div
      aria-hidden
      className={cn('pointer-events-none fixed inset-0 -z-10 overflow-hidden', className)}
    >
      <div className="absolute inset-0 bg-abyss" />

      <motion.div
        className="absolute -top-[30%] left-[8%] size-[68vw] rounded-full blur-[120px]"
        style={{
          background: `radial-gradient(circle, ${primary}, transparent 68%)`,
          opacity: 0.24 * intensity,
        }}
        animate={reduced ? undefined : { x: [0, 60, -30, 0], y: [0, 40, 20, 0] }}
        transition={{ duration: 44, repeat: Infinity, ease: 'easeInOut' }}
      />

      <motion.div
        className="absolute -right-[12%] bottom-[-25%] size-[62vw] rounded-full blur-[130px]"
        style={{
          background: `radial-gradient(circle, ${secondary}, transparent 70%)`,
          opacity: 0.2 * intensity,
        }}
        animate={reduced ? undefined : { x: [0, -50, 30, 0], y: [0, -30, -10, 0] }}
        transition={{ duration: 52, repeat: Infinity, ease: 'easeInOut' }}
      />

      {/* Horizon wash — grounds the composition so the pools do not float. */}
      <div
        className="absolute inset-x-0 bottom-0 h-[45vh]"
        style={{
          background: 'linear-gradient(to top, rgba(4,6,12,0.9), transparent)',
        }}
      />

      <div className="grain-overlay absolute inset-0 opacity-[0.13]" />
    </div>
  )
}
