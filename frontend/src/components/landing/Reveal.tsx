import { motion } from 'framer-motion'
import type { ReactNode } from 'react'

import { useReveal } from '@/hooks/useReveal'
import { cn } from '@/lib/cn'

/**
 * Scroll reveal wrapper.
 *
 * One component instead of each section hand-rolling its own `whileInView`,
 * which is how the page ended up with several slightly different reveal
 * behaviours. Uses the shared hook, so it survives anchor jumps and never
 * strands content at zero opacity.
 */
export function Reveal({
  children,
  className,
  delay = 0,
}: {
  children: ReactNode
  className?: string
  delay?: number
}) {
  const reveal = useReveal({ delay })

  return (
    <motion.div
      ref={reveal.ref as React.RefObject<HTMLDivElement>}
      {...reveal.props}
      className={cn(className)}
    >
      {children}
    </motion.div>
  )
}
