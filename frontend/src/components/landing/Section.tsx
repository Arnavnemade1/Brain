import { motion } from 'framer-motion'
import type { ReactNode } from 'react'

import { useReveal } from '@/hooks/useReveal'
import { cn } from '@/lib/cn'

interface SectionProps {
  id?: string
  label: string
  title: ReactNode
  lede?: ReactNode
  children?: ReactNode
  className?: string
  /** Centres the header block. */
  centered?: boolean
}

/** Shared shell for every landing-page section. */
export function Section({
  id,
  label,
  title,
  lede,
  children,
  className,
  centered = false,
}: SectionProps) {
  const reveal = useReveal()

  return (
    <section
      id={id}
      className={cn(
        'relative mx-auto w-[min(76rem,calc(100vw-2.5rem))] scroll-mt-24 py-28',
        className,
      )}
    >
      <motion.header
        ref={reveal.ref as React.RefObject<HTMLElement>}
        {...reveal.props}
        className={cn('mb-16 max-w-2xl', centered && 'mx-auto text-center')}
      >
        <p className="text-label mb-5">{label}</p>
        <h2 className="text-[clamp(1.9rem,4.2vw,3rem)] leading-[1.08]">{title}</h2>
        {lede && (
          <p className="mt-6 text-[1.02rem] leading-relaxed text-ink-muted">{lede}</p>
        )}
      </motion.header>
      {children}
    </section>
  )
}
