import { motion, type HTMLMotionProps } from 'framer-motion'
import type { ReactNode } from 'react'

import { riseItem } from '@/animations/motion'
import { cn } from '@/lib/cn'

type PanelTone = 'default' | 'strong' | 'faint'

interface PanelProps extends Omit<HTMLMotionProps<'section'>, 'title'> {
  children: ReactNode
  tone?: PanelTone
  className?: string
  /** Adds the standard header block. */
  title?: ReactNode
  label?: ReactNode
  action?: ReactNode
  /** Removes inner padding for panels that host edge-to-edge content. */
  flush?: boolean
  animate?: boolean
}

const TONE: Record<PanelTone, string> = {
  default: 'glass',
  strong: 'glass-strong',
  faint: 'glass-faint',
}

/**
 * The floating glass surface every piece of chrome sits on.
 */
export function Panel({
  children,
  tone = 'default',
  className,
  title,
  label,
  action,
  flush = false,
  animate = true,
  ...rest
}: PanelProps) {
  const hasHeader = Boolean(title || label || action)

  return (
    <motion.section
      variants={animate ? riseItem : undefined}
      initial={animate ? 'initial' : undefined}
      animate={animate ? 'animate' : undefined}
      className={cn(TONE[tone], 'rounded-panel relative overflow-hidden', className)}
      {...rest}
    >
      {hasHeader && (
        <header
          className={cn(
            'flex items-start justify-between gap-4',
            flush ? 'px-5 pt-5 pb-3' : 'px-5 pt-5',
          )}
        >
          <div className="min-w-0">
            {label && <p className="text-label mb-2">{label}</p>}
            {title && (
              <h3 className="truncate text-[0.95rem] font-medium text-ink">{title}</h3>
            )}
          </div>
          {action && <div className="shrink-0">{action}</div>}
        </header>
      )}
      <div className={cn(flush ? '' : 'p-5', hasHeader && !flush && 'pt-4')}>{children}</div>
    </motion.section>
  )
}
