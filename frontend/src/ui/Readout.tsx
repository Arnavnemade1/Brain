import { motion } from 'framer-motion'
import type { ReactNode } from 'react'

import { cn } from '@/lib/cn'
import { clamp } from '@/lib/math'

/* -------------------------------------------------------------------------- */
/* Stat                                                                       */
/* -------------------------------------------------------------------------- */

interface StatProps {
  label: string
  value: ReactNode
  unit?: string
  hint?: string
  tone?: string
  className?: string
}

/** A single labelled number. The workhorse of the telemetry panels. */
export function Stat({ label, value, unit, hint, tone, className }: StatProps) {
  return (
    <div className={cn('min-w-0', className)}>
      <p className="text-label mb-1.5">{label}</p>
      <p className="text-readout flex items-baseline gap-1 text-xl leading-none">
        <span style={tone ? { color: tone } : undefined} className="truncate">
          {value}
        </span>
        {unit && <span className="text-xs text-ink-muted">{unit}</span>}
      </p>
      {hint && <p className="mt-1.5 truncate text-[0.7rem] text-ink-faint">{hint}</p>}
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Meter                                                                      */
/* -------------------------------------------------------------------------- */

interface MeterProps {
  value: number
  label?: string
  /** Hex colour for the fill. */
  tone?: string
  showValue?: boolean
  size?: 'sm' | 'md'
  className?: string
}

/** Horizontal 0–1 bar. */
export function Meter({
  value,
  label,
  tone = 'var(--color-neural-400)',
  showValue = true,
  size = 'md',
  className,
}: MeterProps) {
  const pct = clamp(value) * 100

  return (
    <div className={cn('min-w-0', className)}>
      {(label || showValue) && (
        <div className="mb-1.5 flex items-baseline justify-between gap-2">
          {label && <span className="text-label">{label}</span>}
          {showValue && (
            <span className="text-readout text-[0.7rem] text-ink-soft">
              {pct.toFixed(0)}%
            </span>
          )}
        </div>
      )}
      <div
        role="meter"
        aria-valuenow={Math.round(pct)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={label}
        className={cn(
          'w-full overflow-hidden rounded-full bg-ink/8',
          size === 'sm' ? 'h-1' : 'h-1.5',
        )}
      >
        <motion.div
          className="h-full rounded-full"
          style={{ background: tone, boxShadow: `0 0 12px -2px ${tone}` }}
          initial={false}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
        />
      </div>
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Badge                                                                      */
/* -------------------------------------------------------------------------- */

interface BadgeProps {
  children: ReactNode
  tone?: string
  /** Pulsing dot for live indicators. */
  live?: boolean
  className?: string
}

export function Badge({ children, tone, live = false, className }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-pill border px-2.5 py-1',
        'text-[0.65rem] font-medium tracking-wide uppercase',
        className,
      )}
      style={{
        color: tone ?? 'var(--color-ink-soft)',
        borderColor: tone ? `color-mix(in oklab, ${tone} 32%, transparent)` : undefined,
        background: tone ? `color-mix(in oklab, ${tone} 12%, transparent)` : undefined,
      }}
    >
      {live && (
        <span className="relative flex size-1.5">
          <span
            className="absolute inline-flex size-full animate-pulse-ring rounded-full"
            style={{ background: tone ?? 'currentColor' }}
          />
          <span
            className="relative inline-flex size-1.5 rounded-full"
            style={{ background: tone ?? 'currentColor' }}
          />
        </span>
      )}
      {children}
    </span>
  )
}

/* -------------------------------------------------------------------------- */
/* Ring                                                                       */
/* -------------------------------------------------------------------------- */

interface RingProps {
  value: number
  size?: number
  thickness?: number
  tone?: string
  label?: string
  sublabel?: string
}

/** Circular 0–1 gauge — used for the headline confidence score. */
export function Ring({
  value,
  size = 104,
  thickness = 5,
  tone = 'var(--color-neural-400)',
  label,
  sublabel,
}: RingProps) {
  const clamped = clamp(value)
  const radius = (size - thickness) / 2
  const circumference = 2 * Math.PI * radius

  return (
    <div className="relative inline-grid place-items-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90" aria-hidden>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="color-mix(in oklab, var(--color-ink) 8%, transparent)"
          strokeWidth={thickness}
        />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={tone}
          strokeWidth={thickness}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={false}
          animate={{ strokeDashoffset: circumference * (1 - clamped) }}
          transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
          style={{ filter: `drop-shadow(0 0 6px ${tone})` }}
        />
      </svg>
      <div className="absolute inset-0 grid place-content-center text-center">
        <span className="text-readout text-2xl leading-none">
          {(clamped * 100).toFixed(0)}
          <span className="text-sm text-ink-muted">%</span>
        </span>
        {label && <span className="text-label mt-1.5">{label}</span>}
        {sublabel && <span className="mt-1 text-[0.65rem] text-ink-faint">{sublabel}</span>}
      </div>
    </div>
  )
}
