import { motion } from 'framer-motion'

import { BAND_COLOR } from '@/lib/palette'
import { BAND_RANGES, FREQUENCY_BANDS } from '@/types'
import type { BandPowers } from '@/types'

interface BandBarsProps {
  bands: BandPowers | null
  className?: string
}

/** Relative band power as a horizontal stack plus per-band readouts. */
export function BandBars({ bands, className }: BandBarsProps) {
  const values = FREQUENCY_BANDS.map((band) => ({
    band,
    value: bands?.[band] ?? 0,
    color: BAND_COLOR[band],
    range: BAND_RANGES[band],
  }))

  const total = values.reduce((sum, v) => sum + v.value, 0) || 1

  return (
    <div className={className}>
      {/* Proportional stack */}
      <div className="mb-4 flex h-2 w-full overflow-hidden rounded-full bg-ink/6">
        {values.map(({ band, value, color }) => (
          <motion.div
            key={band}
            initial={false}
            animate={{ flexGrow: value / total }}
            transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
            style={{ background: color, flexBasis: 0 }}
            title={`${band} ${((value / total) * 100).toFixed(0)}%`}
          />
        ))}
      </div>

      <ul className="space-y-2.5">
        {values.map(({ band, value, color, range }) => (
          <li key={band} className="grid grid-cols-[3.6rem_1fr_2.6rem] items-center gap-3">
            <div className="min-w-0">
              <p className="truncate text-[0.78rem] capitalize" style={{ color }}>
                {band}
              </p>
              <p className="text-readout text-[0.6rem] text-ink-faint">
                {range[0]}–{range[1]} Hz
              </p>
            </div>

            <div className="h-1 overflow-hidden rounded-full bg-ink/6">
              <motion.div
                className="h-full rounded-full"
                initial={false}
                animate={{ width: `${Math.min(100, (value / 0.5) * 100)}%` }}
                transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
                style={{ background: color, boxShadow: `0 0 8px -1px ${color}` }}
              />
            </div>

            <span className="text-readout text-right text-[0.72rem] text-ink-soft">
              {(value * 100).toFixed(0)}%
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}
