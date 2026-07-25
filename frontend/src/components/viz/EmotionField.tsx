import { motion } from 'framer-motion'

import { cn } from '@/lib/cn'
import { clamp } from '@/lib/math'
import { EMOTION_COLOR } from '@/lib/palette'
import type { EmotionReading } from '@/types'

interface EmotionFieldProps {
  emotion: EmotionReading | null
  size?: number
  className?: string
}

/** Anchor positions matching the backend's affect model. */
const ANCHORS: Record<string, { valence: number; arousal: number }> = {
  calm: { valence: 0.45, arousal: 0.12 },
  nostalgia: { valence: 0.35, arousal: 0.45 },
  wonder: { valence: 0.6, arousal: 0.72 },
  joy: { valence: 0.85, arousal: 0.6 },
  curiosity: { valence: 0.25, arousal: 0.62 },
  melancholy: { valence: -0.4, arousal: 0.22 },
  stress: { valence: -0.55, arousal: 0.75 },
  fear: { valence: -0.8, arousal: 0.95 },
}

/**
 * The valence–arousal plane with the current reading plotted on it.
 *
 * Shows *where the estimate sits relative to every candidate emotion*, which
 * a bar chart of probabilities cannot: adjacent labels like calm and
 * nostalgia are genuinely close in this space, and seeing the marker fall
 * between them explains the model's uncertainty far better than two similar
 * bars would.
 */
export function EmotionField({ emotion, size = 200, className }: EmotionFieldProps) {
  const valence = emotion?.valence ?? 0
  const arousal = emotion?.arousal ?? 0.5

  const toX = (v: number) => ((v + 1) / 2) * 100
  const toY = (a: number) => (1 - a) * 100

  const primaryColor = emotion ? EMOTION_COLOR[emotion.primary] : 'var(--color-ink-faint)'

  return (
    <div className={cn('relative', className)} style={{ width: size, height: size }}>
      <svg viewBox="0 0 100 100" className="size-full overflow-visible">
        {/* Quadrant grid */}
        <rect x="0" y="0" width="100" height="100" rx="6" className="fill-ink/3" />
        <line x1="50" y1="0" x2="50" y2="100" className="stroke-ink/10" strokeWidth="0.4" />
        <line x1="0" y1="50" x2="100" y2="50" className="stroke-ink/10" strokeWidth="0.4" />

        {/* Candidate anchors, sized by current probability */}
        {Object.entries(ANCHORS).map(([label, anchor]) => {
          const probability = emotion?.distribution?.[label as keyof typeof emotion.distribution]
          const weight = clamp(probability ?? 0)
          const color = EMOTION_COLOR[label as keyof typeof EMOTION_COLOR]

          return (
            <g key={label}>
              <motion.circle
                cx={toX(anchor.valence)}
                cy={toY(anchor.arousal)}
                initial={false}
                animate={{ r: 2 + weight * 9, opacity: 0.16 + weight * 0.5 }}
                transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
                fill={color}
              />
              <text
                x={toX(anchor.valence)}
                y={toY(anchor.arousal) - 5.5}
                textAnchor="middle"
                fontSize="3.4"
                fill={color}
                opacity={0.35 + weight * 0.65}
                style={{ fontFamily: 'ui-monospace, monospace' }}
              >
                {label}
              </text>
            </g>
          )
        })}

        {/* Current reading */}
        <motion.g
          initial={false}
          animate={{ x: toX(valence), y: toY(arousal) }}
          transition={{ type: 'spring', stiffness: 120, damping: 20 }}
        >
          <circle r="6" fill={primaryColor} opacity="0.16" />
          <circle r="2.6" fill={primaryColor} />
          <circle r="2.6" fill="none" stroke={primaryColor} strokeWidth="0.5" opacity="0.7">
            <animate attributeName="r" values="2.6;8;2.6" dur="3s" repeatCount="indefinite" />
            <animate attributeName="opacity" values="0.7;0;0.7" dur="3s" repeatCount="indefinite" />
          </circle>
        </motion.g>
      </svg>

      {/* Axis labels */}
      <span className="text-label absolute -bottom-4 left-0 !text-[0.55rem]">unpleasant</span>
      <span className="text-label absolute -bottom-4 right-0 !text-[0.55rem]">pleasant</span>
      <span className="text-label absolute -top-4 left-1/2 -translate-x-1/2 !text-[0.55rem]">
        high arousal
      </span>
    </div>
  )
}
