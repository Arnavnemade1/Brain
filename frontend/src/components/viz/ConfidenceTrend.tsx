import { useMemo } from 'react'

import { cn } from '@/lib/cn'
import type { NeuralFrame } from '@/types'

interface ConfidenceTrendProps {
  frames: NeuralFrame[]
  height?: number
  className?: string
}

/**
 * Confidence and coherence over the session.
 *
 * Both series share one axis because the interesting signal is the *gap*
 * between them: high confidence with low coherence means the model is sure
 * about each window but they disagree with each other, which is exactly the
 * condition that produces an incoherent reconstruction.
 */
export function ConfidenceTrend({ frames, height = 96, className }: ConfidenceTrendProps) {
  const { confidencePath, coherencePath, areaPath, latest } = useMemo(() => {
    if (frames.length < 2) {
      return { confidencePath: '', coherencePath: '', areaPath: '', latest: null }
    }

    const w = 100
    const h = 100
    const step = w / (frames.length - 1)

    const points = frames.map((frame, i) => ({
      x: i * step,
      confidence: h - frame.neuralConfidence * h,
      coherence: h - frame.coherence * h,
    }))

    const line = (key: 'confidence' | 'coherence') =>
      points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(2)},${p[key].toFixed(2)}`).join(' ')

    const area =
      `M0,${h} ` +
      points.map((p) => `L${p.x.toFixed(2)},${p.confidence.toFixed(2)}`).join(' ') +
      ` L${w},${h} Z`

    return {
      confidencePath: line('confidence'),
      coherencePath: line('coherence'),
      areaPath: area,
      latest: frames[frames.length - 1],
    }
  }, [frames])

  return (
    <div className={cn('relative', className)}>
      <svg
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        style={{ height }}
        className="w-full"
        role="img"
        aria-label="Neural confidence and coherence over time"
      >
        <defs>
          <linearGradient id="ms-conf-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#38c1ef" stopOpacity="0.28" />
            <stop offset="100%" stopColor="#38c1ef" stopOpacity="0" />
          </linearGradient>
        </defs>

        {[0.25, 0.5, 0.75].map((line) => (
          <line
            key={line}
            x1="0"
            y1={line * 100}
            x2="100"
            y2={line * 100}
            className="stroke-ink/6"
            strokeWidth="0.4"
            vectorEffect="non-scaling-stroke"
          />
        ))}

        {areaPath && <path d={areaPath} fill="url(#ms-conf-fill)" />}
        {coherencePath && (
          <path
            d={coherencePath}
            fill="none"
            stroke="#e2a54c"
            strokeWidth="1.2"
            strokeDasharray="3 2"
            vectorEffect="non-scaling-stroke"
            opacity="0.75"
          />
        )}
        {confidencePath && (
          <path
            d={confidencePath}
            fill="none"
            stroke="#38c1ef"
            strokeWidth="1.6"
            vectorEffect="non-scaling-stroke"
            strokeLinejoin="round"
          />
        )}
      </svg>

      <div className="mt-2.5 flex items-center gap-4">
        <span className="flex items-center gap-1.5 text-[0.65rem] text-ink-muted">
          <span className="h-px w-3 bg-neural-400" />
          Confidence
          {latest && (
            <span className="text-readout text-ink-soft">
              {(latest.neuralConfidence * 100).toFixed(0)}%
            </span>
          )}
        </span>
        <span className="flex items-center gap-1.5 text-[0.65rem] text-ink-muted">
          <span className="h-px w-3 border-t border-dashed border-memory-400" />
          Coherence
          {latest && (
            <span className="text-readout text-ink-soft">
              {(latest.coherence * 100).toFixed(0)}%
            </span>
          )}
        </span>
      </div>
    </div>
  )
}
