import { motion } from 'framer-motion'
import { useMemo, useState } from 'react'

import type { ElectrodeInfo } from '@/services/api'

interface ElectrodeMapProps {
  electrodes: ElectrodeInfo[]
  className?: string
}

const REGION_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  occipital: { bg: 'rgba(236, 72, 153, 0.25)', border: '#ec4899', text: 'text-pink-400' },
  parietal: { bg: 'rgba(168, 85, 247, 0.25)', border: '#a855f7', text: 'text-purple-400' },
  frontal: { bg: 'rgba(59, 130, 246, 0.25)', border: '#3b82f6', text: 'text-blue-400' },
  temporal: { bg: 'rgba(16, 185, 129, 0.25)', border: '#10b981', text: 'text-emerald-400' },
  central: { bg: 'rgba(245, 158, 11, 0.25)', border: '#f59e0b', text: 'text-amber-400' },
}

const DEFAULT_COLOR = { bg: 'rgba(148, 163, 184, 0.25)', border: '#94a3b8', text: 'text-slate-400' }

export function ElectrodeMap({ electrodes, className = '' }: ElectrodeMapProps) {
  const [hovered, setHovered] = useState<ElectrodeInfo | null>(null)
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null)

  const regions = useMemo(() => {
    const set = new Set<string>()
    electrodes.forEach((e) => set.add(e.region.toLowerCase()))
    return Array.from(set)
  }, [electrodes])

  return (
    <div className={`glass rounded-panel relative p-6 ${className}`}>
      <div className="flex flex-wrap items-center justify-between gap-4 pb-4">
        <div>
          <h3 className="font-display text-[1rem] tracking-tight">Electrode Montage Map</h3>
          <p className="text-[0.8rem] text-ink-muted">
            {electrodes.length} channel 10-20 layout across brain regions
          </p>
        </div>

        {/* Region filter pills */}
        <div className="flex flex-wrap items-center gap-1.5">
          <button
            onClick={() => setSelectedRegion(null)}
            className={`rounded-pill px-3 py-1 text-[0.72rem] transition-colors ${
              selectedRegion === null
                ? 'bg-neural-400/30 text-neural-200 border border-neural-400/40'
                : 'text-ink-muted hover:text-ink'
            }`}
          >
            All ({electrodes.length})
          </button>
          {regions.map((region) => {
            const style = REGION_COLORS[region] || DEFAULT_COLOR
            const isSelected = selectedRegion === region
            const count = electrodes.filter((e) => e.region.toLowerCase() === region).length
            return (
              <button
                key={region}
                onClick={() => setSelectedRegion(isSelected ? null : region)}
                className={`rounded-pill px-3 py-1 text-[0.72rem] transition-colors capitalize ${
                  isSelected
                    ? `${style.text} bg-ink/10 border border-current`
                    : 'text-ink-muted hover:text-ink'
                }`}
              >
                {region} ({count})
              </button>
            )
          })}
        </div>
      </div>

      {/* Head Map SVG */}
      <div className="relative mx-auto aspect-square max-w-[420px] p-4">
        <svg viewBox="-1.2 -1.2 2.4 2.4" className="size-full overflow-visible">
          {/* Head outline circle */}
          <circle
            cx="0"
            cy="0"
            r="1"
            className="fill-neural-950/40 stroke-ink/15"
            strokeWidth="0.015"
          />

          {/* Nose indicator at top */}
          <polygon
            points="-0.08,-1 -0.04,-1.09 0.04,-1.09 0.08,-1"
            className="fill-ink/20 stroke-ink/30"
            strokeWidth="0.01"
          />

          {/* Ears on sides */}
          <path
            d="M -1.02,-0.12 C -1.09,-0.08 -1.09,0.08 -1.02,0.12"
            className="fill-none stroke-ink/20"
            strokeWidth="0.015"
          />
          <path
            d="M 1.02,-0.12 C 1.09,-0.08 1.09,0.08 1.02,0.12"
            className="fill-none stroke-ink/20"
            strokeWidth="0.015"
          />

          {/* Reference crosshairs */}
          <line x1="0" y1="-0.95" x2="0" y2="0.95" className="stroke-ink/5" strokeWidth="0.005" strokeDasharray="0.02 0.02" />
          <line x1="-0.95" y1="0" x2="0.95" y2="0" className="stroke-ink/5" strokeWidth="0.005" strokeDasharray="0.02 0.02" />

          {/* Electrodes */}
          {electrodes.map((e) => {
            const regionLower = e.region.toLowerCase()
            const color = REGION_COLORS[regionLower] || DEFAULT_COLOR
            const isDimmed = selectedRegion && selectedRegion !== regionLower
            const isHovered = hovered?.label === e.label

            return (
              <g
                key={e.label}
                transform={`translate(${e.x}, ${e.y})`}
                onMouseEnter={() => setHovered(e)}
                onMouseLeave={() => setHovered(null)}
                className="cursor-pointer transition-opacity duration-200"
                style={{ opacity: isDimmed ? 0.25 : 1 }}
              >
                {/* Glow ring when hovered */}
                {isHovered && (
                  <circle
                    r="0.07"
                    fill={color.border}
                    opacity="0.3"
                    className="animate-pulse"
                  />
                )}
                {/* Electrode dot */}
                <circle
                  r={isHovered ? '0.045' : '0.035'}
                  fill={color.bg}
                  stroke={color.border}
                  strokeWidth="0.008"
                />
                {/* Channel Label */}
                <text
                  y="0.01"
                  textAnchor="middle"
                  dominantBaseline="middle"
                  className="select-none font-mono text-[0.035px] font-semibold fill-ink"
                >
                  {e.label}
                </text>
              </g>
            )
          })}
        </svg>

        {/* Floating tooltip */}
        {hovered && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass rounded-panel absolute bottom-4 left-1/2 -translate-x-1/2 border border-ink/15 px-3 py-2 text-center shadow-lg"
          >
            <div className="font-mono text-sm font-bold text-neural-300">{hovered.label}</div>
            <div className="text-[0.72rem] capitalize text-ink-muted">
              Region: <span className="font-medium text-ink">{hovered.region}</span>
            </div>
            <div className="font-mono text-[0.68rem] text-ink-faint">
              x: {hovered.x.toFixed(2)}, y: {hovered.y.toFixed(2)}
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}
