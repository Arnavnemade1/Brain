import { motion } from 'framer-motion'
import { useMemo, useState } from 'react'

import { EASE_GLIDE } from '@/animations/motion'
import { useReveal } from '@/hooks/useReveal'
import { cn } from '@/lib/cn'
import { clamp, mulberry32 } from '@/lib/math'
import { EMOTION_COLOR } from '@/lib/palette'
import { BIOME_META, EMOTIONS } from '@/types'
import type { Biome, EmotionLabel } from '@/types'

import { Section } from './Section'

/* -------------------------------------------------------------------------- */
/* Emotion → world mapping                                                    */
/* -------------------------------------------------------------------------- */

interface EmotionShape {
  biome: Biome
  weather: string
  light: string
  particles: string
  /** 0–1 how tall and dense the structures read. */
  verticality: number
  /** 0–1 how far apart things sit. */
  openness: number
  fog: number
}

const SHAPES: Record<EmotionLabel, EmotionShape> = {
  calm: {
    biome: 'beach_sunset', weather: 'Clear', light: 'Soft, even',
    particles: 'Pollen', verticality: 0.18, openness: 0.92, fog: 0.12,
  },
  nostalgia: {
    biome: 'childhood_home', weather: 'Clear', light: 'Low, golden',
    particles: 'Dust', verticality: 0.45, openness: 0.34, fog: 0.24,
  },
  wonder: {
    biome: 'floating_islands', weather: 'Clear', light: 'Wide, cold',
    particles: 'Motes', verticality: 0.88, openness: 0.86, fog: 0.2,
  },
  stress: {
    biome: 'city_streets', weather: 'Fog', light: 'Flat, hard',
    particles: 'None', verticality: 0.76, openness: 0.14, fog: 0.7,
  },
  curiosity: {
    biome: 'forest', weather: 'Overcast', light: 'Shifting',
    particles: 'Motes', verticality: 0.55, openness: 0.48, fog: 0.28,
  },
  melancholy: {
    biome: 'ocean', weather: 'Rain', light: 'Grey, diffuse',
    particles: 'Rain', verticality: 0.12, openness: 0.78, fog: 0.52,
  },
  joy: {
    biome: 'mountains', weather: 'Clear', light: 'Strong, warm',
    particles: 'Pollen', verticality: 0.7, openness: 0.9, fog: 0.08,
  },
  fear: {
    biome: 'abstract_space', weather: 'Fog', light: 'Failing',
    particles: 'Embers', verticality: 0.62, openness: 0.2, fog: 0.82,
  },
}

/* -------------------------------------------------------------------------- */
/* Preview                                                                    */
/* -------------------------------------------------------------------------- */

interface Structure {
  x: number
  width: number
  height: number
  confidence: number
  /** Horizontal slice breaks, as fractions of the structure's height. */
  breaks: number[]
}

/**
 * A schematic elevation of the reconstruction.
 *
 * SVG rather than a 3D preview: this is a diagram of *how confidence becomes
 * geometry*, and a flat elevation makes the relationship legible in a way a
 * perspective view would obscure. The real engine is one click away.
 */
function ReconstructionPreview({
  emotion,
  confidence,
}: {
  emotion: EmotionLabel
  confidence: number
}) {
  const shape = SHAPES[emotion]
  const tone = EMOTION_COLOR[emotion]

  const structures = useMemo<Structure[]>(() => {
    const random = mulberry32(emotion.length * 977 + 41)
    const count = Math.round(5 + shape.verticality * 7)

    return Array.from({ length: count }, (_, i) => {
      const t = count === 1 ? 0.5 : i / (count - 1)

      // Confidence falls off with distance from the centre of attention, the
      // same rule the real region generator uses.
      const distance = Math.abs(t - 0.5) * 2
      const local = clamp(confidence * (1 - distance * 0.45) * (0.7 + random() * 0.6))

      const height = (0.18 + shape.verticality * 0.72) * (0.5 + random() * 0.8)

      // Below the threshold, geometry breaks into fragments.
      const breakCount = local < 0.55 ? Math.round((1 - local) * 5) : 0
      const breaks = Array.from({ length: breakCount }, () => 0.2 + random() * 0.65).sort()

      return {
        x: 6 + t * 88 + (random() - 0.5) * 6 * shape.openness,
        width: 3 + random() * 7 * (1 - shape.openness * 0.5),
        height,
        confidence: local,
        breaks,
      }
    })
  }, [emotion, confidence, shape.verticality, shape.openness])

  return (
    <div
      className="rounded-card relative aspect-[16/10] w-full overflow-hidden"
      style={{
        background: `linear-gradient(to bottom, color-mix(in oklab, ${tone} 22%, var(--color-void)), var(--color-void))`,
      }}
    >
      <svg viewBox="0 0 100 62" className="absolute inset-0 size-full" preserveAspectRatio="none">
        <defs>
          <linearGradient id="ms-ground" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={tone} stopOpacity="0.22" />
            <stop offset="100%" stopColor={tone} stopOpacity="0.04" />
          </linearGradient>
        </defs>

        {/* Sun / key light */}
        <circle
          cx={72}
          cy={16}
          r={5 + confidence * 3}
          fill={tone}
          opacity={0.12 + confidence * 0.2}
          style={{ filter: 'blur(3px)' }}
        />

        {structures.map((structure, index) => {
          const top = 48 - structure.height * 40
          const resolved = structure.confidence >= 0.55

          if (structure.breaks.length === 0) {
            return (
              <motion.rect
                key={index}
                initial={false}
                animate={{ y: top, height: 48 - top, opacity: 0.35 + structure.confidence * 0.6 }}
                transition={{ duration: 0.8, ease: EASE_GLIDE }}
                x={structure.x}
                width={structure.width}
                fill={resolved ? tone : 'none'}
                stroke={tone}
                strokeWidth={0.35}
              />
            )
          }

          // Fragmented: draw the surviving slices only.
          const edges = [0, ...structure.breaks, 1]
          return (
            <g key={index}>
              {edges.slice(0, -1).map((edge, sliceIndex) => {
                if (sliceIndex % 2 === 1) return null // gaps
                const nextEdge = edges[sliceIndex + 1]
                const sliceTop = 48 - structure.height * 40 * (1 - edge)
                const sliceBottom = 48 - structure.height * 40 * (1 - nextEdge)
                return (
                  <motion.rect
                    key={sliceIndex}
                    initial={false}
                    animate={{ opacity: 0.2 + structure.confidence * 0.55 }}
                    transition={{ duration: 0.7, ease: EASE_GLIDE }}
                    x={structure.x}
                    y={Math.min(sliceTop, sliceBottom)}
                    width={structure.width}
                    height={Math.abs(sliceBottom - sliceTop)}
                    fill={tone}
                  />
                )
              })}
            </g>
          )
        })}

        {/* Ground */}
        <rect x="0" y="48" width="100" height="14" fill="url(#ms-ground)" />
        <line x1="0" y1="48" x2="100" y2="48" stroke={tone} strokeWidth="0.3" opacity="0.5" />
      </svg>

      {/* Atmosphere — thickens as confidence drops */}
      <motion.div
        className="pointer-events-none absolute inset-0"
        initial={false}
        animate={{ opacity: shape.fog * 0.5 + (1 - confidence) * 0.45 }}
        transition={{ duration: 0.8 }}
        style={{
          background: `linear-gradient(to top, color-mix(in oklab, ${tone} 45%, transparent), transparent 65%)`,
          backdropFilter: 'blur(2px)',
        }}
      />

      <div className="absolute top-3 left-3">
        <p className="text-label" style={{ color: `color-mix(in oklab, ${tone} 80%, white)` }}>
          {BIOME_META[shape.biome].label}
        </p>
      </div>
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Section                                                                    */
/* -------------------------------------------------------------------------- */

export function InteractiveViz() {
  const [emotion, setEmotion] = useState<EmotionLabel>('nostalgia')
  const [confidence, setConfidence] = useState(0.62)
  const preview = useReveal()

  const shape = SHAPES[emotion]
  const tone = EMOTION_COLOR[emotion]

  const facts = [
    { label: 'Setting', value: BIOME_META[shape.biome].label },
    { label: 'Lighting', value: shape.light },
    { label: 'Weather', value: shape.weather },
    { label: 'Particles', value: shape.particles },
  ]

  return (
    <Section
      id="visualisation"
      label="Interactive visualisation"
      title="Affect decides what kind of place it is. Confidence decides how much of it exists."
      lede="Move the controls. This is the same mapping the reconstruction engine applies to real inferred state."
    >
      <div className="grid gap-8 lg:grid-cols-[1.15fr_1fr] lg:items-start">
        <motion.div
          ref={preview.ref as React.RefObject<HTMLDivElement>}
          {...preview.props}
          className="glass rounded-panel p-4"
        >
          <ReconstructionPreview emotion={emotion} confidence={confidence} />
        </motion.div>

        <div className="space-y-8">
          <div>
            <p className="text-label mb-4">Detected emotional state</p>
            <div className="flex flex-wrap gap-2">
              {EMOTIONS.map((option) => {
                const active = option === emotion
                const optionTone = EMOTION_COLOR[option]
                return (
                  <button
                    key={option}
                    onClick={() => setEmotion(option)}
                    aria-pressed={active}
                    className={cn(
                      'rounded-pill border px-3.5 py-2 text-sm capitalize transition-all duration-300',
                      active ? 'text-ink' : 'border-ink/8 text-ink-muted hover:text-ink-soft',
                    )}
                    style={
                      active
                        ? {
                            borderColor: `color-mix(in oklab, ${optionTone} 50%, transparent)`,
                            background: `color-mix(in oklab, ${optionTone} 15%, transparent)`,
                          }
                        : undefined
                    }
                  >
                    {option}
                  </button>
                )
              })}
            </div>
          </div>

          <div>
            <div className="mb-3 flex items-baseline justify-between">
              <label htmlFor="confidence-slider" className="text-label">
                Neural confidence
              </label>
              <span className="text-readout text-sm" style={{ color: tone }}>
                {(confidence * 100).toFixed(0)}%
              </span>
            </div>
            <input
              id="confidence-slider"
              type="range"
              min={0.1}
              max={1}
              step={0.01}
              value={confidence}
              onChange={(event) => setConfidence(Number(event.target.value))}
              className="w-full accent-neural-400"
              style={{ accentColor: tone }}
            />
            <p className="mt-3 text-[0.8rem] leading-relaxed text-ink-faint">
              {confidence < 0.4
                ? 'Below the resolution threshold most structures never close. The engine leaves the gaps rather than guessing at what belongs there.'
                : confidence < 0.7
                  ? 'Partial support. Central regions resolve; the periphery stays fragmented and the air thickens where evidence thins.'
                  : 'Strong support across the extent. Geometry closes, haze lifts, and detail the model can actually defend appears.'}
            </p>
          </div>

          <dl className="grid grid-cols-2 gap-x-6 gap-y-5 border-t border-ink/8 pt-6">
            {facts.map((fact) => (
              <div key={fact.label}>
                <dt className="text-label mb-1.5">{fact.label}</dt>
                <dd className="text-sm text-ink-soft">{fact.value}</dd>
              </div>
            ))}
          </dl>
        </div>
      </div>
    </Section>
  )
}
