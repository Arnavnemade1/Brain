import { AnimatePresence, motion } from 'framer-motion'
import { Boxes, Sparkles } from 'lucide-react'

import { popVariants } from '@/animations/motion'
import { cn } from '@/lib/cn'
import { titleize } from '@/lib/format'
import { confidenceColor, EMOTION_COLOR } from '@/lib/palette'
import type {
  MemoryFragment,
  MemoryNarrative,
  ReconstructionStatus,
  SceneParameters,
} from '@/types'
import { BIOME_META } from '@/types'
import { Badge, EmptyState, Meter, Panel, Ring } from '@/ui'

const STATUS_COPY: Record<ReconstructionStatus, string> = {
  idle: 'No session running',
  acquiring: 'Buffering incoming signal',
  processing: 'Accumulating evidence before committing to a setting',
  reconstructing: 'Building geometry from scene parameters',
  stabilised: 'Reconstruction complete and stable',
  error: 'Reconstruction failed',
}

/* -------------------------------------------------------------------------- */
/* Progress                                                                   */
/* -------------------------------------------------------------------------- */

export function ReconstructionProgress({
  status,
  progress,
  confidence,
  coherence,
  resolvedCount,
  totalRegions,
}: {
  status: ReconstructionStatus
  progress: number
  confidence: number
  coherence: number
  resolvedCount: number
  totalRegions: number
}) {
  return (
    <Panel
      label="Memory reconstruction"
      title={titleize(status)}
      action={
        <Badge
          tone={status === 'stabilised' ? 'var(--color-signal-good)' : 'var(--color-neural-300)'}
          live={status === 'reconstructing' || status === 'processing'}
        >
          {(progress * 100).toFixed(0)}%
        </Badge>
      }
    >
      <div className="flex items-center gap-6">
        <Ring value={confidence} label="Confidence" tone={confidenceColor(confidence)} />

        <div className="min-w-0 flex-1 space-y-4">
          <Meter
            label="Reconstruction progress"
            value={progress}
            tone="var(--color-neural-400)"
          />
          <Meter label="Coherence" value={coherence} tone="var(--color-memory-300)" />
          <Meter
            label="Regions resolved"
            value={totalRegions > 0 ? resolvedCount / totalRegions : 0}
            tone="var(--color-emotion-calm)"
          />
        </div>
      </div>

      <p className="mt-5 border-t border-ink/8 pt-4 text-[0.76rem] leading-relaxed text-ink-muted">
        {STATUS_COPY[status]}
        {totalRegions > 0 && status !== 'idle' && (
          <>
            {' '}
            <span className="text-ink-faint">
              {resolvedCount} of {totalRegions} regions above the resolution threshold; the
              rest render as incomplete geometry.
            </span>
          </>
        )}
      </p>
    </Panel>
  )
}

/* -------------------------------------------------------------------------- */
/* Scene summary                                                              */
/* -------------------------------------------------------------------------- */

export function ScenePanel({
  scene,
  regionConfidence,
  onFocusRegion,
}: {
  scene: SceneParameters | null
  regionConfidence: Record<string, number>
  onFocusRegion?: (regionId: string) => void
}) {
  if (!scene) {
    return (
      <Panel label="Scene parameters" title="Not yet committed">
        <EmptyState
          icon={<Boxes className="size-5" />}
          title="Waiting for enough evidence"
          detail="The context model needs several consistent windows before it will name a setting. Nothing is guessed to fill the gap."
        />
      </Panel>
    )
  }

  const meta = BIOME_META[scene.biome]

  const facts: { label: string; value: string }[] = [
    { label: 'Time of day', value: titleize(scene.lighting.timeOfDay) },
    { label: 'Colour temp', value: `${scene.lighting.temperature.toFixed(0)}K` },
    { label: 'Weather', value: titleize(scene.atmosphere.weather) },
    { label: 'Particles', value: titleize(scene.atmosphere.particles) },
    { label: 'Spatial scale', value: `${(scene.spatialScale * 100).toFixed(0)}%` },
    { label: 'Distortion', value: `${(scene.distortion * 100).toFixed(0)}%` },
  ]

  return (
    <Panel
      label="Scene parameters"
      title={meta.label}
      action={
        <Badge tone="var(--color-memory-300)">seed {scene.seed.toString(36)}</Badge>
      }
    >
      <p className="text-[0.78rem] leading-relaxed text-ink-muted">{meta.tagline}</p>

      {/* Palette */}
      <div className="mt-4 flex gap-1.5">
        {(['key', 'fill', 'rim', 'sky', 'ground', 'accent'] as const).map((slot) => (
          <div key={slot} className="flex-1" title={`${slot}: ${scene.palette[slot]}`}>
            <div
              className="h-7 rounded-md border border-ink/10"
              style={{ background: scene.palette[slot] }}
            />
            <p className="text-label mt-1.5 !text-[0.5rem]">{slot}</p>
          </div>
        ))}
      </div>

      <dl className="mt-5 grid grid-cols-3 gap-x-4 gap-y-4 border-t border-ink/8 pt-4">
        {facts.map((fact) => (
          <div key={fact.label}>
            <dt className="text-label mb-1">{fact.label}</dt>
            <dd className="text-[0.78rem] text-ink-soft">{fact.value}</dd>
          </div>
        ))}
      </dl>

      {/* Regions */}
      <div className="mt-5 border-t border-ink/8 pt-4">
        <p className="text-label mb-3">Regions</p>
        <ul className="space-y-1.5">
          {scene.regions.map((region) => {
            const value = regionConfidence[region.id] ?? region.confidence
            const resolved = value >= scene.fragmentThreshold

            return (
              <li key={region.id}>
                <button
                  onClick={() => onFocusRegion?.(region.id)}
                  disabled={!onFocusRegion || resolved}
                  className={cn(
                    'grid w-full grid-cols-[1fr_4.5rem_2.4rem] items-center gap-3 rounded-lg px-2 py-1.5 text-left transition-colors',
                    onFocusRegion && !resolved && 'hover:bg-ink/5',
                    'disabled:cursor-default',
                  )}
                  title={
                    resolved
                      ? 'Resolved'
                      : 'Direct additional evidence at this region'
                  }
                >
                  <span className="min-w-0">
                    <span className="block truncate text-[0.76rem] text-ink-soft">
                      {region.label}
                    </span>
                    <span className="text-label !text-[0.52rem]">{region.archetype}</span>
                  </span>

                  <span className="h-1 overflow-hidden rounded-full bg-ink/6">
                    <motion.span
                      className="block h-full rounded-full"
                      initial={false}
                      animate={{ width: `${value * 100}%` }}
                      transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
                      style={{ background: confidenceColor(value) }}
                    />
                  </span>

                  <span
                    className="text-readout text-right text-[0.68rem]"
                    style={{ color: confidenceColor(value) }}
                  >
                    {(value * 100).toFixed(0)}%
                  </span>
                </button>
              </li>
            )
          })}
        </ul>
        <p className="mt-3 text-[0.68rem] leading-relaxed text-ink-faint">
          Threshold {(scene.fragmentThreshold * 100).toFixed(0)}%. Regions below it stay
          fragmented; select one to spend incoming evidence resolving it.
        </p>
      </div>
    </Panel>
  )
}

/* -------------------------------------------------------------------------- */
/* Fragments                                                                  */
/* -------------------------------------------------------------------------- */

export function FragmentsPanel({
  fragments,
  openFragmentId,
  onOpen,
}: {
  fragments: MemoryFragment[]
  openFragmentId: string | null
  onOpen: (id: string | null) => void
}) {
  return (
    <Panel
      label="Memory fragments"
      title={fragments.length > 0 ? `${fragments.length} recovered` : 'None yet'}
    >
      {fragments.length === 0 ? (
        <EmptyState
          icon={<Sparkles className="size-5" />}
          title="Fragments appear as the session completes"
          detail="Each one marks a window that stood out from its neighbours in affect, spectral character or coherence."
        />
      ) : (
        <ul className="space-y-1.5">
          {fragments.map((fragment) => {
            const open = fragment.id === openFragmentId
            const tone = EMOTION_COLOR[fragment.emotion]

            return (
              <li key={fragment.id}>
                <button
                  onClick={() => onOpen(open ? null : fragment.id)}
                  aria-expanded={open}
                  className={cn(
                    'w-full rounded-xl border px-3.5 py-3 text-left transition-colors',
                    open
                      ? 'border-ink/16 bg-ink/6'
                      : 'border-ink/8 hover:border-ink/14 hover:bg-ink/4',
                  )}
                >
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="truncate text-[0.84rem] text-ink">{fragment.title}</span>
                    <span
                      className="text-readout shrink-0 text-[0.62rem]"
                      style={{ color: tone }}
                    >
                      {(fragment.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="mt-1.5 flex items-center gap-2">
                    <span
                      className="size-1.5 rounded-full"
                      style={{ background: tone }}
                      aria-hidden
                    />
                    <span className="text-[0.68rem] capitalize text-ink-muted">
                      {fragment.emotion}
                    </span>
                    <span className="text-readout text-[0.62rem] text-ink-faint">
                      t+{fragment.sourceTime.toFixed(0)}s
                    </span>
                  </div>

                  <AnimatePresence initial={false}>
                    {open && (
                      <motion.div
                        variants={popVariants}
                        initial="initial"
                        animate="animate"
                        exit="exit"
                        className="mt-3 space-y-3 border-t border-ink/8 pt-3"
                      >
                        <p className="text-[0.76rem] leading-relaxed text-ink-soft">
                          {fragment.description}
                        </p>

                        <div>
                          <p className="text-label mb-1.5">Sensory detail</p>
                          <div className="flex flex-wrap gap-1.5">
                            {fragment.sensoryDetails.map((detail) => (
                              <span
                                key={detail}
                                className="rounded border border-ink/8 px-1.5 py-0.5 text-[0.65rem] text-ink-muted"
                              >
                                {detail}
                              </span>
                            ))}
                          </div>
                        </div>

                        <div>
                          <p className="text-label mb-1.5">Neural evidence</p>
                          <p className="text-readout text-[0.68rem] leading-relaxed text-ink-faint">
                            {fragment.neuralEvidence}
                          </p>
                        </div>

                        {fragment.unlocksRegion && (
                          <p className="text-[0.68rem] text-memory-300">
                            Opening this directed additional evidence at a fragmented region.
                          </p>
                        )}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </button>
              </li>
            )
          })}
        </ul>
      )}
    </Panel>
  )
}

/* -------------------------------------------------------------------------- */
/* Narrative                                                                  */
/* -------------------------------------------------------------------------- */

export function NarrativePanel({ narrative }: { narrative: MemoryNarrative | null }) {
  if (!narrative) return null

  return (
    <Panel
      label="AI memory summary"
      title={narrative.title}
      action={
        <Badge tone={narrative.generatedByModel ? 'var(--color-cortex-300)' : undefined}>
          {narrative.generatedByModel ? 'model' : 'local'}
        </Badge>
      }
    >
      <p className="text-[0.86rem] leading-relaxed text-ink-soft">{narrative.summary}</p>

      {narrative.observations.length > 0 && (
        <div className="mt-6">
          <p className="text-label mb-3">Observations</p>
          <ul className="space-y-2.5">
            {narrative.observations.map((observation) => (
              <li
                key={observation}
                className="flex gap-2.5 text-[0.78rem] leading-relaxed text-ink-muted"
              >
                <span className="mt-[0.55em] size-1 shrink-0 rounded-full bg-neural-400/60" />
                {observation}
              </li>
            ))}
          </ul>
        </div>
      )}

      {narrative.uncertainties.length > 0 && (
        <div className="mt-6 border-t border-ink/8 pt-5">
          <p className="text-label mb-3">What could not be recovered</p>
          <ul className="space-y-2.5">
            {narrative.uncertainties.map((uncertainty) => (
              <li
                key={uncertainty}
                className="flex gap-2.5 text-[0.78rem] leading-relaxed text-ink-faint"
              >
                <span className="mt-[0.55em] size-1 shrink-0 rounded-full bg-memory-400/50" />
                {uncertainty}
              </li>
            ))}
          </ul>
        </div>
      )}
    </Panel>
  )
}
