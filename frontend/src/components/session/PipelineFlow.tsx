import { AnimatePresence, motion } from 'framer-motion'
import { useState } from 'react'

import { cn } from '@/lib/cn'
import { STAGE_BY_ID } from '@/types'
import type { PipelineStageMeta, PipelineStageState, StageStatus } from '@/types'

interface PipelineFlowProps {
  stages: PipelineStageState[]
  className?: string
}

const BAND_TONE: Record<PipelineStageMeta['band'], string> = {
  acquisition: 'var(--color-neural-300)',
  analysis: 'var(--color-emotion-calm)',
  inference: 'var(--color-cortex-400)',
  synthesis: 'var(--color-memory-300)',
}

const STATUS_LABEL: Record<StageStatus, string> = {
  idle: 'Idle',
  active: 'Active',
  complete: 'Complete',
  degraded: 'Degraded',
  error: 'Error',
}

function statusTone(status: StageStatus, bandTone: string): string {
  if (status === 'error') return 'var(--color-signal-poor)'
  if (status === 'degraded') return 'var(--color-signal-fair)'
  if (status === 'idle') return 'var(--color-ink-faint)'
  return bandTone
}

function StageNode({
  state,
  meta,
  expanded,
  onToggle,
}: {
  state: PipelineStageState
  meta: PipelineStageMeta
  expanded: boolean
  onToggle: () => void
}) {
  const bandTone = BAND_TONE[meta.band]
  const tone = statusTone(state.status, bandTone)
  const running = state.status === 'active' || state.status === 'degraded'
  const idle = state.status === 'idle'

  return (
    <li className="relative">
      <button
        onClick={onToggle}
        aria-expanded={expanded}
        className={cn(
          'group relative flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition-colors',
          'hover:bg-ink/5',
          expanded && 'bg-ink/6',
        )}
      >
        {/* Node marker */}
        <span className="relative grid size-7 shrink-0 place-items-center">
          <span
            className="absolute inset-0 rounded-full border transition-all duration-500"
            style={{
              borderColor: `color-mix(in oklab, ${tone} ${idle ? 25 : 60}%, transparent)`,
              background: `color-mix(in oklab, ${tone} ${idle ? 4 : 14}%, transparent)`,
            }}
          />
          {running && (
            <motion.span
              className="absolute inset-0 rounded-full border"
              style={{ borderColor: tone }}
              animate={{ scale: [1, 1.55], opacity: [0.55, 0] }}
              transition={{ duration: 1.8, repeat: Infinity, ease: 'easeOut' }}
            />
          )}
          <span
            className="size-1.5 rounded-full transition-colors duration-500"
            style={{
              background: tone,
              boxShadow: idle ? 'none' : `0 0 8px ${tone}`,
            }}
          />
        </span>

        <span className="min-w-0 flex-1">
          <span className="flex items-baseline justify-between gap-2">
            <span
              className={cn(
                'truncate text-[0.82rem] transition-colors',
                idle ? 'text-ink-faint' : 'text-ink',
              )}
            >
              {meta.label}
            </span>
            <span
              className="text-readout shrink-0 text-[0.6rem]"
              style={{ color: `color-mix(in oklab, ${tone} 80%, transparent)` }}
            >
              {state.latencyMs > 0 ? `${state.latencyMs.toFixed(1)}ms` : STATUS_LABEL[state.status]}
            </span>
          </span>

          <span className="mt-1 block truncate text-[0.68rem] text-ink-faint">
            {state.message || meta.summary}
          </span>

          {/* Per-stage progress */}
          <span className="mt-1.5 block h-px w-full overflow-hidden bg-ink/8">
            <motion.span
              className="block h-full"
              style={{ background: tone }}
              initial={false}
              animate={{ width: `${state.progress * 100}%` }}
              transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
            />
          </span>
        </span>
      </button>

      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
            className="overflow-hidden"
          >
            <div className="ml-10 border-l border-ink/8 py-2 pl-4">
              <p className="text-[0.75rem] leading-relaxed text-ink-muted">{meta.detail}</p>
              <dl className="mt-3 flex flex-wrap gap-x-6 gap-y-2">
                <div>
                  <dt className="text-label !text-[0.55rem]">Throughput</dt>
                  <dd className="text-readout mt-0.5 text-[0.72rem] text-ink-soft">
                    {state.throughput.toLocaleString()}
                  </dd>
                </div>
                <div>
                  <dt className="text-label !text-[0.55rem]">Stage confidence</dt>
                  <dd className="text-readout mt-0.5 text-[0.72rem] text-ink-soft">
                    {(state.confidence * 100).toFixed(0)}%
                  </dd>
                </div>
                <div>
                  <dt className="text-label !text-[0.55rem]">Status</dt>
                  <dd className="mt-0.5 text-[0.72rem]" style={{ color: tone }}>
                    {STATUS_LABEL[state.status]}
                  </dd>
                </div>
              </dl>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </li>
  )
}

/**
 * The twelve-stage pipeline as a live vertical flow.
 *
 * The connecting rail fills to the deepest stage that has actually done work,
 * so the animation is driven by real progress rather than a timer.
 */
export function PipelineFlow({ stages, className }: PipelineFlowProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null)

  const deepestActive = stages.reduce(
    (deepest, stage, index) => (stage.status !== 'idle' ? index : deepest),
    -1,
  )
  const railProgress = stages.length > 0 ? (deepestActive + 1) / stages.length : 0

  return (
    <div className={cn('relative', className)}>
      <div className="absolute top-3 bottom-3 left-[26px] w-px bg-ink/8" aria-hidden />
      <motion.div
        className="absolute top-3 left-[26px] w-px bg-gradient-to-b from-neural-300 via-cortex-400 to-memory-300"
        initial={false}
        animate={{ height: `${railProgress * 100}%` }}
        transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
        aria-hidden
      />

      <ol className="relative space-y-0.5">
        {stages.map((state) => {
          const meta = STAGE_BY_ID[state.id]
          if (!meta) return null
          return (
            <StageNode
              key={state.id}
              state={state}
              meta={meta}
              expanded={expandedId === state.id}
              onToggle={() => setExpandedId(expandedId === state.id ? null : state.id)}
            />
          )
        })}
      </ol>
    </div>
  )
}
