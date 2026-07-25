import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { Film, Play } from 'lucide-react'
import { useState } from 'react'

import { riseItem, staggerContainer } from '@/animations/motion'
import { cn } from '@/lib/cn'
import { api } from '@/services/api'
import { useSessionStore } from '@/stores/sessionStore'
import { useUiStore } from '@/stores/uiStore'
import { Button, ErrorNotice, LoadingState, Panel } from '@/ui'

const SPEEDS = [1, 2, 4, 8] as const

export function SessionLauncher() {
  const [clipId, setClipId] = useState('daybreak')
  const [conditionId, setConditionId] = useState('research')

  const speed = useUiStore((s) => s.preferredSpeed)
  const setSpeed = useUiStore((s) => s.setPreferredSpeed)

  const starting = useSessionStore((s) => s.starting)
  const error = useSessionStore((s) => s.error)
  const clearError = useSessionStore((s) => s.clearError)
  const startSession = useSessionStore((s) => s.startSession)

  const clips = useQuery({ queryKey: ['clips'], queryFn: api.clips })
  const conditions = useQuery({ queryKey: ['conditions'], queryFn: api.conditions })
  const system = useQuery({ queryKey: ['system'], queryFn: api.system })

  const launch = async () => {
    try {
      await startSession(clipId, conditionId, speed)
    } catch {
      // The store holds the message; ErrorNotice renders it below.
    }
  }

  return (
    <motion.div
      variants={staggerContainer}
      initial="initial"
      animate="animate"
      className="mx-auto w-full max-w-2xl space-y-5"
    >
      <motion.header variants={riseItem} className="text-center">
        <p className="text-label mb-4">Memory reconstruction</p>
        <h1 className="text-[clamp(1.7rem,4vw,2.4rem)] leading-tight">
          Pick what the subject watched.
        </h1>
        <p className="mx-auto mt-4 max-w-lg text-sm leading-relaxed text-ink-muted">
          EEG is recorded while the clip plays, then the reconstruction is decoded from
          that signal alone and scored against the original frame by frame.
        </p>
      </motion.header>

      {system.data && !system.data.decodersFitted && (
        <ErrorNotice
          message="Decoder weights are missing, so reconstruction will use the untrained fallback and fidelity will be near zero. Run `python -m tools.train_decoders` in the backend."
        />
      )}

      <Panel label="Stimulus" title="Reference clip" animate={false}>
        {clips.isLoading && <LoadingState title="Loading clips" />}

        {clips.isError && (
          <ErrorNotice
            message="Could not reach the backend. Start it with `make backend`."
            onRetry={() => clips.refetch()}
          />
        )}

        {clips.data && (
          <div className="space-y-2">
            {clips.data.map((clip) => {
              const active = clip.id === clipId
              return (
                <button
                  key={clip.id}
                  onClick={() => setClipId(clip.id)}
                  aria-pressed={active}
                  className={cn(
                    'w-full rounded-xl border px-4 py-3.5 text-left transition-all duration-300',
                    active
                      ? 'border-neural-400/45 bg-neural-400/8'
                      : 'border-ink/8 hover:border-ink/16 hover:bg-ink/4',
                  )}
                >
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="flex items-center gap-2 text-[0.9rem] text-ink">
                      <Film className="size-3.5 text-ink-muted" />
                      {clip.title}
                    </span>
                    <span className="text-readout shrink-0 text-[0.62rem] text-ink-faint">
                      {clip.durationSeconds.toFixed(0)}s · {clip.sceneCount} scenes
                    </span>
                  </div>
                  <p className="mt-1.5 text-[0.78rem] leading-relaxed text-ink-muted">
                    {clip.description}
                  </p>
                  <div className="mt-2.5 flex flex-wrap gap-1.5">
                    {clip.tags.map((tag) => (
                      <span
                        key={tag}
                        className="rounded border border-ink/8 px-1.5 py-0.5 text-[0.6rem] text-ink-faint"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </button>
              )
            })}
          </div>
        )}
      </Panel>

      <Panel label="Acquisition" title="Recording quality" animate={false}>
        {conditions.data && (
          <div className="grid gap-2 sm:grid-cols-2">
            {conditions.data.map((condition) => {
              const active = condition.id === conditionId
              return (
                <button
                  key={condition.id}
                  onClick={() => setConditionId(condition.id)}
                  aria-pressed={active}
                  className={cn(
                    'rounded-xl border px-3.5 py-3 text-left transition-all duration-300',
                    active
                      ? 'border-neural-400/45 bg-neural-400/8'
                      : 'border-ink/8 hover:border-ink/16 hover:bg-ink/4',
                  )}
                >
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="text-[0.84rem] text-ink">{condition.label}</span>
                    <span className="text-readout shrink-0 text-[0.58rem] text-ink-faint">
                      artifact {(condition.artifactLevel * 100).toFixed(0)}%
                    </span>
                  </div>
                  <p className="mt-1.5 text-[0.72rem] leading-relaxed text-ink-muted">
                    {condition.description}
                  </p>
                </button>
              )
            })}
          </div>
        )}

        <div className="mt-5 border-t border-ink/8 pt-4">
          <p className="text-label mb-3">Playback rate</p>
          <div className="flex gap-1.5">
            {SPEEDS.map((option) => (
              <button
                key={option}
                onClick={() => setSpeed(option)}
                aria-pressed={speed === option}
                className={cn(
                  'flex-1 rounded-lg border px-3 py-2 text-[0.78rem] transition-colors',
                  speed === option
                    ? 'border-neural-400/45 bg-neural-400/10 text-ink'
                    : 'border-ink/8 text-ink-muted hover:text-ink-soft',
                )}
              >
                {option}×
              </button>
            ))}
          </div>
          <p className="mt-2.5 text-[0.7rem] text-ink-faint">
            Analysis runs at full fidelity regardless; this only sets how fast windows
            are fed in.
          </p>
        </div>
      </Panel>

      {error && <ErrorNotice message={error} onDismiss={clearError} />}

      <motion.div variants={riseItem}>
        <Button
          variant="primary"
          size="lg"
          fullWidth
          loading={starting}
          onClick={launch}
          icon={<Play className="size-4" />}
        >
          {starting ? 'Opening stream' : 'Begin reconstruction'}
        </Button>
      </motion.div>
    </motion.div>
  )
}
