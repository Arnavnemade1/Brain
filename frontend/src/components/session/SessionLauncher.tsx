import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { FileUp, Play, Waves } from 'lucide-react'
import { useRef, useState } from 'react'

import { riseItem, staggerContainer } from '@/animations/motion'
import { cn } from '@/lib/cn'
import { bytes } from '@/lib/format'
import { api } from '@/services/api'
import { useSessionStore } from '@/stores/sessionStore'
import { useUiStore } from '@/stores/uiStore'
import { Button, ErrorNotice, LoadingState, Panel } from '@/ui'

type Mode = 'simulate' | 'upload'

const DURATIONS = [60, 120, 240] as const
const SPEEDS = [1, 2, 4, 8] as const

const ACCEPTED = '.csv,.tsv,.txt,.edf,.bdf'

export function SessionLauncher({ initialMode = 'simulate' }: { initialMode?: Mode }) {
  const [mode, setMode] = useState<Mode>(initialMode)
  const [profile, setProfile] = useState<string>('coastal_recall')
  const [duration, setDuration] = useState<number>(120)
  const [file, setFile] = useState<File | null>(null)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const speed = useUiStore((s) => s.preferredSpeed)
  const setSpeed = useUiStore((s) => s.setPreferredSpeed)
  const toast = useUiStore((s) => s.toast)

  const starting = useSessionStore((s) => s.starting)
  const error = useSessionStore((s) => s.error)
  const clearError = useSessionStore((s) => s.clearError)
  const startSimulation = useSessionStore((s) => s.startSimulation)
  const startUpload = useSessionStore((s) => s.startUpload)

  const profiles = useQuery({ queryKey: ['profiles'], queryFn: api.profiles })

  const launch = async () => {
    try {
      if (mode === 'simulate') {
        await startSimulation(profile, duration, speed)
      } else if (file) {
        await startUpload(file, speed)
      }
    } catch {
      // The store holds the message; ErrorNotice renders it below.
    }
  }

  const pickFile = (candidate: File | undefined) => {
    if (!candidate) return
    const extension = candidate.name.slice(candidate.name.lastIndexOf('.')).toLowerCase()
    if (!ACCEPTED.split(',').includes(extension)) {
      toast(`${extension || 'That file type'} is not supported. Use CSV, TSV or EDF.`, 'warning')
      return
    }
    setFile(candidate)
  }

  return (
    <motion.div
      variants={staggerContainer}
      initial="initial"
      animate="animate"
      className="mx-auto w-full max-w-2xl space-y-5"
    >
      <motion.header variants={riseItem} className="text-center">
        <p className="text-label mb-4">Neural session</p>
        <h1 className="text-[clamp(1.7rem,4vw,2.4rem)] leading-tight">
          Give the engine a signal to work from.
        </h1>
        <p className="mx-auto mt-4 max-w-md text-sm leading-relaxed text-ink-muted">
          The reconstruction assembles in real time as windows are analysed. Nothing is
          pre-rendered.
        </p>
      </motion.header>

      {/* Mode switch */}
      <motion.div variants={riseItem} className="glass-faint rounded-pill flex gap-1 p-1">
        {(
          [
            { id: 'simulate' as const, label: 'Simulated stream', icon: Waves },
            { id: 'upload' as const, label: 'Upload recording', icon: FileUp },
          ]
        ).map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setMode(id)}
            aria-pressed={mode === id}
            className={cn(
              'relative flex flex-1 items-center justify-center gap-2 rounded-pill px-4 py-2.5 text-sm transition-colors',
              mode === id ? 'text-ink' : 'text-ink-muted hover:text-ink-soft',
            )}
          >
            {mode === id && (
              <motion.span
                layoutId="launcher-mode"
                className="absolute inset-0 rounded-pill bg-ink/8"
                transition={{ type: 'spring', stiffness: 340, damping: 30 }}
              />
            )}
            <Icon className="relative size-3.5" />
            <span className="relative">{label}</span>
          </button>
        ))}
      </motion.div>

      {mode === 'simulate' ? (
        <Panel label="Source" title="Simulation profile" animate={false}>
          {profiles.isLoading && <LoadingState title="Loading profiles" />}

          {profiles.isError && (
            <ErrorNotice
              message="Could not reach the backend. Start it with `make backend`."
              onRetry={() => profiles.refetch()}
            />
          )}

          {profiles.data && (
            <div className="space-y-2">
              {profiles.data.map((option) => {
                const active = option.id === profile
                return (
                  <button
                    key={option.id}
                    onClick={() => setProfile(option.id)}
                    aria-pressed={active}
                    className={cn(
                      'w-full rounded-xl border px-4 py-3.5 text-left transition-all duration-300',
                      active
                        ? 'border-neural-400/45 bg-neural-400/8'
                        : 'border-ink/8 hover:border-ink/16 hover:bg-ink/4',
                    )}
                  >
                    <div className="flex items-baseline justify-between gap-3">
                      <span className="text-[0.9rem] text-ink">{option.name}</span>
                      <span className="text-readout shrink-0 text-[0.6rem] text-ink-faint">
                        artifact {(option.artifactLevel * 100).toFixed(0)}%
                      </span>
                    </div>
                    <p className="mt-1.5 text-[0.78rem] leading-relaxed text-ink-muted">
                      {option.description}
                    </p>
                    <div className="mt-2.5 flex flex-wrap gap-1.5">
                      {option.tags.map((tag) => (
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
      ) : (
        <Panel label="Source" title="EEG recording" animate={false}>
          <div
            onDragOver={(event) => {
              event.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault()
              setDragging(false)
              pickFile(event.dataTransfer.files[0])
            }}
            onClick={() => inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') inputRef.current?.click()
            }}
            className={cn(
              'grid cursor-pointer place-items-center gap-3 rounded-xl border border-dashed px-6 py-12 text-center transition-colors',
              dragging
                ? 'border-neural-400/60 bg-neural-400/8'
                : 'border-ink/14 hover:border-ink/24 hover:bg-ink/3',
            )}
          >
            <FileUp className="size-5 text-ink-muted" />
            {file ? (
              <div>
                <p className="text-sm text-ink">{file.name}</p>
                <p className="text-readout mt-1 text-[0.7rem] text-ink-faint">
                  {bytes(file.size)}
                </p>
              </div>
            ) : (
              <div>
                <p className="text-sm text-ink-soft">Drop a recording, or click to browse</p>
                <p className="mt-1.5 text-[0.72rem] text-ink-faint">
                  CSV or TSV (one column per channel) · EDF · BDF
                </p>
              </div>
            )}
          </div>

          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED}
            className="sr-only"
            onChange={(event) => pickFile(event.target.files?.[0])}
          />

          <p className="mt-4 text-[0.72rem] leading-relaxed text-ink-faint">
            Channel labels are matched against the 10–20 montage where possible. A
            <code className="mx-1 rounded bg-ink/6 px-1 py-0.5">#&nbsp;sample_rate:&nbsp;256</code>
            comment on the first line sets the rate; otherwise a leading time column is
            used, falling back to 256&nbsp;Hz.
          </p>
        </Panel>
      )}

      {/* Parameters */}
      <Panel label="Parameters" animate={false}>
        <div className="grid gap-6 sm:grid-cols-2">
          {mode === 'simulate' && (
            <div>
              <p className="text-label mb-3">Session length</p>
              <div className="flex gap-1.5">
                {DURATIONS.map((option) => (
                  <button
                    key={option}
                    onClick={() => setDuration(option)}
                    aria-pressed={duration === option}
                    className={cn(
                      'flex-1 rounded-lg border px-3 py-2 text-[0.78rem] transition-colors',
                      duration === option
                        ? 'border-neural-400/45 bg-neural-400/10 text-ink'
                        : 'border-ink/8 text-ink-muted hover:text-ink-soft',
                    )}
                  >
                    {option / 60} min
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className={mode === 'upload' ? 'sm:col-span-2' : undefined}>
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
        </div>
      </Panel>

      {error && <ErrorNotice message={error} onDismiss={clearError} />}

      <motion.div variants={riseItem}>
        <Button
          variant="primary"
          size="lg"
          fullWidth
          loading={starting}
          disabled={mode === 'upload' && !file}
          onClick={launch}
          icon={<Play className="size-4" />}
        >
          {starting
            ? 'Opening stream'
            : mode === 'simulate'
              ? 'Begin session'
              : file
                ? 'Analyse recording'
                : 'Choose a recording first'}
        </Button>
      </motion.div>
    </motion.div>
  )
}
