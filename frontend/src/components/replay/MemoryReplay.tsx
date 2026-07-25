import { motion } from 'framer-motion'
import { Eye, Pause, Play } from 'lucide-react'
import { useEffect, useState } from 'react'

import { cn } from '@/lib/cn'
import { clock } from '@/lib/format'
import { confidenceColor } from '@/lib/palette'
import type { ReplayFrame } from '@/stores/sessionStore'
import type { StimulusClip } from '@/types'
import { Badge, Panel } from '@/ui'

import { FrameCanvas } from './FrameCanvas'

interface MemoryReplayProps {
  clip: StimulusClip | null
  replay: ReplayFrame[]
  playhead: number
  following: boolean
  onScrub: (index: number) => void
  onFollowingChange: (following: boolean) => void
  live: boolean
}

type ViewMode = 'side-by-side' | 'overlay' | 'reconstruction'

const MODES: { id: ViewMode; label: string }[] = [
  { id: 'side-by-side', label: 'Compare' },
  { id: 'overlay', label: 'Difference' },
  { id: 'reconstruction', label: 'Reconstruction' },
]

/**
 * The Memory Replay surface.
 *
 * Reconstruction and reference are always presented together. Showing the
 * reconstruction alone would invite the viewer to judge it as an image; shown
 * beside its source, with the fidelity numbers, it reads as what it is — a
 * measurement of how much survived.
 */
export function MemoryReplay({
  clip,
  replay,
  playhead,
  following,
  onScrub,
  onFollowingChange,
  live,
}: MemoryReplayProps) {
  const [mode, setMode] = useState<ViewMode>('side-by-side')
  const [autoplay, setAutoplay] = useState(false)

  const current = replay[playhead] ?? null
  const width = clip?.gridWidth ?? 32
  const height = clip?.gridHeight ?? 18

  // Local playback when the stream has finished and the user wants to watch
  // the reconstruction back at its own rate.
  useEffect(() => {
    if (!autoplay || live || replay.length === 0) return
    const timer = window.setInterval(() => {
      const next = playhead + 1
      if (next >= replay.length) {
        setAutoplay(false)
        return
      }
      onScrub(next)
    }, 120)
    return () => window.clearInterval(timer)
  }, [autoplay, live, playhead, replay.length, onScrub])

  return (
    <Panel
      label="Memory replay"
      title={clip ? clip.title : 'Awaiting stimulus'}
      action={
        <div className="flex items-center gap-2">
          {current && (
            <Badge tone={confidenceColor(Math.max(0, current.fidelity.ssim))}>
              SSIM {current.fidelity.ssim.toFixed(3)}
            </Badge>
          )}
          {live && (
            <Badge tone="var(--color-signal-good)" live>
              live
            </Badge>
          )}
        </div>
      }
      flush
    >
      {/* Mode switch */}
      <div className="flex items-center justify-between gap-3 px-5 pb-3">
        <div className="glass-faint rounded-pill flex gap-1 p-1">
          {MODES.map((option) => (
            <button
              key={option.id}
              onClick={() => setMode(option.id)}
              aria-pressed={mode === option.id}
              className={cn(
                'relative rounded-pill px-3 py-1.5 text-[0.72rem] transition-colors',
                mode === option.id ? 'text-ink' : 'text-ink-muted hover:text-ink-soft',
              )}
            >
              {mode === option.id && (
                <motion.span
                  layoutId="replay-mode"
                  className="absolute inset-0 rounded-pill bg-ink/10"
                  transition={{ type: 'spring', stiffness: 340, damping: 30 }}
                />
              )}
              <span className="relative">{option.label}</span>
            </button>
          ))}
        </div>

        <p className="text-readout text-[0.68rem] text-ink-faint">
          {width}×{height} native
        </p>
      </div>

      {/* Frames */}
      <div className="px-5">
        {mode === 'side-by-side' ? (
          <div className="grid grid-cols-2 gap-3">
            <FramePane
              label="Reference"
              sublabel="what was watched"
              pixels={current?.reference ?? null}
              width={width}
              height={height}
              tone="var(--color-neural-300)"
            />
            <FramePane
              label="Reconstruction"
              sublabel="recovered from EEG"
              pixels={current?.pixels ?? null}
              width={width}
              height={height}
              tone="var(--color-memory-300)"
            />
          </div>
        ) : mode === 'overlay' ? (
          <DifferencePane
            reference={current?.reference ?? null}
            reconstruction={current?.pixels ?? null}
            width={width}
            height={height}
          />
        ) : (
          <FramePane
            label="Reconstruction"
            sublabel="recovered from EEG"
            pixels={current?.pixels ?? null}
            width={width}
            height={height}
            tone="var(--color-memory-300)"
          />
        )}
      </div>

      {/* Transport */}
      <div className="px-5 pt-4 pb-5">
        <div className="mb-3 flex items-center gap-3">
          <button
            onClick={() => setAutoplay((value) => !value)}
            disabled={live || replay.length === 0}
            aria-label={autoplay ? 'Pause replay' : 'Play replay'}
            className="grid size-8 shrink-0 place-items-center rounded-lg text-ink-muted transition-colors hover:bg-ink/8 hover:text-ink disabled:opacity-35"
          >
            {autoplay ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
          </button>

          <input
            type="range"
            min={0}
            max={Math.max(0, replay.length - 1)}
            value={playhead}
            onChange={(event) => onScrub(Number(event.target.value))}
            disabled={replay.length === 0}
            aria-label="Replay position"
            className="min-w-0 flex-1 accent-neural-400"
          />

          <span className="text-readout w-20 shrink-0 text-right text-[0.7rem] text-ink-soft">
            {current ? clock(current.t) : '0:00'}
            <span className="text-ink-faint">
              {' / '}
              {clock(clip?.durationSeconds ?? 0)}
            </span>
          </span>

          <button
            onClick={() => onFollowingChange(!following)}
            aria-pressed={following}
            title="Follow the newest frame"
            className={cn(
              'grid size-8 shrink-0 place-items-center rounded-lg transition-colors',
              following
                ? 'bg-neural-400/15 text-neural-200'
                : 'text-ink-faint hover:bg-ink/8 hover:text-ink',
            )}
          >
            <Eye className="size-3.5" />
          </button>
        </div>

        {/* Per-frame fidelity strip — where the reconstruction did well. */}
        <FidelityStrip replay={replay} playhead={playhead} onScrub={onScrub} />

        {current && (
          <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-4">
            <Comparison
              label="Brightness"
              estimate={current.scene.luminance}
              truth={current.truth.luminance}
            />
            <Comparison
              label="Motion"
              estimate={current.scene.motion}
              truth={current.truth.motion}
            />
            <Comparison
              label="Contrast"
              estimate={current.scene.contrast}
              truth={current.truth.contrast}
            />
            <div>
              <dt className="text-label mb-1.5">Scene</dt>
              <dd className="text-readout text-[0.78rem] text-ink-soft">
                {current.scene.sceneIndex + 1}
                <span className="text-ink-faint">
                  {' '}
                  / actual {current.truth.sceneIndex + 1}
                </span>
              </dd>
            </div>
          </dl>
        )}
      </div>
    </Panel>
  )
}

/* -------------------------------------------------------------------------- */

function FramePane({
  label,
  sublabel,
  pixels,
  width,
  height,
  tone,
}: {
  label: string
  sublabel: string
  pixels: string | null
  width: number
  height: number
  tone: string
}) {
  return (
    <figure>
      <div
        className="rounded-card overflow-hidden border"
        style={{ borderColor: `color-mix(in oklab, ${tone} 28%, transparent)` }}
      >
        <div className="aspect-video bg-void">
          <FrameCanvas pixels={pixels} width={width} height={height} />
        </div>
      </div>
      <figcaption className="mt-2 flex items-baseline justify-between gap-2">
        <span className="text-[0.75rem]" style={{ color: tone }}>
          {label}
        </span>
        <span className="text-[0.65rem] text-ink-faint">{sublabel}</span>
      </figcaption>
    </figure>
  )
}

/**
 * Absolute per-pixel difference between reference and reconstruction.
 *
 * Computed in the browser rather than sent from the server: both frames are
 * already here, and shipping a third image per frame would add 50% to the
 * stream for something derivable.
 */
function DifferencePane({
  reference,
  reconstruction,
  width,
  height,
}: {
  reference: string | null
  reconstruction: string | null
  width: number
  height: number
}) {
  const [difference, setDifference] = useState<string | null>(null)

  useEffect(() => {
    if (!reference || !reconstruction) {
      setDifference(null)
      return
    }
    try {
      const a = atob(reference)
      const b = atob(reconstruction)
      const length = Math.min(a.length, b.length)
      const out = new Uint8Array(length)
      for (let i = 0; i < length; i++) {
        out[i] = Math.abs(a.charCodeAt(i) - b.charCodeAt(i))
      }
      let binary = ''
      for (let i = 0; i < out.length; i++) binary += String.fromCharCode(out[i])
      setDifference(btoa(binary))
    } catch {
      setDifference(null)
    }
  }, [reference, reconstruction])

  return (
    <figure>
      <div className="rounded-card overflow-hidden border border-signal-poor/25">
        <div className="aspect-video bg-void">
          <FrameCanvas pixels={difference} width={width} height={height} />
        </div>
      </div>
      <figcaption className="mt-2 text-[0.68rem] leading-relaxed text-ink-faint">
        Absolute per-pixel error. Dark means agreement; bright regions are where
        the reconstruction departs from what was actually shown.
      </figcaption>
    </figure>
  )
}

function FidelityStrip({
  replay,
  playhead,
  onScrub,
}: {
  replay: ReplayFrame[]
  playhead: number
  onScrub: (index: number) => void
}) {
  if (replay.length === 0) {
    return <div className="h-6 rounded bg-ink/4" aria-hidden />
  }

  return (
    <div
      className="flex h-6 gap-px overflow-hidden rounded"
      role="group"
      aria-label="Per-frame structural similarity"
    >
      {replay.map((frame, index) => {
        const value = Math.max(0, Math.min(1, frame.fidelity.ssim))
        return (
          <button
            key={frame.index}
            onClick={() => onScrub(index)}
            title={`t+${frame.t.toFixed(1)}s · SSIM ${frame.fidelity.ssim.toFixed(3)}`}
            aria-label={`Frame at ${frame.t.toFixed(1)} seconds`}
            className={cn(
              'min-w-0 flex-1 transition-opacity',
              index === playhead ? 'opacity-100' : 'opacity-55 hover:opacity-85',
            )}
            style={{
              background: confidenceColor(value),
              outline: index === playhead ? '1px solid var(--color-ink)' : undefined,
            }}
          />
        )
      })}
    </div>
  )
}

function Comparison({
  label,
  estimate,
  truth,
}: {
  label: string
  estimate: number
  truth: number
}) {
  const error = Math.abs(estimate - truth)
  const tone = error < 0.1 ? 'var(--color-signal-good)' : error < 0.25 ? 'var(--color-signal-fair)' : 'var(--color-signal-poor)'

  return (
    <div>
      <dt className="text-label mb-1.5">{label}</dt>
      <dd className="text-readout text-[0.78rem]">
        <span style={{ color: tone }}>{estimate.toFixed(2)}</span>
        <span className="text-ink-faint"> / actual {truth.toFixed(2)}</span>
      </dd>
    </div>
  )
}
