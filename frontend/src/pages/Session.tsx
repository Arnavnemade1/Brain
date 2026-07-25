import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { Eye, EyeOff, Gauge, Pause, Play, Square } from 'lucide-react'
import { useEffect } from 'react'

import { pageVariants, staggerContainer } from '@/animations/motion'
import { FidelityPanel, ReportPanel } from '@/components/replay/FidelityPanel'
import { MemoryReplay } from '@/components/replay/MemoryReplay'
import { PipelineFlow } from '@/components/session/PipelineFlow'
import { SessionLauncher } from '@/components/session/SessionLauncher'
import {
  LatentPanel,
  SignalQualityPanel,
  SpectrumPanel,
  StatusStrip,
  SyncPanel,
  TopographyPanel,
  VisualResponsePanel,
  WaveformPanel,
} from '@/components/session/Telemetry'
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts'
import { cn } from '@/lib/cn'
import { api } from '@/services/api'
import { selectIsLive, useSessionStore } from '@/stores/sessionStore'
import { useUiStore } from '@/stores/uiStore'
import { Badge, Button, ErrorNotice, Panel } from '@/ui'

const SPEEDS = [1, 2, 4, 8] as const

export default function Session() {
  const sessionId = useSessionStore((s) => s.sessionId)
  const connection = useSessionStore((s) => s.connection)
  const playing = useSessionStore((s) => s.playing)
  const speed = useSessionStore((s) => s.speed)
  const frame = useSessionStore((s) => s.frame)
  const stages = useSessionStore((s) => s.stages)
  const latents = useSessionStore((s) => s.latents)
  const sync = useSessionStore((s) => s.sync)
  const status = useSessionStore((s) => s.status)
  const fidelity = useSessionStore((s) => s.fidelity)
  const replay = useSessionStore((s) => s.replay)
  const playhead = useSessionStore((s) => s.playhead)
  const following = useSessionStore((s) => s.following)
  const metrics = useSessionStore((s) => s.metrics)
  const report = useSessionStore((s) => s.report)
  const error = useSessionStore((s) => s.error)
  const meta = useSessionStore((s) => s.meta)
  const live = useSessionStore(selectIsLive)

  const pause = useSessionStore((s) => s.pause)
  const resume = useSessionStore((s) => s.resume)
  const stop = useSessionStore((s) => s.stop)
  const setSpeed = useSessionStore((s) => s.setSpeed)
  const setPlayhead = useSessionStore((s) => s.setPlayhead)
  const setFollowing = useSessionStore((s) => s.setFollowing)
  const reset = useSessionStore((s) => s.reset)
  const clearError = useSessionStore((s) => s.clearError)

  const showOverlay = useUiStore((s) => s.showScientificOverlay)
  const toggleOverlay = useUiStore((s) => s.toggleScientificOverlay)

  const montage = useQuery({
    queryKey: ['montage'],
    queryFn: api.montage,
    staleTime: Infinity,
  })

  useEffect(() => {
    document.title = sessionId
      ? `Reconstruction ${sessionId} — MindScape`
      : 'Memory Reconstruction — MindScape'
  }, [sessionId])

  // Leaving the page must close the socket; an orphaned stream keeps the
  // backend session alive against the concurrency cap.
  useEffect(() => () => reset(), [reset])

  useKeyboardShortcuts(
    [
      {
        key: ' ',
        run: () => (playing ? pause() : resume()),
        description: 'Pause or resume',
        group: 'Session',
      },
      {
        key: 'v',
        run: toggleOverlay,
        description: 'Toggle scientific overlay',
        group: 'Session',
      },
      {
        key: 'arrowleft',
        run: () => setPlayhead(playhead - 1),
        description: 'Previous frame',
        group: 'Replay',
      },
      {
        key: 'arrowright',
        run: () => setPlayhead(playhead + 1),
        description: 'Next frame',
        group: 'Replay',
      },
      ...SPEEDS.map((option, index) => ({
        key: String(index + 1),
        run: () => setSpeed(option),
        description: `Playback ${option}×`,
        group: 'Session',
      })),
    ],
    Boolean(sessionId),
  )

  if (!sessionId) {
    return (
      <motion.main
        variants={pageVariants}
        initial="initial"
        animate="animate"
        exit="exit"
        className="mx-auto w-[min(76rem,calc(100vw-2.5rem))] px-1 pt-28 pb-20"
      >
        <SessionLauncher />
      </motion.main>
    )
  }

  return (
    <motion.main
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      className="mx-auto w-[min(92rem,calc(100vw-2.5rem))] pt-24 pb-20"
    >
      {/* Transport */}
      <div className="glass-strong rounded-panel sticky top-20 z-30 mb-5 flex flex-wrap items-center gap-x-6 gap-y-3 px-5 py-3.5">
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant="secondary"
            onClick={() => (playing ? pause() : resume())}
            icon={playing ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
            disabled={status === 'complete'}
          >
            {playing ? 'Pause' : 'Resume'}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => void stop()}
            icon={<Square className="size-3.5" />}
            disabled={status === 'complete'}
          >
            Stop
          </Button>
        </div>

        <div className="flex items-center gap-1.5">
          <Gauge className="size-3.5 text-ink-faint" />
          {SPEEDS.map((option) => (
            <button
              key={option}
              onClick={() => setSpeed(option)}
              aria-pressed={speed === option}
              className={cn(
                'rounded-md px-2 py-1 text-[0.7rem] transition-colors',
                speed === option
                  ? 'bg-neural-400/15 text-neural-200'
                  : 'text-ink-faint hover:text-ink-soft',
              )}
            >
              {option}×
            </button>
          ))}
        </div>

        <StatusStrip frame={frame} fidelity={fidelity} className="flex-1" />

        <div className="flex items-center gap-2">
          <Badge
            tone={
              connection === 'open'
                ? 'var(--color-signal-good)'
                : connection === 'reconnecting'
                  ? 'var(--color-signal-fair)'
                  : 'var(--color-signal-poor)'
            }
            live={connection === 'open' && playing}
          >
            {connection}
          </Badge>

          <button
            onClick={toggleOverlay}
            aria-label="Toggle scientific overlay"
            title="Toggle scientific overlay (V)"
            className="grid size-8 place-items-center rounded-lg text-ink-muted transition-colors hover:bg-ink/8 hover:text-ink"
          >
            {showOverlay ? <Eye className="size-4" /> : <EyeOff className="size-4" />}
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-5">
          <ErrorNotice message={error} onDismiss={clearError} />
        </div>
      )}

      <motion.div
        variants={staggerContainer}
        initial="initial"
        animate="animate"
        className="grid gap-5 xl:grid-cols-[19rem_1fr_21rem]"
      >
        {/* Pipeline */}
        <div className="space-y-5">
          <Panel label="Reconstruction engine" title="Pipeline" flush>
            <div className="px-2 pb-3">
              <PipelineFlow stages={stages} />
            </div>
          </Panel>
          <SyncPanel sync={sync} />
        </div>

        {/* Replay + fidelity */}
        <div className="space-y-5">
          <MemoryReplay
            clip={meta?.clip ?? null}
            replay={replay}
            playhead={playhead}
            following={following}
            onScrub={setPlayhead}
            onFollowingChange={setFollowing}
            live={live}
          />
          <FidelityPanel metrics={metrics} live={live} />
          <ReportPanel report={report} />
          {showOverlay && <LatentPanel latents={latents} />}
        </div>

        {/* Telemetry */}
        <div className="space-y-5">
          <VisualResponsePanel frame={frame} />
          <SignalQualityPanel frame={frame} />
          <WaveformPanel
            frame={frame}
            elapsed={frame?.t ?? 0}
            duration={meta?.clip.durationSeconds ?? null}
          />
          {showOverlay && <SpectrumPanel frame={frame} />}
          {showOverlay && (
            <TopographyPanel frame={frame} electrodes={montage.data ?? []} />
          )}
        </div>
      </motion.div>
    </motion.main>
  )
}
