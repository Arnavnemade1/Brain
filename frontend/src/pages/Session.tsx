import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { Eye, EyeOff, Gauge, Maximize2, Pause, Play, Square } from 'lucide-react'
import { lazy, Suspense, useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import { pageVariants, staggerContainer } from '@/animations/motion'
import { PipelineFlow } from '@/components/session/PipelineFlow'
import {
  FragmentsPanel,
  NarrativePanel,
  ReconstructionProgress,
  ScenePanel,
} from '@/components/session/ReconstructionPanel'
import { SessionLauncher } from '@/components/session/SessionLauncher'
import {
  CognitivePanel,
  EmotionPanel,
  SignalQualityPanel,
  SpectrumPanel,
  StatusStrip,
  TopographyPanel,
  WaveformPanel,
} from '@/components/session/Telemetry'
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts'
import { cn } from '@/lib/cn'
import { api } from '@/services/api'
import { selectResolvedCount, useSessionStore } from '@/stores/sessionStore'
import { useUiStore } from '@/stores/uiStore'
import { Badge, Button, ErrorBoundary, ErrorNotice, LoadingState, Panel } from '@/ui'

// The 3D engine is the heaviest thing in the app and is only needed once a
// scene exists, so it stays out of the initial session bundle.
const MemoryViewport = lazy(() =>
  import('@/three/MemoryViewport').then((m) => ({ default: m.MemoryViewport })),
)

const SPEEDS = [1, 2, 4, 8] as const

export default function Session() {
  const [params] = useSearchParams()
  const [immersive, setImmersive] = useState(false)

  const sessionId = useSessionStore((s) => s.sessionId)
  const connection = useSessionStore((s) => s.connection)
  const playing = useSessionStore((s) => s.playing)
  const speed = useSessionStore((s) => s.speed)
  const frame = useSessionStore((s) => s.frame)
  const frames = useSessionStore((s) => s.frames)
  const stages = useSessionStore((s) => s.stages)
  const status = useSessionStore((s) => s.status)
  const progress = useSessionStore((s) => s.progress)
  const confidence = useSessionStore((s) => s.confidence)
  const coherence = useSessionStore((s) => s.coherence)
  const scene = useSessionStore((s) => s.scene)
  const regionConfidence = useSessionStore((s) => s.regionConfidence)
  const fragments = useSessionStore((s) => s.fragments)
  const narrative = useSessionStore((s) => s.narrative)
  const openFragmentId = useSessionStore((s) => s.openFragmentId)
  const error = useSessionStore((s) => s.error)
  const meta = useSessionStore((s) => s.meta)

  const resolvedCount = useSessionStore(selectResolvedCount)

  const pause = useSessionStore((s) => s.pause)
  const resume = useSessionStore((s) => s.resume)
  const stop = useSessionStore((s) => s.stop)
  const setSpeed = useSessionStore((s) => s.setSpeed)
  const focusRegion = useSessionStore((s) => s.focusRegion)
  const openFragment = useSessionStore((s) => s.openFragment)
  const reset = useSessionStore((s) => s.reset)
  const clearError = useSessionStore((s) => s.clearError)

  const showOverlay = useUiStore((s) => s.showScientificOverlay)
  const toggleOverlay = useUiStore((s) => s.toggleScientificOverlay)

  const montage = useQuery({ queryKey: ['montage'], queryFn: api.montage, staleTime: Infinity })

  useEffect(() => {
    document.title = sessionId ? `Session ${sessionId} — MindScape` : 'Neural Session — MindScape'
  }, [sessionId])

  // Leaving the page must close the socket; an orphaned stream keeps the
  // backend session alive and counts against the concurrency cap.
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
        key: 'f',
        run: () => scene && setImmersive((value) => !value),
        description: 'Enter the reconstruction',
        group: 'Session',
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

  /* ---------------------------------------------------------------- launcher */

  if (!sessionId) {
    return (
      <motion.main
        variants={pageVariants}
        initial="initial"
        animate="animate"
        exit="exit"
        className="mx-auto w-[min(76rem,calc(100vw-2.5rem))] px-1 pt-28 pb-20"
      >
        <SessionLauncher
          initialMode={params.get('mode') === 'upload' ? 'upload' : 'simulate'}
        />
      </motion.main>
    )
  }

  /* --------------------------------------------------------------- immersive */

  if (immersive && scene) {
    return (
      <ErrorBoundary
        label="Reconstruction engine"
        fallback={(err, retry) => (
          <div className="grid min-h-dvh place-items-center p-6">
            <div className="max-w-md text-center">
              <p className="text-sm text-ink">The reconstruction could not be rendered.</p>
              <p className="mt-2 text-xs leading-relaxed text-ink-muted">{err.message}</p>
              <div className="mt-5 flex justify-center gap-2">
                <Button size="sm" onClick={retry}>
                  Try again
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setImmersive(false)}>
                  Back to console
                </Button>
              </div>
            </div>
          </div>
        )}
      >
        <Suspense
          fallback={
            <LoadingState
              title="Materialising reconstruction"
              detail="Building geometry from the committed scene parameters."
              className="min-h-dvh"
            />
          }
        >
          <MemoryViewport
            scene={scene}
            regionConfidence={regionConfidence}
            fragments={fragments}
            openFragmentId={openFragmentId}
            onOpenFragment={openFragment}
            onExit={() => setImmersive(false)}
          />
        </Suspense>
      </ErrorBoundary>
    )
  }

  /* ----------------------------------------------------------------- console */

  return (
    <motion.main
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      className="mx-auto w-[min(90rem,calc(100vw-2.5rem))] pt-24 pb-20"
    >
      {/* Transport bar */}
      <div className="glass-strong rounded-panel sticky top-20 z-30 mb-5 flex flex-wrap items-center gap-x-6 gap-y-3 px-5 py-3.5">
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant="secondary"
            onClick={() => (playing ? pause() : resume())}
            icon={playing ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
            disabled={status === 'stabilised'}
          >
            {playing ? 'Pause' : 'Resume'}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => void stop()}
            icon={<Square className="size-3.5" />}
            disabled={status === 'stabilised'}
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

        <StatusStrip frame={frame} className="flex-1" />

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

          <Button
            size="sm"
            variant="primary"
            disabled={!scene}
            onClick={() => setImmersive(true)}
            icon={<Maximize2 className="size-3.5" />}
          >
            Enter
          </Button>
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
        className="grid gap-5 lg:grid-cols-[20rem_1fr_22rem]"
      >
        {/* Pipeline */}
        <div className="space-y-5">
          <Panel label="AI pipeline" title="Processing status" flush>
            <div className="px-2 pb-3">
              <PipelineFlow stages={stages} />
            </div>
          </Panel>
        </div>

        {/* Centre */}
        <div className="space-y-5">
          <ReconstructionProgress
            status={status}
            progress={progress}
            confidence={confidence}
            coherence={coherence}
            resolvedCount={resolvedCount}
            totalRegions={scene?.regions.length ?? 0}
          />

          <WaveformPanel
            frame={frame}
            frames={frames}
            elapsed={frame?.t ?? 0}
            duration={meta?.duration ?? null}
          />

          {showOverlay && <SpectrumPanel frame={frame} />}

          <ScenePanel
            scene={scene}
            regionConfidence={regionConfidence}
            onFocusRegion={focusRegion}
          />

          <NarrativePanel narrative={narrative} />
        </div>

        {/* Telemetry */}
        <div className="space-y-5">
          <SignalQualityPanel frame={frame} />
          <CognitivePanel frame={frame} />
          <EmotionPanel frame={frame} />
          {showOverlay && (
            <TopographyPanel frame={frame} electrodes={montage.data ?? []} />
          )}
          <FragmentsPanel
            fragments={fragments}
            openFragmentId={openFragmentId}
            onOpen={openFragment}
          />
        </div>
      </motion.div>
    </motion.main>
  )
}
