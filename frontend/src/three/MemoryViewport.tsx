import { AdaptiveDpr, Preload } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { AnimatePresence, motion } from 'framer-motion'
import { LogOut, MousePointerClick } from 'lucide-react'
import { Suspense, useCallback, useEffect, useState } from 'react'

import { popVariants } from '@/animations/motion'
import { useReducedMotion } from '@/hooks/useMotionPreference'
import { cn } from '@/lib/cn'
import { confidenceColor, EMOTION_COLOR } from '@/lib/palette'
import { useUiStore } from '@/stores/uiStore'
import { BIOME_META } from '@/types'
import type { MemoryFragment, SceneParameters } from '@/types'
import { Badge, Button } from '@/ui'

import { Atmosphere, Lighting } from './Atmosphere'
import { FirstPersonControls } from './FirstPersonControls'
import { FragmentMarkers } from './FragmentMarkers'
import { Structures } from './Structures'
import { Terrain } from './Terrain'
import { useSceneTheme } from './useSceneTheme'

interface MemoryViewportProps {
  scene: SceneParameters
  regionConfidence: Record<string, number>
  fragments: MemoryFragment[]
  openFragmentId: string | null
  onOpenFragment: (id: string | null) => void
  onExit: () => void
}

/**
 * The explorable reconstruction.
 *
 * Everything visible here is derived from `scene` — nothing is authored. The
 * world is intentionally incomplete wherever confidence is low.
 */
export function MemoryViewport({
  scene,
  regionConfidence,
  fragments,
  openFragmentId,
  onOpenFragment,
  onExit,
}: MemoryViewportProps) {
  const quality = useUiStore((s) => s.renderQuality)
  const reduced = useReducedMotion()

  const [locked, setLocked] = useState(false)
  const [nearest, setNearest] = useState<MemoryFragment | null>(null)
  const revealedIds = useState<string[]>([])[0]

  const averageConfidence =
    scene.regions.length > 0
      ? scene.regions.reduce(
          (sum, region) => sum + (regionConfidence[region.id] ?? region.confidence),
          0,
        ) / scene.regions.length
      : 0.5

  const theme = useSceneTheme(scene, averageConfidence)
  const meta = BIOME_META[scene.biome]

  const openFragment = fragments.find((f) => f.id === openFragmentId) ?? null

  // `E` opens whatever is within reach; Escape steps back out.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key.toLowerCase() === 'e' && nearest && !openFragment) {
        event.preventDefault()
        onOpenFragment(nearest.id)
      }
      if (event.key === 'Escape' && openFragment) {
        onOpenFragment(null)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [nearest, openFragment, onOpenFragment])

  const handleLockChange = useCallback((value: boolean) => setLocked(value), [])

  const resolved = scene.regions.filter(
    (region) => (regionConfidence[region.id] ?? region.confidence) >= scene.fragmentThreshold,
  ).length

  return (
    <div className="fixed inset-0 z-40 bg-void">
      <Canvas
        shadows={quality !== 'low'}
        dpr={quality === 'high' ? [1, 2] : [1, 1.5]}
        camera={{ fov: 68, near: 0.1, far: 900, position: [0, 1.7, 14] }}
        gl={{
          antialias: quality !== 'low',
          powerPreference: 'high-performance',
        }}
      >
        <Suspense fallback={null}>
          <Atmosphere scene={scene} theme={theme} quality={quality} />
          <Lighting theme={theme} quality={quality} />
          <Terrain
            scene={scene}
            theme={theme}
            confidence={averageConfidence}
            segments={quality === 'low' ? 96 : quality === 'high' ? 200 : 156}
          />
          <Structures scene={scene} theme={theme} regionConfidence={regionConfidence} />
          <FragmentMarkers
            fragments={fragments}
            openFragmentId={openFragmentId}
            revealedIds={revealedIds}
            onOpen={onOpenFragment}
            onNearestChange={setNearest}
          />
          <FirstPersonControls enabled={!reduced} onLockChange={handleLockChange} />
          <Preload all />
        </Suspense>
        <AdaptiveDpr pixelated />
      </Canvas>

      {/* ------------------------------------------------------------ HUD */}

      <div className="pointer-events-none absolute inset-0">
        {/* Vignette grounds the frame and hides the far clip plane. */}
        <div
          className="absolute inset-0"
          style={{
            background:
              'radial-gradient(ellipse 78% 68% at 50% 50%, transparent 40%, rgba(4,6,12,0.55) 100%)',
          }}
        />

        {/* Top-left: what this place is */}
        <div className="absolute top-5 left-5 max-w-xs">
          <div className="glass-strong rounded-panel px-4 py-3.5">
            <p className="text-label mb-2">Reconstruction</p>
            <p className="text-[0.95rem] text-ink">{meta.label}</p>
            <p className="mt-1.5 text-[0.72rem] leading-relaxed text-ink-muted">
              {meta.tagline}
            </p>
            <div className="mt-3 flex flex-wrap gap-1.5">
              <Badge tone={confidenceColor(averageConfidence)}>
                {(averageConfidence * 100).toFixed(0)}% confidence
              </Badge>
              <Badge>
                {resolved}/{scene.regions.length} resolved
              </Badge>
            </div>
          </div>
        </div>

        {/* Top-right: exit */}
        <div className="pointer-events-auto absolute top-5 right-5">
          <Button size="sm" variant="secondary" onClick={onExit} icon={<LogOut className="size-3.5" />}>
            Back to console
          </Button>
        </div>

        {/* Centre: crosshair and interaction prompt */}
        {locked && (
          <>
            <div className="absolute top-1/2 left-1/2 size-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-ink/45" />
            <AnimatePresence>
              {nearest && !openFragment && (
                <motion.div
                  variants={popVariants}
                  initial="initial"
                  animate="animate"
                  exit="exit"
                  className="absolute top-[calc(50%+2.2rem)] left-1/2 -translate-x-1/2"
                >
                  <div className="glass-strong rounded-pill flex items-center gap-2 px-3.5 py-2">
                    <kbd className="text-readout rounded border border-ink/15 bg-ink/8 px-1.5 py-0.5 text-[0.62rem]">
                      E
                    </kbd>
                    <span className="text-[0.78rem] text-ink-soft">{nearest.title}</span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </>
        )}

        {/* Click-to-enter prompt */}
        {!locked && !reduced && (
          <div className="absolute inset-0 grid place-items-center">
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4, duration: 0.7 }}
              className="glass-strong rounded-panel px-7 py-6 text-center"
            >
              <MousePointerClick className="mx-auto mb-3 size-5 text-neural-300" />
              <p className="text-[0.92rem] text-ink">Click to look around</p>
              <p className="mt-2 text-[0.75rem] leading-relaxed text-ink-muted">
                <kbd className="text-readout">WASD</kbd> to move ·{' '}
                <kbd className="text-readout">Shift</kbd> to hurry ·{' '}
                <kbd className="text-readout">E</kbd> to open a fragment ·{' '}
                <kbd className="text-readout">Esc</kbd> to release the cursor
              </p>
            </motion.div>
          </div>
        )}

        {reduced && (
          <div className="absolute bottom-5 left-1/2 -translate-x-1/2">
            <div className="glass-strong rounded-pill px-4 py-2">
              <p className="text-[0.72rem] text-ink-muted">
                Movement is disabled while reduced motion is on. Fragments remain clickable.
              </p>
            </div>
          </div>
        )}

        {/* Bottom-left: atmospheric readout */}
        <div className="absolute bottom-5 left-5">
          <div className="glass rounded-card px-3.5 py-2.5">
            <dl className="flex gap-5">
              {[
                { label: 'Light', value: scene.lighting.timeOfDay },
                { label: 'Weather', value: scene.atmosphere.weather },
                { label: 'Distortion', value: `${(scene.distortion * 100).toFixed(0)}%` },
              ].map((item) => (
                <div key={item.label}>
                  <dt className="text-label !text-[0.52rem]">{item.label}</dt>
                  <dd className="mt-1 text-[0.72rem] capitalize text-ink-soft">{item.value}</dd>
                </div>
              ))}
            </dl>
          </div>
        </div>
      </div>

      {/* Fragment detail */}
      <AnimatePresence>
        {openFragment && (
          <motion.aside
            variants={popVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            className={cn(
              'glass-strong rounded-panel absolute top-1/2 right-5 z-10 w-[min(24rem,calc(100vw-2.5rem))]',
              '-translate-y-1/2 p-6',
            )}
          >
            <div className="mb-4 flex items-start justify-between gap-3">
              <div>
                <p className="text-label mb-2">Memory fragment</p>
                <h3 className="text-[1.05rem]">{openFragment.title}</h3>
              </div>
              <Badge tone={EMOTION_COLOR[openFragment.emotion]}>
                {(openFragment.confidence * 100).toFixed(0)}%
              </Badge>
            </div>

            <p className="text-[0.82rem] leading-relaxed text-ink-soft">
              {openFragment.description}
            </p>

            <dl className="mt-5 space-y-4 border-t border-ink/8 pt-4">
              <div>
                <dt className="text-label mb-1.5">Estimated emotion</dt>
                <dd
                  className="text-[0.8rem] capitalize"
                  style={{ color: EMOTION_COLOR[openFragment.emotion] }}
                >
                  {openFragment.emotion}
                </dd>
              </div>

              <div>
                <dt className="text-label mb-1.5">Likely sensory detail</dt>
                <dd className="flex flex-wrap gap-1.5">
                  {openFragment.sensoryDetails.map((detail) => (
                    <span
                      key={detail}
                      className="rounded border border-ink/8 px-1.5 py-0.5 text-[0.68rem] text-ink-muted"
                    >
                      {detail}
                    </span>
                  ))}
                </dd>
              </div>

              <div>
                <dt className="text-label mb-1.5">Timeline position</dt>
                <dd className="text-readout text-[0.75rem] text-ink-soft">
                  t+{openFragment.sourceTime.toFixed(0)}s ·{' '}
                  {(openFragment.timelinePosition * 100).toFixed(0)}% through the session
                </dd>
              </div>

              <div>
                <dt className="text-label mb-1.5">Neural evidence</dt>
                <dd className="text-readout text-[0.7rem] leading-relaxed text-ink-faint">
                  {openFragment.neuralEvidence}
                </dd>
              </div>

              {openFragment.relatedIds.length > 0 && (
                <div>
                  <dt className="text-label mb-1.5">Related fragments</dt>
                  <dd className="flex flex-wrap gap-1.5">
                    {openFragment.relatedIds.map((id) => {
                      const related = fragments.find((f) => f.id === id)
                      if (!related) return null
                      return (
                        <button
                          key={id}
                          onClick={() => onOpenFragment(id)}
                          className="rounded border border-ink/10 px-2 py-1 text-[0.68rem] text-ink-muted transition-colors hover:border-ink/20 hover:text-ink"
                        >
                          {related.title}
                        </button>
                      )
                    })}
                  </dd>
                </div>
              )}
            </dl>

            {openFragment.unlocksRegion && (
              <p className="mt-5 rounded-lg border border-memory-400/25 bg-memory-400/8 px-3 py-2.5 text-[0.72rem] leading-relaxed text-memory-200">
                Opening this fragment directed additional evidence at a region that had not
                resolved. Look for structure appearing where there was none.
              </p>
            )}

            <Button
              size="sm"
              variant="ghost"
              fullWidth
              className="mt-5"
              onClick={() => onOpenFragment(null)}
            >
              Close
            </Button>
          </motion.aside>
        )}
      </AnimatePresence>
    </div>
  )
}
