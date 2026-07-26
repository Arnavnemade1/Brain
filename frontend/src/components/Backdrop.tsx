import { useEffect, useRef } from 'react'

import { useReducedMotion } from '@/hooks/useMotionPreference'
import { cn } from '@/lib/cn'

/**
 * The looping background.
 *
 * A generated dusk vista — layered ridges, drifting mist, a slow aurora —
 * rather than stock footage: the loop is seamless by construction, the
 * palette matches the design tokens exactly, and there is no licensing
 * question attached to the repository.
 *
 * `dimmed` is for the app routes, where dense readouts and tables have no
 * business competing with a mountain range.
 */
export function Backdrop({ dimmed = false }: { dimmed?: boolean }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const reduced = useReducedMotion()

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    if (reduced) {
      video.pause()
      // Hold a representative frame rather than a black rectangle.
      video.currentTime = 4
      return
    }

    // Autoplay can be refused; a still frame is an acceptable fallback and
    // far better than an unhandled rejection in the console.
    void video.play().catch(() => undefined)
  }, [reduced])

  return (
    <>
      <video
        ref={videoRef}
        className="backdrop-video"
        src="/ambient.mp4"
        autoPlay={!reduced}
        loop
        muted
        playsInline
        preload="auto"
        aria-hidden
        tabIndex={-1}
      />
      <div className={cn(dimmed ? 'backdrop-veil-dim' : 'backdrop-veil')} aria-hidden />
    </>
  )
}
