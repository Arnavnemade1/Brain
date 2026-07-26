import { useEffect, useRef } from 'react'

import { useReducedMotion } from '@/hooks/useMotionPreference'

/**
 * The looping background.
 *
 * A generated 16-second light field rather than stock footage: the loop is
 * seamless by construction, the palette matches the design tokens exactly,
 * and there is no licensing question attached to the repository. It sits
 * behind everything at low opacity under a veil, so foreground type never
 * has to fight it.
 */
export function Backdrop() {
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
      <div className="backdrop-veil" aria-hidden />
    </>
  )
}
