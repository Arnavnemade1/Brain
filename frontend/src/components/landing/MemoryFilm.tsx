import { Pause, Play } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import { cn } from '@/lib/cn'

/**
 * The memory video, presented as the centrepiece.
 *
 * Plays muted and looping so it reads as a live demonstration rather than
 * something the visitor has to opt into, with a single control for people who
 * want it still. The caption underneath states what the video is and is not,
 * because the picture alone would over-claim.
 */
export function MemoryFilm() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [playing, setPlaying] = useState(true)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    void video.play().catch(() => setPlaying(false))
  }, [])

  const toggle = () => {
    const video = videoRef.current
    if (!video) return
    if (video.paused) {
      void video.play()
      setPlaying(true)
    } else {
      video.pause()
      setPlaying(false)
    }
  }

  return (
    <figure className="m-0">
      <div className="glass rounded-panel overflow-hidden p-2">
        <div className="overflow-hidden rounded-[0.55rem] bg-void">
          <video
            ref={videoRef}
            className={cn(
              'block aspect-[1120/620] w-full transition-opacity duration-700',
              ready ? 'opacity-100' : 'opacity-0',
            )}
            src="/memory-video.mp4"
            poster="/memory-video-poster.png"
            loop
            muted
            playsInline
            preload="metadata"
            onLoadedData={() => setReady(true)}
            aria-label="Gameplay reconstructed from EEG, side by side with what actually happened"
          />
        </div>

        {/* The control sits below the frame rather than floating over it. The
            render uses every corner — titles, clock, legend, caption — so an
            overlaid button covered something worth reading no matter where it
            was placed. */}
        <div className="flex items-center gap-3 px-2 pt-3 pb-1">
          <button
            onClick={toggle}
            aria-label={playing ? 'Pause the memory video' : 'Play the memory video'}
            className="glass grid size-8 shrink-0 place-items-center rounded-full text-ink transition-colors hover:border-ink/20"
          >
            {playing ? <Pause className="size-3" /> : <Play className="size-3" />}
          </button>
          <p className="text-label">sub-005 · 60 s continuous · loops</p>
        </div>
      </div>

      <figcaption className="mx-auto mt-6 grid max-w-3xl gap-x-10 gap-y-3 text-center sm:grid-cols-3">
        {[
          ['Detection', '76.6% recall · 72.6% precision'],
          ['Decoding rate', 'every 250 ms, no event log'],
          ['Validation', 'leave-one-subject-out'],
        ].map(([label, value]) => (
          <div key={label}>
            <p className="text-label mb-1.5">{label}</p>
            <p className="text-[0.8rem] text-ink-soft">{value}</p>
          </div>
        ))}
      </figcaption>

      <p className="mx-auto mt-7 max-w-2xl text-center text-[0.78rem] leading-relaxed text-ink-faint">
        The rhythm of the session — when things happened, the bursts and the lulls — is
        genuinely recovered. Which particular event occurred at each moment is not, and
        the right panel dims to show it.
      </p>
    </figure>
  )
}
