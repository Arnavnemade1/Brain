import { useEffect, useRef } from 'react'

interface WaveformTraceProps {
  waveform: number[]
  height?: number
  color?: string
  className?: string
}

/**
 * Scrolling montage-average waveform.
 *
 * Keeps its own rolling buffer across frames rather than redrawing each
 * window in isolation, so the trace reads as one continuous recording moving
 * past — which is what an EEG display looks like.
 */
export function WaveformTrace({
  waveform,
  height = 72,
  color = '#6fd8fb',
  className,
}: WaveformTraceProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const buffer = useRef<number[]>([])

  useEffect(() => {
    if (waveform.length === 0) return

    // Hold roughly four windows of history.
    const capacity = waveform.length * 4
    buffer.current = [...buffer.current, ...waveform].slice(-capacity)

    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    const rect = canvas.getBoundingClientRect()
    const w = rect.width
    const h = rect.height
    canvas.width = Math.floor(w * dpr)
    canvas.height = Math.floor(h * dpr)
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)

    const samples = buffer.current
    const mid = h / 2

    // Centre line
    ctx.strokeStyle = 'rgba(238, 242, 249, 0.07)'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(0, mid)
    ctx.lineTo(w, mid)
    ctx.stroke()

    ctx.beginPath()
    samples.forEach((value, i) => {
      const x = (i / Math.max(1, samples.length - 1)) * w
      const y = mid - value * (h * 0.42)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })

    // Older samples fade out toward the left edge.
    const gradient = ctx.createLinearGradient(0, 0, w, 0)
    gradient.addColorStop(0, 'rgba(111, 216, 251, 0.12)')
    gradient.addColorStop(0.55, 'rgba(111, 216, 251, 0.6)')
    gradient.addColorStop(1, color)

    ctx.strokeStyle = gradient
    ctx.lineWidth = 1.2
    ctx.stroke()

    // Leading edge marker
    const last = samples[samples.length - 1] ?? 0
    ctx.fillStyle = color
    ctx.beginPath()
    ctx.arc(w - 1, mid - last * (h * 0.42), 2, 0, Math.PI * 2)
    ctx.fill()
  }, [waveform, color])

  return (
    <canvas
      ref={canvasRef}
      style={{ height }}
      className={className ?? 'w-full'}
      role="img"
      aria-label="Live EEG waveform"
    />
  )
}
