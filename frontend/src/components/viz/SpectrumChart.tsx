import { useEffect, useRef } from 'react'

import { BAND_COLOR, withAlpha } from '@/lib/palette'
import { BAND_RANGES, FREQUENCY_BANDS } from '@/types'
import type { Spectrum } from '@/types'

interface SpectrumChartProps {
  spectrum: Spectrum | null
  height?: number
  className?: string
}

/**
 * Live power spectral density, banded by frequency.
 *
 * Canvas with a persistence trail: each frame the previous plot is faded
 * rather than cleared, so the eye reads the *envelope* of recent activity as
 * well as the current window. A hard clear at 2 Hz makes the plot strobe.
 */
export function SpectrumChart({ spectrum, height = 150, className }: SpectrumChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const previous = useRef<number[] | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !spectrum) return
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

    const { frequencies, magnitudes } = spectrum
    if (frequencies.length === 0) return

    const minHz = frequencies[0]
    const maxHz = frequencies[frequencies.length - 1]

    // Fixed dB window: an auto-scaling axis makes every window look identical
    // and hides the amplitude changes that matter.
    const minDb = -40
    const maxDb = 25

    const xFor = (hz: number) => ((hz - minHz) / (maxHz - minHz)) * w
    const yFor = (db: number) => h - ((db - minDb) / (maxDb - minDb)) * h

    // Band backgrounds
    for (const band of FREQUENCY_BANDS) {
      const [low, high] = BAND_RANGES[band]
      const x0 = xFor(Math.max(low, minHz))
      const x1 = xFor(Math.min(high, maxHz))
      ctx.fillStyle = withAlpha(BAND_COLOR[band], 0.07)
      ctx.fillRect(x0, 0, x1 - x0, h)

      ctx.fillStyle = withAlpha(BAND_COLOR[band], 0.55)
      ctx.font = '9px ui-monospace, monospace'
      ctx.textAlign = 'center'
      ctx.fillText(band, (x0 + x1) / 2, h - 5)
    }

    // Persistence trail from the previous frame
    if (previous.current) {
      ctx.beginPath()
      previous.current.forEach((db, i) => {
        const x = xFor(frequencies[i] ?? 0)
        const y = yFor(db)
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      })
      ctx.strokeStyle = 'rgba(169, 233, 255, 0.18)'
      ctx.lineWidth = 1
      ctx.stroke()
    }

    // Filled curve
    const gradient = ctx.createLinearGradient(0, 0, 0, h)
    gradient.addColorStop(0, 'rgba(56, 193, 239, 0.34)')
    gradient.addColorStop(1, 'rgba(56, 193, 239, 0.02)')

    ctx.beginPath()
    ctx.moveTo(xFor(frequencies[0]), h)
    magnitudes.forEach((db, i) => ctx.lineTo(xFor(frequencies[i]), yFor(db)))
    ctx.lineTo(xFor(frequencies[frequencies.length - 1]), h)
    ctx.closePath()
    ctx.fillStyle = gradient
    ctx.fill()

    ctx.beginPath()
    magnitudes.forEach((db, i) => {
      const x = xFor(frequencies[i])
      const y = yFor(db)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.strokeStyle = '#a9e9ff'
    ctx.lineWidth = 1.4
    ctx.stroke()

    // Dominant frequency marker
    const peakX = xFor(spectrum.peakFrequency)
    ctx.strokeStyle = 'rgba(226, 165, 76, 0.7)'
    ctx.lineWidth = 1
    ctx.setLineDash([2, 3])
    ctx.beginPath()
    ctx.moveTo(peakX, 0)
    ctx.lineTo(peakX, h - 12)
    ctx.stroke()
    ctx.setLineDash([])

    ctx.fillStyle = '#e2a54c'
    ctx.font = '9px ui-monospace, monospace'
    ctx.textAlign = peakX > w - 46 ? 'right' : 'left'
    ctx.fillText(
      `${spectrum.peakFrequency.toFixed(1)} Hz`,
      peakX + (peakX > w - 46 ? -4 : 4),
      11,
    )

    previous.current = magnitudes
  }, [spectrum])

  return (
    <canvas
      ref={canvasRef}
      style={{ height }}
      className={className ?? 'w-full'}
      role="img"
      aria-label={
        spectrum
          ? `Power spectrum, dominant frequency ${spectrum.peakFrequency.toFixed(1)} hertz`
          : 'Power spectrum, awaiting data'
      }
    />
  )
}
