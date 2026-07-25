import { useEffect, useRef } from 'react'

interface TopographyProps {
  /** Electrode label → 0–1 activation. */
  topography: Record<string, number>
  positions: { label: string; x: number; y: number }[]
  size?: number
  className?: string
}

/** Cool → warm ramp for the activation surface. */
const RAMP: [number, [number, number, number]][] = [
  [0.0, [24, 30, 48]],
  [0.25, [40, 82, 140]],
  [0.5, [56, 193, 239]],
  [0.75, [232, 180, 100]],
  [1.0, [240, 120, 90]],
]

function sample(t: number): [number, number, number] {
  const clamped = Math.min(1, Math.max(0, t))
  for (let i = 0; i < RAMP.length - 1; i++) {
    const [t0, c0] = RAMP[i]
    const [t1, c1] = RAMP[i + 1]
    if (clamped >= t0 && clamped <= t1) {
      const k = (clamped - t0) / (t1 - t0)
      return [
        c0[0] + (c1[0] - c0[0]) * k,
        c0[1] + (c1[1] - c0[1]) * k,
        c0[2] + (c1[2] - c0[2]) * k,
      ]
    }
  }
  return RAMP[RAMP.length - 1][1]
}

/**
 * Scalp activation heatmap.
 *
 * Values are known only at ~19 electrodes, so the surface between them is
 * filled by inverse-distance weighting — the standard interpolation for
 * scattered EEG topography. Rendered per-pixel into an ImageData buffer;
 * at this size that is a few thousand samples and stays well inside frame
 * budget, while a mesh or blur-stack approach would misrepresent the falloff.
 */
export function Topography({ topography, positions, size = 190, className }: TopographyProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = Math.floor(size * dpr)
    canvas.height = Math.floor(size * dpr)
    ctx.setTransform(1, 0, 0, 1, 0, 0)
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    const known = positions
      .map((p) => ({ ...p, value: topography[p.label] }))
      .filter((p): p is typeof p & { value: number } => typeof p.value === 'number')

    const px = canvas.width
    const image = ctx.createImageData(px, px)
    const radius = px / 2

    for (let y = 0; y < px; y++) {
      for (let x = 0; x < px; x++) {
        const index = (y * px + x) * 4

        // Normalised head coordinates, +y toward the nasion.
        const nx = (x - radius) / radius
        const ny = -(y - radius) / radius
        const distance = Math.hypot(nx, ny)

        if (distance > 1) {
          image.data[index + 3] = 0
          continue
        }

        let numerator = 0
        let denominator = 0

        if (known.length > 0) {
          for (const electrode of known) {
            const d = Math.hypot(nx - electrode.x, ny - electrode.y)
            if (d < 1e-4) {
              numerator = electrode.value
              denominator = 1
              break
            }
            // Inverse *squared* distance: a plain inverse smears each
            // electrode's influence across the whole scalp.
            const weight = 1 / (d * d)
            numerator += electrode.value * weight
            denominator += weight
          }
        }

        const value = denominator > 0 ? numerator / denominator : 0
        const [r, g, b] = sample(value)

        image.data[index] = r
        image.data[index + 1] = g
        image.data[index + 2] = b
        // Soft edge so the disc does not read as a hard-cut circle.
        image.data[index + 3] = 255 * Math.min(1, (1 - distance) * 7)
      }
    }

    ctx.putImageData(image, 0, 0)

    // Electrode markers, drawn in CSS pixels over the buffer.
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    const center = size / 2
    const scale = size / 2

    ctx.strokeStyle = 'rgba(238, 242, 249, 0.22)'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.arc(center, center, scale - 0.5, 0, Math.PI * 2)
    ctx.stroke()

    // Nasion marker, so orientation is unambiguous.
    ctx.beginPath()
    ctx.moveTo(center - 5, 3)
    ctx.lineTo(center, -3)
    ctx.lineTo(center + 5, 3)
    ctx.stroke()

    for (const electrode of positions) {
      const cx = center + electrode.x * scale * 0.92
      const cy = center - electrode.y * scale * 0.92
      const value = topography[electrode.label]

      ctx.beginPath()
      ctx.arc(cx, cy, 2, 0, Math.PI * 2)
      ctx.fillStyle =
        typeof value === 'number' ? 'rgba(255,255,255,0.85)' : 'rgba(255,255,255,0.2)'
      ctx.fill()
    }
  }, [topography, positions, size])

  return (
    <canvas
      ref={canvasRef}
      style={{ width: size, height: size }}
      className={className}
      role="img"
      aria-label="Scalp activation topography"
    />
  )
}
