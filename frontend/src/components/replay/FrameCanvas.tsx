import { useEffect, useRef } from 'react'

import { cn } from '@/lib/cn'

interface FrameCanvasProps {
  /** Base64-encoded RGB bytes at grid resolution. */
  pixels: string | null
  width: number
  height: number
  className?: string
  /** Render without smoothing, so the true resolution is visible. */
  pixelated?: boolean
}

function decodeBase64(encoded: string): Uint8ClampedArray {
  const binary = atob(encoded)
  const bytes = new Uint8ClampedArray(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return bytes
}

/**
 * Draws one reconstruction or reference frame.
 *
 * The canvas backing store stays at the true grid size and CSS scales it up,
 * with `imageSmoothingEnabled` off by default. That is a deliberate honesty
 * choice: interpolating to a crisp-looking image would imply spatial detail
 * the reconstruction does not have, and the fidelity numbers alongside would
 * then look inexplicably low.
 */
export function FrameCanvas({
  pixels,
  width,
  height,
  className,
  pixelated = true,
}: FrameCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = width
    canvas.height = height
    ctx.imageSmoothingEnabled = !pixelated

    if (!pixels) {
      ctx.fillStyle = '#080b14'
      ctx.fillRect(0, 0, width, height)
      return
    }

    let rgb: Uint8ClampedArray
    try {
      rgb = decodeBase64(pixels)
    } catch {
      ctx.fillStyle = '#080b14'
      ctx.fillRect(0, 0, width, height)
      return
    }

    const expected = width * height * 3
    if (rgb.length < expected) {
      ctx.fillStyle = '#080b14'
      ctx.fillRect(0, 0, width, height)
      return
    }

    // RGB on the wire, RGBA in the canvas.
    const image = ctx.createImageData(width, height)
    for (let i = 0, j = 0; i < expected; i += 3, j += 4) {
      image.data[j] = rgb[i]
      image.data[j + 1] = rgb[i + 1]
      image.data[j + 2] = rgb[i + 2]
      image.data[j + 3] = 255
    }
    ctx.putImageData(image, 0, 0)
  }, [pixels, width, height, pixelated])

  return (
    <canvas
      ref={canvasRef}
      className={cn('block size-full', className)}
      style={{ imageRendering: pixelated ? 'pixelated' : 'auto' }}
    />
  )
}
