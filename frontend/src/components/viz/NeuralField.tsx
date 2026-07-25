import { useEffect, useRef } from 'react'

import { useReducedMotion } from '@/hooks/useMotionPreference'
import { mulberry32 } from '@/lib/math'

interface NeuralFieldProps {
  /** Node count. Scaled down automatically on small viewports. */
  density?: number
  className?: string
  /** 0–1 — how much the field pulses and drifts. */
  activity?: number
  color?: string
  accent?: string
}

interface Node {
  x: number
  y: number
  vx: number
  vy: number
  phase: number
  weight: number
}

/**
 * Ambient neural field for the hero.
 *
 * A slow constellation of drifting nodes with proximity-linked edges and
 * signal pulses travelling along them. Canvas rather than DOM or WebGL: a few
 * hundred nodes with per-frame edge testing is trivial for 2D canvas, and it
 * costs nothing on machines that would rather save the GPU for the actual
 * reconstruction later in the session.
 */
export function NeuralField({
  density = 90,
  className,
  activity = 0.5,
  color = '#38c1ef',
  accent = '#e2a54c',
}: NeuralFieldProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const reduced = useReducedMotion()

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d', { alpha: true })
    if (!ctx) return

    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    let width = 0
    let height = 0
    let nodes: Node[] = []
    let raf = 0
    let start = performance.now()

    const random = mulberry32(0x51de)

    function build() {
      const rect = canvas!.getBoundingClientRect()
      width = rect.width
      height = rect.height
      canvas!.width = Math.floor(width * dpr)
      canvas!.height = Math.floor(height * dpr)
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0)

      // Fewer nodes on small screens — edge testing is O(n²).
      const count = Math.round(density * Math.min(1, (width * height) / 900_000))

      nodes = Array.from({ length: Math.max(24, count) }, () => ({
        x: random() * width,
        y: random() * height,
        vx: (random() - 0.5) * 0.18,
        vy: (random() - 0.5) * 0.18,
        phase: random() * Math.PI * 2,
        weight: 0.35 + random() * 0.65,
      }))
    }

    const linkDistance = 132

    function frame(now: number) {
      const t = (now - start) / 1000
      ctx!.clearRect(0, 0, width, height)

      for (const node of nodes) {
        if (!reduced) {
          node.x += node.vx
          node.y += node.vy
        }
        // Wrap rather than bounce: a bounce reads as a wall, and the field
        // should feel unbounded.
        if (node.x < -20) node.x = width + 20
        if (node.x > width + 20) node.x = -20
        if (node.y < -20) node.y = height + 20
        if (node.y > height + 20) node.y = -20
      }

      // Edges
      for (let i = 0; i < nodes.length; i++) {
        const a = nodes[i]
        for (let j = i + 1; j < nodes.length; j++) {
          const b = nodes[j]
          const dx = a.x - b.x
          const dy = a.y - b.y
          const distanceSq = dx * dx + dy * dy
          if (distanceSq > linkDistance * linkDistance) continue

          const distance = Math.sqrt(distanceSq)
          const strength = 1 - distance / linkDistance

          ctx!.globalAlpha = strength * 0.16 * (0.5 + activity * 0.5)
          ctx!.strokeStyle = color
          ctx!.lineWidth = 0.6
          ctx!.beginPath()
          ctx!.moveTo(a.x, a.y)
          ctx!.lineTo(b.x, b.y)
          ctx!.stroke()

          // A pulse travelling the edge, at a rate set by the pair's phase.
          if (strength > 0.55 && !reduced) {
            const travel = ((t * 0.35 + a.phase) % 1 + 1) % 1
            ctx!.globalAlpha = strength * 0.5 * activity
            ctx!.fillStyle = accent
            ctx!.beginPath()
            ctx!.arc(a.x + -dx * travel, a.y + -dy * travel, 1.1, 0, Math.PI * 2)
            ctx!.fill()
          }
        }
      }

      // Nodes
      for (const node of nodes) {
        const pulse = reduced ? 0.6 : 0.55 + Math.sin(t * 1.1 + node.phase) * 0.45 * activity
        ctx!.globalAlpha = 0.25 + pulse * 0.5
        ctx!.fillStyle = color
        ctx!.beginPath()
        ctx!.arc(node.x, node.y, node.weight * 1.5, 0, Math.PI * 2)
        ctx!.fill()
      }

      ctx!.globalAlpha = 1
      raf = requestAnimationFrame(frame)
    }

    build()
    start = performance.now()
    raf = requestAnimationFrame(frame)

    const observer = new ResizeObserver(build)
    observer.observe(canvas)

    return () => {
      cancelAnimationFrame(raf)
      observer.disconnect()
    }
  }, [density, activity, color, accent, reduced])

  return <canvas ref={canvasRef} className={className} aria-hidden />
}
