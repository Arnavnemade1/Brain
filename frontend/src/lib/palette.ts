/**
 * Colour bindings shared by the UI and the 3D engine.
 *
 * Every emotion, band and cognitive state resolves to one hex value here so a
 * reading is the same colour in a chart, a badge and the reconstruction sky.
 */

import type { FrequencyBand, SignalQualityGrade } from '@/types'

export const BAND_COLOR: Record<FrequencyBand, string> = {
  delta: '#6247c7',
  theta: '#7f66e8',
  alpha: '#38c1ef',
  beta: '#5fd4c4',
  gamma: '#e8b464',
}

export const QUALITY_COLOR: Record<SignalQualityGrade, string> = {
  excellent: '#58d69d',
  good: '#58d69d',
  fair: '#e8c15f',
  poor: '#e0776b',
}

/** Confidence → colour ramp: fragmented (cool, dim) → resolved (warm, bright). */
export function confidenceColor(confidence: number): string {
  const c = Math.min(1, Math.max(0, confidence))
  if (c < 0.34) return '#4e576b'
  if (c < 0.55) return '#6b83c9'
  if (c < 0.72) return '#38c1ef'
  if (c < 0.88) return '#5fd4c4'
  return '#58d69d'
}

/** `rgba()` string from a hex colour and alpha, for canvas fills. */
export function withAlpha(hex: string, alpha: number): string {
  const h = hex.replace('#', '')
  const full =
    h.length === 3
      ? h
          .split('')
          .map((c) => c + c)
          .join('')
      : h
  const r = parseInt(full.slice(0, 2), 16)
  const g = parseInt(full.slice(2, 4), 16)
  const b = parseInt(full.slice(4, 6), 16)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

/** Linear hex blend, used for palette interpolation during scene morphs. */
export function mixHex(a: string, b: string, t: number): string {
  const parse = (hex: string): [number, number, number] => {
    const h = hex.replace('#', '')
    return [
      parseInt(h.slice(0, 2), 16),
      parseInt(h.slice(2, 4), 16),
      parseInt(h.slice(4, 6), 16),
    ]
  }
  const [r1, g1, b1] = parse(a)
  const [r2, g2, b2] = parse(b)
  const k = Math.min(1, Math.max(0, t))
  const to = (x: number) => Math.round(x).toString(16).padStart(2, '0')
  return `#${to(r1 + (r2 - r1) * k)}${to(g1 + (g2 - g1) * k)}${to(b1 + (b2 - b1) * k)}`
}
