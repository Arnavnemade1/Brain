/** Small numeric helpers shared by the visualisations and the 3D engine. */

export const clamp = (v: number, min = 0, max = 1): number => Math.min(max, Math.max(min, v))

export const lerp = (a: number, b: number, t: number): number => a + (b - a) * t

/** Frame-rate independent exponential approach toward `b`. */
export const damp = (a: number, b: number, lambda: number, dt: number): number =>
  lerp(a, b, 1 - Math.exp(-lambda * dt))

export const mapRange = (
  v: number,
  inMin: number,
  inMax: number,
  outMin: number,
  outMax: number,
): number => {
  if (inMax === inMin) return outMin
  return outMin + ((v - inMin) / (inMax - inMin)) * (outMax - outMin)
}

export const smoothstep = (edge0: number, edge1: number, x: number): number => {
  const t = clamp((x - edge0) / (edge1 - edge0))
  return t * t * (3 - 2 * t)
}

export const mean = (values: readonly number[]): number =>
  values.length === 0 ? 0 : values.reduce((a, b) => a + b, 0) / values.length

export const stdDev = (values: readonly number[]): number => {
  if (values.length < 2) return 0
  const m = mean(values)
  return Math.sqrt(mean(values.map((v) => (v - m) ** 2)))
}

/**
 * Deterministic 32-bit hash → seeded PRNG factory (mulberry32).
 * Keeps client-side procedural detail reproducible from a backend seed.
 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function hashString(input: string): number {
  let h = 2166136261
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

/** Pick from a weighted distribution using a 0–1 random draw. */
export function weightedPick<T extends string>(
  weights: Partial<Record<T, number>>,
  draw: number,
): T | null {
  const entries = Object.entries(weights) as [T, number][]
  const total = entries.reduce((sum, [, w]) => sum + w, 0)
  if (total <= 0) return null
  let acc = 0
  const target = draw * total
  for (const [key, w] of entries) {
    acc += w
    if (target <= acc) return key
  }
  return entries[entries.length - 1]?.[0] ?? null
}
