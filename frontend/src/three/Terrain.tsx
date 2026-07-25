import { useMemo } from 'react'
import * as THREE from 'three'

import { mulberry32 } from '@/lib/math'
import type { SceneParameters } from '@/types'

import type { SceneTheme } from './useSceneTheme'

interface TerrainProps {
  scene: SceneParameters
  theme: SceneTheme
  /** 0–1 — low values flatten and sink detail. */
  confidence: number
  segments?: number
}

/**
 * Value noise with smooth interpolation.
 *
 * A dependency-free 2D noise sufficient for terrain relief: fully
 * deterministic from the scene seed, so a reconstruction rebuilds identically
 * every time it is opened.
 */
function makeNoise(seed: number) {
  const random = mulberry32(seed)
  const size = 256
  const table = new Float32Array(size * size)
  for (let i = 0; i < table.length; i++) table[i] = random()

  const at = (x: number, y: number) =>
    table[(((y % size) + size) % size) * size + (((x % size) + size) % size)]

  const smooth = (t: number) => t * t * (3 - 2 * t)

  return (x: number, y: number): number => {
    const xi = Math.floor(x)
    const yi = Math.floor(y)
    const xf = smooth(x - xi)
    const yf = smooth(y - yi)

    const a = at(xi, yi)
    const b = at(xi + 1, yi)
    const c = at(xi, yi + 1)
    const d = at(xi + 1, yi + 1)

    return (
      a * (1 - xf) * (1 - yf) + b * xf * (1 - yf) + c * (1 - xf) * yf + d * xf * yf
    )
  }
}

/**
 * The ground plane of the reconstruction.
 *
 * Relief is fractal noise scaled by the inferred spatial extent. Vertex
 * colours darken with distance from the origin, which is both an aesthetic
 * horizon falloff and an honest one — the model's confidence is highest where
 * the user starts and thins outward.
 */
export function Terrain({ scene, theme, confidence, segments = 156 }: TerrainProps) {
  const geometry = useMemo(() => {
    const size = theme.extent
    const geo = new THREE.PlaneGeometry(size, size, segments, segments)
    geo.rotateX(-Math.PI / 2)

    const noise = makeNoise(scene.seed)
    const positions = geo.attributes.position as THREE.BufferAttribute
    const colors = new Float32Array(positions.count * 3)

    const isWater = scene.biome === 'ocean'
    const isFloating = scene.biome === 'floating_islands' || scene.biome === 'abstract_space'
    const relief = isWater ? theme.relief * 0.08 : theme.relief

    const near = theme.ground.clone().lerp(theme.key, 0.16)
    const far = theme.ground.clone().lerp(theme.fog, 0.72)
    const scratch = new THREE.Color()

    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i)
      const z = positions.getZ(i)

      // Three octaves of fractal noise.
      const frequency = 0.012
      let height =
        noise(x * frequency, z * frequency) * 1.0 +
        noise(x * frequency * 2.3, z * frequency * 2.3) * 0.45 +
        noise(x * frequency * 5.1, z * frequency * 5.1) * 0.2
      height = (height / 1.65 - 0.5) * 2 * relief

      const distance = Math.hypot(x, z)
      const normalised = Math.min(1, distance / (size * 0.5))

      // Flatten a clearing where the user spawns so the first step is stable.
      const clearing = THREE.MathUtils.smoothstep(distance, 6, 26)
      height *= clearing

      // Low confidence sinks and breaks up the far field, so the world
      // visibly runs out rather than stretching on forever.
      const decay = 1 - normalised * (1 - confidence) * 1.35
      height *= Math.max(0.08, decay)

      if (isFloating) {
        // Terrain dissolves into voids beyond the centre.
        const island = noise(x * 0.006 + 40, z * 0.006 + 40)
        if (island < 0.42 && distance > 34) height -= 90 * (0.42 - island)
      }

      positions.setY(i, height)

      scratch.copy(near).lerp(far, Math.min(1, normalised * 1.15))
      colors[i * 3] = scratch.r
      colors[i * 3 + 1] = scratch.g
      colors[i * 3 + 2] = scratch.b
    }

    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    geo.computeVertexNormals()
    return geo
  }, [scene.seed, scene.biome, theme, confidence, segments])

  const isWater = scene.biome === 'ocean'

  return (
    <mesh geometry={geometry} receiveShadow position={[0, -1.2, 0]}>
      <meshStandardMaterial
        vertexColors
        roughness={isWater ? 0.16 : 0.95}
        metalness={isWater ? 0.5 : 0.02}
        flatShading={scene.distortion > 0.5}
      />
    </mesh>
  )
}
