import { useFrame } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'

import { clamp, lerp, mulberry32 } from '@/lib/math'
import type { RegionParams, SceneParameters } from '@/types'

import type { SceneTheme } from './useSceneTheme'

interface StructuresProps {
  scene: SceneParameters
  theme: SceneTheme
  regionConfidence: Record<string, number>
}

interface Piece {
  position: [number, number, number]
  scale: [number, number, number]
  rotation: number
  /** 0–1 — how completely this piece is rendered. */
  solidity: number
  /** Pieces above the region's confidence budget drift out of place. */
  drift: number
}

/** Per-archetype form vocabulary. */
function buildRegion(region: RegionParams, scene: SceneParameters, confidence: number): Piece[] {
  const random = mulberry32(scene.seed ^ region.id.length * 7919)
  const pieces: Piece[] = []

  const [cx, cy, cz] = region.position
  const radius = region.radius

  // A confident region gets more, larger, better-formed pieces. This is the
  // core rule of the whole engine: detail is a function of evidence.
  const density = Math.round(lerp(3, 16, confidence) * (0.5 + scene.objectDensity))
  const count = Math.max(2, Math.min(26, density))

  for (let i = 0; i < count; i++) {
    const angle = random() * Math.PI * 2
    const distance = Math.sqrt(random()) * radius

    // Pieces further from the region centre are less well supported.
    const falloff = 1 - (distance / Math.max(radius, 1)) * 0.55
    const solidity = clamp(confidence * falloff * (0.6 + random() * 0.7))

    let width = 1.5 + random() * 4
    let height = 2 + random() * 8
    let depth = 1.5 + random() * 4

    switch (region.archetype) {
      case 'building':
        height = 4 + random() * 22 * (0.4 + confidence)
        width = 3 + random() * 7
        depth = 3 + random() * 7
        break
      case 'corridor':
        // Two parallel walls with a gap between them.
        width = 1.2 + random() * 1.6
        height = 4 + random() * 6
        depth = 8 + random() * 14
        break
      case 'monument':
        width = 2 + random() * 3
        depth = width
        height = 8 + random() * 26 * (0.3 + confidence)
        break
      case 'grove':
        width = 0.5 + random() * 1.1
        depth = width
        height = 6 + random() * 14
        break
      case 'terrain':
        width = 4 + random() * 12
        depth = 4 + random() * 12
        height = 1 + random() * 5
        break
      case 'water':
        width = 6 + random() * 16
        depth = 6 + random() * 16
        height = 0.3 + random() * 0.6
        break
      case 'void':
        // Void regions are the model admitting it has nothing: thin,
        // near-transparent slabs that read as placeholders.
        width = 2 + random() * 6
        depth = 0.3
        height = 3 + random() * 12
        break
    }

    const corridorOffset =
      region.archetype === 'corridor' ? (i % 2 === 0 ? -3.5 : 3.5) : 0

    // Below the resolution threshold, pieces lift and tilt out of alignment.
    const unresolved = Math.max(0, scene.fragmentThreshold - confidence)
    const drift = unresolved * (0.4 + random()) * 2.4

    pieces.push({
      position: [
        cx + Math.cos(angle) * distance + corridorOffset,
        cy + height / 2 - 1.2 + drift * random() * 4,
        cz + Math.sin(angle) * distance,
      ],
      scale: [width, height, depth],
      rotation: random() * Math.PI * 2 * (0.1 + scene.distortion),
      solidity,
      drift,
    })
  }

  return pieces
}

/**
 * A single region rendered as instanced geometry.
 *
 * Split into two instanced meshes — resolved and fragmented — because they
 * need genuinely different materials. Resolved geometry is opaque and lit;
 * fragmented geometry is translucent and additive so it reads as *incomplete
 * information* rather than as a solid object that happens to be see-through.
 */
function Region({
  region,
  scene,
  theme,
  confidence,
}: {
  region: RegionParams
  scene: SceneParameters
  theme: SceneTheme
  confidence: number
}) {
  const solidRef = useRef<THREE.InstancedMesh>(null)
  const ghostRef = useRef<THREE.InstancedMesh>(null)

  const { solid, ghost } = useMemo(() => {
    const pieces = buildRegion(region, scene, confidence)
    return {
      solid: pieces.filter((p) => p.solidity >= scene.fragmentThreshold),
      ghost: pieces.filter((p) => p.solidity < scene.fragmentThreshold),
    }
  }, [region, scene, confidence])

  // Write instance transforms once per data change rather than every frame.
  useMemo(() => {
    const matrix = new THREE.Matrix4()
    const quaternion = new THREE.Quaternion()
    const euler = new THREE.Euler()

    const write = (mesh: THREE.InstancedMesh | null, pieces: Piece[]) => {
      if (!mesh) return
      pieces.forEach((piece, i) => {
        euler.set(
          piece.drift * 0.12,
          piece.rotation,
          piece.drift * 0.09 * scene.distortion,
        )
        quaternion.setFromEuler(euler)
        matrix.compose(
          new THREE.Vector3(...piece.position),
          quaternion,
          new THREE.Vector3(...piece.scale),
        )
        mesh.setMatrixAt(i, matrix)
      })
      mesh.instanceMatrix.needsUpdate = true
      mesh.count = pieces.length
    }

    write(solidRef.current, solid)
    write(ghostRef.current, ghost)
  }, [solid, ghost, scene.distortion])

  // Fragmented pieces breathe very slowly, as though not fully settled.
  useFrame(({ clock }) => {
    if (!ghostRef.current) return
    const material = ghostRef.current.material as THREE.MeshStandardMaterial
    material.opacity = 0.12 + Math.sin(clock.elapsedTime * 0.6) * 0.04 + confidence * 0.18
  })

  const solidColor = useMemo(
    () => theme.fill.clone().lerp(theme.key, confidence * 0.4),
    [theme, confidence],
  )

  return (
    <group>
      {solid.length > 0 && (
        <instancedMesh
          ref={solidRef}
          args={[undefined, undefined, solid.length]}
          castShadow
          receiveShadow
          frustumCulled={false}
        >
          <boxGeometry args={[1, 1, 1]} />
          <meshStandardMaterial
            color={solidColor}
            roughness={0.82}
            metalness={0.06}
            flatShading
          />
        </instancedMesh>
      )}

      {ghost.length > 0 && (
        <instancedMesh
          ref={ghostRef}
          args={[undefined, undefined, ghost.length]}
          frustumCulled={false}
        >
          <boxGeometry args={[1, 1, 1]} />
          <meshStandardMaterial
            color={theme.rim}
            transparent
            opacity={0.16}
            roughness={1}
            metalness={0}
            depthWrite={false}
            emissive={theme.accent}
            emissiveIntensity={0.35}
          />
        </instancedMesh>
      )}
    </group>
  )
}

/** Every structural region of the reconstruction. */
export function Structures({ scene, theme, regionConfidence }: StructuresProps) {
  return (
    <group>
      {scene.regions.map((region) => (
        <Region
          key={region.id}
          region={region}
          scene={scene}
          theme={theme}
          confidence={clamp(regionConfidence[region.id] ?? region.confidence)}
        />
      ))}
    </group>
  )
}
