import { Billboard, Text } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'
import { useRef, useState } from 'react'
import * as THREE from 'three'

import { EMOTION_COLOR } from '@/lib/palette'
import type { MemoryFragment } from '@/types'

interface FragmentMarkersProps {
  fragments: MemoryFragment[]
  openFragmentId: string | null
  revealedIds: string[]
  onOpen: (id: string) => void
  /** Distance within which a fragment can be opened with the E key. */
  reach?: number
  onNearestChange?: (fragment: MemoryFragment | null) => void
}

/**
 * One floating memory fragment.
 *
 * Rendered as a slowly rotating shard whose brightness tracks its confidence,
 * so a glance across the world reads which episodes are well supported. The
 * label only appears within reading distance — a field of always-on labels
 * turns the memory into a menu.
 */
function Marker({
  fragment,
  open,
  revealed,
  onOpen,
  onDistance,
}: {
  fragment: MemoryFragment
  open: boolean
  revealed: boolean
  onOpen: () => void
  onDistance: (id: string, distance: number) => void
}) {
  const groupRef = useRef<THREE.Group>(null)
  const coreRef = useRef<THREE.Mesh>(null)
  const [hovered, setHovered] = useState(false)
  const [near, setNear] = useState(false)

  const color = EMOTION_COLOR[fragment.emotion]

  useFrame(({ clock, camera }) => {
    const group = groupRef.current
    if (!group) return

    const time = clock.elapsedTime
    group.position.y = fragment.position[1] + Math.sin(time * 0.7 + fragment.position[0]) * 0.28
    group.rotation.y = time * 0.28

    if (coreRef.current) {
      const pulse = 1 + Math.sin(time * 1.6 + fragment.position[2]) * 0.09
      coreRef.current.scale.setScalar(pulse)
    }

    const distance = camera.position.distanceTo(group.position)
    onDistance(fragment.id, distance)

    const isNear = distance < 14
    if (isNear !== near) setNear(isNear)
  })

  const emissiveIntensity = (revealed ? 0.6 : 1.4) * (0.4 + fragment.confidence)
  const scale = 0.6 + fragment.confidence * 0.5

  return (
    <group ref={groupRef} position={fragment.position}>
      <mesh
        ref={coreRef}
        scale={scale}
        onClick={(event) => {
          event.stopPropagation()
          onOpen()
        }}
        onPointerOver={(event) => {
          event.stopPropagation()
          setHovered(true)
          document.body.style.cursor = 'pointer'
        }}
        onPointerOut={() => {
          setHovered(false)
          document.body.style.cursor = ''
        }}
      >
        <octahedronGeometry args={[0.55, 0]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={hovered || open ? emissiveIntensity * 1.7 : emissiveIntensity}
          transparent
          opacity={revealed ? 0.75 : 0.95}
          roughness={0.25}
          metalness={0.3}
        />
      </mesh>

      {/* Halo — gives the shard presence in fog without a post-process bloom. */}
      <mesh scale={scale * 2.6}>
        <sphereGeometry args={[0.55, 12, 10]} />
        <meshBasicMaterial
          color={color}
          transparent
          opacity={hovered ? 0.16 : 0.07}
          depthWrite={false}
          side={THREE.BackSide}
        />
      </mesh>

      {(near || hovered) && (
        <Billboard position={[0, 1.5, 0]}>
          <Text
            fontSize={0.34}
            color={color}
            anchorX="center"
            anchorY="middle"
            outlineWidth={0.014}
            outlineColor="#05070d"
            maxWidth={7}
          >
            {fragment.title}
          </Text>
          <Text
            position={[0, -0.42, 0]}
            fontSize={0.2}
            color="#b9c2d4"
            anchorX="center"
            anchorY="middle"
            outlineWidth={0.01}
            outlineColor="#05070d"
          >
            {`${(fragment.confidence * 100).toFixed(0)}% · ${fragment.emotion}`}
          </Text>
        </Billboard>
      )}
    </group>
  )
}

export function FragmentMarkers({
  fragments,
  openFragmentId,
  revealedIds,
  onOpen,
  reach = 6,
  onNearestChange,
}: FragmentMarkersProps) {
  // Distances are written into a ref, not state: they update every frame for
  // every fragment, and routing that through React would re-render the whole
  // scene 60 times a second.
  const distances = useRef<Map<string, number>>(new Map())
  const nearestId = useRef<string | null>(null)

  const handleDistance = (id: string, distance: number) => {
    distances.current.set(id, distance)
  }

  useFrame(() => {
    if (!onNearestChange) return

    let bestId: string | null = null
    let best = reach

    for (const [id, distance] of distances.current) {
      if (distance < best) {
        best = distance
        bestId = id
      }
    }

    if (bestId !== nearestId.current) {
      nearestId.current = bestId
      onNearestChange(fragments.find((f) => f.id === bestId) ?? null)
    }
  })

  return (
    <group>
      {fragments.map((fragment) => (
        <Marker
          key={fragment.id}
          fragment={fragment}
          open={fragment.id === openFragmentId}
          revealed={revealedIds.includes(fragment.id)}
          onOpen={() => onOpen(fragment.id)}
          onDistance={handleDistance}
        />
      ))}
    </group>
  )
}
