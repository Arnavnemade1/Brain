import { useFrame } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'

import { mulberry32 } from '@/lib/math'
import type { ParticleKind, SceneParameters } from '@/types'

import type { SceneTheme } from './useSceneTheme'

interface AtmosphereProps {
  scene: SceneParameters
  theme: SceneTheme
  quality: 'low' | 'balanced' | 'high'
}

const PARTICLE_BUDGET: Record<'low' | 'balanced' | 'high', number> = {
  low: 400,
  balanced: 1400,
  high: 3200,
}

interface ParticleBehaviour {
  /** Vertical velocity, world units per second. Negative falls. */
  fall: number
  size: number
  opacity: number
  /** How strongly wind pushes the particle sideways. */
  sway: number
}

const BEHAVIOUR: Record<ParticleKind, ParticleBehaviour> = {
  dust: { fall: -0.25, size: 0.09, opacity: 0.5, sway: 1.0 },
  pollen: { fall: -0.15, size: 0.13, opacity: 0.55, sway: 1.4 },
  motes: { fall: 0.08, size: 0.07, opacity: 0.6, sway: 0.7 },
  embers: { fall: 0.55, size: 0.11, opacity: 0.75, sway: 1.1 },
  snow: { fall: -0.9, size: 0.16, opacity: 0.7, sway: 0.9 },
  rain: { fall: -9.0, size: 0.05, opacity: 0.45, sway: 0.15 },
  none: { fall: 0, size: 0, opacity: 0, sway: 0 },
}

/**
 * Airborne particles.
 *
 * One `Points` cloud with per-frame positional updates on a typed array.
 * Particles wrap within a box around the camera rather than being culled and
 * respawned, which keeps density constant wherever the user walks without
 * ever allocating during the frame.
 */
function Particles({ scene, theme, quality }: AtmosphereProps) {
  const pointsRef = useRef<THREE.Points>(null)
  const kind = scene.atmosphere.particles
  const behaviour = BEHAVIOUR[kind]

  const count = Math.round(
    PARTICLE_BUDGET[quality] * (0.3 + scene.atmosphere.particleDensity * 0.9),
  )

  const { positions, offsets, span } = useMemo(() => {
    const random = mulberry32(scene.seed ^ 0x9e37)
    const span = 90
    const positions = new Float32Array(count * 3)
    const offsets = new Float32Array(count)

    for (let i = 0; i < count; i++) {
      positions[i * 3] = (random() - 0.5) * span
      positions[i * 3 + 1] = random() * 42
      positions[i * 3 + 2] = (random() - 0.5) * span
      offsets[i] = random() * Math.PI * 2
    }

    return { positions, offsets, span }
  }, [count, scene.seed])

  useFrame((state, delta) => {
    const points = pointsRef.current
    if (!points || kind === 'none') return

    const attribute = points.geometry.attributes.position as THREE.BufferAttribute
    const array = attribute.array as Float32Array
    const time = state.clock.elapsedTime
    const wind = scene.atmosphere.wind
    const dt = Math.min(delta, 0.1)

    // Follow the camera so the field never runs out.
    const camera = state.camera.position

    for (let i = 0; i < count; i++) {
      const y = i * 3 + 1

      array[y] += behaviour.fall * dt
      array[i * 3] += Math.sin(time * 0.4 + offsets[i]) * behaviour.sway * wind * dt
      array[i * 3 + 2] += Math.cos(time * 0.3 + offsets[i]) * behaviour.sway * wind * dt

      // Wrap within the box centred on the camera.
      if (behaviour.fall < 0 && array[y] < camera.y - 6) array[y] = camera.y + 36
      if (behaviour.fall > 0 && array[y] > camera.y + 36) array[y] = camera.y - 6

      const dx = array[i * 3] - camera.x
      if (dx > span / 2) array[i * 3] -= span
      if (dx < -span / 2) array[i * 3] += span

      const dz = array[i * 3 + 2] - camera.z
      if (dz > span / 2) array[i * 3 + 2] -= span
      if (dz < -span / 2) array[i * 3 + 2] += span
    }

    attribute.needsUpdate = true
  })

  if (kind === 'none' || count === 0) return null

  const color =
    kind === 'embers' ? theme.accent : kind === 'rain' ? theme.fog : theme.key

  return (
    <points ref={pointsRef} frustumCulled={false}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial
        color={color}
        size={behaviour.size}
        transparent
        opacity={behaviour.opacity}
        sizeAttenuation
        depthWrite={false}
        blending={kind === 'embers' || kind === 'motes' ? THREE.AdditiveBlending : THREE.NormalBlending}
      />
    </points>
  )
}

/**
 * Sky dome.
 *
 * A large back-face sphere with a vertical gradient, rather than a physical
 * sky model: the palette here is derived from inferred *affect*, not from
 * atmospheric scattering, and a physically-correct sky would fight the grade
 * the rest of the scene is built around.
 */
function SkyDome({ theme }: { theme: SceneTheme }) {
  const material = useMemo(
    () =>
      new THREE.ShaderMaterial({
        side: THREE.BackSide,
        depthWrite: false,
        uniforms: {
          uTop: { value: theme.sky },
          uHorizon: { value: theme.horizon },
          uGround: { value: theme.fog },
        },
        vertexShader: /* glsl */ `
          varying vec3 vWorld;
          void main() {
            vWorld = normalize(position);
            gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
          }
        `,
        fragmentShader: /* glsl */ `
          uniform vec3 uTop;
          uniform vec3 uHorizon;
          uniform vec3 uGround;
          varying vec3 vWorld;

          void main() {
            float h = vWorld.y;
            vec3 color = h > 0.0
              ? mix(uHorizon, uTop, pow(clamp(h, 0.0, 1.0), 0.55))
              : mix(uHorizon, uGround, pow(clamp(-h, 0.0, 1.0), 0.4));
            gl_FragColor = vec4(color, 1.0);
          }
        `,
      }),
    [theme.sky, theme.horizon, theme.fog],
  )

  return (
    <mesh material={material} frustumCulled={false} renderOrder={-1}>
      <sphereGeometry args={[600, 32, 20]} />
    </mesh>
  )
}

export function Atmosphere({ scene, theme, quality }: AtmosphereProps) {
  return (
    <>
      <SkyDome theme={theme} />
      <fogExp2 attach="fog" args={[theme.fog.getHex(), theme.fogDensity]} />
      <Particles scene={scene} theme={theme} quality={quality} />
    </>
  )
}

/** Key, fill and ambient lighting derived from the scene parameters. */
export function Lighting({
  theme,
  quality,
}: {
  theme: SceneTheme
  quality: 'low' | 'balanced' | 'high'
}) {
  const sun = useMemo(
    () => theme.sunDirection.clone().multiplyScalar(120),
    [theme.sunDirection],
  )

  return (
    <>
      <ambientLight intensity={theme.ambientIntensity} color={theme.fill} />

      <directionalLight
        position={sun}
        intensity={theme.keyIntensity}
        color={theme.key}
        castShadow={quality !== 'low'}
        shadow-mapSize-width={quality === 'high' ? 2048 : 1024}
        shadow-mapSize-height={quality === 'high' ? 2048 : 1024}
        shadow-camera-near={1}
        shadow-camera-far={320}
        shadow-camera-left={-90}
        shadow-camera-right={90}
        shadow-camera-top={90}
        shadow-camera-bottom={-90}
        shadow-bias={-0.0008}
      />

      {/* Rim light from the opposite side keeps silhouettes readable against
          the fog, which otherwise swallows unlit faces entirely. */}
      <directionalLight
        position={[-sun.x, Math.abs(sun.y) * 0.35 + 20, -sun.z]}
        intensity={theme.keyIntensity * 0.22}
        color={theme.rim}
      />

      <hemisphereLight
        args={[theme.sky.getHex(), theme.ground.getHex(), theme.ambientIntensity * 0.8]}
      />
    </>
  )
}
