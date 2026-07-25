import { useFrame, useThree } from '@react-three/fiber'
import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'

import { damp } from '@/lib/math'

interface FirstPersonControlsProps {
  /** Walking speed in world units per second. */
  speed?: number
  /** Multiplier while shift is held. */
  sprint?: number
  eyeHeight?: number
  enabled?: boolean
  onLockChange?: (locked: boolean) => void
}

const KEY_MAP: Record<string, keyof MoveState> = {
  KeyW: 'forward',
  ArrowUp: 'forward',
  KeyS: 'back',
  ArrowDown: 'back',
  KeyA: 'left',
  ArrowLeft: 'left',
  KeyD: 'right',
  ArrowRight: 'right',
}

interface MoveState {
  forward: boolean
  back: boolean
  left: boolean
  right: boolean
  sprint: boolean
}

/**
 * Pointer-lock first-person movement.
 *
 * Hand-rolled rather than using Drei's `PointerLockControls` because the
 * reconstruction needs velocity smoothing, head bob and a ground-following
 * eye height that the stock controller does not provide — and because a
 * memory should feel drifted through, not driven through.
 */
export function FirstPersonControls({
  speed = 7,
  sprint = 2.1,
  eyeHeight = 1.7,
  enabled = true,
  onLockChange,
}: FirstPersonControlsProps) {
  const { camera, gl, scene } = useThree()

  const move = useRef<MoveState>({
    forward: false,
    back: false,
    left: false,
    right: false,
    sprint: false,
  })
  const velocity = useRef(new THREE.Vector3())
  const euler = useRef(new THREE.Euler(0, 0, 0, 'YXZ'))
  const locked = useRef(false)
  const bobPhase = useRef(0)

  const raycaster = useMemo(() => new THREE.Raycaster(), [])
  const down = useMemo(() => new THREE.Vector3(0, -1, 0), [])

  useEffect(() => {
    if (!enabled) return

    const canvas = gl.domElement
    euler.current.setFromQuaternion(camera.quaternion)

    const onKeyDown = (event: KeyboardEvent) => {
      const action = KEY_MAP[event.code]
      if (action) {
        move.current[action] = true
        event.preventDefault()
      }
      if (event.code === 'ShiftLeft' || event.code === 'ShiftRight') {
        move.current.sprint = true
      }
    }

    const onKeyUp = (event: KeyboardEvent) => {
      const action = KEY_MAP[event.code]
      if (action) move.current[action] = false
      if (event.code === 'ShiftLeft' || event.code === 'ShiftRight') {
        move.current.sprint = false
      }
    }

    const onMouseMove = (event: MouseEvent) => {
      if (!locked.current) return
      // Sensitivity is deliberately low; a memory should not whip around.
      euler.current.y -= event.movementX * 0.0018
      euler.current.x -= event.movementY * 0.0018
      euler.current.x = Math.max(
        -Math.PI / 2 + 0.05,
        Math.min(Math.PI / 2 - 0.05, euler.current.x),
      )
      camera.quaternion.setFromEuler(euler.current)
    }

    const onPointerLockChange = () => {
      locked.current = document.pointerLockElement === canvas
      onLockChange?.(locked.current)
      if (!locked.current) {
        // Release every key, or a held direction sticks after unlocking.
        move.current = { forward: false, back: false, left: false, right: false, sprint: false }
      }
    }

    const onClick = () => {
      if (!locked.current) void canvas.requestPointerLock()
    }

    canvas.addEventListener('click', onClick)
    document.addEventListener('keydown', onKeyDown)
    document.addEventListener('keyup', onKeyUp)
    document.addEventListener('mousemove', onMouseMove)
    document.addEventListener('pointerlockchange', onPointerLockChange)

    return () => {
      canvas.removeEventListener('click', onClick)
      document.removeEventListener('keydown', onKeyDown)
      document.removeEventListener('keyup', onKeyUp)
      document.removeEventListener('mousemove', onMouseMove)
      document.removeEventListener('pointerlockchange', onPointerLockChange)
      if (document.pointerLockElement === canvas) document.exitPointerLock()
    }
  }, [camera, gl, enabled, onLockChange])

  useFrame((_, delta) => {
    if (!enabled) return
    const dt = Math.min(delta, 0.1)

    const state = move.current
    const forwardInput = Number(state.forward) - Number(state.back)
    const strafeInput = Number(state.right) - Number(state.left)

    const target = new THREE.Vector3()
    if (forwardInput !== 0 || strafeInput !== 0) {
      const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion)
      forward.y = 0
      forward.normalize()

      const strafe = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion)
      strafe.y = 0
      strafe.normalize()

      target
        .addScaledVector(forward, forwardInput)
        .addScaledVector(strafe, strafeInput)
        .normalize()
        .multiplyScalar(speed * (state.sprint ? sprint : 1))
    }

    // Smooth acceleration and deceleration — instant stops feel mechanical.
    velocity.current.x = damp(velocity.current.x, target.x, 9, dt)
    velocity.current.z = damp(velocity.current.z, target.z, 9, dt)

    camera.position.addScaledVector(velocity.current, dt)

    // Follow the ground. Cast from well above so overhanging geometry does
    // not trap the camera underneath the terrain.
    raycaster.set(
      new THREE.Vector3(camera.position.x, camera.position.y + 60, camera.position.z),
      down,
    )
    raycaster.far = 200
    const hits = raycaster.intersectObjects(scene.children, true)
    const ground = hits.find((hit) => hit.object.type === 'Mesh' && hit.distance > 0.1)

    const speedRatio = velocity.current.length() / speed
    bobPhase.current += dt * speedRatio * 9
    const bob = Math.sin(bobPhase.current) * 0.045 * Math.min(1, speedRatio)

    if (ground) {
      const targetY = ground.point.y + eyeHeight + bob
      camera.position.y = damp(camera.position.y, targetY, 11, dt)
    } else {
      // No ground beneath — over a void. Settle toward the nominal height
      // rather than falling forever.
      camera.position.y = damp(camera.position.y, eyeHeight + bob, 3, dt)
    }
  })

  return null
}
