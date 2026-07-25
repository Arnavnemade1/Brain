import { useMemo } from 'react'
import * as THREE from 'three'

import { clamp, mapRange } from '@/lib/math'
import type { SceneParameters } from '@/types'

export interface SceneTheme {
  sky: THREE.Color
  horizon: THREE.Color
  ground: THREE.Color
  key: THREE.Color
  fill: THREE.Color
  rim: THREE.Color
  fog: THREE.Color
  accent: THREE.Color
  /** Exponential-squared fog density. */
  fogDensity: number
  sunDirection: THREE.Vector3
  keyIntensity: number
  ambientIntensity: number
  /** World units the terrain spans. */
  extent: number
  /** Metres of vertical relief. */
  relief: number
}

/**
 * Convert a colour temperature in Kelvin to a linear RGB multiplier.
 *
 * A piecewise approximation of the Planckian locus — good enough for grading
 * a scene and far cheaper than a spectral integration.
 */
function kelvinToColor(kelvin: number): THREE.Color {
  const t = clamp(kelvin, 1000, 12000) / 100

  let r: number
  let g: number
  let b: number

  if (t <= 66) {
    r = 255
    g = 99.4708025861 * Math.log(t) - 161.1195681661
    b = t <= 19 ? 0 : 138.5177312231 * Math.log(t - 10) - 305.0447927307
  } else {
    r = 329.698727446 * Math.pow(t - 60, -0.1332047592)
    g = 288.1221695283 * Math.pow(t - 60, -0.0755148492)
    b = 255
  }

  return new THREE.Color(
    clamp(r / 255, 0, 1),
    clamp(g / 255, 0, 1),
    clamp(b / 255, 0, 1),
  ).convertSRGBToLinear()
}

/**
 * Derive every renderable property of the world from the scene parameters.
 *
 * Single source of truth: lighting, fog, terrain scale and palette all come
 * from here, so the environment can never disagree with the readouts in the
 * console.
 */
export function useSceneTheme(scene: SceneParameters | null, confidence: number): SceneTheme {
  return useMemo(() => {
    const palette = scene?.palette
    const lighting = scene?.lighting
    const atmosphere = scene?.atmosphere

    const parse = (hex: string | undefined, fallback: string) =>
      new THREE.Color(hex ?? fallback).convertSRGBToLinear()

    const temperature = kelvinToColor(lighting?.temperature ?? 5200)

    const key = parse(palette?.key, '#cfe9ec').multiply(temperature)
    const fill = parse(palette?.fill, '#6fa8b4')
    const rim = parse(palette?.rim, '#a9dce4')
    const sky = parse(palette?.sky, '#93bdd0')
    const ground = parse(palette?.ground, '#5c7f8a')
    const fog = parse(palette?.fog, '#8fb6c2')
    const accent = parse(palette?.accent, '#5fd4c4')

    // Sun position from elevation/azimuth, in a Y-up world.
    const elevation = THREE.MathUtils.degToRad(lighting?.sunElevation ?? 20)
    const azimuth = THREE.MathUtils.degToRad(lighting?.sunAzimuth ?? 140)
    const sunDirection = new THREE.Vector3(
      Math.cos(elevation) * Math.cos(azimuth),
      Math.sin(elevation),
      Math.cos(elevation) * Math.sin(azimuth),
    ).normalize()

    // Fog does double duty: it is the weather, and it is how uncertainty
    // reads. Low confidence thickens the air so the world literally fades
    // out where the evidence does.
    const weatherFog = atmosphere?.fogDensity ?? 0.2
    const uncertainty = 1 - clamp(confidence)
    const fogDensity = mapRange(
      clamp(weatherFog * 0.7 + uncertainty * 0.5),
      0,
      1,
      0.004,
      0.032,
    )

    const spatialScale = scene?.spatialScale ?? 0.5

    return {
      sky,
      horizon: sky.clone().lerp(fog, 0.55),
      ground,
      key,
      fill,
      rim,
      fog,
      accent,
      fogDensity,
      sunDirection,
      keyIntensity: (lighting?.intensity ?? 0.8) * 2.4,
      ambientIntensity: (lighting?.ambient ?? 0.3) * 1.6,
      extent: mapRange(spatialScale, 0, 1, 180, 480),
      relief: mapRange(spatialScale, 0, 1, 3, 34),
    }
  }, [scene, confidence])
}
