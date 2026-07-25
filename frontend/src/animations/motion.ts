/**
 * Shared Framer Motion presets.
 *
 * Centralised so the whole app moves with one vocabulary: long, soft eases
 * for anything spatial; short, crisp ones for controls.
 */

import type { Transition, Variants } from 'framer-motion'

export const EASE_GLIDE = [0.22, 1, 0.36, 1] as const
export const EASE_SETTLE = [0.16, 1, 0.3, 1] as const
export const EASE_DRIFT = [0.4, 0, 0.2, 1] as const

export const spring: Transition = { type: 'spring', stiffness: 240, damping: 30, mass: 0.9 }
export const softSpring: Transition = { type: 'spring', stiffness: 130, damping: 22, mass: 1 }
export const glide: Transition = { duration: 0.7, ease: EASE_GLIDE }
export const quick: Transition = { duration: 0.22, ease: EASE_DRIFT }

/** Page-level crossfade with a slight rise. */
export const pageVariants: Variants = {
  initial: { opacity: 0, y: 14, filter: 'blur(6px)' },
  animate: {
    opacity: 1,
    y: 0,
    filter: 'blur(0px)',
    transition: { duration: 0.6, ease: EASE_GLIDE },
  },
  exit: {
    opacity: 0,
    y: -10,
    filter: 'blur(6px)',
    transition: { duration: 0.32, ease: EASE_DRIFT },
  },
}

/** Panels arriving as a group. */
export const staggerContainer: Variants = {
  initial: {},
  animate: { transition: { staggerChildren: 0.07, delayChildren: 0.08 } },
}

export const riseItem: Variants = {
  initial: { opacity: 0, y: 18 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.65, ease: EASE_GLIDE } },
}

export const fadeItem: Variants = {
  initial: { opacity: 0 },
  animate: { opacity: 1, transition: { duration: 0.5, ease: EASE_DRIFT } },
}

/** Scale-in used by dialogs, fragment cards and popovers. */
export const popVariants: Variants = {
  initial: { opacity: 0, scale: 0.96, y: 8 },
  animate: { opacity: 1, scale: 1, y: 0, transition: spring },
  exit: { opacity: 0, scale: 0.97, y: 4, transition: quick },
}

/** Section reveal on scroll — used across the landing page. */
export const revealOnScroll = {
  initial: { opacity: 0, y: 40 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: '-12% 0px -12% 0px' },
  transition: { duration: 0.85, ease: EASE_GLIDE },
} as const

/** Slow ambient float for decorative elements. */
export const floatTransition: Transition = {
  duration: 9,
  ease: 'easeInOut',
  repeat: Infinity,
  repeatType: 'reverse',
}

/**
 * Strip motion from a variant set when the user has asked for reduced motion.
 * Opacity is kept — a hard cut reads as a glitch, not as calm.
 */
export function staticVariants(variants: Variants): Variants {
  const flatten = (value: unknown): unknown => {
    if (typeof value !== 'object' || value === null) return value
    const { opacity, transition, ...rest } = value as Record<string, unknown>
    void rest
    void transition
    return { opacity: opacity ?? 1, transition: { duration: 0.15 } }
  }

  return Object.fromEntries(
    Object.entries(variants).map(([key, value]) => [key, flatten(value)]),
  ) as Variants
}
