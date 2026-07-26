/**
 * Shared Framer Motion presets.
 *
 * Centralised so the whole app moves with one vocabulary: long, soft eases
 * for anything spatial; short, crisp ones for controls. Only presets that are
 * actually used live here — an animation library nobody calls is just another
 * thing to keep consistent.
 */

import type { Transition, Variants } from 'framer-motion'

const EASE_GLIDE = [0.22, 1, 0.36, 1] as const
const EASE_DRIFT = [0.4, 0, 0.2, 1] as const

export const spring: Transition = { type: 'spring', stiffness: 240, damping: 30, mass: 0.9 }
export const quick: Transition = { duration: 0.22, ease: EASE_DRIFT }

/**
 * Page-level crossfade with a slight rise.
 *
 * Deliberately no `filter: blur()`. Animating a filter on the page root forces
 * the whole document — including the fixed video backdrop behind it — through
 * a compositing pass on every route change, and it leaves the entire page
 * smeared for the duration. Opacity and transform are enough.
 */
export const pageVariants: Variants = {
  initial: { opacity: 0, y: 14 },
  animate: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, ease: EASE_GLIDE },
  },
  exit: {
    opacity: 0,
    y: -10,
    transition: { duration: 0.28, ease: EASE_DRIFT },
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

/** Scale-in used by dialogs, fragment cards and popovers. */
export const popVariants: Variants = {
  initial: { opacity: 0, scale: 0.96, y: 8 },
  animate: { opacity: 1, scale: 1, y: 0, transition: spring },
  exit: { opacity: 0, scale: 0.97, y: 4, transition: quick },
}
