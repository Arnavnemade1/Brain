import { useEffect, useRef, useState } from 'react'

import { useReducedMotion } from './useMotionPreference'

export interface RevealOptions {
  /** Fraction of the element that must be visible. Defaults to a sliver. */
  amount?: number
  /** Extra delay before revealing, in seconds. */
  delay?: number
}

export interface Reveal {
  ref: React.RefObject<HTMLElement | null>
  revealed: boolean
  /** Spread onto a `motion` element for the standard rise-and-fade. */
  props: {
    initial: { opacity: number; y: number }
    animate: { opacity: number; y: number }
    transition: { duration: number; ease: readonly [number, number, number, number]; delay: number }
  }
}

const EASE = [0.22, 1, 0.36, 1] as const

/**
 * Reveal-on-scroll that survives jump navigation.
 *
 * `whileInView` alone is subtly broken for this: it reveals on intersection,
 * but an element the user has already scrolled *past* — via an anchor link,
 * a restored scroll position, or a deep link — never intersects again and
 * stays invisible forever. Scrolling back up to a blank section is a bad bug
 * and an easy one to miss in development, where you always scroll down.
 *
 * So the predicate is "in view **or** above the viewport", evaluated on mount
 * as well as on intersection.
 */
export function useReveal({ amount = 0.08, delay = 0 }: RevealOptions = {}): Reveal {
  const ref = useRef<HTMLElement | null>(null)
  const [revealed, setRevealed] = useState(false)
  const reduced = useReducedMotion()

  useEffect(() => {
    const element = ref.current
    if (!element || revealed) return

    /** True once the element is in view or has been scrolled past. */
    const seen = () => {
      const rect = element.getBoundingClientRect()
      // A zero-height viewport (hidden pane, collapsed iframe) can satisfy
      // any comparison trivially, so require a real viewport first.
      if (window.innerHeight <= 0) return false
      return rect.top < window.innerHeight * (1 - amount)
    }

    if (seen()) {
      setRevealed(true)
      return
    }

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting || entry.boundingClientRect.top < 0) {
            setRevealed(true)
            observer.disconnect()
            return
          }
        }
      },
      { threshold: amount, rootMargin: '0px 0px -8% 0px' },
    )
    observer.observe(element)

    // Re-check on resize. The observer only reports *changes* in
    // intersection, so an element that was above a viewport which has since
    // resized — a collapsed pane reopening, an orientation change — can end
    // up permanently unrevealed with no further callback coming.
    const onResize = () => {
      if (seen()) {
        setRevealed(true)
        observer.disconnect()
      }
    }
    window.addEventListener('resize', onResize)

    // Safety net. Content that starts at opacity 0 and waits for an observer
    // is permanently invisible if that observer never fires — a zero-height
    // viewport, a display:none ancestor, an unsupported browser. Failing open
    // after a short delay means the worst case is a missed animation rather
    // than a blank page.
    const failsafe = window.setTimeout(() => setRevealed(true), 1600)

    return () => {
      observer.disconnect()
      window.removeEventListener('resize', onResize)
      window.clearTimeout(failsafe)
    }
  }, [amount, revealed])

  return {
    ref,
    revealed,
    props: {
      initial: { opacity: 0, y: reduced ? 0 : 28 },
      animate: revealed ? { opacity: 1, y: 0 } : { opacity: 0, y: reduced ? 0 : 28 },
      transition: { duration: reduced ? 0.2 : 0.8, ease: EASE, delay: reduced ? 0 : delay },
    },
  }
}
