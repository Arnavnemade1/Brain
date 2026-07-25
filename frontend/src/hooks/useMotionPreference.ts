import { useEffect, useState } from 'react'

import { useUiStore } from '@/stores/uiStore'

/** Live value of a CSS media query. */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() =>
    typeof window === 'undefined' ? false : window.matchMedia(query).matches,
  )

  useEffect(() => {
    const list = window.matchMedia(query)
    const onChange = (event: MediaQueryListEvent) => setMatches(event.matches)
    setMatches(list.matches)
    list.addEventListener('change', onChange)
    return () => list.removeEventListener('change', onChange)
  }, [query])

  return matches
}

/**
 * Whether motion should be suppressed.
 *
 * True if the OS asks for reduced motion *or* the user has turned it off in
 * settings — an explicit in-app preference can only ever add restriction, so
 * a user who wants stillness gets it regardless of their system config.
 */
export function useReducedMotion(): boolean {
  const systemPrefers = useMediaQuery('(prefers-reduced-motion: reduce)')
  const userPrefers = useUiStore((s) => s.reduceMotion)
  return systemPrefers || userPrefers
}

/** Coarse pointer — used to swap hover affordances for tap targets. */
export function useIsTouch(): boolean {
  return useMediaQuery('(pointer: coarse)')
}

export function useIsCompact(): boolean {
  return useMediaQuery('(max-width: 1024px)')
}
