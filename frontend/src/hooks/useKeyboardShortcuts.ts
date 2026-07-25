import { useEffect, useMemo } from 'react'

export interface Shortcut {
  /** Lowercase `event.key`, e.g. `'k'`, `' '`, `'escape'`. */
  key: string
  meta?: boolean
  shift?: boolean
  run: () => void
  description: string
  group: string
  /** Allow while typing in an input. Off by default. */
  allowInField?: boolean
}

function isTextEntry(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  if (target.isContentEditable) return true
  const tag = target.tagName
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
}

/**
 * Binds a set of keyboard shortcuts for as long as the component is mounted.
 *
 * Keystrokes inside form fields are ignored unless a shortcut opts in, so
 * typing "s" in the library search box does not stop the session.
 */
export function useKeyboardShortcuts(shortcuts: Shortcut[], enabled = true): void {
  // Serialise the binding shape so callers can pass an inline array without
  // re-registering the listener on every render.
  const signature = shortcuts
    .map((s) => `${s.key}${s.meta ? '+meta' : ''}${s.shift ? '+shift' : ''}`)
    .join('|')

  const bindings = useMemo(() => shortcuts, [signature]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!enabled) return

    const onKeyDown = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase()
      const typing = isTextEntry(event.target)

      for (const shortcut of bindings) {
        if (key !== shortcut.key.toLowerCase()) continue
        if (Boolean(shortcut.meta) !== (event.metaKey || event.ctrlKey)) continue
        if (Boolean(shortcut.shift) !== event.shiftKey) continue
        if (typing && !shortcut.allowInField) continue

        event.preventDefault()
        shortcut.run()
        return
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [bindings, enabled])
}

/** Human-readable form of a binding, e.g. `⌘K` or `Shift + F`. */
export function formatShortcut(shortcut: Pick<Shortcut, 'key' | 'meta' | 'shift'>): string {
  const isApple =
    typeof navigator !== 'undefined' && /mac|iphone|ipad/i.test(navigator.platform ?? '')

  const parts: string[] = []
  if (shortcut.meta) parts.push(isApple ? '⌘' : 'Ctrl')
  if (shortcut.shift) parts.push(isApple ? '⇧' : 'Shift')

  const key = shortcut.key === ' ' ? 'Space' : shortcut.key.toUpperCase()
  parts.push(key)

  return parts.join(isApple ? '' : ' + ')
}
