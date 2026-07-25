import { motion } from 'framer-motion'
import { Command, Library, Settings2, Waves } from 'lucide-react'
import { NavLink, useLocation } from 'react-router-dom'

import { cn } from '@/lib/cn'
import { useSessionStore } from '@/stores/sessionStore'
import { useUiStore } from '@/stores/uiStore'
import { Badge } from '@/ui'

const LINKS = [
  { to: '/session', label: 'Session', icon: Waves },
  { to: '/library', label: 'Library', icon: Library },
  { to: '/settings', label: 'Settings', icon: Settings2 },
] as const

/** Persistent top navigation for the application shell. */
export function Nav() {
  const location = useLocation()
  const toggleShortcuts = useUiStore((s) => s.toggleShortcuts)
  const status = useSessionStore((s) => s.status)
  const playing = useSessionStore((s) => s.playing)

  const live = playing && status !== 'idle'

  return (
    <header className="fixed inset-x-0 top-0 z-50">
      <div className="glass-strong mx-auto mt-3 flex h-14 w-[min(76rem,calc(100vw-1.5rem))] items-center justify-between gap-4 rounded-pill px-3 pl-5">
        <NavLink to="/" className="group flex items-center gap-2.5">
          <span className="relative grid size-6 place-items-center">
            <span className="absolute inset-0 rounded-full border border-neural-300/40" />
            <span className="size-1.5 rounded-full bg-memory-300 shadow-[0_0_10px] shadow-memory-400" />
          </span>
          <span className="font-display text-[0.95rem] tracking-tight">MindScape</span>
        </NavLink>

        <nav className="flex items-center gap-1" aria-label="Primary">
          {LINKS.map(({ to, label, icon: Icon }) => {
            const active = location.pathname.startsWith(to)
            return (
              <NavLink
                key={to}
                to={to}
                className={cn(
                  'relative flex items-center gap-2 rounded-pill px-3.5 py-2 text-sm transition-colors',
                  active ? 'text-ink' : 'text-ink-muted hover:text-ink-soft',
                )}
              >
                {active && (
                  <motion.span
                    layoutId="nav-active"
                    className="absolute inset-0 rounded-pill bg-ink/8"
                    transition={{ type: 'spring', stiffness: 380, damping: 32 }}
                  />
                )}
                <Icon className="relative size-3.5" />
                <span className="relative hidden sm:inline">{label}</span>
              </NavLink>
            )
          })}
        </nav>

        <div className="flex items-center gap-2">
          {live && (
            <Badge tone="var(--color-signal-good)" live className="hidden md:inline-flex">
              Streaming
            </Badge>
          )}
          <button
            onClick={() => toggleShortcuts()}
            aria-label="Keyboard shortcuts"
            className="grid size-9 place-items-center rounded-full text-ink-muted transition-colors hover:bg-ink/8 hover:text-ink"
          >
            <Command className="size-4" />
          </button>
        </div>
      </div>
    </header>
  )
}
