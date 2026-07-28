import { motion } from 'framer-motion'
import { useEffect, useState } from 'react'
import { NavLink, useLocation } from 'react-router-dom'

import { cn } from '@/lib/cn'

const LINKS = [
  { to: '/research', label: 'Research' },
  { to: '/session', label: 'Pipeline' },
  { to: '/library', label: 'Library' },
] as const

/**
 * The single navigation for the whole product.
 *
 * One bar on every route including the landing page. An earlier version gave
 * the landing page its own separate header, which meant two different sets of
 * links, two alignments and two behaviours for the same job.
 */
export function Nav() {
  const location = useLocation()

  // The bar is transparent over the hero, where it looks best, but the page
  // scrolls bright content underneath it — the memory video especially — and
  // unbacked labels became unreadable on top of it. So it earns a backdrop the
  // moment anything has scrolled past.
  const [scrolled, setScrolled] = useState(false)
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <header
      className={cn(
        'fixed inset-x-0 top-0 z-50 transition-colors duration-300',
        scrolled && 'border-b border-ink/8 bg-abyss/70 backdrop-blur-xl',
      )}
    >
      <div className="shell flex h-16 items-center justify-between">
        <NavLink to="/" className="flex items-center gap-2.5" aria-label="MindScape home">
          <span className="size-1.5 rounded-full bg-neural-300" />
          <span className="font-display text-[0.95rem] tracking-tight text-ink">
            MindScape
          </span>
        </NavLink>

        {/* `whitespace-nowrap` and a tighter mobile gutter: at 375 px the
            three labels were wrapping mid-phrase ("Real / EEG") and colliding
            with the brand. */}
        <nav className="flex items-center gap-0.5 sm:gap-1" aria-label="Primary">
          {LINKS.map(({ to, label }) => {
            const active = location.pathname.startsWith(to)
            return (
              <NavLink
                key={to}
                to={to}
                className={cn(
                  'relative rounded-full px-2.5 py-1.5 text-[0.8rem] whitespace-nowrap transition-colors sm:px-3.5 sm:text-[0.82rem]',
                  active ? 'text-ink' : 'text-ink-muted hover:text-ink-soft',
                )}
              >
                {active && (
                  <motion.span
                    layoutId="nav-active"
                    className="absolute inset-0 rounded-full bg-ink/8"
                    transition={{ type: 'spring', stiffness: 380, damping: 32 }}
                  />
                )}
                <span className="relative">{label}</span>
              </NavLink>
            )
          })}
        </nav>
      </div>
    </header>
  )
}
