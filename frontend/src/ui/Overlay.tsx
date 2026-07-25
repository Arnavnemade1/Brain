import { AnimatePresence, motion } from 'framer-motion'
import { X } from 'lucide-react'
import { useEffect, useRef, type ReactNode } from 'react'
import { createPortal } from 'react-dom'

import { popVariants, quick } from '@/animations/motion'
import { cn } from '@/lib/cn'

interface DialogProps {
  open: boolean
  onClose: () => void
  title?: ReactNode
  label?: ReactNode
  children: ReactNode
  /** Width class, e.g. `max-w-2xl`. */
  width?: string
  /** Hides the close affordance for flows that must be completed. */
  dismissible?: boolean
}

/**
 * Centred modal.
 *
 * Handles the accessibility work a `<div>` overlay otherwise skips: focus is
 * moved in on open and restored on close, Escape dismisses, and focus is
 * trapped while open so keyboard users cannot tab into the inert page behind.
 */
export function Dialog({
  open,
  onClose,
  title,
  label,
  children,
  width = 'max-w-lg',
  dismissible = true,
}: DialogProps) {
  const panelRef = useRef<HTMLDivElement>(null)
  const restoreTo = useRef<HTMLElement | null>(null)

  useEffect(() => {
    if (!open) return

    restoreTo.current = document.activeElement as HTMLElement | null
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'

    // Move focus into the dialog once it has mounted.
    const focusTimer = window.setTimeout(() => {
      const target = panelRef.current?.querySelector<HTMLElement>(
        '[data-autofocus], button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
      )
      ;(target ?? panelRef.current)?.focus()
    }, 30)

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && dismissible) {
        event.stopPropagation()
        onClose()
        return
      }

      if (event.key !== 'Tab' || !panelRef.current) return

      const focusable = Array.from(
        panelRef.current.querySelectorAll<HTMLElement>(
          'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ),
      ).filter((el) => el.offsetParent !== null)

      if (focusable.length === 0) return

      const first = focusable[0]
      const last = focusable[focusable.length - 1]

      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }

    document.addEventListener('keydown', onKeyDown, true)

    return () => {
      window.clearTimeout(focusTimer)
      document.removeEventListener('keydown', onKeyDown, true)
      document.body.style.overflow = previousOverflow
      restoreTo.current?.focus?.()
    }
  }, [open, onClose, dismissible])

  return createPortal(
    <AnimatePresence>
      {open && (
        <div className="fixed inset-0 z-[70] grid place-items-center p-4">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={quick}
            onClick={dismissible ? onClose : undefined}
            className="absolute inset-0 bg-void/70 backdrop-blur-md"
          />
          <motion.div
            ref={panelRef}
            role="dialog"
            aria-modal="true"
            aria-label={typeof title === 'string' ? title : 'Dialog'}
            tabIndex={-1}
            variants={popVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            className={cn(
              'glass-strong rounded-panel relative w-full outline-none',
              'max-h-[min(85dvh,52rem)] overflow-y-auto',
              width,
            )}
          >
            {(title || label) && (
              <header className="flex items-start justify-between gap-4 px-6 pt-6">
                <div className="min-w-0">
                  {label && <p className="text-label mb-2">{label}</p>}
                  {title && <h2 className="text-lg">{title}</h2>}
                </div>
                {dismissible && (
                  <button
                    onClick={onClose}
                    aria-label="Close dialog"
                    className="-mt-1 -mr-1 shrink-0 rounded-lg p-2 text-ink-faint transition-colors hover:bg-ink/6 hover:text-ink"
                  >
                    <X className="size-4" />
                  </button>
                )}
              </header>
            )}
            <div className="p-6">{children}</div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>,
    document.body,
  )
}

/* -------------------------------------------------------------------------- */
/* Tooltip                                                                    */
/* -------------------------------------------------------------------------- */

/**
 * Lightweight hover/focus tooltip.
 *
 * CSS-driven rather than JS-positioned: these annotate dense telemetry, and
 * a dozen of them running position calculations on a 60 fps page is a cost
 * with no visible benefit at this size.
 */
export function Tooltip({
  content,
  children,
  side = 'top',
  className,
}: {
  content: ReactNode
  children: ReactNode
  side?: 'top' | 'bottom'
  className?: string
}) {
  return (
    <span className={cn('group/tip relative inline-flex', className)}>
      {children}
      <span
        role="tooltip"
        className={cn(
          'glass-strong pointer-events-none absolute left-1/2 z-50 w-max max-w-[16rem]',
          '-translate-x-1/2 rounded-lg px-2.5 py-1.5 text-[0.7rem] leading-snug text-ink-soft',
          'opacity-0 transition-opacity duration-150',
          'group-hover/tip:opacity-100 group-focus-within/tip:opacity-100',
          side === 'top' ? 'bottom-[calc(100%+8px)]' : 'top-[calc(100%+8px)]',
        )}
      >
        {content}
      </span>
    </span>
  )
}
