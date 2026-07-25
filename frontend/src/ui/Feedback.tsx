import { AnimatePresence, motion } from 'framer-motion'
import { AlertTriangle, Check, Info, X } from 'lucide-react'
import { Component, type ErrorInfo, type ReactNode, useEffect } from 'react'

import { popVariants, quick } from '@/animations/motion'
import { cn } from '@/lib/cn'
import { useUiStore, type Toast as ToastModel } from '@/stores/uiStore'

import { Button } from './Button'

/* -------------------------------------------------------------------------- */
/* Loading                                                                    */
/* -------------------------------------------------------------------------- */

/**
 * Loading state for anything that takes real time.
 *
 * Shows what is being waited on rather than a bare spinner — this app's waits
 * are for signal processing, and naming the work is part of the experience.
 */
export function LoadingState({
  title = 'Preparing',
  detail,
  className,
}: {
  title?: string
  detail?: string
  className?: string
}) {
  return (
    <div className={cn('grid place-items-center gap-4 py-16 text-center', className)}>
      <div className="relative grid size-14 place-items-center">
        <span className="absolute inset-0 animate-pulse-ring rounded-full border border-neural-400/40" />
        <span
          className="absolute inset-0 animate-pulse-ring rounded-full border border-neural-400/30"
          style={{ animationDelay: '1.1s' }}
        />
        <span className="size-2 rounded-full bg-neural-300 shadow-[0_0_16px] shadow-neural-400" />
      </div>
      <div>
        <p className="shimmer-text text-sm font-medium">{title}</p>
        {detail && <p className="mt-1.5 max-w-xs text-xs text-ink-muted">{detail}</p>}
      </div>
    </div>
  )
}

/** Skeleton block for content-shaped placeholders. */
export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      className={cn('relative overflow-hidden rounded-lg bg-ink/6', className)}
      aria-hidden
    >
      <div className="absolute inset-0 -translate-x-full animate-[shimmer_2s_infinite] bg-gradient-to-r from-transparent via-ink/8 to-transparent" />
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Empty                                                                      */
/* -------------------------------------------------------------------------- */

export function EmptyState({
  icon,
  title,
  detail,
  action,
}: {
  icon?: ReactNode
  title: string
  detail?: string
  action?: ReactNode
}) {
  return (
    <div className="grid place-items-center gap-4 px-6 py-20 text-center">
      {icon && (
        <div className="glass-faint grid size-12 place-items-center rounded-2xl text-ink-muted">
          {icon}
        </div>
      )}
      <div className="max-w-sm">
        <p className="text-[0.95rem] text-ink">{title}</p>
        {detail && <p className="mt-2 text-sm leading-relaxed text-ink-muted">{detail}</p>}
      </div>
      {action}
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Toasts                                                                     */
/* -------------------------------------------------------------------------- */

const TOAST_TONE: Record<ToastModel['tone'], { color: string; icon: ReactNode }> = {
  info: { color: 'var(--color-neural-300)', icon: <Info className="size-4" /> },
  success: { color: 'var(--color-signal-good)', icon: <Check className="size-4" /> },
  warning: { color: 'var(--color-signal-fair)', icon: <AlertTriangle className="size-4" /> },
  error: { color: 'var(--color-signal-poor)', icon: <AlertTriangle className="size-4" /> },
}

function ToastRow({ toast }: { toast: ToastModel }) {
  const dismiss = useUiStore((s) => s.dismissToast)
  const tone = TOAST_TONE[toast.tone]

  useEffect(() => {
    const timer = window.setTimeout(() => dismiss(toast.id), 6000)
    return () => window.clearTimeout(timer)
  }, [toast.id, dismiss])

  return (
    <motion.li
      layout
      variants={popVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      className="glass-strong rounded-card flex items-start gap-3 px-4 py-3 shadow-2xl"
    >
      <span className="mt-0.5 shrink-0" style={{ color: tone.color }}>
        {tone.icon}
      </span>
      <p className="min-w-0 flex-1 text-sm leading-snug text-ink-soft">{toast.message}</p>
      <button
        onClick={() => dismiss(toast.id)}
        aria-label="Dismiss notification"
        className="-mr-1 shrink-0 rounded p-1 text-ink-faint transition-colors hover:text-ink"
      >
        <X className="size-3.5" />
      </button>
    </motion.li>
  )
}

export function ToastViewport() {
  const toasts = useUiStore((s) => s.toasts)

  return (
    <ul
      aria-live="polite"
      aria-atomic="false"
      className="pointer-events-none fixed right-4 bottom-4 z-[80] flex w-[min(22rem,calc(100vw-2rem))] flex-col gap-2"
    >
      <AnimatePresence initial={false}>
        {toasts.map((toast) => (
          <div key={toast.id} className="pointer-events-auto">
            <ToastRow toast={toast} />
          </div>
        ))}
      </AnimatePresence>
    </ul>
  )
}

/* -------------------------------------------------------------------------- */
/* Inline error                                                               */
/* -------------------------------------------------------------------------- */

export function ErrorNotice({
  message,
  onRetry,
  onDismiss,
}: {
  message: string
  onRetry?: () => void
  onDismiss?: () => void
}) {
  return (
    <motion.div
      variants={popVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      transition={quick}
      role="alert"
      className="rounded-card flex items-start gap-3 border border-signal-poor/25 bg-signal-poor/8 px-4 py-3"
    >
      <AlertTriangle className="mt-0.5 size-4 shrink-0 text-signal-poor" />
      <p className="min-w-0 flex-1 text-sm leading-snug text-ink-soft">{message}</p>
      <div className="flex shrink-0 gap-1">
        {onRetry && (
          <Button size="sm" variant="ghost" onClick={onRetry}>
            Retry
          </Button>
        )}
        {onDismiss && (
          <button
            onClick={onDismiss}
            aria-label="Dismiss error"
            className="rounded p-1.5 text-ink-faint transition-colors hover:text-ink"
          >
            <X className="size-3.5" />
          </button>
        )}
      </div>
    </motion.div>
  )
}

/* -------------------------------------------------------------------------- */
/* Error boundary                                                             */
/* -------------------------------------------------------------------------- */

interface BoundaryProps {
  children: ReactNode
  /** Shown instead of the default panel. */
  fallback?: (error: Error, reset: () => void) => ReactNode
  label?: string
}

interface BoundaryState {
  error: Error | null
}

/**
 * Catches render failures so one broken panel — most likely the WebGL scene —
 * cannot take down the whole session.
 */
export class ErrorBoundary extends Component<BoundaryProps, BoundaryState> {
  state: BoundaryState = { error: null }

  static getDerivedStateFromError(error: Error): BoundaryState {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error(`[MindScape] ${this.props.label ?? 'Component'} failed:`, error, info)
  }

  private reset = () => this.setState({ error: null })

  render(): ReactNode {
    const { error } = this.state
    if (!error) return this.props.children

    if (this.props.fallback) return this.props.fallback(error, this.reset)

    return (
      <div className="glass rounded-panel grid place-items-center gap-4 p-10 text-center">
        <AlertTriangle className="size-6 text-signal-fair" />
        <div>
          <p className="text-sm text-ink">
            {this.props.label ?? 'This panel'} stopped unexpectedly.
          </p>
          <p className="mx-auto mt-2 max-w-sm text-xs leading-relaxed text-ink-muted">
            {error.message}
          </p>
        </div>
        <Button size="sm" onClick={this.reset}>
          Try again
        </Button>
      </div>
    )
  }
}
