import { motion, type HTMLMotionProps } from 'framer-motion'
import type { ReactNode } from 'react'

import { cn } from '@/lib/cn'

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger'
type Size = 'sm' | 'md' | 'lg'

interface ButtonProps extends Omit<HTMLMotionProps<'button'>, 'children'> {
  children: ReactNode
  variant?: Variant
  size?: Size
  icon?: ReactNode
  loading?: boolean
  fullWidth?: boolean
}

const VARIANTS: Record<Variant, string> = {
  primary: cn(
    'bg-neural-400/90 text-abyss font-medium',
    'hover:bg-neural-300 shadow-[0_10px_30px_-12px] shadow-neural-400/60',
  ),
  secondary: cn(
    'glass text-ink hover:border-ink/20',
    'hover:bg-ink/8',
  ),
  ghost: 'text-ink-soft hover:text-ink hover:bg-ink/6',
  danger: 'bg-signal-poor/85 text-abyss font-medium hover:bg-signal-poor',
}

const SIZES: Record<Size, string> = {
  sm: 'h-8 px-3 text-[0.8rem] gap-1.5 rounded-lg',
  md: 'h-10 px-4 text-sm gap-2 rounded-xl',
  lg: 'h-12 px-6 text-[0.95rem] gap-2.5 rounded-xl',
}

export function Button({
  children,
  variant = 'secondary',
  size = 'md',
  icon,
  loading = false,
  fullWidth = false,
  className,
  disabled,
  ...rest
}: ButtonProps) {
  return (
    <motion.button
      whileHover={disabled || loading ? undefined : { y: -1 }}
      whileTap={disabled || loading ? undefined : { y: 0, scale: 0.985 }}
      transition={{ duration: 0.16, ease: [0.4, 0, 0.2, 1] }}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={cn(
        'inline-flex select-none items-center justify-center whitespace-nowrap',
        'transition-colors duration-200',
        'disabled:pointer-events-none disabled:opacity-45',
        VARIANTS[variant],
        SIZES[size],
        fullWidth && 'w-full',
        className,
      )}
      {...rest}
    >
      {loading ? (
        <span
          className="size-3.5 animate-spin rounded-full border-[1.5px] border-current border-t-transparent"
          aria-hidden
        />
      ) : (
        icon
      )}
      {children}
    </motion.button>
  )
}
