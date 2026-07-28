import { useEffect, useRef, useState } from 'react'

import { cn } from '@/lib/cn'

/**
 * Chart primitives for the research page.
 *
 * Hand-built SVG rather than a charting library, for the same reason the
 * decoders are linear: every mark on screen should be traceable to a number,
 * and a library's defaults — rounded domains, auto-hidden ticks, tooltips that
 * interpolate — quietly editorialise data that is the entire point.
 *
 * Two rules hold throughout. Axes start at zero unless the caption says
 * otherwise, so a bar's length means what it looks like it means. And the
 * shuffled control is drawn on the same axis as the result it controls,
 * because a chart that shows only the signal is an argument with its rebuttal
 * cropped out.
 */

/** Reveals a chart once, when it first scrolls into view. */
export function useInView<T extends Element>(threshold = 0.25) {
  const ref = useRef<T | null>(null)
  const [seen, setSeen] = useState(false)

  useEffect(() => {
    const element = ref.current
    if (!element || seen) return

    // Already past it (deep link, restored scroll) counts as seen — an
    // observer alone leaves such elements permanently un-drawn.
    const rect = element.getBoundingClientRect()
    if (rect.top < window.innerHeight * (1 - threshold)) {
      setSeen(true)
      return
    }

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setSeen(true)
          observer.disconnect()
        }
      },
      { threshold },
    )
    observer.observe(element)
    return () => observer.disconnect()
  }, [seen, threshold])

  return { ref, seen }
}

export function Figure({
  label,
  title,
  caption,
  children,
  className,
}: {
  label: string
  title: string
  caption: string
  children: React.ReactNode
  className?: string
}) {
  return (
    <figure className={cn('glass rounded-panel m-0 p-6 sm:p-8', className)}>
      <p className="text-label mb-3 text-neural-300">{label}</p>
      <h3 className="mb-2 text-[1.05rem] leading-snug text-ink">{title}</h3>
      <p className="mb-7 max-w-2xl text-[0.82rem] leading-relaxed text-ink-faint">
        {caption}
      </p>
      {/* Charts scroll rather than squash: a compressed axis lies. */}
      <div className="-mx-2 overflow-x-auto px-2">{children}</div>
    </figure>
  )
}

export interface SeriesPoint {
  x: number
  y: number
}

/**
 * Line chart with a threshold rule and an optional control series.
 *
 * The x axis is categorical-but-ordered (grid resolutions), so points are
 * evenly spaced and labelled with their real values rather than positioned by
 * them — spacing 1x1 to 12x12 linearly would bunch five of the seven readings
 * into the first fifth of the axis.
 */
export function LineChart({
  series,
  control,
  labels,
  threshold,
  thresholdLabel,
  yMax = 0.7,
  height = 260,
  animate = true,
}: {
  series: number[]
  control?: number[]
  labels: string[]
  threshold?: number
  thresholdLabel?: string
  yMax?: number
  height?: number
  animate?: boolean
}) {
  const { ref, seen } = useInView<HTMLDivElement>()

  const width = 720
  const pad = { top: 18, right: 26, bottom: 40, left: 46 }
  const plotW = width - pad.left - pad.right
  const plotH = height - pad.top - pad.bottom

  const x = (i: number) =>
    pad.left + (series.length === 1 ? plotW / 2 : (i / (series.length - 1)) * plotW)
  const y = (v: number) => pad.top + plotH - (v / yMax) * plotH

  const path = series.map((v, i) => `${i === 0 ? 'M' : 'L'} ${x(i)} ${y(v)}`).join(' ')
  const controlPath = control
    ? control.map((v, i) => `${i === 0 ? 'M' : 'L'} ${x(i)} ${y(v)}`).join(' ')
    : null

  const ticks = [0, 0.2, 0.4, 0.6].filter((t) => t <= yMax)

  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full min-w-[540px]" role="img">
        {ticks.map((t) => (
          <g key={t}>
            <line
              x1={pad.left}
              x2={width - pad.right}
              y1={y(t)}
              y2={y(t)}
              stroke="currentColor"
              className="text-ink/8"
            />
            <text
              x={pad.left - 10}
              y={y(t) + 4}
              textAnchor="end"
              className="fill-current text-[11px] text-ink-faint"
            >
              {t.toFixed(1)}
            </text>
          </g>
        ))}

        {threshold != null && (
          <>
            <line
              x1={pad.left}
              x2={width - pad.right}
              y1={y(threshold)}
              y2={y(threshold)}
              stroke="currentColor"
              strokeDasharray="4 4"
              className="text-memory-400/70"
            />
            {thresholdLabel && (
              <text
                x={width - pad.right}
                y={y(threshold) - 7}
                textAnchor="end"
                className="fill-current text-[11px] text-memory-300"
              >
                {thresholdLabel}
              </text>
            )}
          </>
        )}

        {controlPath && (
          <path
            d={controlPath}
            fill="none"
            stroke="currentColor"
            strokeWidth={1.5}
            strokeDasharray="3 3"
            className="text-ink-faint/70"
          />
        )}

        <path
          d={path}
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
          strokeLinecap="round"
          className="text-neural-300"
          style={
            animate
              ? {
                  strokeDasharray: 2000,
                  strokeDashoffset: seen ? 0 : 2000,
                  transition: 'stroke-dashoffset 1.4s cubic-bezier(0.22,1,0.36,1)',
                }
              : undefined
          }
        />

        {series.map((v, i) => (
          <circle
            key={i}
            cx={x(i)}
            cy={y(v)}
            r={3.5}
            className="fill-current text-neural-300"
            style={{
              opacity: seen ? 1 : 0,
              transition: `opacity 400ms ease ${300 + i * 90}ms`,
            }}
          />
        ))}

        {labels.map((text, i) => (
          <text
            key={text}
            x={x(i)}
            y={height - 14}
            textAnchor="middle"
            className="fill-current text-[11px] text-ink-faint"
          >
            {text}
          </text>
        ))}
      </svg>
    </div>
  )
}

/**
 * Heatmap of per-cell correlation across a square grid.
 *
 * Sequential colour, not diverging: these are correlations that never go
 * negative in this data, and a diverging ramp would imply a meaningful
 * midpoint that does not exist.
 */
export function HeatGrid({
  values,
  side,
  vMin = 0,
  vMax = 0.7,
}: {
  values: number[]
  side: number
  vMin?: number
  vMax?: number
}) {
  const { ref, seen } = useInView<HTMLDivElement>()
  const cell = 34
  const gap = 3
  const size = side * cell + (side - 1) * gap

  return (
    <div ref={ref} className="flex flex-wrap items-end gap-8">
      <svg
        viewBox={`0 0 ${size} ${size}`}
        className="h-auto w-[min(320px,100%)]"
        role="img"
        aria-label={`Per-cell decoding correlation over a ${side} by ${side} grid`}
      >
        {values.map((v, i) => {
          const row = Math.floor(i / side)
          const col = i % side
          const t = Math.max(0, Math.min(1, (v - vMin) / (vMax - vMin)))
          return (
            <rect
              key={i}
              x={col * (cell + gap)}
              y={row * (cell + gap)}
              width={cell}
              height={cell}
              rx={3}
              fill={`color-mix(in oklab, var(--color-neural-300) ${(t * 100).toFixed(0)}%, var(--color-abyss))`}
              style={{
                opacity: seen ? 1 : 0,
                transition: `opacity 500ms ease ${(row + col) * 45}ms`,
              }}
            />
          )
        })}
      </svg>

      <div className="min-w-[9rem]">
        <p className="text-label mb-3">Correlation</p>
        <div
          className="h-2 w-full rounded-full"
          style={{
            background:
              'linear-gradient(90deg, var(--color-abyss), var(--color-neural-300))',
          }}
        />
        <div className="mt-1.5 flex justify-between text-[0.7rem] text-ink-faint">
          <span>{vMin.toFixed(1)}</span>
          <span>{vMax.toFixed(1)}</span>
        </div>
      </div>
    </div>
  )
}

/**
 * Paired bars: how much variance a component holds, and whether it decodes.
 *
 * The two quantities share an x position but not a scale, so they are drawn as
 * separate rows rather than stacked — overlaying them on one axis would invite
 * reading a ratio between numbers that have no ratio.
 */
export function ComponentBars({
  components,
  threshold,
  height = 300,
}: {
  components: { index: number; variance: number; r: number }[]
  threshold: number
  height?: number
}) {
  const { ref, seen } = useInView<HTMLDivElement>()

  const width = 720
  const pad = { top: 22, right: 20, bottom: 46, left: 46 }
  const plotW = width - pad.left - pad.right
  const rowH = (height - pad.top - pad.bottom) / 2 - 16

  const slot = plotW / components.length
  const barW = Math.min(30, slot * 0.56)
  const maxVariance = Math.max(...components.map((c) => c.variance))
  const rMax = 0.7

  return (
    <div ref={ref}>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full min-w-[560px]" role="img">
        <text x={pad.left} y={14} className="fill-current text-[11px] text-ink-faint">
          Share of grid variance
        </text>

        {components.map((c, i) => {
          const h = (c.variance / maxVariance) * rowH
          const cx = pad.left + slot * i + slot / 2
          return (
            <rect
              key={`v${c.index}`}
              x={cx - barW / 2}
              y={pad.top + rowH - h}
              width={barW}
              height={seen ? h : 0}
              rx={2}
              className="fill-current text-ink/20"
              style={{
                transition: `height 700ms cubic-bezier(0.22,1,0.36,1) ${i * 45}ms, y 700ms cubic-bezier(0.22,1,0.36,1) ${i * 45}ms`,
                ...(seen ? {} : { y: pad.top + rowH }),
              }}
            />
          )
        })}

        <text
          x={pad.left}
          y={pad.top + rowH + 34}
          className="fill-current text-[11px] text-ink-faint"
        >
          Decoding correlation
        </text>

        {(() => {
          const base = pad.top + rowH + 44 + rowH
          return (
            <>
              <line
                x1={pad.left}
                x2={width - pad.right}
                y1={base - (threshold / rMax) * rowH}
                y2={base - (threshold / rMax) * rowH}
                stroke="currentColor"
                strokeDasharray="4 4"
                className="text-memory-400/70"
              />
              <text
                x={width - pad.right}
                y={base - (threshold / rMax) * rowH - 6}
                textAnchor="end"
                className="fill-current text-[11px] text-memory-300"
              >
                r = {threshold}
              </text>

              {components.map((c, i) => {
                const h = (c.r / rMax) * rowH
                const cx = pad.left + slot * i + slot / 2
                const passes = c.r > threshold
                return (
                  <g key={`r${c.index}`}>
                    <rect
                      x={cx - barW / 2}
                      y={base - (seen ? h : 0)}
                      width={barW}
                      height={seen ? h : 0}
                      rx={2}
                      className={cn(
                        'fill-current',
                        passes ? 'text-neural-300' : 'text-ink/18',
                      )}
                      style={{
                        transition: `height 700ms cubic-bezier(0.22,1,0.36,1) ${i * 45}ms, y 700ms cubic-bezier(0.22,1,0.36,1) ${i * 45}ms`,
                      }}
                    />
                    <text
                      x={cx}
                      y={height - 16}
                      textAnchor="middle"
                      className={cn(
                        'fill-current text-[11px]',
                        passes ? 'text-ink-soft' : 'text-ink-faint',
                      )}
                    >
                      {c.index}
                    </text>
                  </g>
                )
              })}
            </>
          )
        })()}
      </svg>
    </div>
  )
}
