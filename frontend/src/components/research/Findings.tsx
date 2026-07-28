import { cn } from '@/lib/cn'
import type { Finding, Status } from '@/data/findings'

import { useInView } from './Chart'

/**
 * Results as tracks, each with its own chance level marked.
 *
 * Every measurement in this project has a different baseline — 50% for a
 * two-way decision, 20% for five classes, 0.5% for retrieval over 200
 * photographs, and a shuffled correlation for the regressions. Putting them on
 * a shared axis without that baseline drawn would be the single most
 * misleading thing this page could do: 36.7% is a strong result against 20%
 * and a failure against 50%.
 *
 * So chance is a tick on every track, and the region below it is shaded. A bar
 * that stops short of the tick has visibly failed, which is exactly what the
 * emotion results should look like.
 */

const TONE: Record<Status, { bar: string; text: string; label: string }> = {
  works: { bar: 'bg-neural-300', text: 'text-neural-300', label: 'recovered' },
  weak: { bar: 'bg-ink/45', text: 'text-ink-soft', label: 'weak' },
  fails: { bar: 'bg-rose-400/70', text: 'text-rose-300', label: 'at or below chance' },
}

export function ResultLadder({ findings }: { findings: Finding[] }) {
  const { ref, seen } = useInView<HTMLDivElement>(0.15)

  return (
    <div ref={ref} className="space-y-3.5">
      {findings.map((finding, index) => {
        const tone = TONE[finding.status]
        const value = Math.max(0, Math.min(1, finding.value))
        const chance = Math.max(0, Math.min(1, finding.chance))

        return (
          <div key={finding.label}>
            <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-0.5">
              <span className="text-[0.86rem] text-ink">{finding.label}</span>
              <span className="flex items-baseline gap-2">
                <span className={cn('text-readout text-[0.82rem]', tone.text)}>
                  {finding.unit === 'r'
                    ? `r = ${finding.value.toFixed(2)}`
                    : `${(finding.value * 100).toFixed(1)}%`}
                </span>
                <span className="text-[0.7rem] text-ink-faint">
                  {finding.unit === 'r'
                    ? `shuffled ${finding.chance.toFixed(2)}`
                    : `chance ${(finding.chance * 100).toFixed(1)}%`}
                </span>
              </span>
            </div>

            <div className="relative h-2 w-full overflow-hidden rounded-full bg-ink/8">
              {/* Everything below chance is inert territory, shaded so the
                  eye reads the bar's useful length rather than its total. */}
              <div
                className="absolute inset-y-0 left-0 bg-ink/10"
                style={{ width: `${chance * 100}%` }}
              />
              <div
                className={cn('absolute inset-y-0 left-0 rounded-full', tone.bar)}
                style={{
                  width: seen ? `${value * 100}%` : '0%',
                  transition: `width 900ms cubic-bezier(0.22,1,0.36,1) ${index * 70}ms`,
                }}
              />
              <div
                className="absolute inset-y-0 w-px bg-ink/55"
                style={{ left: `${chance * 100}%` }}
                aria-hidden
              />
            </div>

            {finding.note && (
              <p className="mt-1.5 text-[0.72rem] leading-relaxed text-ink-faint">
                {finding.note}
              </p>
            )}
          </div>
        )
      })}
    </div>
  )
}

/**
 * Bars against a single shared chance line, for the artifact controls.
 *
 * These two tests are the reason the motor result can be called cortical at
 * all, so they get their own chart rather than a sentence. If muscle were
 * driving the decode, the high-frequency bars would be the tall ones and the
 * jaw site would not sit at chance.
 */
export function ControlBars({
  rows,
  chance,
  chanceLabel = 'chance',
  highlight,
}: {
  rows: { name: string; value: number; muted?: boolean }[]
  chance: number
  chanceLabel?: string
  highlight?: string
}) {
  const { ref, seen } = useInView<HTMLDivElement>(0.2)
  const max = Math.max(...rows.map((r) => r.value), chance) * 1.15

  return (
    <div ref={ref} className="space-y-3">
      {rows.map((row, index) => {
        const atChance = row.value <= chance * 1.15
        return (
          <div key={row.name}>
            <div className="mb-1.5 flex items-baseline justify-between gap-4">
              <span
                className={cn(
                  'text-[0.84rem]',
                  row.name === highlight ? 'text-ink' : 'text-ink-soft',
                )}
              >
                {row.name}
              </span>
              <span
                className={cn(
                  'text-readout text-[0.8rem]',
                  atChance ? 'text-ink-faint' : 'text-neural-300',
                )}
              >
                {(row.value * 100).toFixed(1)}%
              </span>
            </div>
            <div className="relative h-2 rounded-full bg-ink/8">
              <div
                className={cn(
                  'absolute inset-y-0 left-0 rounded-full',
                  atChance ? 'bg-ink/25' : 'bg-neural-300',
                )}
                style={{
                  width: seen ? `${(row.value / max) * 100}%` : '0%',
                  transition: `width 850ms cubic-bezier(0.22,1,0.36,1) ${index * 80}ms`,
                }}
              />
              <div
                className="absolute inset-y-[-4px] w-px bg-memory-400/80"
                style={{ left: `${(chance / max) * 100}%` }}
                aria-hidden
              />
            </div>
          </div>
        )
      })}
      <p className="text-[0.72rem] text-ink-faint">
        <span className="text-memory-300">│</span> {chanceLabel}{' '}
        {(chance * 100).toFixed(0)}%
      </p>
    </div>
  )
}
