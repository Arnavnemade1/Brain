import { motion } from 'framer-motion'
import { FlaskConical } from 'lucide-react'

import { cn } from '@/lib/cn'
import { clamp, mapRange } from '@/lib/math'
import { confidenceColor } from '@/lib/palette'
import { METRIC_DESCRIPTORS } from '@/types'
import type { FidelityMetrics, ReconstructionReport } from '@/types'
import { Badge, EmptyState, Panel, Ring, Tooltip } from '@/ui'

/**
 * Measured fidelity, with each number shown against its own baseline and
 * ceiling.
 *
 * A raw SSIM of 0.36 is uninterpretable on its own. Placed on a track between
 * what a trivial baseline scores and what EEG's information content permits,
 * the same number becomes a readable result.
 */
export function FidelityPanel({
  metrics,
  live,
}: {
  metrics: FidelityMetrics | null
  live: boolean
}) {
  if (!metrics) {
    return (
      <Panel label="Fidelity" title="Not yet scored">
        <EmptyState
          icon={<FlaskConical className="size-5" />}
          title={live ? 'Scoring as frames arrive' : 'Awaiting reconstruction'}
          detail="Every metric is computed at native reconstruction resolution against the reference clip, so nothing is flattered by upsampling."
        />
      </Panel>
    )
  }

  return (
    <Panel
      label="Fidelity"
      title="Measured against the reference"
      action={
        <Badge tone={confidenceColor(metrics.composite)}>
          {(metrics.composite * 100).toFixed(0)}% composite
        </Badge>
      }
    >
      <div className="mb-6 flex items-center gap-6">
        <Ring
          value={metrics.composite}
          label="Composite"
          tone={confidenceColor(metrics.composite)}
        />
        <p className="text-[0.78rem] leading-relaxed text-ink-muted">
          Weighted across every measure below. A mean-frame baseline — the same
          average image held for the whole clip — scores around{' '}
          <span className="text-readout text-ink-soft">19%</span> on this composite,
          because it cannot track anything that changes over time.
        </p>
      </div>

      <ul className="space-y-4">
        {METRIC_DESCRIPTORS.map((descriptor) => {
          const raw = metrics[descriptor.key] as number
          // Position on the baseline→ceiling track, which is what makes the
          // number legible.
          const position = clamp(
            mapRange(raw, descriptor.baseline, descriptor.ceiling, 0, 1),
          )
          const display =
            descriptor.unit === 'dB'
              ? `${raw.toFixed(1)} dB`
              : descriptor.unit === 'r'
                ? `r = ${raw >= 0 ? '+' : ''}${raw.toFixed(2)}`
                : raw.toFixed(3)

          return (
            <li key={descriptor.key}>
              <div className="mb-1.5 flex items-baseline justify-between gap-3">
                <Tooltip content={descriptor.description}>
                  <span className="cursor-help border-b border-dotted border-ink/20 text-[0.78rem] text-ink-soft">
                    {descriptor.label}
                  </span>
                </Tooltip>
                <span
                  className="text-readout shrink-0 text-[0.76rem]"
                  style={{ color: confidenceColor(position) }}
                >
                  {display}
                </span>
              </div>

              <div className="relative h-1.5 overflow-hidden rounded-full bg-ink/8">
                <motion.div
                  className="absolute inset-y-0 left-0 rounded-full"
                  initial={false}
                  animate={{ width: `${position * 100}%` }}
                  transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
                  style={{ background: confidenceColor(position) }}
                />
              </div>

              <div className="mt-1 flex justify-between text-[0.6rem] text-ink-faint">
                <span>
                  baseline{' '}
                  {descriptor.unit === 'dB'
                    ? `${descriptor.baseline} dB`
                    : descriptor.baseline.toFixed(2)}
                </span>
                <span>
                  practical ceiling{' '}
                  {descriptor.unit === 'dB'
                    ? `${descriptor.ceiling} dB`
                    : descriptor.ceiling.toFixed(2)}
                </span>
              </div>
            </li>
          )
        })}
      </ul>
    </Panel>
  )
}

/** The closing account of what was and was not recovered. */
export function ReportPanel({ report }: { report: ReconstructionReport | null }) {
  if (!report) return null

  return (
    <Panel label="Reconstruction report" title={report.title}>
      <p className="text-[0.86rem] leading-relaxed text-ink-soft">{report.summary}</p>

      {report.findings.length > 0 && (
        <div className="mt-6">
          <p className="text-label mb-3">Findings</p>
          <ul className="space-y-2.5">
            {report.findings.map((finding) => (
              <li
                key={finding}
                className="flex gap-2.5 text-[0.78rem] leading-relaxed text-ink-muted"
              >
                <span className="mt-[0.55em] size-1 shrink-0 rounded-full bg-signal-good/60" />
                {finding}
              </li>
            ))}
          </ul>
        </div>
      )}

      {report.limitations.length > 0 && (
        <div className={cn('mt-6 border-t border-ink/8 pt-5')}>
          <p className="text-label mb-3">What the signal could not support</p>
          <ul className="space-y-2.5">
            {report.limitations.map((limitation) => (
              <li
                key={limitation}
                className="flex gap-2.5 text-[0.78rem] leading-relaxed text-ink-faint"
              >
                <span className="mt-[0.55em] size-1 shrink-0 rounded-full bg-signal-fair/50" />
                {limitation}
              </li>
            ))}
          </ul>
        </div>
      )}
    </Panel>
  )
}
