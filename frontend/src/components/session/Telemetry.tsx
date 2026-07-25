import { motion } from 'framer-motion'
import { AlertTriangle } from 'lucide-react'

import { BandBars } from '@/components/viz/BandBars'
import { SpectrumChart } from '@/components/viz/SpectrumChart'
import { Topography } from '@/components/viz/Topography'
import { WaveformTrace } from '@/components/viz/WaveformTrace'
import { cn } from '@/lib/cn'
import { clock, titleize } from '@/lib/format'
import { confidenceColor, QUALITY_COLOR } from '@/lib/palette'
import type { ElectrodeInfo } from '@/services/api'
import type { LatentPoint, NeuralFrame } from '@/types'
import type { SyncInfo } from '@/stores/sessionStore'
import { Badge, Meter, Panel, Stat, Tooltip } from '@/ui'

/* -------------------------------------------------------------------------- */
/* Signal quality                                                             */
/* -------------------------------------------------------------------------- */

export function SignalQualityPanel({ frame }: { frame: NeuralFrame | null }) {
  const quality = frame?.signalQuality
  const tone = quality ? QUALITY_COLOR[quality.grade] : 'var(--color-ink-faint)'

  return (
    <Panel
      label="Signal quality"
      title={quality ? titleize(quality.grade) : 'Awaiting signal'}
      action={quality && <Badge tone={tone}>{(quality.score * 100).toFixed(0)}%</Badge>}
    >
      <Meter value={quality?.score ?? 0} tone={tone} showValue={false} />

      <dl className="mt-5 grid grid-cols-2 gap-4">
        <Stat
          label="Artifact load"
          value={quality ? (quality.artifactRatio * 100).toFixed(2) : '—'}
          unit="%"
          hint="samples attenuated"
        />
        <Stat
          label="Line noise"
          value={quality ? (quality.lineNoise * 100).toFixed(0) : '—'}
          unit="%"
          hint="residual after notch"
        />
      </dl>

      {quality && quality.droppedChannels.length > 0 && (
        <div className="mt-4 flex items-start gap-2 rounded-lg border border-signal-fair/25 bg-signal-fair/8 px-3 py-2.5">
          <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-signal-fair" />
          <p className="text-[0.72rem] leading-relaxed text-ink-muted">
            {quality.droppedChannels.length} channel
            {quality.droppedChannels.length > 1 ? 's' : ''} excluded:{' '}
            <span className="text-readout text-ink-soft">
              {quality.droppedChannels.join(', ')}
            </span>
          </p>
        </div>
      )}
    </Panel>
  )
}

/* -------------------------------------------------------------------------- */
/* Synchronization                                                            */
/* -------------------------------------------------------------------------- */

export function SyncPanel({ sync }: { sync: SyncInfo | null }) {
  const usable = sync ? sync.markersMatched >= 2 && sync.residualMs < 120 : false

  return (
    <Panel
      label="Signal synchronization"
      title={sync ? (usable ? 'Aligned' : 'Poorly aligned') : 'Not yet aligned'}
      action={
        sync && (
          <Badge
            tone={usable ? 'var(--color-signal-good)' : 'var(--color-signal-poor)'}
          >
            {sync.markersMatched}/{sync.markersExpected} markers
          </Badge>
        )
      }
    >
      {sync ? (
        <>
          <dl className="grid grid-cols-2 gap-4">
            <Stat
              label="Clock offset"
              value={sync.offsetSeconds.toFixed(3)}
              unit="s"
              hint="recording vs playback"
            />
            <Stat
              label="Residual error"
              value={sync.residualMs.toFixed(1)}
              unit="ms"
              hint="after correction"
              tone={usable ? undefined : 'var(--color-signal-poor)'}
            />
          </dl>
          <p className="mt-4 text-[0.72rem] leading-relaxed text-ink-faint">
            {sync.driftPpm === 0
              ? 'Clock drift was not estimable from this marker geometry, so offset alone was fitted. Over a clip this short the resulting misalignment is a few milliseconds — far below one analysis window.'
              : `Drift fitted at ${sync.driftPpm.toFixed(0)} ppm across the marker span.`}
          </p>
        </>
      ) : (
        <p className="text-[0.78rem] text-ink-faint">
          Alignment is fitted once, from event markers written at playback start and at
          each scene transition.
        </p>
      )}
    </Panel>
  )
}

/* -------------------------------------------------------------------------- */
/* Visual response                                                            */
/* -------------------------------------------------------------------------- */

const RESPONSE_ROWS: {
  key: keyof NonNullable<NeuralFrame['visual']>
  label: string
  hint: string
  signed?: boolean
}[] = [
  {
    key: 'occipitalDrive',
    label: 'Occipital drive',
    hint: 'Broadband response over V1 — tracks overall brightness',
  },
  {
    key: 'alphaSuppression',
    label: 'Alpha suppression',
    hint: 'Posterior alpha falls as visual engagement rises',
  },
  {
    key: 'motionResponse',
    label: 'Motion response',
    hint: 'Beta/gamma over lateral occipito-temporal sites',
  },
  {
    key: 'transient',
    label: 'Evoked transient',
    hint: 'Time-locked deflection — spikes at scene changes',
  },
  {
    key: 'detailResponse',
    label: 'Detail response',
    hint: 'Gamma activity, tracks contrast and spatial detail',
  },
  {
    key: 'lateralBalance',
    label: 'Lateral balance',
    hint: 'Left/right hemisphere asymmetry — coarse horizontal layout',
    signed: true,
  },
  {
    key: 'verticalBalance',
    label: 'Vertical balance',
    hint: 'Dorsal/ventral balance — coarse vertical layout',
    signed: true,
  },
]

/**
 * The encoder's output: the quantities the reconstruction is actually built
 * from. Everything the system claims about the video passes through these
 * seven numbers.
 */
export function VisualResponsePanel({ frame }: { frame: NeuralFrame | null }) {
  const visual = frame?.visual

  return (
    <Panel
      label="Temporal neural encoder"
      title="Visual response"
      action={
        frame && (
          <Badge tone={confidenceColor(frame.confidence)}>
            {(frame.confidence * 100).toFixed(0)}% usable
          </Badge>
        )
      }
    >
      <ul className="space-y-3">
        {RESPONSE_ROWS.map((row) => {
          const raw = visual ? (visual[row.key] as number) : 0
          // Signed measures are centred, so the bar grows from the middle.
          const magnitude = row.signed ? Math.abs(raw) : raw
          const tone = row.signed
            ? raw >= 0
              ? 'var(--color-neural-400)'
              : 'var(--color-memory-400)'
            : 'var(--color-neural-400)'

          return (
            <li key={row.key}>
              <div className="mb-1 flex items-baseline justify-between gap-2">
                <Tooltip content={row.hint}>
                  <span className="cursor-help border-b border-dotted border-ink/20 text-[0.75rem] text-ink-soft">
                    {row.label}
                  </span>
                </Tooltip>
                <span className="text-readout text-[0.7rem] text-ink-muted">
                  {row.signed && raw >= 0 ? '+' : ''}
                  {raw.toFixed(2)}
                </span>
              </div>

              {row.signed ? (
                <div className="relative h-1 rounded-full bg-ink/6">
                  <span className="absolute inset-y-0 left-1/2 w-px bg-ink/15" />
                  <motion.div
                    className="absolute inset-y-0 rounded-full"
                    initial={false}
                    animate={{
                      width: `${(magnitude / 2) * 100}%`,
                      left: raw >= 0 ? '50%' : `${50 - (magnitude / 2) * 100}%`,
                    }}
                    transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
                    style={{ background: tone }}
                  />
                </div>
              ) : (
                <div className="h-1 overflow-hidden rounded-full bg-ink/6">
                  <motion.div
                    className="h-full rounded-full"
                    initial={false}
                    animate={{ width: `${magnitude * 100}%` }}
                    transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
                    style={{ background: tone, boxShadow: `0 0 8px -2px ${tone}` }}
                  />
                </div>
              )}
            </li>
          )
        })}
      </ul>
    </Panel>
  )
}

/* -------------------------------------------------------------------------- */
/* Latent space                                                               */
/* -------------------------------------------------------------------------- */

/**
 * The latent trajectory as a heat strip: dimensions down, time across.
 *
 * Shows the representation the reconstruction reads, and makes structure
 * visible — a scene change appears as a vertical discontinuity across
 * several dimensions at once.
 */
export function LatentPanel({ latents }: { latents: LatentPoint[] }) {
  const recent = latents.slice(-80)
  const dimensions = recent[0]?.vector.length ?? 16

  return (
    <Panel
      label="Neural memory latent space"
      title={`${dimensions}-dimensional trajectory`}
      action={<Badge>{latents.length} points</Badge>}
    >
      {recent.length === 0 ? (
        <p className="text-[0.78rem] text-ink-faint">
          The latent is the interface between measurement and reconstruction. Nothing
          downstream reads the raw signal.
        </p>
      ) : (
        <>
          <div
            className="flex gap-px overflow-hidden rounded"
            style={{ height: dimensions * 6 }}
          >
            {recent.map((point) => (
              <div key={point.t} className="flex min-w-0 flex-1 flex-col gap-px">
                {point.vector.map((value, index) => {
                  // Latent values are roughly centred; map to a diverging ramp.
                  const magnitude = Math.min(1, Math.abs(value))
                  const positive = value >= 0
                  return (
                    <div
                      key={index}
                      className="flex-1"
                      style={{
                        background: positive
                          ? `color-mix(in oklab, var(--color-neural-400) ${magnitude * 100}%, transparent)`
                          : `color-mix(in oklab, var(--color-memory-400) ${magnitude * 100}%, transparent)`,
                      }}
                    />
                  )
                })}
              </div>
            ))}
          </div>
          <p className="mt-3 text-[0.68rem] leading-relaxed text-ink-faint">
            Time runs left to right, one column per analysis window. Blue is positive,
            amber negative. Vertical banding marks a scene boundary.
          </p>
        </>
      )}
    </Panel>
  )
}

/* -------------------------------------------------------------------------- */
/* Spectrum, topography, waveform                                             */
/* -------------------------------------------------------------------------- */

export function SpectrumPanel({ frame }: { frame: NeuralFrame | null }) {
  return (
    <Panel
      label="Brainwave spectrum"
      title="Power spectral density"
      action={
        frame && (
          <Badge tone="var(--color-memory-300)">
            peak {frame.spectrum.peakFrequency.toFixed(1)} Hz
          </Badge>
        )
      }
      flush
    >
      <div className="px-5 pb-2">
        <SpectrumChart spectrum={frame?.spectrum ?? null} height={140} />
      </div>
      <div className="px-5 pb-5">
        <BandBars bands={frame?.bandPowers ?? null} />
      </div>
    </Panel>
  )
}

export function TopographyPanel({
  frame,
  electrodes,
}: {
  frame: NeuralFrame | null
  electrodes: ElectrodeInfo[]
}) {
  return (
    <Panel label="Neural activity" title="Scalp topography">
      <div className="grid place-items-center py-2">
        <Topography topography={frame?.topography ?? {}} positions={electrodes} size={188} />
      </div>
      <p className="mt-3 text-center text-[0.68rem] leading-relaxed text-ink-faint">
        {electrodes.length} electrodes, weighted posterior — the visual hierarchy is
        occipital, and that is where the recoverable signal is.
      </p>
    </Panel>
  )
}

export function WaveformPanel({
  frame,
  elapsed,
  duration,
}: {
  frame: NeuralFrame | null
  elapsed: number
  duration: number | null
}) {
  return (
    <Panel
      label="Acquisition"
      title={`${clock(elapsed)}${duration ? ` / ${clock(duration)}` : ''}`}
      action={frame && <Badge tone="var(--color-neural-300)">window {frame.index + 1}</Badge>}
      flush
    >
      <div className="px-5 pb-5">
        <WaveformTrace waveform={frame?.waveform ?? []} height={64} />
        <p className="mt-2 text-[0.66rem] text-ink-faint">
          Montage average after artifact removal, on the stimulus timeline.
        </p>
      </div>
    </Panel>
  )
}

/* -------------------------------------------------------------------------- */
/* Status strip                                                               */
/* -------------------------------------------------------------------------- */

export function StatusStrip({
  frame,
  fidelity,
  className,
}: {
  frame: NeuralFrame | null
  fidelity: number
  className?: string
}) {
  const items = [
    {
      label: 'Signal',
      value: frame ? titleize(frame.signalQuality.grade) : '—',
      tone: frame ? QUALITY_COLOR[frame.signalQuality.grade] : undefined,
    },
    {
      label: 'Drive',
      value: frame ? frame.visual.occipitalDrive.toFixed(2) : '—',
    },
    {
      label: 'Motion',
      value: frame ? frame.visual.motionResponse.toFixed(2) : '—',
    },
    {
      label: 'Fidelity',
      value: `${(fidelity * 100).toFixed(0)}%`,
      tone: confidenceColor(fidelity),
    },
  ]

  return (
    <div className={cn('flex flex-wrap items-center gap-x-7 gap-y-3', className)}>
      {items.map((item) => (
        <div key={item.label}>
          <p className="text-label mb-1">{item.label}</p>
          <p
            className="text-readout text-[0.82rem]"
            style={item.tone ? { color: item.tone } : undefined}
          >
            {item.value}
          </p>
        </div>
      ))}
    </div>
  )
}
