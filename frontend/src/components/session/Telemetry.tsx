import { motion } from 'framer-motion'
import { AlertTriangle } from 'lucide-react'

import { BandBars } from '@/components/viz/BandBars'
import { ConfidenceTrend } from '@/components/viz/ConfidenceTrend'
import { EmotionField } from '@/components/viz/EmotionField'
import { SpectrumChart } from '@/components/viz/SpectrumChart'
import { Topography } from '@/components/viz/Topography'
import { WaveformTrace } from '@/components/viz/WaveformTrace'
import { cn } from '@/lib/cn'
import { clock, titleize } from '@/lib/format'
import { COGNITIVE_COLOR, EMOTION_COLOR, QUALITY_COLOR } from '@/lib/palette'
import type { ElectrodeInfo } from '@/services/api'
import type { NeuralFrame } from '@/types'
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
      action={
        quality && (
          <Badge tone={tone}>{(quality.score * 100).toFixed(0)}%</Badge>
        )
      }
    >
      <Meter value={quality?.score ?? 0} tone={tone} showValue={false} />

      <dl className="mt-5 grid grid-cols-2 gap-4">
        <Stat
          label="Artifact load"
          value={quality ? `${(quality.artifactRatio * 100).toFixed(2)}` : '—'}
          unit="%"
          hint="samples compressed"
        />
        <Stat
          label="Line noise"
          value={quality ? `${(quality.lineNoise * 100).toFixed(0)}` : '—'}
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
/* Cognitive state                                                            */
/* -------------------------------------------------------------------------- */

export function CognitivePanel({ frame }: { frame: NeuralFrame | null }) {
  const cognitive = frame?.cognitive
  const tone = cognitive ? COGNITIVE_COLOR[cognitive.state] : 'var(--color-ink-faint)'

  const ranked = cognitive
    ? Object.entries(cognitive.distribution)
        .sort(([, a], [, b]) => (b ?? 0) - (a ?? 0))
        .slice(0, 4)
    : []

  return (
    <Panel
      label="Cognitive state"
      title={cognitive ? titleize(cognitive.state) : 'Unclassified'}
      action={
        cognitive && (
          <Tooltip content="Winning margin over the runner-up, not the raw probability">
            <Badge tone={tone}>{(cognitive.confidence * 100).toFixed(0)}% conf</Badge>
          </Tooltip>
        )
      }
    >
      <ul className="space-y-2">
        {ranked.map(([state, probability]) => (
          <li key={state} className="grid grid-cols-[5.2rem_1fr_2.4rem] items-center gap-3">
            <span className="truncate text-[0.75rem] capitalize text-ink-soft">{state}</span>
            <div className="h-1 overflow-hidden rounded-full bg-ink/6">
              <motion.div
                className="h-full rounded-full"
                initial={false}
                animate={{ width: `${(probability ?? 0) * 100}%` }}
                transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
                style={{
                  background: COGNITIVE_COLOR[state as keyof typeof COGNITIVE_COLOR],
                }}
              />
            </div>
            <span className="text-readout text-right text-[0.7rem] text-ink-muted">
              {((probability ?? 0) * 100).toFixed(0)}%
            </span>
          </li>
        ))}
        {ranked.length === 0 && (
          <li className="text-[0.75rem] text-ink-faint">Awaiting classification.</li>
        )}
      </ul>

      <dl className="mt-5 grid grid-cols-2 gap-4 border-t border-ink/8 pt-4">
        <Stat
          label="Engagement"
          value={cognitive ? (cognitive.engagement * 100).toFixed(0) : '—'}
          unit="%"
          hint="β / (α + θ)"
        />
        <Stat
          label="Cognitive load"
          value={cognitive ? (cognitive.load * 100).toFixed(0) : '—'}
          unit="%"
          hint="θ / β"
        />
      </dl>
    </Panel>
  )
}

/* -------------------------------------------------------------------------- */
/* Emotion                                                                    */
/* -------------------------------------------------------------------------- */

export function EmotionPanel({ frame }: { frame: NeuralFrame | null }) {
  const emotion = frame?.emotion
  const tone = emotion ? EMOTION_COLOR[emotion.primary] : 'var(--color-ink-faint)'

  return (
    <Panel
      label="Emotional state"
      title={emotion ? titleize(emotion.primary) : 'Unresolved'}
      action={
        emotion?.secondary && (
          <Badge tone={EMOTION_COLOR[emotion.secondary]}>
            + {emotion.secondary}
          </Badge>
        )
      }
    >
      <div className="flex flex-col items-center gap-6 sm:flex-row sm:items-start">
        <div className="shrink-0 px-5 pt-4">
          <EmotionField emotion={emotion ?? null} size={176} />
        </div>

        <dl className="grid w-full grid-cols-2 gap-4 sm:grid-cols-1">
          <Stat
            label="Valence"
            value={emotion ? (emotion.valence >= 0 ? '+' : '') + emotion.valence.toFixed(2) : '—'}
            tone={tone}
            hint="frontal alpha asymmetry"
          />
          <Stat
            label="Arousal"
            value={emotion ? emotion.arousal.toFixed(2) : '—'}
            hint="fast-band share"
          />
          <Stat
            label="Intensity"
            value={emotion ? (emotion.intensity * 100).toFixed(0) : '—'}
            unit="%"
            hint="affective activation"
          />
        </dl>
      </div>
    </Panel>
  )
}

/* -------------------------------------------------------------------------- */
/* Spectrum                                                                   */
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
        <SpectrumChart spectrum={frame?.spectrum ?? null} height={148} />
      </div>
      <div className="px-5 pb-5">
        <BandBars bands={frame?.bandPowers ?? null} />
      </div>
    </Panel>
  )
}

/* -------------------------------------------------------------------------- */
/* Topography                                                                 */
/* -------------------------------------------------------------------------- */

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
        <Topography
          topography={frame?.topography ?? {}}
          positions={electrodes}
          size={188}
        />
      </div>
      <p className="mt-3 text-center text-[0.68rem] text-ink-faint">
        Inverse-distance interpolation across {electrodes.length} electrodes
      </p>
    </Panel>
  )
}

/* -------------------------------------------------------------------------- */
/* Waveform + trend                                                           */
/* -------------------------------------------------------------------------- */

export function WaveformPanel({
  frame,
  frames,
  elapsed,
  duration,
}: {
  frame: NeuralFrame | null
  frames: NeuralFrame[]
  elapsed: number
  duration: number | null
}) {
  return (
    <Panel
      label="Session timeline"
      title={`${clock(elapsed)}${duration ? ` / ${clock(duration)}` : ''}`}
      action={
        frame && (
          <Badge tone="var(--color-neural-300)">window {frame.index + 1}</Badge>
        )
      }
      flush
    >
      <div className="px-5">
        <WaveformTrace waveform={frame?.waveform ?? []} height={64} />
      </div>
      <div className="px-5 pt-3 pb-5">
        <ConfidenceTrend frames={frames} height={88} />
      </div>
    </Panel>
  )
}

/* -------------------------------------------------------------------------- */
/* Compact status strip                                                       */
/* -------------------------------------------------------------------------- */

export function StatusStrip({
  frame,
  className,
}: {
  frame: NeuralFrame | null
  className?: string
}) {
  const items = [
    {
      label: 'Signal',
      value: frame ? titleize(frame.signalQuality.grade) : '—',
      tone: frame ? QUALITY_COLOR[frame.signalQuality.grade] : undefined,
    },
    {
      label: 'State',
      value: frame ? titleize(frame.cognitive.state) : '—',
      tone: frame ? COGNITIVE_COLOR[frame.cognitive.state] : undefined,
    },
    {
      label: 'Affect',
      value: frame ? titleize(frame.emotion.primary) : '—',
      tone: frame ? EMOTION_COLOR[frame.emotion.primary] : undefined,
    },
    {
      label: 'Peak',
      value: frame ? `${frame.spectrum.peakFrequency.toFixed(1)} Hz` : '—',
    },
  ]

  return (
    <div className={cn('flex flex-wrap items-center gap-x-7 gap-y-3', className)}>
      {items.map((item) => (
        <div key={item.label}>
          <p className="text-label mb-1">{item.label}</p>
          <p
            className="text-[0.82rem]"
            style={item.tone ? { color: item.tone } : undefined}
          >
            {item.value}
          </p>
        </div>
      ))}
    </div>
  )
}
