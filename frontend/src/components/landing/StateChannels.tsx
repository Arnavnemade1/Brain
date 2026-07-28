import { cn } from '@/lib/cn'

/**
 * The bindings between brain signal and world, split by how much they claim.
 *
 * This table is the honest core of the terrain reconstruction, so it is shown
 * rather than summarised. A channel marked *measured* is a direct reading off
 * the recording with no inference in it; a channel marked *interpretation*
 * assumes a cognitive meaning on top of a measurement, and every one of those
 * assumptions is contested somewhere in the literature.
 *
 * The two are deliberately given different colours and are never mixed in one
 * list, because the whole difference between this and a mood visualiser is
 * whether a viewer can tell which is which.
 */

export interface StateChannel {
  signal: string
  drives: string
  status: 'measured' | 'interpretation'
}

/** Mirrors `CHANNELS` in backend/app/scene/cognitive.py. */
export const STATE_CHANNELS: StateChannel[] = [
  { signal: 'Signal intensity', drives: 'how vivid the scene is', status: 'measured' },
  { signal: 'Gamma power', drives: 'sharpness of detail', status: 'measured' },
  { signal: 'Posterior alpha', drives: 'how far the world recedes', status: 'interpretation' },
  { signal: 'Alpha suppression', drives: 'clarity of the air', status: 'interpretation' },
  { signal: 'Beta / alpha', drives: 'colour, and speed of travel', status: 'interpretation' },
  { signal: 'Frontal theta', drives: 'how bare the ground is', status: 'interpretation' },
  { signal: 'Theta / beta', drives: 'how the light fades', status: 'interpretation' },
  { signal: 'Frontal asymmetry', drives: 'warmth of the light', status: 'interpretation' },
]

export function StateChannels() {
  const measured = STATE_CHANNELS.filter((c) => c.status === 'measured')
  const interpreted = STATE_CHANNELS.filter((c) => c.status === 'interpretation')

  return (
    <div className="card-grid md:grid-cols-2">
      <Group
        title="Measured"
        tone="measured"
        caption="Read straight off the recording. Alpha power over occipital cortex is alpha power over occipital cortex — there is no prediction here, so there is no accuracy figure either."
        channels={measured}
      />
      <Group
        title="Interpretation"
        tone="interpretation"
        caption="A cognitive meaning assumed on top of a measurement. Standard readings, each contested at the edges. Frontal asymmetry moves the least of anything — it is the one this project tested and found wanting."
        channels={interpreted}
      />
    </div>
  )
}

function Group({
  title,
  tone,
  caption,
  channels,
}: {
  title: string
  tone: 'measured' | 'interpretation'
  caption: string
  channels: StateChannel[]
}) {
  const accent = tone === 'measured' ? 'text-neural-300' : 'text-memory-300'
  const dot = tone === 'measured' ? 'bg-neural-300' : 'bg-memory-300'

  return (
    <div className="glass rounded-panel flex flex-col p-6">
      <div className="mb-4 flex items-center gap-2">
        <span className={cn('size-1.5 rounded-full', dot)} />
        <p className={cn('text-label', accent)}>{title}</p>
      </div>

      <ul className="space-y-2.5">
        {channels.map((channel) => (
          <li
            key={channel.signal}
            className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-0.5 border-b border-ink/6 pb-2.5 last:border-0"
          >
            <span className="text-[0.86rem] text-ink">{channel.signal}</span>
            <span className="text-[0.78rem] text-ink-faint">{channel.drives}</span>
          </li>
        ))}
      </ul>

      <p className="mt-5 text-[0.76rem] leading-relaxed text-ink-faint">{caption}</p>
    </div>
  )
}
