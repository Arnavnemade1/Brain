import { useMutation, useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { Brain, Check, RefreshCw, X } from 'lucide-react'
import { useEffect, useState } from 'react'

import { pageVariants, riseItem, staggerContainer } from '@/animations/motion'
import { cn } from '@/lib/cn'
import { api } from '@/services/api'
import type { RealViewing, RetrievalCandidate } from '@/types'
import { Badge, Button, EmptyState, ErrorNotice, LoadingState, Panel, Stat } from '@/ui'

const TRIAL_OPTIONS = [1, 4, 8, 12] as const

/**
 * Reconstruction from real human EEG.
 *
 * The whole page is one comparison: what a person actually looked at, beside
 * the objects the system recovered from their brain activity. Held-out data
 * throughout — the templates being matched against never saw these trials.
 */
export default function RealSubjects() {
  const [subject, setSubject] = useState<string | null>(null)
  const [trials, setTrials] = useState<number>(8)
  const [seed, setSeed] = useState(0)

  const status = useQuery({ queryKey: ['realStatus'], queryFn: api.realStatus })
  const results = useQuery({ queryKey: ['realResults'], queryFn: api.realResults })

  useEffect(() => {
    document.title = 'Real Subjects — MindScape'
  }, [])

  useEffect(() => {
    if (!subject && status.data?.subjects.length) {
      setSubject(status.data.subjects[0])
    }
  }, [status.data, subject])

  const reconstruct = useMutation({
    mutationFn: () =>
      api.realReconstruct(subject!, { trials, seed }),
  })

  const viewing = reconstruct.data

  const run = () => {
    setSeed((value) => value + 1)
    reconstruct.mutate()
  }

  return (
    <motion.main
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      className="mx-auto w-[min(80rem,calc(100vw-2.5rem))] pt-24 pb-20"
    >
      <motion.div variants={staggerContainer} initial="initial" animate="animate">
        <motion.header variants={riseItem} className="mb-8 max-w-3xl">
          <p className="text-label mb-4">Real human EEG</p>
          <h1 className="text-[clamp(1.7rem,4vw,2.5rem)] leading-tight">
            What they saw, and what their brain activity gave back.
          </h1>
          <p className="mt-5 text-[0.95rem] leading-relaxed text-ink-muted">
            Recordings from people viewing 200 objects in rapid succession. The system
            ranks all 200 candidates from the EEG alone, on trials it was never trained
            on. Where the first candidate is right, the reconstruction is the photograph
            they actually looked at.
          </p>
          {status.data && (
            <p className="mt-4 text-[0.72rem] leading-relaxed text-ink-faint">
              {status.data.citation}. Stimulus order is randomised within each stream,
              which is what makes this decoding trustworthy — the widely-cited
              EEG-to-image results used block designs and were later shown to be
              decoding time, not vision.
            </p>
          )}
        </motion.header>

        {status.isLoading && <LoadingState title="Checking for real subject data" />}

        {status.data && !status.data.available && (
          <Panel label="Real data" title="Not downloaded">
            <EmptyState
              icon={<Brain className="size-5" />}
              title="The EEG recordings are not present"
              detail={status.data.hint ?? 'Run the fetch tool in the backend.'}
            />
          </Panel>
        )}

        {status.data?.available && (
          <div className="grid gap-5 lg:grid-cols-[17rem_1fr]">
            {/* Controls + measured accuracy */}
            <div className="space-y-5">
              <Panel label="Subject" title="Recording" animate={false}>
                <div className="space-y-1.5">
                  {status.data.subjects.map((option) => (
                    <button
                      key={option}
                      onClick={() => setSubject(option)}
                      aria-pressed={subject === option}
                      className={cn(
                        'w-full rounded-lg border px-3 py-2 text-left text-[0.8rem] transition-colors',
                        subject === option
                          ? 'border-neural-400/45 bg-neural-400/10 text-ink'
                          : 'border-ink/8 text-ink-muted hover:text-ink-soft',
                      )}
                    >
                      {option}
                    </button>
                  ))}
                </div>

                <div className="mt-5 border-t border-ink/8 pt-4">
                  <p className="text-label mb-3">Viewings averaged</p>
                  <div className="flex gap-1.5">
                    {TRIAL_OPTIONS.map((option) => (
                      <button
                        key={option}
                        onClick={() => setTrials(option)}
                        aria-pressed={trials === option}
                        className={cn(
                          'flex-1 rounded-lg border px-2 py-1.5 text-[0.75rem] transition-colors',
                          trials === option
                            ? 'border-neural-400/45 bg-neural-400/10 text-ink'
                            : 'border-ink/8 text-ink-muted hover:text-ink-soft',
                        )}
                      >
                        {option}
                      </button>
                    ))}
                  </div>
                  <p className="mt-2.5 text-[0.68rem] leading-relaxed text-ink-faint">
                    How many times the person saw this object. More exposures mean a
                    cleaner signal — the same reason a memory you revisited is sharper.
                  </p>
                </div>

                <Button
                  variant="primary"
                  fullWidth
                  className="mt-5"
                  loading={reconstruct.isPending}
                  disabled={!subject}
                  onClick={run}
                  icon={<RefreshCw className="size-3.5" />}
                >
                  Reconstruct a viewing
                </Button>
              </Panel>

              {results.data?.available && (
                <Panel label="Measured accuracy" title="Across all subjects" animate={false}>
                  <dl className="space-y-4">
                    <Stat
                      label="Exact match, 200-way"
                      value={((results.data.reconstruction?.exact ?? 0) * 100).toFixed(1)}
                      unit="%"
                      hint="chance 0.5%"
                    />
                    <Stat
                      label="Correct in top 5"
                      value={((results.data.reconstruction?.top5 ?? 0) * 100).toFixed(1)}
                      unit="%"
                      hint="chance 2.5%"
                    />
                    <Stat
                      label="Animate vs inanimate"
                      value={((results.data.animacy ?? 0) * 100).toFixed(1)}
                      unit="%"
                      hint={`chance 50% · shuffled ${((results.data.animacyShuffled ?? 0) * 100).toFixed(1)}%`}
                    />
                    <Stat
                      label="Category, 10-way"
                      value={((results.data.category ?? 0) * 100).toFixed(1)}
                      unit="%"
                      hint="chance 10%"
                    />
                  </dl>
                  <p className="mt-4 border-t border-ink/8 pt-3 text-[0.68rem] leading-relaxed text-ink-faint">
                    Pre-stimulus decoding sits at{' '}
                    {((results.data.preStimulus ?? 0) * 100).toFixed(1)}% and peaks at{' '}
                    {(results.data.peakLatencyMs ?? 0).toFixed(0)} ms — chance before the
                    image appears, rising only after it does.
                  </p>
                </Panel>
              )}
            </div>

            {/* The comparison */}
            <div className="space-y-5">
              {reconstruct.isError && (
                <ErrorNotice
                  message={
                    reconstruct.error instanceof Error
                      ? reconstruct.error.message
                      : 'Reconstruction failed.'
                  }
                  onRetry={run}
                />
              )}

              {!viewing && !reconstruct.isPending && (
                <Panel label="Reconstruction" title="Nothing reconstructed yet">
                  <EmptyState
                    icon={<Brain className="size-5" />}
                    title="Pick a subject and reconstruct a viewing"
                    detail="A held-out moment is chosen at random, its EEG matched against all 200 candidate objects."
                  />
                </Panel>
              )}

              {reconstruct.isPending && (
                <Panel label="Reconstruction" title="Decoding">
                  <LoadingState
                    title="Matching brain activity against 200 candidates"
                    detail="First run for a subject loads a 500 MB recording."
                  />
                </Panel>
              )}

              {viewing && <ViewingResult viewing={viewing} />}
            </div>
          </div>
        )}
      </motion.div>
    </motion.main>
  )
}

function ViewingResult({ viewing }: { viewing: RealViewing }) {
  const rank = viewing.truthRank
  const tone = viewing.exact
    ? 'var(--color-signal-good)'
    : viewing.inTop5
      ? 'var(--color-signal-fair)'
      : 'var(--color-signal-poor)'

  return (
    <>
      <Panel
        label="Memory replay"
        title={viewing.exact ? 'Exact match' : viewing.inTop5 ? 'Within the top five' : 'Missed'}
        action={
          <Badge tone={tone}>
            {rank === null ? 'unranked' : `rank ${rank + 1} of 200`}
          </Badge>
        }
      >
        <div className="grid gap-6 sm:grid-cols-[10rem_1fr] sm:items-start">
          <figure>
            <div
              className="rounded-card overflow-hidden border-2"
              style={{ borderColor: 'color-mix(in oklab, var(--color-neural-300) 45%, transparent)' }}
            >
              {viewing.truth.image ? (
                <img
                  src={`data:image/png;base64,${viewing.truth.image}`}
                  alt={`Object shown: ${viewing.truth.label}`}
                  className="block aspect-square w-full bg-void object-contain"
                />
              ) : (
                <div className="aspect-square w-full bg-deep" />
              )}
            </div>
            <figcaption className="mt-2">
              <p className="text-label">What they saw</p>
              <p className="mt-1 text-[0.85rem] text-ink capitalize">
                {viewing.truth.label}
              </p>
              <p className="text-[0.68rem] text-ink-faint">
                {viewing.truth.levelA} · {viewing.truth.levelB}
              </p>
            </figcaption>
          </figure>

          <div>
            <p className="text-label mb-3">
              Recovered from EEG · {viewing.trialsUsed} viewing
              {viewing.trialsUsed > 1 ? 's' : ''} averaged
            </p>
            <ol className="grid grid-cols-4 gap-2.5 sm:grid-cols-6">
              {viewing.candidates.map((candidate) => (
                <CandidateTile key={candidate.stimulusNumber} candidate={candidate} />
              ))}
            </ol>

            <dl className="mt-6 grid grid-cols-3 gap-4 border-t border-ink/8 pt-4">
              <div>
                <dt className="text-label mb-1.5">Top-1 animacy</dt>
                <dd className="flex items-center gap-1.5 text-[0.8rem]">
                  {viewing.correctAnimacy ? (
                    <Check className="size-3.5 text-signal-good" />
                  ) : (
                    <X className="size-3.5 text-signal-poor" />
                  )}
                  <span className="text-ink-soft">
                    {viewing.correctAnimacy ? 'correct' : 'wrong'}
                  </span>
                </dd>
              </div>
              <div>
                <dt className="text-label mb-1.5">Top-1 category</dt>
                <dd className="flex items-center gap-1.5 text-[0.8rem]">
                  {viewing.correctCategory ? (
                    <Check className="size-3.5 text-signal-good" />
                  ) : (
                    <X className="size-3.5 text-signal-poor" />
                  )}
                  <span className="text-ink-soft">
                    {viewing.correctCategory ? 'correct' : 'wrong'}
                  </span>
                </dd>
              </div>
              <div>
                <dt className="text-label mb-1.5">Confidence</dt>
                <dd className="text-readout text-[0.8rem] text-ink-soft">
                  {((viewing.candidates[0]?.probability ?? 0) * 100).toFixed(0)}%
                </dd>
              </div>
            </dl>
          </div>
        </div>
      </Panel>

      <Panel label="Reading this" title="What the ranking means" animate={false}>
        <p className="text-[0.82rem] leading-relaxed text-ink-muted">
          The correct object is the single best match about 6% of the time and inside
          the top five about 18% of the time, against chance rates of 0.5% and 2.5%.
          That is roughly twelve times chance for an exact hit — genuine, and a long
          way from perfect. Most of the time the top candidate is wrong.
        </p>
        <p className="mt-4 text-[0.82rem] leading-relaxed text-ink-muted">
          Notice what the near misses are. When the exact object is not first, the
          candidates that beat it are usually from the same category — plants for
          plants, tools for tools. The EEG reliably carries what *kind* of thing was
          seen well before it carries *which* one.
        </p>
        <p className="mt-4 text-[0.78rem] leading-relaxed text-ink-faint">
          Retrieval has a real limit worth stating: the system can only return objects
          in its candidate set. It recognises, it does not imagine. Generating pixels
          instead would produce a picture that looked like a reconstruction while being
          mostly invention, because scalp EEG does not carry per-pixel detail.
        </p>
      </Panel>
    </>
  )
}

function CandidateTile({ candidate }: { candidate: RetrievalCandidate }) {
  return (
    <li className="min-w-0">
      <div
        className={cn(
          'rounded-card overflow-hidden border-2 transition-colors',
          candidate.correct ? 'border-signal-good' : 'border-ink/10',
        )}
      >
        {candidate.image ? (
          <img
            src={`data:image/png;base64,${candidate.image}`}
            alt={candidate.label}
            className="block aspect-square w-full bg-void object-contain"
          />
        ) : (
          <div className="aspect-square w-full bg-deep" />
        )}
      </div>
      <p className="mt-1.5 truncate text-[0.68rem] capitalize text-ink-soft">
        {candidate.rank + 1}. {candidate.label}
      </p>
      <div className="mt-1 h-0.5 overflow-hidden rounded-full bg-ink/8">
        <div
          className="h-full rounded-full"
          style={{
            width: `${Math.min(100, candidate.probability * 100)}%`,
            background: candidate.correct
              ? 'var(--color-signal-good)'
              : 'var(--color-neural-400)',
          }}
        />
      </div>
    </li>
  )
}
