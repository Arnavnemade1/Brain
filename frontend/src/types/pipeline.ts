/**
 * The reconstruction pipeline: raw EEG in, scene parameters out.
 *
 * Stage identifiers are stable and shared with the backend, which emits
 * `pipeline` events keyed by these ids as work flows through each stage.
 */

export const PIPELINE_STAGE_IDS = [
  'ingest',
  'cleaning',
  'denoise',
  'features',
  'frequency',
  'temporal',
  'memory_inference',
  'emotion',
  'context',
  'scene_params',
  'engine',
  'environment',
] as const

export type PipelineStageId = (typeof PIPELINE_STAGE_IDS)[number]

export type StageStatus = 'idle' | 'active' | 'complete' | 'degraded' | 'error'

export interface PipelineStageMeta {
  id: PipelineStageId
  label: string
  /** One-line description surfaced in the flow diagram. */
  summary: string
  /** Longer copy for the expanded inspector. */
  detail: string
  /** Which layer of the system this belongs to — drives colour grouping. */
  band: 'acquisition' | 'analysis' | 'inference' | 'synthesis'
}

/** Live state of one stage, streamed from the backend. */
export interface PipelineStageState {
  id: PipelineStageId
  status: StageStatus
  /** 0–1 completion of the current unit of work. */
  progress: number
  /** Milliseconds spent in the most recent pass. */
  latencyMs: number
  /** Units processed since session start (windows, components, features…). */
  throughput: number
  /** Human-readable current operation, e.g. "notch 60 Hz · 19 ch". */
  message: string
  /** Stage-local confidence in its own output, 0–1. */
  confidence: number
}

/** Static description of every stage, in execution order. */
export const PIPELINE_STAGES: readonly PipelineStageMeta[] = [
  {
    id: 'ingest',
    label: 'Raw EEG Data',
    summary: 'Buffered acquisition from device or file',
    detail:
      'Samples arrive as a continuous multi-channel stream. MindScape buffers them into overlapping analysis windows and timestamps each against the session clock.',
    band: 'acquisition',
  },
  {
    id: 'cleaning',
    label: 'Signal Cleaning',
    summary: 'Re-reference, detrend, band-limit',
    detail:
      'Channels are re-referenced to a common average, linearly detrended, and band-limited to 0.5–45 Hz. Flat or railed channels are flagged and excluded downstream.',
    band: 'acquisition',
  },
  {
    id: 'denoise',
    label: 'Noise Reduction',
    summary: 'Notch filtering and artifact rejection',
    detail:
      'Line noise is notched at 50/60 Hz. Ocular and muscular artifacts are identified by amplitude and kurtosis thresholds, then attenuated rather than deleted so temporal continuity survives.',
    band: 'acquisition',
  },
  {
    id: 'features',
    label: 'Feature Extraction',
    summary: 'Hjorth, entropy, connectivity',
    detail:
      'Each cleaned window yields Hjorth descriptors, spectral entropy, zero-crossing statistics and inter-channel coherence — the numeric substrate every later stage reasons over.',
    band: 'analysis',
  },
  {
    id: 'frequency',
    label: 'Frequency Analysis',
    summary: 'Welch PSD across five canonical bands',
    detail:
      'Power spectral density is estimated with Welch’s method, then integrated across delta, theta, alpha, beta and gamma to produce the relative band profile.',
    band: 'analysis',
  },
  {
    id: 'temporal',
    label: 'Temporal Pattern Analysis',
    summary: 'Cross-window dynamics and drift',
    detail:
      'Band trajectories are tracked over time to expose transitions, recurrences and drift. Stable recurring motifs are treated as candidate memory episodes.',
    band: 'analysis',
  },
  {
    id: 'memory_inference',
    label: 'Memory Feature Inference',
    summary: 'Episodic content estimation',
    detail:
      'The inference model maps spectral and temporal descriptors onto latent memory attributes: spatial scale, familiarity, population density, object salience and vividness.',
    band: 'inference',
  },
  {
    id: 'emotion',
    label: 'Emotion Detection',
    summary: 'Valence, arousal, affect distribution',
    detail:
      'Frontal alpha asymmetry and beta/gamma balance drive a valence–arousal estimate, projected onto a discrete affect distribution the renderer can act on.',
    band: 'inference',
  },
  {
    id: 'context',
    label: 'Context Reconstruction',
    summary: 'Setting, era and narrative framing',
    detail:
      'Memory attributes and affect are fused into a coherent context: probable setting, time of day, era, weather and the narrative arc that ties fragments together.',
    band: 'inference',
  },
  {
    id: 'scene_params',
    label: 'Scene Generation Parameters',
    summary: 'Deterministic render descriptor',
    detail:
      'Context becomes an explicit parameter set — biome, palette, lighting, fog, structures, particles, soundscape — plus a seed that makes the reconstruction reproducible.',
    band: 'synthesis',
  },
  {
    id: 'engine',
    label: 'Memory Reconstruction Engine',
    summary: 'Procedural world synthesis',
    detail:
      'The engine builds geometry from the parameter set, deliberately leaving low-confidence regions fragmented. Additional neural evidence stabilises them over time.',
    band: 'synthesis',
  },
  {
    id: 'environment',
    label: 'Interactive 3D Environment',
    summary: 'Explorable reconstruction',
    detail:
      'The reconstruction is handed to the renderer, where it can be walked through in first person while continuing to resolve as the session progresses.',
    band: 'synthesis',
  },
]

export const STAGE_BY_ID: Record<PipelineStageId, PipelineStageMeta> = Object.fromEntries(
  PIPELINE_STAGES.map((s) => [s.id, s]),
) as Record<PipelineStageId, PipelineStageMeta>
