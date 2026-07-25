/**
 * Memory reconstruction types.
 *
 * The deliverable is a reconstructed video plus its measured fidelity against
 * the reference. Everything here serves that comparison.
 */

/* -------------------------------------------------------------------------- */
/* Stimulus                                                                   */
/* -------------------------------------------------------------------------- */

export type StimulusSource = 'synthetic' | 'uploaded'

export interface StimulusClip {
  id: string
  title: string
  description: string
  durationSeconds: number
  fps: number
  /** Reconstruction grid width. Deliberately small — see `frames` stage. */
  gridWidth: number
  gridHeight: number
  source: StimulusSource
  tags: string[]
  /** Number of scene transitions the clip contains. */
  sceneCount: number
}

/**
 * Per-frame visual properties of the reference.
 *
 * Both the simulator (which turns these into EEG) and the evaluator (which
 * scores the reconstruction against them) read this same structure, so the
 * ground truth is unambiguous.
 */
export interface VisualFeatures {
  t: number
  /** Mean luminance, 0–1. */
  luminance: number
  /** RMS contrast, 0–1. */
  contrast: number
  /** Global motion energy, 0–1. */
  motion: number
  /** Dominant motion direction in radians, or null when static. */
  motionDirection: number | null
  /** Edge density, 0–1 — a proxy for spatial detail. */
  edgeDensity: number
  /** Mean colour as normalised RGB. */
  color: [number, number, number]
  /** True on the first frame of a new scene. */
  sceneCut: boolean
  /** Index of the scene this frame belongs to. */
  sceneIndex: number
}

/* -------------------------------------------------------------------------- */
/* Latent space                                                               */
/* -------------------------------------------------------------------------- */

export const LATENT_DIMENSIONS = 16

export interface LatentPoint {
  t: number
  /** Fixed-length latent vector, roughly zero-centred. */
  vector: number[]
  /** 0–1 confidence that this point is supported by usable signal. */
  confidence: number
}

/* -------------------------------------------------------------------------- */
/* Reconstruction                                                             */
/* -------------------------------------------------------------------------- */

/** Scene-level properties decoded from the latent. */
export interface SceneEstimate {
  t: number
  luminance: number
  contrast: number
  motion: number
  motionDirection: number | null
  color: [number, number, number]
  sceneCut: boolean
  sceneIndex: number
  confidence: number
}

/**
 * One reconstructed frame.
 *
 * Pixels are base64-encoded RGB bytes at the clip's grid resolution. A raw
 * number array would be roughly four times the size on the wire for the same
 * data, and these stream continuously.
 */
export interface ReconstructedFrame {
  index: number
  t: number
  /** Base64 RGB, length gridWidth × gridHeight × 3 once decoded. */
  pixels: string
  confidence: number
}

export type ReconstructionStatus =
  | 'idle'
  | 'synchronising'
  | 'encoding'
  | 'reconstructing'
  | 'refining'
  | 'complete'
  | 'error'

/* -------------------------------------------------------------------------- */
/* Fidelity                                                                   */
/* -------------------------------------------------------------------------- */

/**
 * Measured agreement between reconstruction and reference.
 *
 * Every field is computed at the reconstruction grid resolution against the
 * downsampled reference, so no metric is flattered by upsampling.
 */
export interface FidelityMetrics {
  /** Structural similarity, −1–1. The headline spatial metric. */
  ssim: number
  /** Peak signal-to-noise ratio in dB. */
  psnr: number
  /** Pearson correlation of per-frame mean luminance. */
  luminanceCorrelation: number
  /** Mean per-channel correlation of frame colour. */
  colorCorrelation: number
  /** Pearson correlation of per-frame motion energy. */
  motionCorrelation: number
  /** F1 of detected scene boundaries against true cuts. */
  sceneCutF1: number
  /** Correlation of the coarse spatial layout, averaged over frames. */
  layoutCorrelation: number
  /** Weighted 0–1 summary. */
  composite: number
}

/** Fidelity for a single frame, for the per-frame trace. */
export interface FrameFidelity {
  index: number
  t: number
  ssim: number
  psnr: number
  layoutCorrelation: number
}

/**
 * What a metric means, and what value would count as chance.
 *
 * Displayed alongside every number: an SSIM of 0.4 is meaningless without
 * knowing that a static grey frame scores 0.25 on this material.
 */
export interface MetricDescriptor {
  key: keyof FidelityMetrics
  label: string
  unit: string
  description: string
  /** Score a trivial baseline achieves. */
  baseline: number
  /** Practical ceiling given EEG's information content. */
  ceiling: number
  /** Higher is better for every metric here, but stated explicitly. */
  higherIsBetter: boolean
}

export const METRIC_DESCRIPTORS: readonly MetricDescriptor[] = [
  {
    key: 'ssim',
    label: 'Structural similarity',
    unit: '',
    description:
      'Agreement in local structure between reconstruction and reference. The headline spatial measure.',
    baseline: 0.25,
    ceiling: 0.7,
    higherIsBetter: true,
  },
  {
    key: 'psnr',
    label: 'Peak SNR',
    unit: 'dB',
    description: 'Pixel-level error, log-scaled. Sensitive to overall brightness accuracy.',
    baseline: 8,
    ceiling: 22,
    higherIsBetter: true,
  },
  {
    key: 'luminanceCorrelation',
    label: 'Luminance tracking',
    unit: 'r',
    description:
      'How well reconstructed brightness follows the reference over time. The most recoverable property in EEG — occipital response scales with luminance.',
    baseline: 0,
    ceiling: 0.9,
    higherIsBetter: true,
  },
  {
    key: 'motionCorrelation',
    label: 'Motion tracking',
    unit: 'r',
    description:
      'Agreement in per-frame motion energy. Supported by motion-sensitive visual areas.',
    baseline: 0,
    ceiling: 0.8,
    higherIsBetter: true,
  },
  {
    key: 'colorCorrelation',
    label: 'Colour tracking',
    unit: 'r',
    description:
      'Agreement in dominant colour. Weakly represented in scalp EEG; expect modest values.',
    baseline: 0,
    ceiling: 0.5,
    higherIsBetter: true,
  },
  {
    key: 'sceneCutF1',
    label: 'Scene boundary F1',
    unit: '',
    description:
      'Detection of scene transitions. Visual transients produce strong evoked responses, so boundaries are among the most reliable recoveries.',
    baseline: 0.1,
    ceiling: 0.95,
    higherIsBetter: true,
  },
  {
    key: 'layoutCorrelation',
    label: 'Spatial layout',
    unit: 'r',
    description:
      'Correlation of the coarse brightness layout. Limited by the spatial resolution scalp EEG can carry.',
    baseline: 0,
    ceiling: 0.55,
    higherIsBetter: true,
  },
]

/* -------------------------------------------------------------------------- */
/* Session records                                                            */
/* -------------------------------------------------------------------------- */

export interface ReconstructionReport {
  /** Plain-language account of what was and was not recovered. */
  summary: string
  title: string
  /** Findings supported by the metrics. */
  findings: string[]
  /** Explicit statement of what the signal could not support. */
  limitations: string[]
}

export interface SessionSummary {
  id: string
  title: string
  createdAt: string
  durationSeconds: number
  stimulusId: string
  stimulusTitle: string
  source: StimulusSource
  /** Composite fidelity, 0–1. */
  fidelity: number
  ssim: number
  frameCount: number
  bookmarked: boolean
  version: number
}

export interface SessionRecord extends SessionSummary {
  clip: StimulusClip
  metrics: FidelityMetrics
  frameFidelity: FrameFidelity[]
  scenes: SceneEstimate[]
  report: ReconstructionReport
  /** Sampled latent trajectory, for the latent-space view. */
  latents: LatentPoint[]
  history: SessionVersion[]
}

export interface SessionVersion {
  version: number
  createdAt: string
  fidelity: number
  ssim: number
  delta: string
}

/* -------------------------------------------------------------------------- */
/* Real-subject reconstruction                                                */
/* -------------------------------------------------------------------------- */

/** One ranked hypothesis about what a person was looking at. */
export interface RetrievalCandidate {
  rank: number
  stimulusNumber: number
  label: string
  levelA: string
  levelB: string
  score: number
  /** Normalised confidence across the 200 candidates. */
  probability: number
  correct: boolean
  /** Base64 PNG of the candidate object. */
  image: string
}

export interface RealViewing {
  subject: string
  trialsUsed: number
  truth: {
    stimulusNumber: number | null
    label: string
    levelA: string
    levelB: string
    image: string
  }
  /** 0-based rank of the true object among the 200 candidates. */
  truthRank: number | null
  exact: boolean
  inTop5: boolean
  correctAnimacy: boolean | null
  correctCategory: boolean | null
  candidates: RetrievalCandidate[]
}

export interface RealDataStatus {
  available: boolean
  subjects: string[]
  dataset: string
  citation: string
  stimuli: number
  hint: string | null
}

/** Measured accuracy from the offline decoding and reconstruction harnesses. */
export interface RealResults {
  available: boolean
  subjects?: number
  animacy?: number
  animacyShuffled?: number
  category?: number
  preStimulus?: number
  peakLatencyMs?: number
  identification?: Record<string, { top1: number; top5: number; pairwise: number }>
  reconstruction?: {
    exact: number
    top5: number
    animacy: number
    category: number
  }
}
