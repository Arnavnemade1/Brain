/**
 * Neural domain types.
 *
 * These mirror the Pydantic models in `backend/app/models`. The backend
 * serialises with a camelCase alias generator, so the wire format matches
 * these declarations exactly.
 */

/** Canonical EEG frequency bands used across the pipeline. */
export const FREQUENCY_BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma'] as const
export type FrequencyBand = (typeof FREQUENCY_BANDS)[number]

/** Absolute band edges in Hz, matching `backend/app/processing/bands.py`. */
export const BAND_RANGES: Record<FrequencyBand, readonly [number, number]> = {
  delta: [0.5, 4],
  theta: [4, 8],
  alpha: [8, 13],
  beta: [13, 30],
  gamma: [30, 45],
}

/** 10–20 system subset the simulator and montage renderer use. */
export const ELECTRODES = [
  'Fp1',
  'Fp2',
  'F7',
  'F3',
  'Fz',
  'F4',
  'F8',
  'T3',
  'C3',
  'Cz',
  'C4',
  'T4',
  'T5',
  'P3',
  'Pz',
  'P4',
  'T6',
  'O1',
  'O2',
] as const
export type Electrode = (typeof ELECTRODES)[number]

export type CognitiveState =
  | 'focused'
  | 'relaxed'
  | 'drowsy'
  | 'dreaming'
  | 'recalling'
  | 'alert'
  | 'meditative'

export const COGNITIVE_STATES: readonly CognitiveState[] = [
  'focused',
  'relaxed',
  'drowsy',
  'dreaming',
  'recalling',
  'alert',
  'meditative',
]

export type EmotionLabel =
  | 'calm'
  | 'nostalgia'
  | 'wonder'
  | 'stress'
  | 'curiosity'
  | 'melancholy'
  | 'joy'
  | 'fear'

export const EMOTIONS: readonly EmotionLabel[] = [
  'calm',
  'nostalgia',
  'wonder',
  'stress',
  'curiosity',
  'melancholy',
  'joy',
  'fear',
]

export type SignalQualityGrade = 'excellent' | 'good' | 'fair' | 'poor'

export interface SignalQuality {
  /** 0–1 composite of impedance, artifact load and channel agreement. */
  score: number
  grade: SignalQualityGrade
  /** Proportion of the last window rejected as artifact, 0–1. */
  artifactRatio: number
  /** Channels currently flagged as unusable. */
  droppedChannels: Electrode[]
  /** Estimated line-noise contamination at 50/60 Hz, 0–1. */
  lineNoise: number
}

/** Normalised (sums to ~1) relative power per band. */
export type BandPowers = Record<FrequencyBand, number>

/** Log-scaled power spectral density sampled on a shared frequency axis. */
export interface Spectrum {
  /** Frequency axis in Hz. */
  frequencies: number[]
  /** Power in dB, same length as `frequencies`. */
  magnitudes: number[]
  peakFrequency: number
}

export interface CognitiveReading {
  state: CognitiveState
  confidence: number
  /** Probability mass over every cognitive state; keys sum to ~1. */
  distribution: Partial<Record<CognitiveState, number>>
  /** Beta/alpha derived engagement index, 0–1. */
  engagement: number
  /** Theta/beta derived cognitive-load index, 0–1. */
  load: number
}

export interface EmotionReading {
  primary: EmotionLabel
  secondary: EmotionLabel | null
  /** 0–1 magnitude of affective activation. */
  intensity: number
  /** −1 (unpleasant) to +1 (pleasant). */
  valence: number
  /** 0–1 physiological activation. */
  arousal: number
  distribution: Partial<Record<EmotionLabel, number>>
}

/** One decoded window of EEG — the atomic unit the pipeline consumes. */
export interface NeuralFrame {
  sessionId: string
  /** Monotonic frame index from session start. */
  index: number
  /** Seconds since session start. */
  t: number
  bandPowers: BandPowers
  spectrum: Spectrum
  /** Per-electrode activation, 0–1, for the scalp heatmap. */
  topography: Record<string, number>
  /** Short raw-ish trace for the live waveform, already decimated. */
  waveform: number[]
  signalQuality: SignalQuality
  cognitive: CognitiveReading
  emotion: EmotionReading
  /** 0–1 rolling confidence the reconstruction engine consumes. */
  neuralConfidence: number
  /** 0–1 stability of the inferred memory trace across recent windows. */
  coherence: number
}
