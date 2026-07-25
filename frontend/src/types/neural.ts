/**
 * Neural signal types.
 *
 * Mirrors the Pydantic models in `backend/app/models`. The backend serialises
 * with a camelCase alias generator, so the wire format matches exactly.
 */

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

/**
 * Montage used for visual reconstruction.
 *
 * Weighted toward posterior coverage: the visual hierarchy is occipital and
 * occipito-temporal, and those channels carry nearly all of the recoverable
 * signal. Frontal sites are retained mainly for artifact reference.
 */
export const ELECTRODES = [
  'Fp1',
  'Fp2',
  'F3',
  'Fz',
  'F4',
  'T7',
  'C3',
  'Cz',
  'C4',
  'T8',
  'P7',
  'P3',
  'Pz',
  'P4',
  'P8',
  'PO7',
  'PO3',
  'POz',
  'PO4',
  'PO8',
  'O1',
  'Oz',
  'O2',
] as const
export type Electrode = (typeof ELECTRODES)[number]

export type SignalQualityGrade = 'excellent' | 'good' | 'fair' | 'poor'

export interface SignalQuality {
  /** 0–1 composite of impedance, artifact load and channel agreement. */
  score: number
  grade: SignalQualityGrade
  /** Proportion of the last window attenuated as artifact, 0–1. */
  artifactRatio: number
  droppedChannels: string[]
  /** Estimated residual line noise, 0–1. */
  lineNoise: number
}

/** Normalised relative power per band. */
export type BandPowers = Record<FrequencyBand, number>

export interface Spectrum {
  frequencies: number[]
  /** Power in dB, same length as `frequencies`. */
  magnitudes: number[]
  /** Strongest oscillatory peak after removing the 1/f slope. */
  peakFrequency: number
}

/**
 * Measures of the visual system's response, extracted per window.
 *
 * These are what the encoder actually reads — the quantities scalp EEG can
 * carry about what someone was looking at.
 */
export interface VisualResponse {
  /** Occipital broadband response amplitude, tracks luminance. 0–1. */
  occipitalDrive: number
  /** Posterior alpha desynchronisation — rises with visual engagement. 0–1. */
  alphaSuppression: number
  /** Occipito-temporal beta/gamma, tracks motion energy. 0–1. */
  motionResponse: number
  /** Transient evoked-response magnitude; spikes at scene changes. 0–1. */
  transient: number
  /** Gamma-band activity, tracks spatial detail and contrast. 0–1. */
  detailResponse: number
  /** Left/right posterior asymmetry, carries coarse horizontal layout. −1–1. */
  lateralBalance: number
  /** Dorsal/ventral posterior balance, carries coarse vertical layout. −1–1. */
  verticalBalance: number
}

/** One decoded window of EEG. */
export interface NeuralFrame {
  sessionId: string
  index: number
  /** Seconds on the *stimulus* timeline, after synchronization. */
  t: number
  bandPowers: BandPowers
  spectrum: Spectrum
  /** Per-electrode activation, 0–1, for the scalp map. */
  topography: Record<string, number>
  /** Decimated trace for the live waveform. */
  waveform: number[]
  signalQuality: SignalQuality
  visual: VisualResponse
  /** 0–1 confidence that this window supports reconstruction. */
  confidence: number
}
