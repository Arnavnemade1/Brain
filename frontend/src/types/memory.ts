/**
 * Memory reconstruction types — the scene descriptor the 3D engine consumes,
 * the fragments scattered through it, and the session records that persist.
 */

import type { CognitiveState, EmotionLabel } from './neural'

/** Environments the context model can settle on. */
export const BIOMES = [
  'childhood_home',
  'school_hallway',
  'beach_sunset',
  'forest',
  'city_streets',
  'mountains',
  'ancient_ruins',
  'dreamscape',
  'ocean',
  'floating_islands',
  'abstract_space',
  'grand_architecture',
] as const

export type Biome = (typeof BIOMES)[number]

export interface BiomeMeta {
  id: Biome
  label: string
  /** Short line shown in the reconstruction header. */
  tagline: string
  /** Emotions this setting is most often selected for. */
  affinity: EmotionLabel[]
}

export const BIOME_META: Record<Biome, BiomeMeta> = {
  childhood_home: {
    id: 'childhood_home',
    label: 'Childhood Home',
    tagline: 'Interior architecture with high familiarity weighting',
    affinity: ['nostalgia', 'melancholy', 'calm'],
  },
  school_hallway: {
    id: 'school_hallway',
    label: 'School Hallway',
    tagline: 'Repeating institutional corridor, strong episodic anchoring',
    affinity: ['nostalgia', 'stress', 'curiosity'],
  },
  beach_sunset: {
    id: 'beach_sunset',
    label: 'Beach at Sunset',
    tagline: 'Open horizon, low sun angle, rhythmic auditory trace',
    affinity: ['calm', 'nostalgia', 'joy'],
  },
  forest: {
    id: 'forest',
    label: 'Forest',
    tagline: 'Dense vertical occlusion with diffuse canopy light',
    affinity: ['calm', 'curiosity', 'melancholy'],
  },
  city_streets: {
    id: 'city_streets',
    label: 'City Streets',
    tagline: 'High object density, elevated ambient motion',
    affinity: ['stress', 'curiosity', 'joy'],
  },
  mountains: {
    id: 'mountains',
    label: 'Mountains',
    tagline: 'Extreme spatial scale, sparse detail, cold palette',
    affinity: ['wonder', 'calm', 'melancholy'],
  },
  ancient_ruins: {
    id: 'ancient_ruins',
    label: 'Ancient Ruins',
    tagline: 'Fragmented monumental geometry, deep time signature',
    affinity: ['wonder', 'melancholy', 'curiosity'],
  },
  dreamscape: {
    id: 'dreamscape',
    label: 'Dream Landscape',
    tagline: 'Non-euclidean layout, unstable object permanence',
    affinity: ['wonder', 'fear', 'curiosity'],
  },
  ocean: {
    id: 'ocean',
    label: 'Ocean',
    tagline: 'Continuous surface, minimal landmarks, deep low-frequency bed',
    affinity: ['calm', 'melancholy', 'wonder'],
  },
  floating_islands: {
    id: 'floating_islands',
    label: 'Floating Islands',
    tagline: 'Suspended terrain, impossible support structures',
    affinity: ['wonder', 'joy', 'curiosity'],
  },
  abstract_space: {
    id: 'abstract_space',
    label: 'Abstract Emotional Space',
    tagline: 'No literal setting recovered — affect rendered directly',
    affinity: ['fear', 'stress', 'wonder'],
  },
  grand_architecture: {
    id: 'grand_architecture',
    label: 'Grand Architecture',
    tagline: 'Monumental interior volume, strong symmetry priors',
    affinity: ['wonder', 'stress', 'nostalgia'],
  },
}

export type WeatherKind = 'clear' | 'overcast' | 'fog' | 'rain' | 'snow' | 'dust' | 'ashfall'
export type TimeOfDay = 'dawn' | 'morning' | 'noon' | 'afternoon' | 'golden' | 'dusk' | 'night'
export type ParticleKind = 'dust' | 'pollen' | 'embers' | 'snow' | 'rain' | 'motes' | 'none'
export type SoundLayerKind =
  | 'wind'
  | 'waves'
  | 'rain'
  | 'birds'
  | 'traffic'
  | 'voices'
  | 'hum'
  | 'chimes'
  | 'heartbeat'

export interface SoundLayer {
  kind: SoundLayerKind
  /** 0–1 mix level. */
  gain: number
  /** Confidence that this layer belongs to the memory, 0–1. */
  confidence: number
}

/** Colour grade for the whole reconstruction, hex strings. */
export interface Palette {
  key: string
  fill: string
  rim: string
  fog: string
  sky: string
  ground: string
  accent: string
}

export interface LightingParams {
  timeOfDay: TimeOfDay
  /** Sun/key elevation in degrees above horizon. */
  sunElevation: number
  sunAzimuth: number
  /** 0–1. */
  intensity: number
  /** Colour temperature in Kelvin; nostalgia pushes this warm. */
  temperature: number
  ambient: number
  /** 0–1 strength of god rays / volumetric shafts. */
  volumetric: number
}

export interface AtmosphereParams {
  weather: WeatherKind
  /** Exponential fog density. */
  fogDensity: number
  particles: ParticleKind
  particleDensity: number
  /** 0–1 wind strength driving foliage and particle drift. */
  wind: number
}

/** A structural region of the reconstruction with its own confidence. */
export interface RegionParams {
  id: string
  label: string
  /** World-space centre. */
  position: [number, number, number]
  radius: number
  /** 0–1 — below `fragmentThreshold` the region renders as broken geometry. */
  confidence: number
  /** Structure archetype the generator instantiates. */
  archetype: 'building' | 'terrain' | 'monument' | 'corridor' | 'grove' | 'water' | 'void'
  /** Regions gated behind a fragment reveal start hidden. */
  lockedBy: string | null
}

export interface SceneParameters {
  seed: number
  biome: Biome
  palette: Palette
  lighting: LightingParams
  atmosphere: AtmosphereParams
  /** 0–1 how far apart the world reads — drives terrain and structure scale. */
  spatialScale: number
  /** 0–1 inferred density of people present in the memory. */
  populationDensity: number
  /** 0–1 density of discrete objects. */
  objectDensity: number
  /** 0–1 how strongly the memory reads as emotionally charged. */
  emotionalIntensity: number
  /** 0–1 ambient motion of the world itself. */
  movement: number
  /** Below this confidence a region stays fragmented. */
  fragmentThreshold: number
  /** 0–1 how strongly geometry departs from euclidean expectations. */
  distortion: number
  regions: RegionParams[]
  soundscape: SoundLayer[]
}

/** A single inferred episode within the memory. */
export interface MemoryFragment {
  id: string
  title: string
  /** AI-written description of the inferred moment. */
  description: string
  position: [number, number, number]
  emotion: EmotionLabel
  confidence: number
  /** Inferred sensory particulars, e.g. "warm air", "distant voices". */
  sensoryDetails: string[]
  relatedIds: string[]
  /** 0–1 position along the reconstructed narrative. */
  timelinePosition: number
  /** Plain-language account of what in the signal produced this. */
  neuralEvidence: string
  /** Session-relative timestamp of the window this was inferred from. */
  sourceTime: number
  /** Region revealed when this fragment is opened, if any. */
  unlocksRegion: string | null
  revealed: boolean
}

export type ReconstructionStatus =
  | 'idle'
  | 'acquiring'
  | 'processing'
  | 'reconstructing'
  | 'stabilised'
  | 'error'

export interface ReconstructionState {
  status: ReconstructionStatus
  /** 0–1 overall build progress of the environment. */
  progress: number
  /** 0–1 aggregate neural confidence backing the reconstruction. */
  confidence: number
  /** 0–1 internal consistency of the reconstruction across time. */
  coherence: number
  /** Region id → 0–1 confidence, updated as evidence accumulates. */
  regionConfidence: Record<string, number>
  scene: SceneParameters | null
  fragments: MemoryFragment[]
}

export interface MemoryNarrative {
  /** One-paragraph AI account of the reconstruction. */
  summary: string
  /** Short evocative title, e.g. "Late Afternoon, Familiar Interior". */
  title: string
  /** Bulleted interpretation notes. */
  observations: string[]
  /** Explicit statement of what the model could not recover. */
  uncertainties: string[]
  /** True when written by an LLM rather than the local template engine. */
  generatedByModel: boolean
}

export type SessionSource = 'simulated' | 'uploaded' | 'device'

export interface SessionSummary {
  id: string
  title: string
  createdAt: string
  durationSeconds: number
  source: SessionSource
  biome: Biome
  primaryEmotion: EmotionLabel
  dominantCognitiveState: CognitiveState
  confidence: number
  coherence: number
  fragmentCount: number
  bookmarked: boolean
  /** Increments each time the session is re-reconstructed with new evidence. */
  version: number
}

export interface SessionRecord extends SessionSummary {
  scene: SceneParameters
  fragments: MemoryFragment[]
  narrative: MemoryNarrative
  /** Confidence sampled over the session, for the progression chart. */
  confidenceTrace: { t: number; confidence: number; coherence: number }[]
  /** Emotion probabilities sampled over the session. */
  emotionTrace: { t: number; emotion: EmotionLabel; intensity: number }[]
  /** Prior versions, newest first — powers the evolution comparison. */
  history: SessionVersion[]
}

export interface SessionVersion {
  version: number
  createdAt: string
  confidence: number
  coherence: number
  fragmentCount: number
  /** What changed relative to the previous version. */
  delta: string
}
