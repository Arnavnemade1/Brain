/**
 * Every measured result in the project, in one place.
 *
 * Collected here rather than written into the page so that a figure appears
 * exactly once. The backing numbers live in `backend/*.md` alongside the code
 * that produced them, and the `source` field on each domain says which.
 *
 * Two conventions hold throughout, and both exist to stop a chart flattering
 * the work:
 *
 * - **Chance is carried with the value, never assumed.** A 200-way task and a
 *   two-way task cannot share an axis without it, and "36.7%" means something
 *   very different against 20% than against 50%.
 * - **Failures are typed the same as successes.** A result that did not work
 *   is a `status: 'fails'` entry with its number intact, not an omission. The
 *   emotion work is here in full.
 */

export type Status = 'works' | 'weak' | 'fails'

export interface Finding {
  label: string
  /** 0-1. An accuracy, or a correlation for `unit: 'r'`. */
  value: number
  /** 0-1. What the same measurement returns on a signal carrying nothing. */
  chance: number
  unit: 'accuracy' | 'r'
  status: Status
  note?: string
}

export interface Domain {
  id: string
  eyebrow: string
  title: string
  summary: string
  /** Where the numbers come from. */
  source: string
  dataset: string
  video?: {
    src: string
    poster: string
    label: string
    meta: string
    note: string
  }
  findings: Finding[]
}

export const DOMAINS: Domain[] = [
  {
    id: 'temporal',
    eyebrow: 'Temporal structure',
    title: 'When something happened is the easiest thing to recover.',
    summary:
      'Someone playing a game with a headset on, decoded every 250 ms with no event log at inference. Detection of *that* an event occurred is the strongest result in the project. Identifying *which* event it was fails outright — it sits below its own chance level.',
    source: 'backend/GAMEPLAY.md',
    dataset: 'ds003517 · 17 subjects · leave-one-subject-out',
    video: {
      src: '/memory-video.mp4',
      poster: '/memory-video-poster.png',
      label: 'Gameplay reconstructed from EEG beside what actually happened',
      meta: 'sub-005 · 60 s continuous',
      note: 'The rhythm of the session is genuinely recovered. Which particular event occurred at each moment is not, and the right panel dims to show it.',
    },
    findings: [
      { label: 'Event detection — recall', value: 0.766, chance: 0.5, unit: 'accuracy', status: 'works' },
      { label: 'Event detection — precision', value: 0.726, chance: 0.5, unit: 'accuracy', status: 'works' },
      { label: 'Reward vs error', value: 0.607, chance: 0.5, unit: 'accuracy', status: 'weak' },
      {
        label: 'Which event (3-class)',
        value: 0.302,
        chance: 0.333,
        unit: 'accuracy',
        status: 'fails',
        note: 'Below chance. Detection works; classification does not.',
      },
    ],
  },
  {
    id: 'motor',
    eyebrow: 'Physical action',
    title: 'What the body did, from sensorimotor rhythm.',
    summary:
      'Movement suppresses the mu and beta rhythms over the motor strip, contralateral to the limb. It is a large, spatially organised effect that does not depend on recovering fine detail — which is why it beats anything visual here.',
    source: 'backend/MOTOR.md',
    dataset: 'eegmmidb · 20 subjects · leave-one-run-out',
    video: {
      src: '/motor-video.mp4',
      poster: '/motor-video-poster.png',
      label: 'Limb movement recovered from EEG beside the movement performed',
      meta: 'S002 · 75 s · real time',
      note: 'The right figure lights each limb by the decoder’s odds rather than its single best guess, so uncertainty is shown rather than resolved.',
    },
    findings: [
      { label: 'Moving vs still', value: 0.763, chance: 0.5, unit: 'accuracy', status: 'works' },
      { label: 'Fists vs feet', value: 0.755, chance: 0.5, unit: 'accuracy', status: 'works' },
      { label: 'Which fist', value: 0.62, chance: 0.5, unit: 'accuracy', status: 'weak' },
      {
        label: 'Exact action (5-class)',
        value: 0.367,
        chance: 0.2,
        unit: 'accuracy',
        status: 'weak',
        note: 'Ranges 24–60% between people.',
      },
    ],
  },
  {
    id: 'appearance',
    eyebrow: 'Appearance',
    title: 'How a scene looked, rather than what was in it.',
    summary:
      'Naming the object on screen works 7.6% of the time. Regressed onto the visual statistics of the same images — edges, brightness, colour, contrast — the same recordings do far better. A landscape is exactly the kind of output low-level statistics can honestly drive.',
    source: 'backend/ENVIRONMENT.md',
    dataset: 'ds004018 · 200 stimuli · held out',
    video: {
      src: '/environment-video.mp4',
      poster: '/environment-video-poster.png',
      label: 'Two landscapes, one graded from the photograph and one from EEG alone',
      meta: 'sub-01 · 60 s · truth vs decoded',
      note: 'Same terrain, same camera, same sun — only appearance is decoded, and a property must clear r = 0.35 held out before it may move anything.',
    },
    findings: [
      { label: 'Edge density', value: 0.69, chance: 0.07, unit: 'r', status: 'works' },
      { label: 'Mean red', value: 0.64, chance: 0.02, unit: 'r', status: 'works' },
      { label: 'Saturation', value: 0.64, chance: 0.05, unit: 'r', status: 'works' },
      { label: 'Luminance', value: 0.62, chance: 0.01, unit: 'r', status: 'works' },
      { label: 'Contrast', value: 0.58, chance: 0.01, unit: 'r', status: 'works' },
      { label: 'Coverage', value: 0.53, chance: 0.05, unit: 'r', status: 'works' },
      {
        label: 'Per-moment agreement',
        value: 0.38,
        chance: 0.0,
        unit: 'r',
        status: 'weak',
        note: 'Whole-scene, decoded against true grading.',
      },
    ],
  },
  {
    id: 'semantic',
    eyebrow: 'Semantic content',
    title: 'Which particular thing was seen is mostly out of reach.',
    summary:
      'Retrieval over 200 real photographs, held out throughout. What comes back reliably is the kind of thing seen, not which one. This is the weakest domain in the project and the reason the image-matching feature was retired from the product.',
    source: 'backend/REALLIFE.md',
    dataset: 'ds004018 · 3 subjects · 8 held-out viewings',
    findings: [
      { label: 'Same object in top 5', value: 0.424, chance: 0.1, unit: 'accuracy', status: 'weak' },
      { label: 'Top-1 correct animacy', value: 0.624, chance: 0.5, unit: 'accuracy', status: 'weak' },
      { label: 'Top-1 correct category', value: 0.25, chance: 0.1, unit: 'accuracy', status: 'weak' },
      { label: 'Exact photo in top 5', value: 0.201, chance: 0.025, unit: 'accuracy', status: 'weak' },
      {
        label: 'Exact photo recovered',
        value: 0.076,
        chance: 0.005,
        unit: 'accuracy',
        status: 'weak',
        note: 'Fifteen times chance, and still wrong nine times in ten.',
      },
    ],
  },
  {
    id: 'emotion',
    eyebrow: 'Affect',
    title: 'Emotion decoding fails in every regime tested.',
    summary:
      'Forty people watching emotionally evocative film clips. A model fitted on eight of them does not read the ninth — the shuffled control scored higher than the real model. A later 294-correlation search found nothing surviving correction, and adding the missing control to the within-subject regime showed the shuffle beats the real model there too. Nothing affective is decoded here in any regime tested.',
    source: 'backend/EMOTION_CORRELATION.md',
    dataset: 'ds003751 · 9 subjects · 95 trials',
    findings: [
      {
        label: 'Quadrant — shuffled control',
        value: 0.389,
        chance: 0.25,
        unit: 'accuracy',
        status: 'fails',
        note: 'The control. It beat the real model.',
      },
      { label: 'Quadrant (4-way)', value: 0.347, chance: 0.25, unit: 'accuracy', status: 'fails' },
      { label: 'Valence, cross-subject', value: 0.492, chance: 0.5, unit: 'accuracy', status: 'fails' },
      { label: 'Arousal, cross-subject', value: 0.479, chance: 0.5, unit: 'accuracy', status: 'fails' },
      {
        label: 'Arousal, within subject',
        value: 0.588,
        chance: 0.631,
        unit: 'accuracy',
        status: 'fails',
        note: 'Previously reported as the one surviving affective reading, against an assumed 50% chance. Its shuffled control scores 63.1% — leave-one-out over ten trials is inflated on its own — so the real model sits below its own baseline. Claim withdrawn.',
      },
      {
        label: 'Widest search: 294 correlations',
        value: 0.0,
        chance: 0.0,
        unit: 'r',
        status: 'fails',
        note: '98 predictors x 3 targets, permutation nulls, FDR. Zero survive at q < 0.05; 19 hits at uncorrected p < 0.05 against 15 expected by chance.',
      },
    ],
  },
]

/** Frequency profile from the artifact controls. Chance is 20%. */
export const FREQUENCY_CONTROL = [
  { band: 'mu 8–13', value: 0.328 },
  { band: 'beta 13–30', value: 0.401 },
  { band: '30–45', value: 0.305 },
  { band: '45–70', value: 0.271 },
]

/** Spatial profile from the artifact controls. Chance is 20%. */
export const SPATIAL_CONTROL = [
  { site: 'Motor cortex', value: 0.345, verdict: 'cortical' as const },
  { site: 'Occipital', value: 0.31, verdict: 'cortical' as const },
  { site: 'Temporalis (jaw muscle)', value: 0.224, verdict: 'chance' as const },
]

export const CONTROL_CHANCE = 0.2
