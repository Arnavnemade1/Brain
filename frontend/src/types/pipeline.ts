/**
 * The Memory Reconstruction Engine pipeline.
 *
 * Video in, reconstructed video out. Stage identifiers are stable and shared
 * with the backend, which emits `pipeline` events keyed by these ids.
 */

export const PIPELINE_STAGE_IDS = [
  'stimulus',
  'acquisition',
  'synchronization',
  'artifact_removal',
  'encoder',
  'latent',
  'scene',
  'frames',
  'refinement',
  'replay',
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
  band: 'stimulus' | 'acquisition' | 'encoding' | 'reconstruction'
}

/** Live state of one stage, streamed from the backend. */
export interface PipelineStageState {
  id: PipelineStageId
  status: StageStatus
  /** 0–1 completion of the current unit of work. */
  progress: number
  /** Milliseconds spent in the most recent pass. */
  latencyMs: number
  /** Units processed since session start (frames, windows, latents…). */
  throughput: number
  /** Human-readable current operation. */
  message: string
  /** Stage-local confidence in its own output, 0–1. */
  confidence: number
}

export const PIPELINE_STAGES: readonly PipelineStageMeta[] = [
  {
    id: 'stimulus',
    label: 'Video',
    summary: 'Reference stimulus with known ground truth',
    detail:
      'The clip the subject watched. Because the source exists, every downstream claim can be checked against it — this is why MindScape works on memories of a stimulus rather than on dreams.',
    band: 'stimulus',
  },
  {
    id: 'acquisition',
    label: 'EEG Recording',
    summary: 'Stimulus-locked multi-channel acquisition',
    detail:
      'Continuous EEG captured while the clip played, with event markers written at playback start and at each scene transition so the two timelines can be tied together.',
    band: 'acquisition',
  },
  {
    id: 'synchronization',
    label: 'Signal Synchronization',
    summary: 'Marker alignment and clock-drift correction',
    detail:
      'Recording and playback clocks run independently and drift apart over minutes. Markers are matched to stimulus events and a linear drift model is fitted, so a neural response is attributed to the frame that actually caused it.',
    band: 'acquisition',
  },
  {
    id: 'artifact_removal',
    label: 'Artifact Removal',
    summary: 'Re-reference, band-limit, suppress non-neural signal',
    detail:
      'Blinks, muscle activity and mains hum are attenuated rather than excised, preserving the temporal continuity the encoder depends on. Channels beyond recovery are excluded and reported.',
    band: 'acquisition',
  },
  {
    id: 'encoder',
    label: 'Temporal Neural Encoder',
    summary: 'Windowed signal to feature sequence',
    detail:
      'Overlapping windows are reduced to spectral, spatial and evoked-response descriptors across the visual hierarchy — occipital luminance response, motion-sensitive activity, transient responses at scene changes.',
    band: 'encoding',
  },
  {
    id: 'latent',
    label: 'Neural Memory Latent Space',
    summary: 'Compact representation of remembered content',
    detail:
      'Encoder output is projected into a fixed-dimensional latent trajectory. This is the interface between neural measurement and visual reconstruction; everything after it reads only the latent.',
    band: 'encoding',
  },
  {
    id: 'scene',
    label: 'Scene Reconstruction',
    summary: 'Global visual properties per moment',
    detail:
      'The latent is decoded into scene-level quantities: overall brightness, dominant colour, motion magnitude and direction, and the boundaries where the scene changed.',
    band: 'reconstruction',
  },
  {
    id: 'frames',
    label: 'Frame Reconstruction',
    summary: 'Spatial layout at recoverable resolution',
    detail:
      'Scene properties and the latent are decoded into a coarse spatial grid. The resolution is set by what the signal actually supports, not by what looks impressive — EEG carries spatial gist, not detail.',
    band: 'reconstruction',
  },
  {
    id: 'refinement',
    label: 'Video Refinement',
    summary: 'Temporal consistency and upsampling',
    detail:
      'Independently decoded frames flicker. Refinement enforces continuity across time, holds structure through scene boundaries, and upsamples for display without inventing detail the metrics would then score.',
    band: 'reconstruction',
  },
  {
    id: 'replay',
    label: 'Memory Replay',
    summary: 'Reconstruction scored against the reference',
    detail:
      'The reconstruction is played beside the original and scored — SSIM, PSNR, luminance and motion correlation, scene-boundary detection. Fidelity is the deliverable, not the picture.',
    band: 'reconstruction',
  },
]

export const STAGE_BY_ID: Record<PipelineStageId, PipelineStageMeta> = Object.fromEntries(
  PIPELINE_STAGES.map((s) => [s.id, s]),
) as Record<PipelineStageId, PipelineStageMeta>
