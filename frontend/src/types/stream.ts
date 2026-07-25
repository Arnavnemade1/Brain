/**
 * WebSocket protocol between the reconstruction console and the backend.
 */

import type { NeuralFrame } from './neural'
import type { PipelineStageId, PipelineStageState } from './pipeline'
import type {
  FidelityMetrics,
  FrameFidelity,
  LatentPoint,
  ReconstructedFrame,
  ReconstructionReport,
  ReconstructionStatus,
  SceneEstimate,
  StimulusClip,
  VisualFeatures,
} from './memory'

export interface SessionOpenedEvent {
  type: 'session.opened'
  sessionId: string
  sampleRate: number
  channels: string[]
  clip: StimulusClip
}

/** Synchronization result, emitted once before reconstruction begins. */
export interface SyncEvent {
  type: 'sync'
  /** Offset in seconds between the recording and playback clocks. */
  offsetSeconds: number
  /** Fitted clock drift in parts per million. */
  driftPpm: number
  /** Residual alignment error in milliseconds after correction. */
  residualMs: number
  markersMatched: number
  markersExpected: number
}

export interface FrameEvent {
  type: 'frame'
  frame: NeuralFrame
}

export interface PipelineEvent {
  type: 'pipeline'
  stages: PipelineStageState[]
}

export interface LatentEvent {
  type: 'latent'
  point: LatentPoint
}

/** A reconstructed frame, paired with the reference it will be scored against. */
export interface ReconstructionFrameEvent {
  type: 'reconstruction.frame'
  frame: ReconstructedFrame
  /** Reference frame at the same grid resolution, base64 RGB. */
  reference: string
  scene: SceneEstimate
  truth: VisualFeatures
  fidelity: FrameFidelity
}

export interface ProgressEvent {
  type: 'progress'
  status: ReconstructionStatus
  progress: number
  /** Running composite fidelity so far. */
  fidelity: number
}

export interface MetricsEvent {
  type: 'metrics'
  metrics: FidelityMetrics
}

export interface ReportEvent {
  type: 'report'
  report: ReconstructionReport
}

export interface SessionClosedEvent {
  type: 'session.closed'
  sessionId: string
  reason: 'complete' | 'stopped' | 'error'
  message: string | null
}

export interface StreamErrorEvent {
  type: 'error'
  stage: PipelineStageId | null
  message: string
  recoverable: boolean
}

export type StreamEvent =
  | SessionOpenedEvent
  | SyncEvent
  | FrameEvent
  | PipelineEvent
  | LatentEvent
  | ReconstructionFrameEvent
  | ProgressEvent
  | MetricsEvent
  | ReportEvent
  | SessionClosedEvent
  | StreamErrorEvent

export type StreamEventType = StreamEvent['type']

/* -------------------------------------------------------------------------- */
/* Client → server                                                            */
/* -------------------------------------------------------------------------- */

export interface StartCommand {
  type: 'start'
  speed: number
}

export interface PauseCommand {
  type: 'pause'
}

export interface ResumeCommand {
  type: 'resume'
}

export interface StopCommand {
  type: 'stop'
}

export interface SeekCommand {
  type: 'seek'
  t: number
}

export interface SpeedCommand {
  type: 'speed'
  speed: number
}

export type StreamCommand =
  | StartCommand
  | PauseCommand
  | ResumeCommand
  | StopCommand
  | SeekCommand
  | SpeedCommand

export type ConnectionStatus =
  | 'idle'
  | 'connecting'
  | 'open'
  | 'reconnecting'
  | 'closed'
  | 'failed'
