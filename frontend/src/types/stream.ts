/**
 * WebSocket protocol between the session console and the backend.
 *
 * The backend pushes a discriminated union of events over `/ws/session/{id}`;
 * the client sends a small set of control commands back on the same socket.
 */

import type { NeuralFrame } from './neural'
import type { PipelineStageId, PipelineStageState } from './pipeline'
import type {
  MemoryFragment,
  MemoryNarrative,
  ReconstructionStatus,
  SceneParameters,
} from './memory'

export interface SessionOpenedEvent {
  type: 'session.opened'
  sessionId: string
  sampleRate: number
  channels: string[]
  /** Total session length in seconds; null for open-ended live streams. */
  duration: number | null
  source: 'simulated' | 'uploaded' | 'device'
}

export interface FrameEvent {
  type: 'frame'
  frame: NeuralFrame
}

export interface PipelineEvent {
  type: 'pipeline'
  stages: PipelineStageState[]
}

/** Emitted once the context model first commits to a scene. */
export interface SceneEvent {
  type: 'scene'
  scene: SceneParameters
}

/** Incremental refinement of an existing scene as evidence accumulates. */
export interface SceneUpdateEvent {
  type: 'scene.update'
  regionConfidence: Record<string, number>
  /** Regions promoted from fragmented to resolved this tick. */
  resolvedRegions: string[]
  /** Parameter drift as the interpretation sharpens. */
  patch: Partial<SceneParameters>
}

export interface FragmentEvent {
  type: 'fragment'
  fragment: MemoryFragment
}

export interface ReconstructionEvent {
  type: 'reconstruction'
  status: ReconstructionStatus
  progress: number
  confidence: number
  coherence: number
}

export interface NarrativeEvent {
  type: 'narrative'
  narrative: MemoryNarrative
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
  | FrameEvent
  | PipelineEvent
  | SceneEvent
  | SceneUpdateEvent
  | FragmentEvent
  | ReconstructionEvent
  | NarrativeEvent
  | SessionClosedEvent
  | StreamErrorEvent

export type StreamEventType = StreamEvent['type']

/* -------------------------------------------------------------------------- */
/* Client → server                                                            */
/* -------------------------------------------------------------------------- */

export interface StartCommand {
  type: 'start'
  /** Playback rate multiplier, 0.25–4. */
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
  /** Target time in seconds from session start. */
  t: number
}

export interface SpeedCommand {
  type: 'speed'
  speed: number
}

/** Ask the engine to spend extra evidence resolving one region. */
export interface FocusCommand {
  type: 'focus'
  regionId: string
}

export type StreamCommand =
  | StartCommand
  | PauseCommand
  | ResumeCommand
  | StopCommand
  | SeekCommand
  | SpeedCommand
  | FocusCommand

export type ConnectionStatus =
  | 'idle'
  | 'connecting'
  | 'open'
  | 'reconnecting'
  | 'closed'
  | 'failed'
