/**
 * Live reconstruction session state.
 *
 * Owns the socket, the rolling neural history, and the reconstructed frames
 * paired with the reference they are scored against.
 */

import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'

import { api, type SessionCreated } from '@/services/api'
import { SessionSocket } from '@/services/socket'
import { PIPELINE_STAGE_IDS } from '@/types'
import type {
  ConnectionStatus,
  FidelityMetrics,
  FrameFidelity,
  LatentPoint,
  NeuralFrame,
  PipelineStageState,
  ReconstructionReport,
  ReconstructionStatus,
  SceneEstimate,
  StreamCommand,
  StreamEvent,
  VisualFeatures,
} from '@/types'

/** Neural frames retained for trend charts. */
const NEURAL_HISTORY = 240

/**
 * One reconstructed moment, paired with its reference.
 *
 * Held together deliberately: every claim the UI makes about a frame is a
 * comparison, so separating them would let the two drift out of step.
 */
export interface ReplayFrame {
  index: number
  t: number
  /** Base64 RGB at grid resolution. */
  pixels: string
  reference: string
  scene: SceneEstimate
  truth: VisualFeatures
  fidelity: FrameFidelity
}

export interface SyncInfo {
  offsetSeconds: number
  driftPpm: number
  residualMs: number
  markersMatched: number
  markersExpected: number
}

function idleStages(): PipelineStageState[] {
  return PIPELINE_STAGE_IDS.map((id) => ({
    id,
    status: 'idle',
    progress: 0,
    latencyMs: 0,
    throughput: 0,
    message: 'Awaiting data',
    confidence: 0,
  }))
}

interface SessionState {
  sessionId: string | null
  meta: SessionCreated | null
  connection: ConnectionStatus
  error: string | null
  starting: boolean

  playing: boolean
  speed: number

  frame: NeuralFrame | null
  frames: NeuralFrame[]
  stages: PipelineStageState[]
  latents: LatentPoint[]

  sync: SyncInfo | null
  status: ReconstructionStatus
  progress: number
  fidelity: number

  replay: ReplayFrame[]
  metrics: FidelityMetrics | null
  report: ReconstructionReport | null

  /** Index into `replay` currently shown in the player. */
  playhead: number
  /** Follow the newest frame while streaming. */
  following: boolean

  startSession: (clip: string, condition: string, speed?: number) => Promise<void>
  send: (command: StreamCommand) => void
  pause: () => void
  resume: () => void
  stop: () => Promise<void>
  setSpeed: (speed: number) => void
  setPlayhead: (index: number) => void
  setFollowing: (following: boolean) => void
  reset: () => void
  clearError: () => void
}

let socket: SessionSocket | null = null

const initial = {
  sessionId: null,
  meta: null,
  connection: 'idle' as ConnectionStatus,
  error: null,
  starting: false,
  playing: false,
  speed: 2,
  frame: null,
  frames: [] as NeuralFrame[],
  stages: idleStages(),
  latents: [] as LatentPoint[],
  sync: null,
  status: 'idle' as ReconstructionStatus,
  progress: 0,
  fidelity: 0,
  replay: [] as ReplayFrame[],
  metrics: null,
  report: null,
  playhead: 0,
  following: true,
}

export const useSessionStore = create<SessionState>()(
  subscribeWithSelector((set, get) => {
    function handleEvent(event: StreamEvent): void {
      switch (event.type) {
        case 'session.opened':
          set({ status: 'synchronising', playing: true })
          break

        case 'sync':
          set({
            sync: {
              offsetSeconds: event.offsetSeconds,
              driftPpm: event.driftPpm,
              residualMs: event.residualMs,
              markersMatched: event.markersMatched,
              markersExpected: event.markersExpected,
            },
            status: 'encoding',
          })
          break

        case 'frame':
          set((state) => {
            const frames = [...state.frames, event.frame]
            if (frames.length > NEURAL_HISTORY) {
              frames.splice(0, frames.length - NEURAL_HISTORY)
            }
            return { frame: event.frame, frames }
          })
          break

        case 'pipeline':
          set({ stages: event.stages })
          break

        case 'latent':
          set((state) => {
            const latents = [...state.latents, event.point]
            if (latents.length > NEURAL_HISTORY) {
              latents.splice(0, latents.length - NEURAL_HISTORY)
            }
            return { latents }
          })
          break

        case 'reconstruction.frame':
          set((state) => {
            const replay = [
              ...state.replay,
              {
                index: event.frame.index,
                t: event.frame.t,
                pixels: event.frame.pixels,
                reference: event.reference,
                scene: event.scene,
                truth: event.truth,
                fidelity: event.fidelity,
              },
            ]
            return {
              replay,
              // Only advance the playhead while following, so scrubbing back
              // through earlier frames is not yanked forward by new arrivals.
              playhead: state.following ? replay.length - 1 : state.playhead,
            }
          })
          break

        case 'progress':
          set({
            status: event.status,
            progress: event.progress,
            fidelity: event.fidelity,
          })
          break

        case 'metrics':
          set({ metrics: event.metrics, fidelity: event.metrics.composite })
          break

        case 'report':
          set({ report: event.report })
          break

        case 'session.closed':
          // The server closes the socket right after this; tell the client so
          // a normal completion is not mistaken for a dropped connection.
          socket?.finish()
          set({
            playing: false,
            status: event.reason === 'error' ? 'error' : 'complete',
            error: event.reason === 'error' ? event.message : null,
          })
          break

        case 'error':
          if (event.recoverable) {
            set((state) => ({
              stages: state.stages.map((stage) =>
                stage.id === event.stage
                  ? { ...stage, status: 'degraded', message: event.message }
                  : stage,
              ),
            }))
          } else {
            set({ error: event.message, status: 'error', playing: false })
          }
          break
      }
    }

    function connect(meta: SessionCreated, speed: number): void {
      socket?.close()
      socket = new SessionSocket(meta.sessionId, {
        onEvent: handleEvent,
        onStatus: (connection) => set({ connection }),
        onError: (message) => set({ error: message }),
      })
      socket.connect()
      socket.send({ type: 'start', speed })
    }

    return {
      ...initial,

      async startSession(clip, condition, speed = 2) {
        get().reset()
        set({ starting: true, error: null, speed })
        try {
          const meta = await api.startSession({ clip, condition, speed })
          set({ sessionId: meta.sessionId, meta, starting: false })
          connect(meta, speed)
        } catch (error) {
          set({
            starting: false,
            error:
              error instanceof Error ? error.message : 'Could not start the session.',
          })
          throw error
        }
      },

      send(command) {
        socket?.send(command)
      },

      pause() {
        socket?.send({ type: 'pause' })
        set({ playing: false })
      },

      resume() {
        socket?.send({ type: 'resume' })
        set({ playing: true })
      },

      async stop() {
        const { sessionId } = get()
        socket?.send({ type: 'stop' })
        set({ playing: false })
        if (sessionId) {
          await api.stopSession(sessionId).catch(() => undefined)
        }
      },

      setSpeed(speed) {
        set({ speed })
        socket?.send({ type: 'speed', speed })
      },

      setPlayhead(index) {
        const { replay } = get()
        const clamped = Math.max(0, Math.min(index, replay.length - 1))
        set({ playhead: clamped, following: clamped >= replay.length - 1 })
      },

      setFollowing(following) {
        set((state) => ({
          following,
          playhead: following ? Math.max(0, state.replay.length - 1) : state.playhead,
        }))
      },

      reset() {
        socket?.close()
        socket = null
        set({ ...initial, stages: idleStages(), frames: [], replay: [], latents: [] })
      },

      clearError() {
        set({ error: null })
      },
    }
  }),
)

/* -------------------------------------------------------------------------- */
/* Selectors                                                                  */
/* -------------------------------------------------------------------------- */

export const selectCurrentReplay = (state: SessionState): ReplayFrame | null =>
  state.replay[state.playhead] ?? null

export const selectIsLive = (state: SessionState): boolean =>
  state.connection === 'open' && state.playing
