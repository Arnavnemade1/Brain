/**
 * Live session state.
 *
 * Owns the socket, the rolling frame history and the reconstruction being
 * assembled. Everything the console and the 3D engine render comes from here.
 */

import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'

import { api, type SessionCreated } from '@/services/api'
import { SessionSocket } from '@/services/socket'
import { PIPELINE_STAGE_IDS } from '@/types'
import type {
  ConnectionStatus,
  MemoryFragment,
  MemoryNarrative,
  NeuralFrame,
  PipelineStageState,
  ReconstructionStatus,
  SceneParameters,
  StreamCommand,
  StreamEvent,
} from '@/types'

/** Frames retained for the timeline and trend charts. */
const FRAME_HISTORY = 320

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
  /* --- connection --- */
  sessionId: string | null
  meta: SessionCreated | null
  connection: ConnectionStatus
  error: string | null
  starting: boolean

  /* --- transport --- */
  playing: boolean
  speed: number

  /* --- live data --- */
  frame: NeuralFrame | null
  frames: NeuralFrame[]
  stages: PipelineStageState[]

  /* --- reconstruction --- */
  status: ReconstructionStatus
  progress: number
  confidence: number
  coherence: number
  scene: SceneParameters | null
  regionConfidence: Record<string, number>
  /** Regions that crossed the threshold this session, newest last. */
  resolvedRegions: string[]
  fragments: MemoryFragment[]
  narrative: MemoryNarrative | null

  /* --- fragment interaction --- */
  openFragmentId: string | null
  revealedFragmentIds: string[]

  /* --- actions --- */
  startSimulation: (profile: string, duration: number, speed?: number) => Promise<void>
  startUpload: (file: File, speed?: number) => Promise<void>
  send: (command: StreamCommand) => void
  pause: () => void
  resume: () => void
  stop: () => Promise<void>
  setSpeed: (speed: number) => void
  focusRegion: (regionId: string) => void
  openFragment: (id: string | null) => void
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
  speed: 1,
  frame: null,
  frames: [] as NeuralFrame[],
  stages: idleStages(),
  status: 'idle' as ReconstructionStatus,
  progress: 0,
  confidence: 0,
  coherence: 0,
  scene: null,
  regionConfidence: {} as Record<string, number>,
  resolvedRegions: [] as string[],
  fragments: [] as MemoryFragment[],
  narrative: null,
  openFragmentId: null,
  revealedFragmentIds: [] as string[],
}

export const useSessionStore = create<SessionState>()(
  subscribeWithSelector((set, get) => {
    function handleEvent(event: StreamEvent): void {
      switch (event.type) {
        case 'session.opened':
          set({ status: 'acquiring', playing: true })
          break

        case 'frame':
          set((state) => {
            const frames = [...state.frames, event.frame]
            if (frames.length > FRAME_HISTORY) frames.splice(0, frames.length - FRAME_HISTORY)
            return { frame: event.frame, frames }
          })
          break

        case 'pipeline':
          set({ stages: event.stages })
          break

        case 'scene':
          set({
            scene: event.scene,
            regionConfidence: Object.fromEntries(
              event.scene.regions.map((r) => [r.id, r.confidence]),
            ),
            status: 'reconstructing',
          })
          break

        case 'scene.update':
          set((state) => ({
            regionConfidence: { ...state.regionConfidence, ...event.regionConfidence },
            resolvedRegions: [...state.resolvedRegions, ...event.resolvedRegions],
            scene: state.scene
              ? { ...state.scene, ...event.patch, regions: state.scene.regions }
              : state.scene,
          }))
          break

        case 'fragment':
          set((state) =>
            state.fragments.some((f) => f.id === event.fragment.id)
              ? state
              : { fragments: [...state.fragments, event.fragment] },
          )
          break

        case 'reconstruction':
          set({
            status: event.status,
            progress: event.progress,
            confidence: event.confidence,
            coherence: event.coherence,
          })
          break

        case 'narrative':
          set({ narrative: event.narrative })
          break

        case 'session.closed':
          set({
            playing: false,
            status: event.reason === 'error' ? 'error' : 'stabilised',
            error: event.reason === 'error' ? event.message : null,
          })
          break

        case 'error':
          // Recoverable errors annotate the affected stage but leave the
          // session running; only fatal ones surface as a page-level error.
          if (event.recoverable) {
            set((state) => ({
              stages: state.stages.map((stage) =>
                stage.id === event.stage
                  ? { ...stage, status: 'error', message: event.message }
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

      async startSimulation(profile, duration, speed = 1) {
        get().reset()
        set({ starting: true, error: null, speed })
        try {
          const meta = await api.startSimulation({ profile, duration, speed })
          set({ sessionId: meta.sessionId, meta, starting: false })
          connect(meta, speed)
        } catch (error) {
          set({
            starting: false,
            error: error instanceof Error ? error.message : 'Could not start the session.',
          })
          throw error
        }
      },

      async startUpload(file, speed = 1) {
        get().reset()
        set({ starting: true, error: null, speed })
        try {
          const meta = await api.uploadRecording(file, { speed })
          set({ sessionId: meta.sessionId, meta, starting: false })
          connect(meta, speed)
        } catch (error) {
          set({
            starting: false,
            error: error instanceof Error ? error.message : 'Could not read that recording.',
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
          // Best effort: the socket teardown also persists server-side, so a
          // failure here is not worth surfacing to the user.
          await api.stopSession(sessionId).catch(() => undefined)
        }
      },

      setSpeed(speed) {
        set({ speed })
        socket?.send({ type: 'speed', speed })
      },

      focusRegion(regionId) {
        socket?.send({ type: 'focus', regionId })
      },

      openFragment(id) {
        set((state) => ({
          openFragmentId: id,
          revealedFragmentIds:
            id && !state.revealedFragmentIds.includes(id)
              ? [...state.revealedFragmentIds, id]
              : state.revealedFragmentIds,
        }))

        // Opening a low-confidence fragment spends evidence resolving the
        // region it gates — this is the "memory becoming clearer" mechanic.
        if (id) {
          const fragment = get().fragments.find((f) => f.id === id)
          if (fragment?.unlocksRegion) {
            socket?.send({ type: 'focus', regionId: fragment.unlocksRegion })
          }
        }
      },

      reset() {
        socket?.close()
        socket = null
        set({ ...initial, stages: idleStages(), frames: [] })
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

export const selectElapsed = (state: SessionState): number => state.frame?.t ?? 0

export const selectResolvedCount = (state: SessionState): number => {
  if (!state.scene) return 0
  return state.scene.regions.filter(
    (region) =>
      (state.regionConfidence[region.id] ?? region.confidence) >=
      state.scene!.fragmentThreshold,
  ).length
}

export const selectIsLive = (state: SessionState): boolean =>
  state.connection === 'open' && state.playing
