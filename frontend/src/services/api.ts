/**
 * REST client.
 *
 * One thin wrapper so every call surfaces backend `detail` messages as real
 * errors — the UI shows those verbatim rather than a generic failure.
 */

import type {
  RealDataStatus,
  RealResults,
  RealViewing,
  SessionRecord,
  SessionSummary,
  StimulusClip,
} from '@/types'

export class ApiError extends Error {
  readonly status: number
  readonly path: string

  constructor(message: string, status: number, path: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.path = path
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, {
      ...init,
      headers: {
        ...(init?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
        ...init?.headers,
      },
    })
  } catch (cause) {
    throw new ApiError(
      'Cannot reach the MindScape backend. Is it running on port 8000?',
      0,
      path,
    )
  }

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      if (typeof body?.detail === 'string') detail = body.detail
      else if (Array.isArray(body?.detail)) detail = body.detail[0]?.msg ?? detail
    } catch {
      // Response body was not JSON — keep the status text.
    }
    throw new ApiError(detail, response.status, path)
  }

  if (response.status === 204) return undefined as T
  const contentType = response.headers.get('content-type') ?? ''
  if (!contentType.includes('application/json')) {
    return (await response.text()) as T
  }
  return (await response.json()) as T
}

/* -------------------------------------------------------------------------- */
/* Types                                                                      */
/* -------------------------------------------------------------------------- */

export interface SystemInfo {
  version: string
  sampleRate: number
  windowSeconds: number
  windowOverlap: number
  lineFrequency: number
  llmNarratives: boolean
  stageOrder: string[]
  storedSessions: number
  activeSessions: number
  decodersFitted: boolean
  decoderTrainedOn: string[]
}

export interface RecordingCondition {
  id: string
  label: string
  description: string
  artifactLevel: number
  responseGain: number
}

export interface SessionCreated {
  sessionId: string
  sampleRate: number
  channels: string[]
  clip: StimulusClip
  condition: string
  streamUrl: string
  windowSeconds: number
  hopSeconds: number
}

export interface ElectrodeInfo {
  label: string
  x: number
  y: number
  region: string
}

/* -------------------------------------------------------------------------- */
/* Endpoints                                                                  */
/* -------------------------------------------------------------------------- */

export const api = {
  system: () => request<SystemInfo>('/api/system'),

  montage: () => request<ElectrodeInfo[]>('/api/montage'),

  clips: () => request<StimulusClip[]>('/api/sessions/clips'),

  conditions: () => request<RecordingCondition[]>('/api/sessions/conditions'),

  startSession: (body: { clip: string; condition: string; speed?: number }) =>
    request<SessionCreated>('/api/sessions', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  stopSession: (sessionId: string) =>
    request<{ stopped: string }>(`/api/sessions/${sessionId}`, { method: 'DELETE' }),

  listSessions: (params?: { q?: string; bookmarked?: boolean; limit?: number }) => {
    const search = new URLSearchParams()
    if (params?.q) search.set('q', params.q)
    if (params?.bookmarked != null) search.set('bookmarked', String(params.bookmarked))
    if (params?.limit) search.set('limit', String(params.limit))
    const suffix = search.toString() ? `?${search}` : ''
    return request<SessionSummary[]>(`/api/library/sessions${suffix}`)
  },

  getSession: (id: string) => request<SessionRecord>(`/api/library/sessions/${id}`),

  deleteSession: (id: string) =>
    request<{ deleted: string }>(`/api/library/sessions/${id}`, { method: 'DELETE' }),

  setBookmark: (id: string, bookmarked: boolean) =>
    request<SessionSummary>(`/api/library/sessions/${id}/bookmark`, {
      method: 'POST',
      body: JSON.stringify({ bookmarked }),
    }),

  /* --- real-subject EEG --- */

  realStatus: () => request<RealDataStatus>('/api/real/status'),

  realResults: () => request<RealResults>('/api/real/results'),

  realViewings: (subject: string) =>
    request<{ subject: string; viewings: number[]; trainingTrials: number }>(
      `/api/real/subjects/${subject}/viewings`,
    ),

  realReconstruct: (
    subject: string,
    params?: { stimulus?: number; trials?: number; seed?: number },
  ) => {
    const search = new URLSearchParams()
    if (params?.stimulus != null) search.set('stimulus', String(params.stimulus))
    if (params?.trials != null) search.set('trials', String(params.trials))
    if (params?.seed != null) search.set('seed', String(params.seed))
    const suffix = search.toString() ? `?${search}` : ''
    return request<RealViewing>(`/api/real/subjects/${subject}/reconstruct${suffix}`)
  },

  reportUrl: (id: string) => `/api/library/sessions/${id}/report`,

  report: (id: string) => request<string>(`/api/library/sessions/${id}/report`),
}
