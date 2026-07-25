/**
 * REST client.
 *
 * One thin wrapper so every call surfaces backend `detail` messages as real
 * errors — the UI shows those verbatim rather than a generic failure.
 */

import type { SessionRecord, SessionSummary } from '@/types'

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
}

export interface SimulationProfile {
  id: string
  name: string
  description: string
  tags: string[]
  artifactLevel: number
}

export interface SessionCreated {
  sessionId: string
  sampleRate: number
  channels: string[]
  duration: number
  source: 'simulated' | 'uploaded' | 'device'
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

  profiles: () => request<SimulationProfile[]>('/api/sessions/profiles'),

  startSimulation: (body: {
    profile: string
    duration: number
    speed?: number
    title?: string
  }) =>
    request<SessionCreated>('/api/sessions/simulate', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  uploadRecording: (file: File, options?: { speed?: number; title?: string }) => {
    const form = new FormData()
    form.append('file', file)
    if (options?.speed != null) form.append('speed', String(options.speed))
    if (options?.title) form.append('title', options.title)
    return request<SessionCreated>('/api/sessions/upload', { method: 'POST', body: form })
  },

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

  reportUrl: (id: string) => `/api/library/sessions/${id}/report`,

  report: (id: string) => request<string>(`/api/library/sessions/${id}/report`),
}
