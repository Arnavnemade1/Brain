/**
 * Session WebSocket client.
 *
 * Wraps a single session stream with typed events, queued commands and
 * bounded reconnection. Reconnect exists for transient network drops; a
 * session the server has finished or forgotten is terminal, and retrying
 * those would spin forever against a socket that will never succeed.
 */

import type { ConnectionStatus, StreamCommand, StreamEvent } from '@/types'

export interface SessionSocketOptions {
  onEvent: (event: StreamEvent) => void
  onStatus?: (status: ConnectionStatus) => void
  onError?: (message: string) => void
  /** Attempts before giving up. Defaults to 4. */
  maxRetries?: number
}

/** Close codes the server uses for "this session is gone, do not retry". */
const TERMINAL_CLOSE_CODES = new Set([4404, 1008])

export class SessionSocket {
  private socket: WebSocket | null = null
  private retries = 0
  private closedByClient = false
  private queue: StreamCommand[] = []
  private reconnectTimer: number | null = null

  private readonly sessionId: string
  private readonly options: SessionSocketOptions

  constructor(sessionId: string, options: SessionSocketOptions) {
    this.sessionId = sessionId
    this.options = options
  }

  get status(): ConnectionStatus {
    if (this.closedByClient) return 'closed'
    if (!this.socket) return 'idle'
    switch (this.socket.readyState) {
      case WebSocket.CONNECTING:
        return this.retries > 0 ? 'reconnecting' : 'connecting'
      case WebSocket.OPEN:
        return 'open'
      default:
        return 'closed'
    }
  }

  connect(): void {
    if (this.socket && this.socket.readyState <= WebSocket.OPEN) return

    this.closedByClient = false
    this.setStatus(this.retries > 0 ? 'reconnecting' : 'connecting')

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const url = `${protocol}//${window.location.host}/ws/session/${this.sessionId}`

    const socket = new WebSocket(url)
    this.socket = socket

    socket.onopen = () => {
      this.retries = 0
      this.setStatus('open')
      // Anything sent before the socket opened is flushed in order.
      const pending = this.queue
      this.queue = []
      pending.forEach((command) => this.send(command))
    }

    socket.onmessage = (message) => {
      let event: StreamEvent
      try {
        event = JSON.parse(message.data) as StreamEvent
      } catch {
        this.options.onError?.('Received a malformed event from the server.')
        return
      }
      this.options.onEvent(event)
    }

    socket.onerror = () => {
      // The error event carries no detail; `onclose` decides what happens next.
      this.options.onError?.('Connection to the neural stream was interrupted.')
    }

    socket.onclose = (event) => {
      this.socket = null
      if (this.closedByClient) {
        this.setStatus('closed')
        return
      }

      const terminal =
        TERMINAL_CLOSE_CODES.has(event.code) ||
        this.retries >= (this.options.maxRetries ?? 4)

      if (terminal) {
        this.setStatus('failed')
        this.options.onError?.(
          TERMINAL_CLOSE_CODES.has(event.code)
            ? 'This session is no longer available on the server.'
            : 'Lost connection to the neural stream.',
        )
        return
      }

      this.retries += 1
      this.setStatus('reconnecting')
      // Exponential backoff, capped so a long outage still retries promptly.
      const delay = Math.min(800 * 2 ** (this.retries - 1), 6000)
      this.reconnectTimer = window.setTimeout(() => this.connect(), delay)
    }
  }

  send(command: StreamCommand): void {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(command))
    } else {
      this.queue.push(command)
    }
  }

  close(): void {
    this.closedByClient = true
    if (this.reconnectTimer != null) {
      window.clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    this.queue = []
    this.socket?.close(1000, 'client closed')
    this.socket = null
    this.setStatus('closed')
  }

  private setStatus(status: ConnectionStatus): void {
    this.options.onStatus?.(status)
  }
}
