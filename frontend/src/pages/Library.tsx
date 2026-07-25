import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import {
  Bookmark,
  BookmarkCheck,
  Download,
  FileText,
  Plus,
  Search,
  Trash2,
} from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router-dom'

import { pageVariants } from '@/animations/motion'
import { api } from '@/services/api'
import type { SessionSummary } from '@/types'
import { Badge, Button, EmptyState, ErrorNotice, LoadingState, Stat } from '@/ui'

export default function Library() {
  const queryClient = useQueryClient()
  const [searchQuery, setSearchQuery] = useState('')
  const [onlyBookmarked, setOnlyBookmarked] = useState(false)
  const [sortBy, setSortBy] = useState<'newest' | 'fidelity' | 'ssim'>('newest')
  const [deletingId, setDeletingId] = useState<string | null>(null)

  const {
    data: sessions = [],
    isLoading,
    isError,
    error,
    refetch,
  } = useQuery<SessionSummary[]>({
    queryKey: ['sessions', searchQuery, onlyBookmarked],
    queryFn: () =>
      api.listSessions({
        q: searchQuery.trim() || undefined,
        bookmarked: onlyBookmarked ? true : undefined,
      }),
  })

  const bookmarkMutation = useMutation({
    mutationFn: ({ id, bookmarked }: { id: string; bookmarked: boolean }) =>
      api.setBookmark(id, bookmarked),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sessions'] })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (id: string) => api.deleteSession(id),
    onSuccess: () => {
      setDeletingId(null)
      queryClient.invalidateQueries({ queryKey: ['sessions'] })
    },
  })

  const sortedSessions = [...sessions].sort((a, b) => {
    if (sortBy === 'fidelity') return b.fidelity - a.fidelity
    if (sortBy === 'ssim') return b.ssim - a.ssim
    return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
  })

  return (
    <motion.main
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      className="mx-auto max-w-6xl px-6 pt-24 pb-16"
    >
      {/* Top Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 pb-8 border-b border-ink/10">
        <div>
          <h1 className="font-display text-2xl md:text-3xl tracking-tight">Memory Library</h1>
          <p className="mt-1 text-sm text-ink-muted">
            Stored neural reconstruction sessions, metrics, and fidelity reports
          </p>
        </div>
        <Link to="/session">
          <Button variant="primary" icon={<Plus className="size-4" />}>
            New Session
          </Button>
        </Link>
      </div>

      {/* Controls: Search, Filter, Sort */}
      <div className="mt-8 flex flex-wrap items-center justify-between gap-4">
        {/* Search input */}
        <div className="relative flex-1 min-w-[240px]">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 size-4 text-ink-muted" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search sessions by title, stimulus, or ID..."
            className="w-full rounded-pill border border-ink/15 bg-neural-950/60 pl-10 pr-4 py-2 text-sm text-ink placeholder:text-ink-faint focus:border-neural-400 focus:outline-none"
          />
        </div>

        {/* Filters */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center rounded-pill border border-ink/15 bg-neural-950/40 p-1">
            <button
              onClick={() => setOnlyBookmarked(false)}
              className={`rounded-pill px-3 py-1 text-xs transition-colors ${
                !onlyBookmarked ? 'bg-neural-400/20 text-neural-200' : 'text-ink-muted hover:text-ink'
              }`}
            >
              All
            </button>
            <button
              onClick={() => setOnlyBookmarked(true)}
              className={`flex items-center gap-1.5 rounded-pill px-3 py-1 text-xs transition-colors ${
                onlyBookmarked ? 'bg-neural-400/20 text-neural-200' : 'text-ink-muted hover:text-ink'
              }`}
            >
              <Bookmark className="size-3" />
              Bookmarked
            </button>
          </div>

          <div className="flex items-center gap-2 text-xs text-ink-muted">
            <span>Sort:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'newest' | 'fidelity' | 'ssim')}
              className="rounded-pill border border-ink/15 bg-neural-950/60 px-3 py-1.5 text-xs text-ink focus:border-neural-400 focus:outline-none"
            >
              <option value="newest">Newest First</option>
              <option value="fidelity">Highest Fidelity</option>
              <option value="ssim">Highest SSIM</option>
            </select>
          </div>
        </div>
      </div>

      {/* Content Area */}
      <div className="mt-8">
        {isLoading ? (
          <LoadingState title="Fetching memory reconstructions..." className="py-20" />
        ) : isError ? (
          <ErrorNotice
            error={error as Error}
            title="Failed to load sessions"
            onRetry={() => refetch()}
          />
        ) : sortedSessions.length === 0 ? (
          <EmptyState
            title={onlyBookmarked || searchQuery ? 'No matching sessions' : 'No memory reconstructions yet'}
            description={
              onlyBookmarked || searchQuery
                ? 'Try clearing your search query or bookmarked filter.'
                : 'Begin a new session to reconstruct visual memories from EEG brainwaves.'
            }
            action={
              <Link to="/session">
                <Button variant="secondary" icon={<Plus className="size-4" />}>
                  Start a Reconstruction
                </Button>
              </Link>
            }
          />
        ) : (
          <div className="grid gap-5 md:grid-cols-2 lg:grid-cols-3">
            {sortedSessions.map((session) => (
              <SessionCard
                key={session.id}
                session={session}
                isDeleting={deletingId === session.id}
                onBookmark={(bookmarked) =>
                  bookmarkMutation.mutate({ id: session.id, bookmarked })
                }
                onDeleteClick={() => setDeletingId(session.id)}
                onConfirmDelete={() => deleteMutation.mutate(session.id)}
                onCancelDelete={() => setDeletingId(null)}
              />
            ))}
          </div>
        )}
      </div>
    </motion.main>
  )
}

function SessionCard({
  session,
  isDeleting,
  onBookmark,
  onDeleteClick,
  onConfirmDelete,
  onCancelDelete,
}: {
  session: SessionSummary
  isDeleting: boolean
  onBookmark: (bookmarked: boolean) => void
  onDeleteClick: () => void
  onConfirmDelete: () => void
  onCancelDelete: () => void
}) {
  const formattedDate = new Date(session.createdAt).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  })

  return (
    <div className="glass rounded-panel relative flex flex-col justify-between overflow-hidden p-5 transition-all hover:border-ink/25">
      <div>
        {/* Card Header: Title & Bookmark */}
        <div className="flex items-start justify-between gap-3">
          <div>
            <span className="font-mono text-[0.72rem] text-neural-300">#{session.id}</span>
            <h3 className="font-display text-[1.05rem] leading-snug mt-0.5">
              <Link to={`/library/${session.id}`} className="hover:underline">
                {session.title || session.stimulusTitle}
              </Link>
            </h3>
          </div>
          <button
            onClick={() => onBookmark(!session.bookmarked)}
            className="text-ink-muted hover:text-neural-300 transition-colors"
            title={session.bookmarked ? 'Remove bookmark' : 'Bookmark session'}
          >
            {session.bookmarked ? (
              <BookmarkCheck className="size-4 text-neural-300 fill-neural-300/20" />
            ) : (
              <Bookmark className="size-4" />
            )}
          </button>
        </div>

        {/* Metadata pills */}
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <Badge variant={session.source === 'synthetic' ? 'neutral' : 'active'}>
            {session.source}
          </Badge>          <span className="text-[0.75rem] text-ink-muted">{formattedDate}</span>
          <span className="text-[0.75rem] text-ink-faint">· {session.durationSeconds}s</span>
        </div>

        {/* Metrics Grid */}
        <div className="mt-5 grid grid-cols-2 gap-3 border-t border-b border-ink/8 py-3">
          <Stat
            label="Composite Fidelity"
            value={`${(session.fidelity * 100).toFixed(0)}%`}
            subtext="Overall reconstruction"
          />
          <Stat
            label="SSIM Score"
            value={session.ssim.toFixed(3)}
            subtext="Structural similarity"
          />
        </div>
      </div>

      {/* Card Footer Actions */}
      <div className="mt-5 flex items-center justify-between gap-2 pt-1">
        <Link to={`/library/${session.id}`}>
          <Button variant="ghost" size="sm" icon={<FileText className="size-3.5" />}>
            Inspect
          </Button>
        </Link>

        <div className="flex items-center gap-2">
          <a
            href={api.reportUrl(session.id)}
            target="_blank"
            rel="noreferrer"
            className="text-ink-muted hover:text-ink p-1.5 transition-colors"
            title="Download Markdown Report"
          >
            <Download className="size-4" />
          </a>

          {isDeleting ? (
            <div className="flex items-center gap-1">
              <button
                onClick={onConfirmDelete}
                className="rounded-pill bg-rose-500/20 border border-rose-500/40 px-2.5 py-1 text-xs text-rose-300 hover:bg-rose-500/30"
              >
                Confirm
              </button>
              <button
                onClick={onCancelDelete}
                className="px-2 py-1 text-xs text-ink-muted hover:text-ink"
              >
                Cancel
              </button>
            </div>
          ) : (
            <button
              onClick={onDeleteClick}
              className="text-ink-muted hover:text-rose-400 p-1.5 transition-colors"
              title="Delete session"
            >
              <Trash2 className="size-4" />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
