import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import {
  ArrowLeft,
  Bookmark,
  BookmarkCheck,
  CheckCircle2,
  Download,
  FileCode,
  FileText,
  History,
  ShieldAlert,
  Trash2,
} from 'lucide-react'
import { useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'

import { pageVariants } from '@/animations/motion'
import { api } from '@/services/api'
import { METRIC_DESCRIPTORS, type SessionRecord } from '@/types'
import { Badge, ErrorNotice, LoadingState, Meter, Stat } from '@/ui'

export default function MemoryDetail() {
  const { sessionId } = useParams<{ sessionId: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [activeTab, setActiveTab] = useState<'overview' | 'findings' | 'history' | 'markdown'>('overview')

  const {
    data: session,
    isLoading,
    isError,
    error,
  } = useQuery<SessionRecord>({
    queryKey: ['session', sessionId],
    queryFn: () => api.getSession(sessionId!),
    enabled: Boolean(sessionId),
  })

  const { data: reportMarkdown } = useQuery<string>({
    queryKey: ['sessionReport', sessionId],
    queryFn: () => api.report(sessionId!),
    enabled: Boolean(sessionId) && activeTab === 'markdown',
  })

  const bookmarkMutation = useMutation({
    mutationFn: (bookmarked: boolean) => api.setBookmark(sessionId!, bookmarked),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['session', sessionId] })
      queryClient.invalidateQueries({ queryKey: ['sessions'] })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: () => api.deleteSession(sessionId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sessions'] })
      navigate('/library')
    },
  })

  if (isLoading) {
    return <LoadingState title="Loading reconstruction session..." className="py-32" />
  }

  if (isError || !session) {
    return (
      <div className="shell page-y">
        <Link to="/library" className="inline-flex items-center gap-2 text-sm text-ink-muted hover:text-ink">
          <ArrowLeft className="size-4" /> Back to Library
        </Link>
        <div className="mt-8">
          <ErrorNotice
            message={String(error as Error || new Error('Session not found'))}
          />
        </div>
      </div>
    )
  }

  const { metrics, report, clip, history } = session

  return (
    <motion.main
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      className="shell page-y"
    >
      {/* Navigation & Header */}
      <div className="pb-6 border-b border-ink/10">
        <Link to="/library" className="inline-flex items-center gap-2 text-sm text-ink-muted hover:text-ink transition-colors">
          <ArrowLeft className="size-4" /> Back to Memory Library
        </Link>

        <div className="mt-6 flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-3">
              <span className="font-mono text-xs text-neural-300">#{session.id}</span>
              <Badge>
                {session.source}
              </Badge>              <span className="text-xs text-ink-muted">v{session.version}</span>
            </div>

            <h1 className="font-display text-2xl md:text-3xl tracking-tight mt-1">
              {session.title || session.stimulusTitle}
            </h1>
            <p className="mt-1 text-sm text-ink-muted">
              Reconstructed from EEG recording on {new Date(session.createdAt).toLocaleDateString()} · Stimulus: {session.stimulusTitle} ({clip.durationSeconds}s)
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => bookmarkMutation.mutate(!session.bookmarked)}
              className="glass rounded-pill px-3 py-2 text-xs flex items-center gap-1.5 text-ink hover:border-ink/20"
            >
              {session.bookmarked ? (
                <>
                  <BookmarkCheck className="size-4 text-neural-300" /> Bookmarked
                </>
              ) : (
                <>
                  <Bookmark className="size-4" /> Bookmark
                </>
              )}
            </button>

            <a
              href={api.reportUrl(session.id)}
              target="_blank"
              rel="noreferrer"
              className="glass rounded-pill px-3 py-2 text-xs flex items-center gap-1.5 text-ink hover:border-ink/20"
            >
              <Download className="size-4" /> Export Report
            </a>

            <button
              onClick={() => {
                if (window.confirm('Delete this reconstruction session?')) {
                  deleteMutation.mutate()
                }
              }}
              className="glass rounded-pill p-2 text-ink-muted hover:text-rose-400 transition-colors"
              title="Delete session"
            >
              <Trash2 className="size-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="mt-6 flex items-center gap-2 border-b border-ink/10 pb-3">
        <TabButton
          active={activeTab === 'overview'}
          onClick={() => setActiveTab('overview')}
          icon={<FileText className="size-4" />}
          label="Metrics & Overview"
        />
        <TabButton
          active={activeTab === 'findings'}
          onClick={() => setActiveTab('findings')}
          icon={<CheckCircle2 className="size-4" />}
          label={`Findings (${report.findings.length})`}
        />
        <TabButton
          active={activeTab === 'history'}
          onClick={() => setActiveTab('history')}
          icon={<History className="size-4" />}
          label={`History (${history.length})`}
        />
        <TabButton
          active={activeTab === 'markdown'}
          onClick={() => setActiveTab('markdown')}
          icon={<FileCode className="size-4" />}
          label="Raw Report"
        />
      </div>

      {/* Tab Contents */}
      <div className="mt-8">
        {activeTab === 'overview' && (
          <div className="space-y-8">
            {/* Top Stat Cards */}
            <div className="grid gap-4 md:grid-cols-3">
              <div className="glass rounded-panel p-6">
                <Stat
                  label="Composite Fidelity"
                  value={`${(metrics.composite * 100).toFixed(0)}%`}
                  hint="Weighted total score"
                />
                <Meter value={metrics.composite} className="mt-4" />
              </div>
              <div className="glass rounded-panel p-6">
                <Stat
                  label="Structural Similarity (SSIM)"
                  value={metrics.ssim.toFixed(3)}
                  hint="Spatial structure agreement"
                />
                <Meter value={metrics.ssim} className="mt-4" />
              </div>
              <div className="glass rounded-panel p-6">
                <Stat
                  label="Peak SNR"
                  value={`${metrics.psnr.toFixed(1)} dB`}
                  hint="Pixel-level signal to noise"
                />
                <Meter value={Math.min(1, metrics.psnr / 30)} className="mt-4" />
              </div>
            </div>

            {/* Summary Box */}
            <div className="glass rounded-panel p-6">
              <h3 className="font-display text-[1rem] tracking-tight mb-2">Reconstruction Summary</h3>
              <p className="text-sm leading-relaxed text-ink-muted">{report.summary}</p>
            </div>

            {/* Detailed Metrics Table vs Baselines & Ceilings */}
            <div className="glass rounded-panel p-6">
              <h3 className="font-display text-[1rem] tracking-tight mb-4">Detailed Benchmark Breakdown</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-ink/10 text-ink-muted">
                      <th className="pb-3 font-medium">Metric</th>
                      <th className="pb-3 font-medium">Measured Value</th>
                      <th className="pb-3 font-medium">Baseline (Chance)</th>
                      <th className="pb-3 font-medium">EEG Ceiling</th>
                      <th className="pb-3 font-medium">Description</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-ink/5">
                    {METRIC_DESCRIPTORS.map((descriptor) => {
                      const value = metrics[descriptor.key]
                      const formattedValue =
                        descriptor.unit === 'r'
                          ? `r = ${value > 0 ? '+' : ''}${value.toFixed(2)}`
                          : descriptor.unit === 'dB'
                          ? `${value.toFixed(1)} dB`
                          : descriptor.key === 'ssim' || descriptor.key === 'sceneCutF1'
                          ? value.toFixed(3)
                          : `${(value * 100).toFixed(0)}%`

                      return (
                        <tr key={descriptor.key} className="hover:bg-ink/5">
                          <td className="py-3 font-medium text-ink">{descriptor.label}</td>
                          <td className="py-3 font-mono font-semibold text-neural-300">{formattedValue}</td>
                          <td className="py-3 font-mono text-ink-faint">{descriptor.baseline} {descriptor.unit}</td>
                          <td className="py-3 font-mono text-ink-faint">{descriptor.ceiling} {descriptor.unit}</td>
                          <td className="py-3 text-ink-muted max-w-xs leading-normal">{descriptor.description}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'findings' && (
          <div className="space-y-6">
            <div className="glass rounded-panel p-6">
              <h3 className="flex items-center gap-2 font-display text-[1.05rem] text-emerald-400 mb-4">
                <CheckCircle2 className="size-5" /> Supported Findings
              </h3>
              <ul className="space-y-3">
                {report.findings.map((finding, idx) => (
                  <li key={idx} className="flex items-start gap-3 text-sm text-ink-muted">
                    <span className="mt-1 size-1.5 rounded-full bg-emerald-400 shrink-0" />
                    <span>{finding}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="glass rounded-panel p-6 border-amber-500/20">
              <h3 className="flex items-center gap-2 font-display text-[1.05rem] text-amber-400 mb-4">
                <ShieldAlert className="size-5" /> Signal Limitations
              </h3>
              <ul className="space-y-3">
                {report.limitations.map((limitation, idx) => (
                  <li key={idx} className="flex items-start gap-3 text-sm text-ink-muted">
                    <span className="mt-1 size-1.5 rounded-full bg-amber-400 shrink-0" />
                    <span>{limitation}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}

        {activeTab === 'history' && (
          <div className="glass rounded-panel p-6">
            <h3 className="font-display text-[1rem] mb-4">Re-reconstruction Iterations</h3>
            {history.length === 0 ? (
              <p className="text-sm text-ink-muted">This is the initial version (v1) of this reconstruction.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-ink/10 text-ink-muted">
                      <th className="pb-3 font-medium">Version</th>
                      <th className="pb-3 font-medium">Date</th>
                      <th className="pb-3 font-medium">Fidelity</th>
                      <th className="pb-3 font-medium">SSIM</th>
                      <th className="pb-3 font-medium">Delta Summary</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-ink/5">
                    {history.map((ver) => (
                      <tr key={ver.version}>
                        <td className="py-3 font-mono text-neural-300">v{ver.version}</td>
                        <td className="py-3 text-ink-muted">{new Date(ver.createdAt).toLocaleDateString()}</td>
                        <td className="py-3 font-mono">{(ver.fidelity * 100).toFixed(0)}%</td>
                        <td className="py-3 font-mono">{ver.ssim.toFixed(3)}</td>
                        <td className="py-3 text-ink-muted">{ver.delta}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {activeTab === 'markdown' && (
          <div className="glass rounded-panel p-6 font-mono text-xs leading-relaxed overflow-x-auto whitespace-pre-wrap text-ink-muted">
            {reportMarkdown || 'Loading markdown report...'}
          </div>
        )}
      </div>
    </motion.main>
  )
}

function TabButton({
  active,
  onClick,
  icon,
  label,
}: {
  active: boolean
  onClick: () => void
  icon: React.ReactNode
  label: string
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-3 py-2 text-xs transition-colors rounded-t-md border-b-2 ${
        active
          ? 'border-neural-400 text-neural-200 bg-neural-400/10'
          : 'border-transparent text-ink-muted hover:text-ink'
      }`}
    >
      {icon}
      <span>{label}</span>
    </button>
  )
}
