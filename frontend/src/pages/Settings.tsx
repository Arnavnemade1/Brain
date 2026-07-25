import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import {
  Activity,
  Cpu,
  Database,
  Layers,
  Sliders,
  Sparkles,
  Zap,
} from 'lucide-react'

import { pageVariants } from '@/animations/motion'
import { ElectrodeMap } from '@/components/settings/ElectrodeMap'
import { api, type ElectrodeInfo, type SystemInfo } from '@/services/api'
import { useUiStore } from '@/stores/uiStore'
import { Badge, ErrorNotice, LoadingState, Stat } from '@/ui'

export default function Settings() {
  const reducedMotion = useUiStore((s) => s.reducedMotion)
  const setReducedMotion = useUiStore((s) => s.setReducedMotion)
  const soundEnabled = useUiStore((s) => s.soundEnabled)
  const toggleSound = useUiStore((s) => s.toggleSound)

  const {
    data: system,
    isLoading: isLoadingSystem,
    isError: isErrorSystem,
    error: errorSystem,
  } = useQuery<SystemInfo>({
    queryKey: ['systemInfo'],
    queryFn: () => api.system(),
  })

  const {
    data: montage = [],
    isLoading: isLoadingMontage,
  } = useQuery<ElectrodeInfo[]>({
    queryKey: ['electrodeMontage'],
    queryFn: () => api.montage(),
  })

  if (isLoadingSystem || isLoadingMontage) {
    return <LoadingState title="Loading system status and montage configuration..." className="py-32" />
  }

  if (isErrorSystem) {
    return (
      <div className="mx-auto max-w-4xl px-6 pt-24 pb-16">
        <ErrorNotice
          error={errorSystem as Error}
          title="Failed to load system parameters"
        />
      </div>
    )
  }

  return (
    <motion.main
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      className="mx-auto max-w-6xl px-6 pt-24 pb-16 space-y-8"
    >
      {/* Header */}
      <div className="pb-6 border-b border-ink/10">
        <div className="flex items-center gap-3">
          <Sliders className="size-6 text-neural-300" />
          <h1 className="font-display text-2xl md:text-3xl tracking-tight">System Settings & Status</h1>
        </div>
        <p className="mt-1 text-sm text-ink-muted">
          Pipeline configuration, EEG hardware montage, decoders state, and UI preferences
        </p>
      </div>

      {/* Grid: System Overview & Decoders */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Hardware & Processing Parameters */}
        <div className="glass rounded-panel p-6 space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="flex items-center gap-2 font-display text-[1rem]">
              <Cpu className="size-4 text-neural-300" /> Signal Processing Pipeline
            </h3>
            <Badge variant="active">v{system?.version}</Badge>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <Stat label="Sampling Rate" value={`${system?.sampleRate || 1000} Hz`} subtext="Native EEG acquisition" />
            <Stat label="Window Length" value={`${system?.windowSeconds || 1.0}s`} subtext="Temporal analysis frame" />
            <Stat label="Window Overlap" value={`${((system?.windowOverlap || 0.5) * 100).toFixed(0)}%`} subtext="Slide step ratio" />
            <Stat label="Line Notch" value={`${system?.lineFrequency || 60} Hz`} subtext="Mains noise filter" />
          </div>

          <div className="border-t border-ink/10 pt-4 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-ink-muted">Bandpass Filter Range</span>
              <span className="font-mono text-ink">0.5 Hz – 45.0 Hz</span>
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-ink-muted">Artifact Removal</span>
              <span className="font-mono text-emerald-400">Robust MAD (0.5µV clamp)</span>
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-ink-muted">Referencing</span>
              <span className="font-mono text-ink">Common Average Reference (CAR)</span>
            </div>
          </div>
        </div>

        {/* Decoder Models & Memory Store */}
        <div className="glass rounded-panel p-6 space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="flex items-center gap-2 font-display text-[1rem]">
              <Database className="size-4 text-neural-300" /> Decoders & Storage
            </h3>
            <Badge variant={system?.decodersFitted ? 'active' : 'neutral'}>
              {system?.decodersFitted ? 'Decoders Fitted' : 'Simulated Model'}
            </Badge>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <Stat label="Stored Sessions" value={system?.storedSessions || 0} subtext="Persisted in JSON store" />
            <Stat label="Active Streams" value={system?.activeSessions || 0} subtext="Live WebSocket sessions" />
          </div>

          <div className="border-t border-ink/10 pt-4 space-y-3">
            <div className="text-xs text-ink-muted font-medium">Decoder Training Corpora:</div>
            <div className="flex flex-wrap gap-1.5">
              {(system?.decoderTrainedOn || ['synthetic-vis-v1', 'ds004018-rsvp']).map((corpus) => (
                <span
                  key={corpus}
                  className="rounded-pill bg-neural-400/15 border border-neural-400/30 px-2.5 py-1 font-mono text-[0.72rem] text-neural-200"
                >
                  {corpus}
                </span>
              ))}
            </div>

            <div className="flex items-center justify-between text-xs pt-2">
              <span className="flex items-center gap-1.5 text-ink-muted">
                <Sparkles className="size-3.5 text-amber-400" /> LLM Report Synthesis
              </span>
              <span className={`font-mono text-xs ${system?.llmNarratives ? 'text-emerald-400' : 'text-ink-faint'}`}>
                {system?.llmNarratives ? 'Active' : 'Disabled (Fallback)'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* 2D Electrode Montage Map */}
      <ElectrodeMap electrodes={montage} />

      {/* Interface Preferences */}
      <div className="glass rounded-panel p-6 space-y-6">
        <h3 className="flex items-center gap-2 font-display text-[1rem]">
          <Zap className="size-4 text-neural-300" /> Interface & Accessibility Preferences
        </h3>

        <div className="grid gap-4 md:grid-cols-2">
          <div className="flex items-center justify-between p-3 rounded-panel border border-ink/10 bg-neural-950/40">
            <div>
              <div className="text-sm font-medium">Reduced Motion</div>
              <div className="text-xs text-ink-muted">Minimize background canvas animations</div>
            </div>
            <button
              onClick={() => setReducedMotion(!reducedMotion)}
              className={`relative h-6 w-11 rounded-full transition-colors ${
                reducedMotion ? 'bg-neural-400' : 'bg-ink/20'
              }`}
            >
              <span
                className={`inline-block size-4 rounded-full bg-white transition-transform ${
                  reducedMotion ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>

          <div className="flex items-center justify-between p-3 rounded-panel border border-ink/10 bg-neural-950/40">
            <div>
              <div className="text-sm font-medium">Audio Ambience</div>
              <div className="text-xs text-ink-muted">Play subtle neural background ambience sound</div>
            </div>
            <button
              onClick={toggleSound}
              className={`relative h-6 w-11 rounded-full transition-colors ${
                soundEnabled ? 'bg-neural-400' : 'bg-ink/20'
              }`}
            >
              <span
                className={`inline-block size-4 rounded-full bg-white transition-transform ${
                  soundEnabled ? 'translate-x-6' : 'translate-x-1'
                }`}
              />
            </button>
          </div>
        </div>
      </div>
    </motion.main>
  )
}
