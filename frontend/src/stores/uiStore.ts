/**
 * UI preferences and app-level chrome state.
 *
 * Persisted to localStorage so quality settings and onboarding progress
 * survive reloads.
 */

import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export type RenderQuality = 'low' | 'balanced' | 'high'
export type PanelDensity = 'comfortable' | 'compact'

export interface Toast {
  id: string
  message: string
  tone: 'info' | 'success' | 'warning' | 'error'
}

interface UiState {
  /* --- preferences (persisted) --- */
  renderQuality: RenderQuality
  density: PanelDensity
  /** Master switch for ambient audio in the reconstruction. */
  audioEnabled: boolean
  audioVolume: number
  /** Honour the OS setting by default; users can force motion off entirely. */
  reduceMotion: boolean
  showScientificOverlay: boolean
  onboardingComplete: boolean
  /** Playback speed remembered between sessions. */
  preferredSpeed: number

  /* --- ephemeral --- */
  shortcutsOpen: boolean
  toasts: Toast[]

  /* --- actions --- */
  setRenderQuality: (quality: RenderQuality) => void
  setDensity: (density: PanelDensity) => void
  setAudioEnabled: (enabled: boolean) => void
  setAudioVolume: (volume: number) => void
  setReduceMotion: (reduce: boolean) => void
  toggleScientificOverlay: () => void
  completeOnboarding: () => void
  restartOnboarding: () => void
  setPreferredSpeed: (speed: number) => void
  toggleShortcuts: (open?: boolean) => void
  toast: (message: string, tone?: Toast['tone']) => void
  dismissToast: (id: string) => void
}

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      renderQuality: 'balanced',
      density: 'comfortable',
      audioEnabled: false,
      audioVolume: 0.5,
      reduceMotion: false,
      showScientificOverlay: true,
      onboardingComplete: false,
      preferredSpeed: 2,

      shortcutsOpen: false,
      toasts: [],

      setRenderQuality: (renderQuality) => set({ renderQuality }),
      setDensity: (density) => set({ density }),
      setAudioEnabled: (audioEnabled) => set({ audioEnabled }),
      setAudioVolume: (audioVolume) => set({ audioVolume }),
      setReduceMotion: (reduceMotion) => set({ reduceMotion }),
      toggleScientificOverlay: () =>
        set((state) => ({ showScientificOverlay: !state.showScientificOverlay })),
      completeOnboarding: () => set({ onboardingComplete: true }),
      restartOnboarding: () => set({ onboardingComplete: false }),
      setPreferredSpeed: (preferredSpeed) => set({ preferredSpeed }),
      toggleShortcuts: (open) =>
        set((state) => ({ shortcutsOpen: open ?? !state.shortcutsOpen })),

      toast: (message, tone = 'info') =>
        set((state) => ({
          toasts: [
            ...state.toasts,
            { id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`, message, tone },
          ].slice(-4),
        })),

      dismissToast: (id) =>
        set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) })),
    }),
    {
      name: 'mindscape.ui',
      // Toasts and transient panels must never come back on reload.
      partialize: (state) => ({
        renderQuality: state.renderQuality,
        density: state.density,
        audioEnabled: state.audioEnabled,
        audioVolume: state.audioVolume,
        reduceMotion: state.reduceMotion,
        showScientificOverlay: state.showScientificOverlay,
        onboardingComplete: state.onboardingComplete,
        preferredSpeed: state.preferredSpeed,
      }),
    },
  ),
)
