import { AnimatePresence } from 'framer-motion'
import { lazy, Suspense, useEffect } from 'react'
import { Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom'

import { Ambience } from '@/components/Ambience'
import { Nav } from '@/components/Nav'
import { ShortcutsDialog } from '@/components/ShortcutsDialog'
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts'
import { useUiStore } from '@/stores/uiStore'
import { ErrorBoundary, LoadingState, ToastViewport } from '@/ui'

import Landing from '@/pages/Landing'

// Three.js and the session console are large and only needed once the user
// commits to entering the experience — keep them out of the landing bundle.
const Session = lazy(() => import('@/pages/Session'))
const Library = lazy(() => import('@/pages/Library'))
const MemoryDetail = lazy(() => import('@/pages/MemoryDetail'))
const SettingsPage = lazy(() => import('@/pages/Settings'))

function AppRoutes() {
  const location = useLocation()

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<Landing />} />
        <Route path="/session" element={<Session />} />
        <Route path="/library" element={<Library />} />
        <Route path="/library/:sessionId" element={<MemoryDetail />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AnimatePresence>
  )
}

export default function App() {
  const location = useLocation()
  const navigate = useNavigate()
  const toggleShortcuts = useUiStore((s) => s.toggleShortcuts)

  const onLanding = location.pathname === '/'

  useKeyboardShortcuts([
    {
      key: 'k',
      meta: true,
      run: () => toggleShortcuts(),
      description: 'Keyboard shortcuts',
      group: 'Global',
      allowInField: true,
    },
  ])

  // `G` then `S`/`L` navigation, the convention Linear and GitHub use.
  useEffect(() => {
    let armed = false
    let timer: number | undefined

    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      if (target && (target.isContentEditable || /INPUT|TEXTAREA|SELECT/.test(target.tagName))) {
        return
      }

      const key = event.key.toLowerCase()

      if (armed) {
        armed = false
        window.clearTimeout(timer)
        if (key === 's') navigate('/session')
        else if (key === 'l') navigate('/library')
        else if (key === 'h') navigate('/')
        return
      }

      if (key === 'g') {
        armed = true
        timer = window.setTimeout(() => (armed = false), 1200)
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      window.clearTimeout(timer)
    }
  }, [navigate])

  return (
    <>
      <Ambience intensity={onLanding ? 1 : 0.6} />
      {!onLanding && <Nav />}

      <ErrorBoundary label="MindScape">
        <Suspense fallback={<LoadingState title="Loading interface" className="min-h-dvh" />}>
          <AppRoutes />
        </Suspense>
      </ErrorBoundary>

      <ShortcutsDialog />
      <ToastViewport />
    </>
  )
}
