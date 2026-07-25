import { useUiStore } from '@/stores/uiStore'
import { Dialog } from '@/ui'

interface Binding {
  keys: string
  description: string
}

const GROUPS: { name: string; bindings: Binding[] }[] = [
  {
    name: 'Global',
    bindings: [
      { keys: '⌘ K', description: 'Open keyboard shortcuts' },
      { keys: 'G then S', description: 'Go to session' },
      { keys: 'G then L', description: 'Go to library' },
      { keys: 'Esc', description: 'Close dialog or exit pointer lock' },
    ],
  },
  {
    name: 'Session',
    bindings: [
      { keys: 'Space', description: 'Pause or resume the stream' },
      { keys: '1 – 4', description: 'Set playback speed' },
      { keys: 'V', description: 'Toggle scientific overlay' },
      { keys: 'F', description: 'Enter the reconstruction' },
    ],
  },
  {
    name: 'Exploration',
    bindings: [
      { keys: 'W A S D', description: 'Move through the memory' },
      { keys: 'Mouse', description: 'Look around' },
      { keys: 'Shift', description: 'Move faster' },
      { keys: 'E', description: 'Open the nearest memory fragment' },
      { keys: 'Esc', description: 'Release the cursor' },
    ],
  },
]

export function ShortcutsDialog() {
  const open = useUiStore((s) => s.shortcutsOpen)
  const toggle = useUiStore((s) => s.toggleShortcuts)

  return (
    <Dialog
      open={open}
      onClose={() => toggle(false)}
      label="Reference"
      title="Keyboard shortcuts"
      width="max-w-xl"
    >
      <div className="grid gap-7 sm:grid-cols-2">
        {GROUPS.map((group) => (
          <section key={group.name}>
            <p className="text-label mb-3">{group.name}</p>
            <ul className="space-y-2.5">
              {group.bindings.map((binding) => (
                <li key={binding.keys} className="flex items-baseline justify-between gap-3">
                  <span className="text-sm text-ink-soft">{binding.description}</span>
                  <kbd className="text-readout shrink-0 rounded-md border border-ink/10 bg-ink/5 px-2 py-1 text-[0.68rem] text-ink-muted">
                    {binding.keys}
                  </kbd>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </Dialog>
  )
}
