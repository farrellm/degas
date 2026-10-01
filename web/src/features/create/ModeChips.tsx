import { MODE_LABELS } from './modes'

export interface ModeChipsProps {
  modes: string[]
  mode: string | undefined
  onChoose: (mode: string) => void
}

/** What the job starts from: text, an image, an edit… */
export function ModeChips({ modes, mode, onChoose }: ModeChipsProps) {
  return (
    <div className="mode-chips" role="group" aria-label="Start from">
      {modes.map((m) => (
        <button
          key={m}
          type="button"
          aria-pressed={m === mode}
          onClick={() => {
            onChoose(m)
          }}
        >
          {MODE_LABELS[m] ?? m}
        </button>
      ))}
    </div>
  )
}
