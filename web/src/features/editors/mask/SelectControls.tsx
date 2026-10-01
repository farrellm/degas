import type { Selection, SelectPoint } from '@/api/types'

import type { Combine } from './canvas'

const MAX_GROW = 128 // image px of margin a selection can be grown by

interface SelectProps {
  note: string | null
  points: SelectPoint[]
  exclude: boolean
  onExclude: (exclude: boolean) => void
  text: string
  onText: (text: string) => void
  onFind: () => void
  pending: boolean
  error: string | null
  selection: Selection | null
  shown: number
  onShown: (i: number) => void
  growBy: number
  onGrow: (px: number) => void
  onCombine: (op: Combine) => void
  onClear: () => void
}

/** SAM 3: taps and a description make a selection, which then adds to the mask. */
export function SelectControls(p: SelectProps) {
  if (p.note) return <p className="row-note mask-note">{p.note}</p>
  const count = p.selection?.candidates.length ?? 0
  return (
    <>
      {/* Not a <form>: the editor renders inside the Create form, and a nested submit would
          bubble up and queue a generation. */}
      <div className="mask-row describe">
        <label htmlFor="select-text" className="visually-hidden">
          Describe what to select
        </label>
        <input
          id="select-text"
          type="text"
          placeholder="Describe it, or tap the image"
          value={p.text}
          enterKeyHint="search"
          onChange={(e) => {
            p.onText(e.target.value)
          }}
          onKeyDown={(e) => {
            if (e.key !== 'Enter') return
            e.preventDefault()
            if (p.text.trim() && !p.pending) p.onFind()
          }}
        />
        <button
          type="button"
          className="btn quiet small"
          disabled={!p.text.trim() || p.pending}
          onClick={p.onFind}
        >
          Find
        </button>
      </div>
      <div className="mask-row">
        <div className="seed-modes" role="group" aria-label="Taps">
          <button
            type="button"
            aria-pressed={!p.exclude}
            onClick={() => {
              p.onExclude(false)
            }}
          >
            Include
          </button>
          <button
            type="button"
            aria-pressed={p.exclude}
            onClick={() => {
              p.onExclude(true)
            }}
          >
            Exclude
          </button>
        </div>
        {(p.points.length > 0 || p.selection) && (
          <button type="button" className="btn quiet small" onClick={p.onClear}>
            Start over
          </button>
        )}
      </div>
      <p className="row-note mask-note" aria-live="polite">
        {p.pending
          ? 'Selecting… the first time on a session copies SAM 3 to the GPU.'
          : p.error
            ? null
            : p.selection && count === 0
              ? 'Nothing matched. Tap it instead, or describe it differently.'
              : p.selection
                ? 'Shown as an outline. Add it to the mask, or tap to refine.'
                : 'Tap what to select. Long-press, or choose Exclude, to leave something out.'}
      </p>
      {p.error && <p role="alert">{p.error}</p>}
      <div className="mask-row">
        <label htmlFor="grow">Grow</label>
        <input
          id="grow"
          type="range"
          min={0}
          max={MAX_GROW}
          value={p.growBy}
          onChange={(e) => {
            p.onGrow(Number(e.target.value))
          }}
        />
        <output htmlFor="grow">{p.growBy} px</output>
      </div>
      {count > 0 && (
        <>
          <div className="mask-row">
            <button
              type="button"
              className="btn quiet small"
              disabled={p.shown <= 0}
              onClick={() => {
                p.onShown(p.shown - 1)
              }}
            >
              Smaller
            </button>
            <button
              type="button"
              className="btn quiet small"
              disabled={p.shown >= count - 1}
              onClick={() => {
                p.onShown(p.shown + 1)
              }}
            >
              Bigger
            </button>
          </div>
          <div className="mask-row combine">
            <button
              type="button"
              className="btn small"
              onClick={() => {
                p.onCombine('add')
              }}
            >
              Add
            </button>
            <button
              type="button"
              className="btn quiet small"
              onClick={() => {
                p.onCombine('subtract')
              }}
            >
              Subtract
            </button>
            <button
              type="button"
              className="btn quiet small"
              onClick={() => {
                p.onCombine('replace')
              }}
            >
              Replace
            </button>
          </div>
        </>
      )}
    </>
  )
}
