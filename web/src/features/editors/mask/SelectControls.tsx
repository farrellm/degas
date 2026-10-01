import { RangeRow } from './RangeRow'
import type { SelectionState } from './useSelection'

const MAX_GROW = 128 // image px of margin a selection can be grown by

export interface SelectControlsProps {
  /** Why Select can't be used right now, shown in place of the controls. */
  note: string | null
  sam: SelectionState
}

/** SAM 3: taps and a description make a selection, which then adds to the mask. */
export function SelectControls({ note, sam: p }: SelectControlsProps) {
  if (note) return <p className="row-note mask-note">{note}</p>
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
            p.setText(e.target.value)
          }}
          onKeyDown={(e) => {
            if (e.key !== 'Enter') return
            e.preventDefault()
            if (p.text.trim() && !p.pending) p.find()
          }}
        />
        <button
          type="button"
          className="btn quiet small"
          disabled={!p.text.trim() || p.pending}
          onClick={p.find}
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
              p.setExclude(false)
            }}
          >
            Include
          </button>
          <button
            type="button"
            aria-pressed={p.exclude}
            onClick={() => {
              p.setExclude(true)
            }}
          >
            Exclude
          </button>
        </div>
        {(p.points.length > 0 || p.selection) && (
          <button type="button" className="btn quiet small" onClick={p.clear}>
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
      <RangeRow
        id="grow"
        label="Grow"
        min={0}
        max={MAX_GROW}
        value={p.growBy}
        output={`${String(p.growBy)} px`}
        onChange={p.setGrowBy}
      />
      {count > 0 && (
        <>
          <div className="mask-row">
            <button
              type="button"
              className="btn quiet small"
              disabled={p.shown <= 0}
              onClick={() => {
                p.setShown(p.shown - 1)
              }}
            >
              Smaller
            </button>
            <button
              type="button"
              className="btn quiet small"
              disabled={p.shown >= count - 1}
              onClick={() => {
                p.setShown(p.shown + 1)
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
                p.combine('add')
              }}
            >
              Add
            </button>
            <button
              type="button"
              className="btn quiet small"
              onClick={() => {
                p.combine('subtract')
              }}
            >
              Subtract
            </button>
            <button
              type="button"
              className="btn quiet small"
              onClick={() => {
                p.combine('replace')
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
