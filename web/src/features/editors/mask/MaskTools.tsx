export interface MaskToolsProps {
  canUndo: boolean
  canRedo: boolean
  /** The mask has loaded, so it can be inverted or cleared. */
  ready: boolean
  /** Whether the blur preview is on; undefined when the form has no mask blur to show. */
  showBlur?: boolean
  onUndo: () => void
  onRedo: () => void
  onInvert: () => void
  onClear: () => void
  onShowBlur: (show: boolean) => void
}

/** The mask editor's tool row: undo, redo, invert, clear, and the blur preview. */
export function MaskTools({
  canUndo,
  canRedo,
  ready,
  showBlur,
  onUndo,
  onRedo,
  onInvert,
  onClear,
  onShowBlur,
}: MaskToolsProps) {
  return (
    <div className="editor-tools mask-tools">
      <button type="button" className="tool" aria-label="Undo" disabled={!canUndo} onClick={onUndo}>
        <svg viewBox="0 0 20 20" aria-hidden>
          <path d="M7 4.5 4 7.5l3 3" />
          <path d="M4.5 7.5H12a4 4 0 0 1 0 8H9" />
        </svg>
      </button>
      <button type="button" className="tool" aria-label="Redo" disabled={!canRedo} onClick={onRedo}>
        <svg viewBox="0 0 20 20" aria-hidden>
          <path d="m13 4.5 3 3-3 3" />
          <path d="M15.5 7.5H8a4 4 0 0 0 0 8h3" />
        </svg>
      </button>
      <button type="button" className="tool" disabled={!ready} onClick={onInvert}>
        Invert
      </button>
      <button type="button" className="tool" disabled={!ready} onClick={onClear}>
        Clear
      </button>
      {showBlur !== undefined && (
        <button
          type="button"
          className="tool"
          aria-pressed={showBlur}
          onClick={() => {
            onShowBlur(!showBlur)
          }}
        >
          Blur
        </button>
      )}
    </div>
  )
}
