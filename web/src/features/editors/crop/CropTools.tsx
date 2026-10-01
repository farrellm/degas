import type { Aspect } from './crop'

export interface CropToolsProps {
  /** The shapes on offer, and the one chosen. */
  aspects: { id: Aspect; label: string }[]
  aspect: Aspect | undefined
  /** The image hasn't loaded yet. */
  disabled: boolean
  flipped: boolean
  /** Whether the crop is resized to the output size; undefined when it never is (a free crop). */
  resize?: boolean
  onAspect: (aspect: Aspect) => void
  onRotate: () => void
  onFlip: () => void
  onResize: (resize: boolean) => void
  onReset: () => void
}

/** The crop editor's controls: the shape chips, then rotate, flip, resize and reset. */
export function CropTools({
  aspects,
  aspect,
  disabled,
  flipped,
  resize,
  onAspect,
  onRotate,
  onFlip,
  onResize,
  onReset,
}: CropToolsProps) {
  return (
    <>
      <div className="aspect-chips" role="group" aria-label="Shape">
        {aspects.map((a) => (
          <button
            key={a.id}
            type="button"
            aria-pressed={aspect === a.id}
            disabled={disabled}
            onClick={() => {
              onAspect(a.id)
            }}
          >
            {a.label}
          </button>
        ))}
      </div>
      <div className="editor-tools">
        <button type="button" className="tool" disabled={disabled} onClick={onRotate}>
          <svg viewBox="0 0 20 20" aria-hidden>
            <path d="M6 4.5 3.5 7 6 9.5" />
            <path d="M3.8 7H12a4.5 4.5 0 0 1 0 9H8" />
          </svg>
          Rotate
        </button>
        <button
          type="button"
          className="tool"
          aria-pressed={flipped}
          disabled={disabled}
          onClick={onFlip}
        >
          <svg viewBox="0 0 20 20" aria-hidden>
            <path d="M10 2.5v15" strokeDasharray="2 2" />
            <path d="M7.5 5 3 14.5h4.5z" />
            <path d="M12.5 5 17 14.5h-4.5z" />
          </svg>
          Flip
        </button>
        {resize !== undefined && (
          <button
            type="button"
            className="tool"
            aria-pressed={resize}
            disabled={disabled}
            onClick={() => {
              onResize(!resize)
            }}
          >
            <svg viewBox="0 0 20 20" aria-hidden>
              <path d="M3 8V3h5" />
              <path d="M17 12v5h-5" />
              <path d="M3 3l14 14" />
            </svg>
            Resize
          </button>
        )}
        <button type="button" className="tool" disabled={disabled} onClick={onReset}>
          Reset
        </button>
      </div>
    </>
  )
}
