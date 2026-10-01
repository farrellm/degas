import { MaskThumb } from '@/features/editors/mask/MaskThumb'

export interface MaskRowProps {
  /** The source image's sha. */
  source: string
  /** The mask painted over that source, if there is one. */
  mask: string | null
  /** What became of the mask when the source last changed. */
  note: string | null
  onPaint: () => void
  onClear: () => void
}

/** The area an inpaint redraws. */
export function MaskRow({ source, mask, note, onPaint, onClear }: MaskRowProps) {
  return (
    <div className="source-row">
      <button type="button" className="setting setting-button" onClick={onPaint}>
        <span className="setting-label">Mask</span>{' '}
        <span className={mask ? 'setting-value' : 'setting-value none'}>
          {mask ? (
            <>
              <MaskThumb source={source} mask={mask} />
              Edit mask
            </>
          ) : (
            'Paint the area to redraw'
          )}
        </span>
      </button>
      {note && <p className="row-note">{note}</p>}
      {mask && (
        <div className="row-buttons source-actions">
          <button type="button" className="btn quiet small" onClick={onClear}>
            Clear
          </button>
        </div>
      )}
    </div>
  )
}
