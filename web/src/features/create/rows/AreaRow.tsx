import type { ReactNode } from 'react'

import { MaskThumb } from '@/features/editors/mask/MaskThumb'

export interface AreaRowProps {
  /** "Mask" for an inpaint, "Area" for a unit. */
  label: string
  /** The painted area and the picture it was painted over, if there is one. */
  area: { source: string; mask: string } | null
  /** What the row says with and without an area. */
  editText: string
  emptyText: string
  disabled?: boolean
  /** Notes between the row and its Clear button, and after it. */
  before?: ReactNode
  after?: ReactNode
  onOpen: () => void
  onClear: () => void
}

/** A painted area (an inpaint's mask, or what a unit is limited to): opens the mask editor. */
export function AreaRow({
  label,
  area,
  editText,
  emptyText,
  disabled,
  before,
  after,
  onOpen,
  onClear,
}: AreaRowProps) {
  return (
    <div className="source-row">
      <button type="button" className="setting setting-button" disabled={disabled} onClick={onOpen}>
        <span className="setting-label">{label}</span>{' '}
        <span className={area ? 'setting-value' : 'setting-value none'}>
          {area ? (
            <>
              <MaskThumb source={area.source} mask={area.mask} />
              {editText}
            </>
          ) : (
            emptyText
          )}
        </span>
      </button>
      {before}
      {area && (
        <div className="row-buttons source-actions">
          <button type="button" className="btn quiet small" onClick={onClear}>
            Clear
          </button>
        </div>
      )}
      {after}
    </div>
  )
}
