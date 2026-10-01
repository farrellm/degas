import type { Fit } from '@/api/types'
import { FitSelect } from '@/components/FitSelect'

export interface FitRowProps {
  fit: Fit
  onFit: (fit: Fit) => void
  /** Offered while letterboxing, when the model can draw the bars instead. */
  onOutpaint?: () => void
}

/** How a source of another shape is fitted to the output. */
export function FitRow({ fit, onFit, onOutpaint }: FitRowProps) {
  return (
    <div className="fit-row">
      <FitSelect id="fit" value={fit} onChange={onFit} />
      {fit === 'pad' && onOutpaint && (
        <p className="row-note">
          <button type="button" className="link" onClick={onOutpaint}>
            Outpaint the bars
          </button>{' '}
          to draw what’s beyond the image instead.
        </p>
      )}
    </div>
  )
}
