import { thumbUrl } from '@/api/urls'
import { formatSize } from '@/lib/format'
import type { Source } from '@/lib/image'

export interface SourceRowProps {
  /** "Source", or "Image 1" when more images follow it. */
  label: string
  source: Source | null
  /** The image is no longer stored on the server. */
  gone: boolean
  /** The source is a clip's last frame, which the job continues. */
  continuesClip: boolean
  onPick: () => void
  onCrop: () => void
  onRemove: () => void
  onGone: () => void
}

/** The image a job starts from: above the model. */
export function SourceRow({
  label,
  source,
  gone,
  continuesClip,
  onPick,
  onCrop,
  onRemove,
  onGone,
}: SourceRowProps) {
  return (
    <div className={gone ? 'source-row missing' : 'source-row'}>
      <button type="button" className="setting setting-button" onClick={onPick}>
        <span className="setting-label">{label}</span>{' '}
        <span className={source ? 'setting-value' : 'setting-value none'}>
          {source ? (
            <>
              <img className="source-thumb" src={thumbUrl(source.sha)} alt="" onError={onGone} />
              {formatSize(source.width, source.height)}
            </>
          ) : (
            'Choose an image'
          )}
        </span>
      </button>
      {gone && <p className="row-warning">This image is no longer stored. Choose another.</p>}
      {continuesClip && source && !gone && (
        <p className="row-note">Continues a clip from its last frame.</p>
      )}
      {source && (
        <div className="row-buttons source-actions">
          <button type="button" className="btn quiet small" disabled={gone} onClick={onCrop}>
            Crop
          </button>
          <button type="button" className="btn quiet small" onClick={onRemove}>
            Remove
          </button>
        </div>
      )}
    </div>
  )
}
