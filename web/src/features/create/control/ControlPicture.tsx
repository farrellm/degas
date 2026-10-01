import { blobUrl } from '@/api/urls'
import type { Source } from '@/lib/image'

import { type ControlUnit, photoOf, traceInfo, underlay } from './control'

export interface ControlPictureProps {
  unit: ControlUnit
  /** Create's source image, offered as the picture to trace. */
  source: Source | null
  /** A trace is being made on the GPU. */
  busy: boolean
  /** Whether the trace is shown laid over the photo it came from. */
  overPhoto: boolean
  onOverPhoto: (over: boolean) => void
  onCrop: (sha: string) => void
  onChoose: () => void
  onUseSource: (source: Source) => void
}

/** A unit's control image (what the model will read), or the invitation to choose a picture. */
export function ControlPicture({
  unit,
  source,
  busy,
  overPhoto,
  onOverPhoto,
  onCrop,
  onChoose,
  onUseSource,
}: ControlPictureProps) {
  const photo = photoOf(unit)
  const photoUnder = underlay(unit)
  const canOverlay = !!unit.trace && !!photoUnder && photoUnder !== unit.image?.sha

  return (
    <>
      {unit.image ? (
        <div className="control-preview">
          <div
            className={busy ? 'control-picture sketch indeterminate' : 'control-picture'}
            style={{
              aspectRatio: `${String(unit.image.width || 1)} / ${String(unit.image.height || 1)}`,
            }}
          >
            {overPhoto && canOverlay && photoUnder && (
              <img src={blobUrl(photoUnder)} alt="" className="control-photo" />
            )}
            <img
              src={blobUrl(unit.image.sha)}
              alt={unit.trace ? `${traceInfo(unit.trace.id).label} trace` : 'Control image'}
              className={overPhoto && canOverlay ? 'control-trace over' : 'control-trace'}
            />
          </div>
          <div className="row-buttons control-picture-actions">
            {canOverlay && (
              <button
                type="button"
                className="btn quiet small"
                aria-pressed={overPhoto}
                onClick={() => {
                  onOverPhoto(!overPhoto)
                }}
              >
                Over the picture
              </button>
            )}
            <span className="spacer" />
            <button
              type="button"
              className="btn quiet small"
              disabled={busy || !photo}
              onClick={() => {
                if (photo) onCrop(photo.sha)
              }}
            >
              Crop
            </button>
            <button type="button" className="btn quiet small" disabled={busy} onClick={onChoose}>
              Change
            </button>
          </div>
        </div>
      ) : (
        <div className="control-empty">
          <p>Choose the picture whose layout the image should follow.</p>
          <div className="row-buttons">
            {source && (
              <button
                type="button"
                className="btn small"
                disabled={busy}
                onClick={() => {
                  onUseSource(source)
                }}
              >
                Use the source
              </button>
            )}
            <button
              type="button"
              className={source ? 'btn quiet small' : 'btn small'}
              disabled={busy}
              onClick={onChoose}
            >
              Choose image
            </button>
          </div>
        </div>
      )}
      {unit.image && source && photo?.sha !== source.sha && (
        <p className="row-note">
          <button
            type="button"
            className="link"
            disabled={busy}
            onClick={() => {
              onUseSource(source)
            }}
          >
            Use the source
          </button>{' '}
          instead of this picture.
        </p>
      )}
    </>
  )
}
