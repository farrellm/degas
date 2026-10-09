import { thumbUrl } from '@/api/urls'
import type { Source } from '@/lib/image'

import { FaceTile } from './FaceTile'
import { MAX_PICTURES } from './imagePrompt'

export interface PromptPicturesProps {
  pictures: Source[]
  /** The model reads who a face is (FaceID), so each picture shows the face found in it. */
  faceid: boolean
  gpuReady: boolean
  onCrop: (index: number) => void
  onRemove: (index: number) => void
  onAdd: () => void
}

/** An image prompt's pictures, and what the model makes of them. */
export function PromptPictures({
  pictures,
  faceid,
  gpuReady,
  onCrop,
  onRemove,
  onAdd,
}: PromptPicturesProps) {
  return (
    <>
      <div className="prompt-pictures" role="group" aria-label="Pictures">
        {pictures.map((picture, i) => {
          const n = String(i + 1)
          return (
            <figure key={`${picture.sha}-${n}`} className="prompt-picture">
              {/* Square, because the model sees the middle square. */}
              {faceid ? (
                <FaceTile sha={picture.sha} n={n} gpuReady={gpuReady} />
              ) : (
                <img src={thumbUrl(picture.sha)} alt={`Picture ${n}`} />
              )}
              <figcaption className="row-buttons">
                <button
                  type="button"
                  className="btn quiet small"
                  aria-label={`Crop picture ${n}`}
                  onClick={() => onCrop(i)}
                >
                  Crop
                </button>
                <button
                  type="button"
                  className="lora-remove"
                  aria-label={`Remove picture ${n}`}
                  onClick={() => onRemove(i)}
                >
                  <svg viewBox="0 0 12 12" aria-hidden>
                    <path d="M2 2l8 8M10 2l-8 8" />
                  </svg>
                </button>
              </figcaption>
            </figure>
          )
        })}
        {pictures.length < MAX_PICTURES && (
          <button type="button" className="prompt-add" onClick={onAdd}>
            {pictures.length === 0 ? 'Choose a picture' : 'Add picture'}
          </button>
        )}
      </div>
      {faceid && pictures.length > 0 && !gpuReady && (
        <p className="row-note">The face in each picture is found on the GPU when it runs.</p>
      )}
      <p className="row-note">
        {faceid && pictures.length > 0
          ? 'The model reads who the face is, not the rest of the picture.'
          : pictures.length === 0
            ? 'The image will take after this picture, the way it takes after the prompt.'
            : pictures.length > 1
              ? 'The model reads them together.'
              : 'The model sees the middle square, at low resolution: fine detail and text don’t carry.'}
      </p>
    </>
  )
}
