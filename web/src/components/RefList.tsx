import { thumbUrl } from '../api'
import type { Source } from '../draft'
import { size } from '../format'

// The model reads up to 10 images: the source and 9 more.
export const MAX_REFS = 9

interface Props {
  refs: Source[]
  onChange: (refs: Source[]) => void
  onAdd: () => void
  onCrop: (index: number) => void
}

/** The images a Qwen edit reads after the source, numbered from 2 in the order it reads them. */
export function RefList({ refs, onChange, onAdd, onCrop }: Props) {
  return (
    <div className="ref-group" role="group" aria-labelledby="refs-label">
      <div className="setting">
        <span className="setting-label" id="refs-label">
          Images
        </span>
        <button
          type="button"
          className="row-action"
          aria-label="Add image"
          disabled={refs.length >= MAX_REFS}
          onClick={onAdd}
        >
          Add
        </button>
      </div>
      {refs.map((ref, i) => {
        const n = String(i + 2)
        return (
          <div key={`${ref.sha}-${n}`} className="ref">
            <img className="source-thumb" src={thumbUrl(ref.sha)} alt="" />
            <span className="ref-text">
              <span className="ref-name">Image {n}</span>
              {ref.width > 0 && <span className="ref-meta">{size(ref.width, ref.height)}</span>}
            </span>
            <button
              type="button"
              className="btn quiet small"
              aria-label={`Crop image ${n}`}
              onClick={() => {
                onCrop(i)
              }}
            >
              Crop
            </button>
            <button
              type="button"
              className="btn quiet small"
              aria-label={`Move image ${n} earlier`}
              disabled={i === 0}
              onClick={() => {
                onChange([
                  ...refs.slice(0, i - 1),
                  ref,
                  ...refs.slice(i - 1).filter((r) => r !== ref),
                ])
              }}
            >
              Earlier
            </button>
            <button
              type="button"
              className="lora-remove"
              aria-label={`Remove image ${n}`}
              onClick={() => {
                onChange(refs.filter((_, j) => j !== i))
              }}
            >
              <svg viewBox="0 0 12 12" aria-hidden>
                <path d="M2 2l8 8M10 2l-8 8" />
              </svg>
            </button>
          </div>
        )
      })}
      <p className="row-note">
        {refs.length > 0
          ? 'Refer to them by number: “the jacket from image 2”. The source is image 1.'
          : 'Add pictures the prompt can use, like a face, an outfit or a place.'}
      </p>
    </div>
  )
}
