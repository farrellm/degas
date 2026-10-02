import { useEffect } from 'react'

import { blobUrl } from '@/api/urls'
import { useDialog } from '@/hooks/useDialog'

/** One of an image's image prompt pictures, with its unit's lines for the label. */
export interface PromptPicture {
  sha: string
  model: string
  summary: string
}

export interface PictureViewerProps {
  pictures: PromptPicture[]
  index: number
  onIndex: (i: number) => void
  onClose: () => void
}

/** An image prompt's picture opened over the viewer, as large as the model read it. */
export function PictureViewer({ pictures, index, onIndex, onClose }: PictureViewerProps) {
  const ref = useDialog()
  const picture = pictures[index]
  const last = pictures.length - 1

  // Caught before the viewer underneath hears them, so its image stays put.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!['Escape', 'ArrowLeft', 'ArrowRight'].includes(e.key)) return
      e.stopPropagation()
      if (e.key === 'Escape') onClose()
      if (e.key === 'ArrowRight' && index < last) onIndex(index + 1)
      if (e.key === 'ArrowLeft' && index > 0) onIndex(index - 1)
    }
    document.addEventListener('keydown', onKey, true)
    return () => {
      document.removeEventListener('keydown', onKey, true)
    }
  }, [index, last, onIndex, onClose])

  if (!picture) return null
  return (
    <div
      ref={ref}
      className="viewer picture-viewer"
      role="dialog"
      aria-modal="true"
      aria-label="Image prompt picture"
      tabIndex={-1}
    >
      <div className="viewer-bar">
        <button type="button" className="btn quiet small" onClick={onClose}>
          Back
        </button>
        {last > 0 && (
          <>
            <span className="position">
              {index + 1} of {pictures.length}
            </span>
            <span className="sheet-actions">
              <button
                type="button"
                className="btn quiet small"
                aria-label="Previous picture"
                disabled={index === 0}
                onClick={() => {
                  onIndex(index - 1)
                }}
              >
                ‹
              </button>
              <button
                type="button"
                className="btn quiet small"
                aria-label="Next picture"
                disabled={index === last}
                onClick={() => {
                  onIndex(index + 1)
                }}
              >
                ›
              </button>
            </span>
          </>
        )}
      </div>
      <div className="viewer-image">
        <img src={blobUrl(picture.sha)} alt="" draggable={false} />
      </div>
      <div className="wall-label">
        <p className="lines">
          <span>{picture.model}</span>
          <span>{picture.summary}</span>
        </p>
      </div>
    </div>
  )
}
