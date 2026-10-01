import { useState } from 'react'

import { SaveToPhotos } from '@/components/Viewer/SaveToPhotos'

import type { Kept } from './kept'

export function LibraryActions({
  item,
  deleting,
  error,
  onRemix,
  onExtend,
  onUseAsSource,
  onDelete,
}: {
  item: Kept
  deleting: boolean
  error: string | undefined
  onRemix: () => void
  onExtend: (() => void) | undefined
  onUseAsSource: (() => void) | undefined
  onDelete: () => void
}) {
  const [confirming, setConfirming] = useState(false)
  if (confirming) {
    return (
      <div className="confirm wide" role="group" aria-label="Confirm delete">
        <p>
          Delete this {item.kind === 'video' ? 'clip' : 'image'} from the library?
          {item.source_result_id ? '' : ' This can’t be undone.'}
        </p>
        <button type="button" className="btn danger" disabled={deleting} onClick={onDelete}>
          Delete
        </button>
        <button
          type="button"
          className="btn quiet"
          onClick={() => {
            setConfirming(false)
          }}
        >
          Cancel
        </button>
      </div>
    )
  }
  return (
    <>
      <button type="button" className="btn wide" onClick={onRemix}>
        Remix
      </button>
      <SaveToPhotos item={item} />
      {onExtend && (
        <button type="button" className="btn quiet" onClick={onExtend}>
          Extend
        </button>
      )}
      {onUseAsSource && (
        <button type="button" className="btn quiet" onClick={onUseAsSource}>
          Use as source
        </button>
      )}
      <button
        type="button"
        className="btn quiet"
        onClick={() => {
          setConfirming(true)
        }}
      >
        Delete
      </button>
      {error && (
        <p className="viewer-note" role="alert">
          {error}
        </p>
      )}
    </>
  )
}
