import { useState } from 'react'

import { blobUrl } from '@/api/urls'

import type { ViewerItem } from './Viewer'

const EXTENSIONS: Record<string, string> = {
  'image/png': 'png',
  'image/jpeg': 'jpg',
  'image/webp': 'webp',
  'video/mp4': 'mp4',
  'video/webm': 'webm',
  'video/quicktime': 'mov',
}

/** Export to the phone: the share sheet where there is one, else a download. */
export function SaveToPhotos({
  item,
  className = 'btn quiet',
}: {
  item: ViewerItem
  className?: string
}) {
  const [error, setError] = useState<string | null>(null)

  const share = async () => {
    setError(null)
    const blob = await (await fetch(blobUrl(item.blob_sha))).blob()
    const ext = EXTENSIONS[item.media_type] ?? 'png'
    const file = new File([blob], `degas-${String(item.seed)}.${ext}`, { type: item.media_type })
    if ('canShare' in navigator && navigator.canShare({ files: [file] })) {
      await navigator.share({ files: [file] })
    } else {
      const a = document.createElement('a')
      a.href = blobUrl(item.blob_sha)
      a.download = file.name
      a.click()
    }
  }

  return (
    <>
      <button
        type="button"
        className={className}
        onClick={() => {
          void share().catch((e: unknown) => {
            if (!(e instanceof DOMException && e.name === 'AbortError')) {
              setError('Saving failed. Try again, or long-press the image.')
            }
          })
        }}
      >
        Save to Photos
      </button>
      {error && (
        <p className="viewer-note" role="alert">
          {error}
        </p>
      )}
    </>
  )
}
