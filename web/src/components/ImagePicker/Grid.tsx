import type { CSSProperties } from 'react'

import { Tile } from '@/components/Tile'
import { isVideo } from '@/lib/image'

import type { Picked } from './types'

export interface GridItem {
  id: string
  blob_sha: string
  media_type: string
  width: number | null
  height: number | null
  duration: number | null
  label: string
}

export function Grid({
  items,
  empty,
  onPick,
}: {
  items: GridItem[]
  empty: string
  onPick: (p: Picked) => void
}) {
  if (items.length === 0) return <p className="asset-empty">{empty}</p>
  return (
    <div className="picker-grid">
      {items.map((it) => (
        <Tile
          key={it.id}
          id={it.id}
          blobSha={it.blob_sha}
          mediaType={it.media_type}
          duration={it.duration}
          style={
            { '--ratio': `${String(it.width ?? 1)} / ${String(it.height ?? 1)}` } as CSSProperties
          }
          label={`${isVideo(it.media_type) ? 'Video' : 'Image'}: ${it.label}`}
          coveredLabel={`Show ${isVideo(it.media_type) ? 'video' : 'image'}`}
          onOpen={() => {
            onPick({
              sha256: it.blob_sha,
              media_type: it.media_type,
              width: it.width,
              height: it.height,
              duration: it.duration,
            })
          }}
        />
      ))}
    </div>
  )
}
