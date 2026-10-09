import './Tile.css'

import type { CSSProperties } from 'react'

import { thumbUrl } from '@/api/urls'
import { useCover } from '@/hooks/useDiscretion'
import { formatClock } from '@/lib/format'
import { isVideo } from '@/lib/image'

export interface TileProps {
  /** What uncovers together: the result or library item id. */
  id: string
  blobSha: string
  mediaType: string
  duration?: number | null
  kept?: boolean
  style?: CSSProperties
  /** The tile's name when it's clear, and when it's covered in discretion mode. */
  label: string
  coveredLabel: string
  onOpen: () => void
}

/** A finished image or clip in a grid; under glassine in discretion mode until tapped. */
export function Tile({
  id,
  blobSha,
  mediaType,
  duration,
  kept,
  style,
  label,
  coveredLabel,
  onOpen,
}: TileProps) {
  const { covered, peek, press, tap } = useCover(id)
  const classes = ['tile', kept && 'kept', covered && 'covered', peek && 'peek']
  return (
    <button
      type="button"
      className={classes.filter(Boolean).join(' ')}
      style={style}
      aria-label={covered ? coveredLabel : label}
      {...press}
      onClick={() => tap(onOpen)}
    >
      <img src={thumbUrl(blobSha)} alt="" loading="lazy" draggable={false} />
      {isVideo(mediaType) && duration != null && (
        <span className="tile-duration">{formatClock(duration)}</span>
      )}
    </button>
  )
}
