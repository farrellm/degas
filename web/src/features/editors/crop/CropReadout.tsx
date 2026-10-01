import { formatSize } from '@/lib/format'
import { ratioDiffers, type Rect, type Size } from '@/lib/geometry'

import { UPSCALE_WARN } from './crop'

export interface CropReadoutProps {
  /** The crop in image px, and the size it's meant to be output at. */
  crop: Rect
  out: Size
  /** The original's size. */
  natural: Size
  /** How far the output scales the crop up. */
  scale: number
  /** Whether the crop is resized to `out` now, or kept at its own size. */
  resize: boolean
  free: boolean
  square: boolean
}

/** What applying the crop will give: its size, and a warning when it's scaled up a lot. */
export function CropReadout({ crop, out, natural, scale, resize, free, square }: CropReadoutProps) {
  return (
    <p className={scale > UPSCALE_WARN ? 'readout warn' : 'readout'} aria-live="polite">
      {resize ? formatSize(out.w, out.h) : formatSize(crop.w, crop.h)} from{' '}
      {formatSize(natural.w, natural.h)}
      {!resize && !free && (out.w !== crop.w || out.h !== crop.h) && (
        <>
          <br />
          Kept at its own size; it’s fitted to {formatSize(out.w, out.h)} when it’s used.
        </>
      )}
      {square && ratioDiffers(out.w / out.h, 1) && (
        <>
          <br />
          The model sees the middle square.
        </>
      )}
      {scale > UPSCALE_WARN && (
        <>
          <br />
          {free ? 'The model scales it up' : 'Scaled up'} {scale.toFixed(1)}×, so fine detail will
          be soft.
        </>
      )}
    </p>
  )
}
