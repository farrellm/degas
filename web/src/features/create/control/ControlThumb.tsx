import { useEffect, useRef } from 'react'

import { blobUrl, thumbUrl } from '@/api/urls'

const SIZE = 68 // drawn at 2× the 34 px slot
const OUTSIDE = 0.25 // how much of the trace shows outside an area
const MAX_GAIN = 8

/**
 * A control image as a chalk study on the paper (the chalk look is CSS: greyscale, blended
 * onto the ground). With an area, the trace fades outside it: the unit only guides there.
 */
export function ControlThumb({ image, area }: { image: string; area: string | null }) {
  const ref = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const ctx = ref.current?.getContext('2d')
    if (!ctx) return
    let live = true
    const load = (src: string) => {
      const img = new Image()
      img.src = src
      return img.decode().then(() => img)
    }
    Promise.all([load(thumbUrl(image)), area ? load(blobUrl(area)) : null])
      .then(([picture, painted]) => {
        if (!live) return
        // Cover the square, like the other row thumbnails.
        const k = SIZE / Math.min(picture.naturalWidth, picture.naturalHeight)
        const w = picture.naturalWidth * k
        const h = picture.naturalHeight * k
        const x = (SIZE - w) / 2
        const y = (SIZE - h) / 2
        ctx.clearRect(0, 0, SIZE, SIZE)
        ctx.drawImage(picture, x, y, w, h)
        const out = ctx.getImageData(0, 0, SIZE, SIZE)
        const px = out.data
        // Thin lines (edges, a pose) average away at this size: stretch the levels so the
        // brightest stroke is full chalk.
        let max = 1
        for (let i = 0; i < px.length; i += 4)
          max = Math.max(max, px[i] ?? 0, px[i + 1] ?? 0, px[i + 2] ?? 0)
        const gain = Math.min(MAX_GAIN, 255 / max)
        let inside: Uint8ClampedArray | null = null
        if (painted) {
          const layer = document.createElement('canvas')
          layer.width = SIZE
          layer.height = SIZE
          const lc = layer.getContext('2d')
          if (lc) {
            lc.drawImage(painted, x, y, w, h)
            inside = lc.getImageData(0, 0, SIZE, SIZE).data
          }
        }
        for (let i = 0; i < px.length; i += 4) {
          // Outside the area, fade towards black, which the chalk blend turns into paper.
          const m = inside ? (inside[i] ?? 0) / 255 : 1
          const k = gain * (OUTSIDE + (1 - OUTSIDE) * m)
          px[i] = (px[i] ?? 0) * k
          px[i + 1] = (px[i + 1] ?? 0) * k
          px[i + 2] = (px[i + 2] ?? 0) * k
        }
        ctx.putImageData(out, 0, 0)
      })
      .catch(() => {
        // the row still names the unit
      })
    return () => {
      live = false
    }
  }, [image, area])

  return (
    <span className="control-thumb" aria-hidden>
      <canvas ref={ref} width={SIZE} height={SIZE} />
    </span>
  )
}
