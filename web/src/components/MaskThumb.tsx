import { useEffect, useRef } from 'react'
import { blobUrl, thumbUrl } from '../api'
import { lumaToAlpha } from '../mask'

const SIZE = 68 // drawn at 2× the 34 px slot

/** The source's thumbnail with its mask laid over in rose, for the Mask row. */
export function MaskThumb({ source, mask }: { source: string; mask: string }) {
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
    Promise.all([load(thumbUrl(source)), load(blobUrl(mask))])
      .then(([image, painted]) => {
        if (!live) return
        // Cover the square, like the source thumbnail's object-fit.
        const k = SIZE / Math.min(image.naturalWidth, image.naturalHeight)
        const w = image.naturalWidth * k
        const h = image.naturalHeight * k
        const x = (SIZE - w) / 2
        const y = (SIZE - h) / 2
        ctx.clearRect(0, 0, SIZE, SIZE)
        ctx.drawImage(image, x, y, w, h)
        const layer = document.createElement('canvas')
        layer.width = SIZE
        layer.height = SIZE
        const lc = layer.getContext('2d')
        if (!lc) return
        lc.drawImage(painted, x, y, w, h)
        const data = lc.getImageData(0, 0, SIZE, SIZE)
        lumaToAlpha(data.data)
        lc.putImageData(data, 0, 0)
        lc.globalCompositeOperation = 'source-in'
        lc.fillStyle =
          getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() ||
          '#efa3b5'
        lc.fillRect(0, 0, SIZE, SIZE)
        ctx.globalAlpha = 0.7
        ctx.drawImage(layer, 0, 0)
        ctx.globalAlpha = 1
      })
      .catch(() => {
        // the row still says the mask is painted
      })
    return () => {
      live = false
    }
  }, [source, mask])

  return <canvas ref={ref} className="source-thumb" width={SIZE} height={SIZE} aria-hidden />
}
