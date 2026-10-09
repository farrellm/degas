import { api } from '@/api/client'
import type { Size } from '@/lib/geometry'
import { type Source, toSource } from '@/lib/image'

/** A plain grey picture of the output's shape, to paint an area over when there's no source. */
export async function blankCanvas(target: Size): Promise<Source> {
  const canvas = document.createElement('canvas')
  canvas.width = target.w
  canvas.height = target.h
  const g = canvas.getContext('2d')
  if (g) {
    g.fillStyle = '#80868c'
    g.fillRect(0, 0, target.w, target.h)
  }
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/png'))
  if (!blob) throw new Error('Couldn’t make a canvas to paint the area on.')
  return toSource(await api.upload(blob))
}
