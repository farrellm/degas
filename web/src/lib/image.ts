import type { BlobInfo } from '@/api/types'

/** A source image in the form, with its pixel size (for the fit hint and the crop editor). */
export interface Source {
  sha: string
  width: number
  height: number
}

/** An inpaint mask, and the source image it was painted over. */
export interface MaskRef {
  sha: string
  source: string
}

export const toSource = (b: BlobInfo): Source => ({
  sha: b.sha256,
  width: b.width ?? 0,
  height: b.height ?? 0,
})

export const isVideo = (mediaType: string) => mediaType.startsWith('video/')

/** `sha256:…`, the way a job spec names a blob. */
export const ref = (sha: string) => `sha256:${sha}`
export const unref = (value: string) => value.replace(/^sha256:/, '')
