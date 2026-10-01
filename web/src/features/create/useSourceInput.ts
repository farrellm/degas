import { useState } from 'react'

import { api } from '@/api/client'
import type { BlobInfo, Fit } from '@/api/types'
import type { Place } from '@/features/editors/place/place'
import type { Size } from '@/lib/geometry'
import type { MaskRef, Source } from '@/lib/image'

import type { FamilyDraft } from './draft'

/**
 * The image a job starts from and what's drawn over it: the fit, the clip it continues, the
 * inpaint mask and the outpaint placement. A new source resets what belonged to the old one.
 */
export function useSourceInput(draft: Partial<FamilyDraft>) {
  const [source, setSource] = useState<Source | null>(draft.source ?? null)
  // The server no longer stores the image (its thumbnail failed to load).
  const [gone, setGone] = useState(false)
  const [fit, setFit] = useState<Fit>(draft.fit ?? 'crop')
  const [extendsClip, setExtends] = useState(draft.extends ?? null)
  const [mask, setMask] = useState<MaskRef | null>(draft.mask ?? null)
  const [maskNote, setMaskNote] = useState<string | null>(null)
  const [place, setPlace] = useState<Place | null>(draft.place ?? null)

  /**
   * Take a picked or cropped image as the source, returning it. `fallback` is the size to
   * assume for an image whose size isn't known.
   */
  const take = (image: BlobInfo, fromCrop: boolean, fallback: Size): Source => {
    const taken = {
      sha: image.sha256,
      width: image.width ?? fallback.w,
      height: image.height ?? fallback.h,
    }
    const previous = source
    setSource(taken)
    setGone(false)
    setExtends(null)
    setPlace(null)
    setMaskNote(null)
    // A new crop of the same image carries the mask with it; another image drops it.
    if (!mask || image.sha256 === mask.source) return taken
    if (!fromCrop || previous?.sha !== mask.source) {
      setMask(null)
      return taken
    }
    api
      .remapMask(mask.sha, mask.source, image.sha256)
      .then((moved) => {
        if (moved.empty) {
          setMask(null)
          setMaskNote('The crop left out the whole mask.')
        } else {
          setMask({ sha: moved.sha256, source: image.sha256 })
          setMaskNote('The mask moved with the crop.')
        }
      })
      .catch(() => {
        setMask(null)
        setMaskNote('The mask couldn’t follow the crop. Paint it again.')
      })
    return taken
  }

  return {
    source,
    gone,
    fit,
    extendsClip,
    mask,
    /** Whether the mask was painted over the current source. */
    maskFits: !!mask && !!source && mask.source === source.sha,
    maskNote,
    place,
    setFit,
    setPlace,
    take,
    markGone: () => {
      setGone(true)
    },
    remove: () => {
      setSource(null)
      setExtends(null)
      setGone(false)
    },
    /** Set the mask painted over the current source, or clear it. */
    setMask: (painted: BlobInfo | null) => {
      setMask(painted && source ? { sha: painted.sha256, source: source.sha } : null)
      setMaskNote(null)
    },
  }
}
