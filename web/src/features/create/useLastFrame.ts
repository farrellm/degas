import { useState } from 'react'

import type { BlobInfo } from '@/api/types'
import type { Size } from '@/lib/geometry'
import type { Source } from '@/lib/image'

/** First and last frame: the last frame, which is fitted to the output like the source. */
export function useLastFrame(initial: Source | null | undefined) {
  const [image, setImage] = useState<Source | null>(initial ?? null)
  // The server no longer stores the last frame (its thumbnail failed to load).
  const [gone, setGone] = useState(false)

  return {
    image,
    gone,
    /** `fallback` is the size to assume for an image whose size isn't known. */
    take: (taken: BlobInfo, fallback: Size) => {
      setImage({
        sha: taken.sha256,
        width: taken.width ?? fallback.w,
        height: taken.height ?? fallback.h,
      })
      setGone(false)
    },
    remove: () => {
      setImage(null)
      setGone(false)
    },
    markGone: () => setGone(true),
  }
}
