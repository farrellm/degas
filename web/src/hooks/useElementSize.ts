import { type RefObject, useLayoutEffect, useState } from 'react'

import type { Size } from '@/lib/geometry'

/**
 * The laid-out size of `ref`'s element, kept up to date. `fallback` stands in until it has
 * been measured (and in tests, which have no layout).
 */
export function useElementSize(ref: RefObject<HTMLElement | null>, fallback: Size): Size {
  const [size, setSize] = useState(fallback)

  useLayoutEffect(() => {
    const el = ref.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(([entry]) => {
      const r = entry?.contentRect
      if (r && r.width > 0 && r.height > 0) setSize({ w: r.width, h: r.height })
    })
    observer.observe(el)
    return () => {
      observer.disconnect()
    }
  }, [ref])

  return size
}
