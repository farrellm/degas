import { useMutation } from '@tanstack/react-query'
import { type RefObject, useDeferredValue, useEffect, useMemo, useState } from 'react'

import { api } from '@/api/client'
import type { Selection, SelectPoint } from '@/api/types'
import type { Size } from '@/lib/geometry'

import { type Combine, context, grow, loadLayer, outline } from './canvas'

/**
 * SAM 3 selection in the mask editor: taps and a description find candidate masks in
 * `picture`, one of which is outlined until it's
 * combined with the mask or cleared.
 */
export function useSelection({
  picture,
  outlineRef,
  work,
  toWork,
  zoom,
  onCombine,
}: {
  picture: string
  /** The canvas the selection's outline is drawn on. */
  outlineRef: RefObject<HTMLCanvasElement | null>
  /** The working size, and working px per image px. */
  work: Size
  toWork: number
  /** Stage px per working px, to keep the outline a constant width on screen. */
  zoom: number
  /** Add, subtract or replace: `layer` is the selection, grown by its margin. */
  onCombine: (op: Combine, layer: HTMLCanvasElement) => void
}) {
  const [points, setPoints] = useState<SelectPoint[]>([])
  const [exclude, setExclude] = useState(false)
  const [text, setText] = useState('')
  const [selection, setSelection] = useState<Selection | null>(null)
  const [shown, setShown] = useState(0)
  const [growBy, setGrowBy] = useState(8)
  const [layers, setLayers] = useState<Record<string, HTMLCanvasElement>>({})

  const select = useMutation({
    mutationFn: (req: { points: SelectPoint[]; text: string }) =>
      api.select(picture, { points: req.points, text: req.text || undefined }),
    onSuccess: async (found) => {
      const loaded: Record<string, HTMLCanvasElement> = {}
      await Promise.all(
        found.candidates.map(async (c) => {
          loaded[c.sha256] = layers[c.sha256] ?? (await loadLayer(c.sha256, work))
        }),
      )
      setLayers((prev) => ({ ...prev, ...loaded }))
      setSelection(found)
      setShown(found.chosen ?? 0)
    },
  })
  const candidate = selection?.candidates[shown]
  const layer = candidate ? layers[candidate.sha256] : undefined
  // The outline shows the selection as it will be added: grown by the slider's margin.
  const previewGrow = useDeferredValue(growBy)
  const grown = useMemo(
    () => (layer ? grow(layer, previewGrow * toWork) : undefined),
    [layer, previewGrow, toWork],
  )

  useEffect(() => {
    const ctx = context(outlineRef.current)
    if (!ctx) return
    ctx.clearRect(0, 0, work.w, work.h)
    if (!grown) return
    const accent =
      getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#efa3b5'
    ctx.drawImage(outline(grown, Math.max(1, 2 / zoom), accent), 0, 0)
  }, [outlineRef, grown, zoom, work.w, work.h])

  const clear = () => {
    setPoints([])
    setSelection(null)
    select.reset()
  }

  return {
    /** The taps so far, in image px. */
    points,
    exclude,
    setExclude,
    text,
    setText,
    selection,
    shown,
    setShown,
    growBy,
    setGrowBy,
    pending: select.isPending,
    error: select.error?.message ?? null,
    /** Look again with the taps so far and the description. */
    find: () => {
      select.mutate({ points, text: text.trim() })
    },
    /** A tap at `x, y` in working px: look again with it added. */
    addPoint: (x: number, y: number, include: boolean) => {
      const next = [...points, { x: x / toWork, y: y / toWork, include }]
      setPoints(next)
      select.mutate({ points: next, text: text.trim() })
    },
    clear,
    combine: (op: Combine) => {
      if (!layer || !grown) return
      onCombine(op, previewGrow === growBy ? grown : grow(layer, growBy * toWork))
      clear()
    },
  }
}

export type SelectionState = ReturnType<typeof useSelection>
