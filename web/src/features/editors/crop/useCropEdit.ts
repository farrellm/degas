import { type KeyboardEvent, type PointerEvent, type RefObject, useRef, useState } from 'react'

import type { Op } from '@/api/types'
import { keyMove, pinch, type Point, stagePoint } from '@/features/editors/gestures'
import { ratioMatches, type Rect, type Size, type View } from '@/lib/geometry'

import {
  type Aspect,
  centered,
  clampView,
  type Constraints,
  cropOf,
  dragCorner,
  exactCrop,
  frameFor,
  frameOf,
  intersect,
  mirror,
  modelUpscale,
  outputSize,
  parseOps,
  ratio,
  reshape,
  rotated,
  type Rotation,
  upscale,
  viewFor,
  zoomAt,
} from './crop'

export const CORNERS = ['nw', 'ne', 'sw', 'se'] as const
type Corner = (typeof CORNERS)[number]

/**
 * What the editor is set to. The crop is in (rotated) image pixels, unrounded; the frame
 * and the image's placement on screen are derived from it and the stage size.
 */
export interface Edit {
  rot: Rotation
  flip: boolean
  aspect: Aspect
  crop: Rect
  /** Resize to the output size now; if not, the crop keeps its pixels and is fitted at submit. */
  resize: boolean
}

/** While a corner of a free crop is dragged, the frame moves and the image stays put. */
interface CornerDrag {
  corner: Corner
  x: number
  y: number
  frame: Rect
  view: View
}

/**
 * The crop being made: the image pans and zooms under a fixed frame, turns and flips, and
 * (in a free crop) the frame's corners drag. Everything on screen follows from `edit`.
 */
export function useCropEdit({
  stageRef,
  stage,
  saved,
  target,
  constraints,
  free,
  square,
  refsKeepSize,
}: {
  stageRef: RefObject<HTMLElement | null>
  stage: Size
  /** How the image being cropped was made, once known: its earlier crop is where this starts. */
  saved: { ops: Op[] } | undefined
  target: Size
  constraints: Constraints
  free: boolean
  square: boolean
  refsKeepSize: boolean
}) {
  // The original's size, once its picture has loaded.
  const [natural, setNatural] = useState<Size | null>(null)
  const [edited, setEdit] = useState<Edit | null>(null)
  const [corner, setCorner] = useState<CornerDrag | null>(null)
  const pointers = useRef(new Map<number, Point>())

  const targetRatio = target.w / target.h

  /** Where a crop starts: all of a free crop's image, a square in the middle of an image
   * prompt's, else the target's shape, centred. */
  const initial = (image: Size): Pick<Edit, 'aspect' | 'crop'> =>
    square
      ? { aspect: '1:1', crop: centered(image, 1) }
      : free
        ? { aspect: 'free', crop: { x: 0, y: 0, ...image } }
        : { aspect: 'match', crop: centered(image, targetRatio) }

  // Until edited: the earlier crop of a derived image, else where a crop starts.
  let edit = edited
  if (!edit && natural && saved) {
    const { rot, flip, crop, resized } = parseOps(saved.ops)
    const img = rotated(natural, rot)
    if (crop) {
      const match = !free && ratioMatches(crop.w / crop.h, targetRatio)
      edit = { rot, flip, aspect: match ? 'match' : 'free', crop, resize: resized }
    } else {
      edit = { rot, flip, resize: true, ...initial(img) }
    }
  }

  const img = natural && edit ? rotated(natural, edit.rot) : null
  const frame = corner?.frame ?? (edit ? frameFor(stage, edit.crop.w / edit.crop.h) : null)
  const view = corner?.view ?? (edit && frame ? viewFor(edit.crop, frame) : null)
  const crop = view && frame && img ? cropOf(view, frame, img) : null
  const out =
    crop && edit ? (free ? crop : outputSize(crop, edit.aspect, target, constraints)) : null
  const scale =
    crop && out && !square
      ? free
        ? refsKeepSize
          ? 1
          : modelUpscale(crop, target)
        : upscale(crop, out)
      : 1

  /** Move or zoom the image under the frame. */
  const moveView = (change: (v: View) => View) => {
    if (!edit || !frame || !view || !img) return
    setEdit({ ...edit, crop: exactCrop(clampView(change(view), frame, img), frame) })
  }
  const setAspect = (aspect: Aspect) => {
    if (!edit || !img) return
    const r = ratio(aspect, target)
    setEdit({ ...edit, aspect, crop: r === null ? edit.crop : reshape(edit.crop, r, img) })
  }

  const rotate = () => {
    if (!edit || !natural) return
    const rot = ((edit.rot + 270) % 360) as Rotation // a quarter turn anticlockwise
    const im = rotated(natural, rot)
    const r = ratio(edit.aspect, target)
    if (free && r === null) {
      setEdit({ ...edit, rot, crop: { x: 0, y: 0, ...im } })
      return
    }
    setEdit({ ...edit, rot, crop: centered(im, r ?? edit.crop.h / edit.crop.w) })
  }

  const flip = () => {
    if (!edit || !img) return
    setEdit({ ...edit, flip: !edit.flip, crop: mirror(edit.crop, img) })
  }

  const reset = () => {
    if (!natural) return
    setEdit({ rot: 0, flip: false, resize: true, ...initial(natural) })
  }

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    const handle = (e.target as HTMLElement).dataset.corner as Corner | undefined
    e.currentTarget.setPointerCapture(e.pointerId)
    const p = stagePoint(stageRef.current, e)
    if (handle && edit?.aspect === 'free' && frame && view) {
      setCorner({ corner: handle, ...p, frame, view })
      return
    }
    pointers.current.set(e.pointerId, p)
  }

  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    const p = stagePoint(stageRef.current, e)
    if (corner && img) {
      const bounds = intersect(frameOf({ x: 0, y: 0, w: img.w, h: img.h }, corner.view), {
        x: 0,
        y: 0,
        ...stage,
      })
      setCorner({
        ...corner,
        ...p,
        frame: dragCorner(corner.frame, corner.corner, p.x - corner.x, p.y - corner.y, bounds),
      })
      return
    }
    const prev = pointers.current.get(e.pointerId)
    if (!prev) return
    const other = [...pointers.current].find(([id]) => id !== e.pointerId)?.[1]
    pointers.current.set(e.pointerId, p)
    if (!other) {
      moveView((v) => ({ ...v, tx: v.tx + p.x - prev.x, ty: v.ty + p.y - prev.y }))
      return
    }
    moveView((v) => pinch(v, prev, p, other))
  }

  const onPointerUp = (e: PointerEvent<HTMLDivElement>) => {
    pointers.current.delete(e.pointerId)
    if (corner && edit) {
      // The new shape settles back into the middle of the stage.
      setEdit({ ...edit, crop: exactCrop(corner.view, corner.frame) })
      setCorner(null)
    }
  }

  const onKeyDown = (e: KeyboardEvent) => {
    if (!frame) return
    const centre = { x: frame.x + frame.w / 2, y: frame.y + frame.h / 2 }
    const move = keyMove(e.key, e.shiftKey ? 40 : 10, 1.1, centre)
    if (move) {
      e.preventDefault()
      moveView(move)
    }
  }

  return {
    natural,
    setNatural,
    edit,
    /** The image as turned, where it sits on the stage, and the frame over it. */
    img,
    view,
    frame,
    /** The crop in image px, the size it will be output at, and how far that scales it up. */
    crop,
    out,
    scale,
    setAspect,
    rotate,
    flip,
    reset,
    setResize: (resize: boolean) => {
      if (edit) setEdit({ ...edit, resize })
    },
    /** Zoom by `factor` about a point on the stage. */
    zoom: (factor: number, cx: number, cy: number) => moveView((v) => zoomAt(v, factor, cx, cy)),
    onPointerDown,
    onPointerMove,
    onPointerUp,
    onKeyDown,
  }
}
