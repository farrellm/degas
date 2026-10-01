import { useMutation, useQuery } from '@tanstack/react-query'
import {
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  useRef,
  useState,
} from 'react'

import { api } from '@/api/client'
import { queries } from '@/api/queries'
import type { BlobInfo } from '@/api/types'
import { blobUrl } from '@/api/urls'
import { pinch, stagePoint } from '@/features/editors/gestures'
import { useDialog } from '@/hooks/useDialog'
import { useElementSize } from '@/hooks/useElementSize'
import { formatSize } from '@/lib/format'
import { ratioDiffers, ratioMatches, type Rect, type Size, type View } from '@/lib/geometry'

import {
  type Aspect,
  ASPECTS,
  buildOps,
  centered,
  clampView,
  type Constraints,
  cropOf,
  dragCorner,
  exactCrop,
  frameFor,
  frameOf,
  FREE_ASPECTS,
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
  UPSCALE_WARN,
  viewFor,
  zoomAt,
} from './crop'
import { CropTools } from './CropTools'

export interface CropEditorProps {
  /** The image in the slot; a derived image reopens on its original with its crop. */
  sha: string
  /** The form's output size. */
  target: Size
  constraints: Constraints
  /**
   * A free crop, for a reference: any shape, kept at its own size, because the model sizes
   * each reference itself (to the output's pixel count).
   */
  free?: boolean
  /** The model scales references down only (FLUX.2 [klein]), so a small crop isn't enlarged. */
  refsKeepSize?: boolean
  /**
   * An image prompt's picture: a free crop that starts square, because the image encoder sees
   * a small square from the middle. Its size doesn't matter, so there's no upscale warning.
   */
  square?: boolean
  /** With the size it's meant for: the resize's, even when the crop was kept at its own. */
  onApply: (image: BlobInfo, out: Size) => void
  onCancel: () => void
}

// Used until the stage has been measured (and in tests, which have no layout).
const FALLBACK_STAGE: Size = { w: 360, h: 480 }
const CORNERS = ['nw', 'ne', 'sw', 'se'] as const
type Corner = (typeof CORNERS)[number]

/**
 * What the editor is set to. The crop is in (rotated) image pixels, unrounded; the frame
 * and the image's placement on screen are derived from it and the stage size.
 */
interface Edit {
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

/** Full-screen crop, rotate and flip on the image well (design §8.2, ux.md Phase 4). */
export function CropEditor({
  sha,
  target,
  constraints,
  free: freeCrop = false,
  refsKeepSize = false,
  square = false,
  onApply,
  onCancel,
}: CropEditorProps) {
  const free = freeCrop || square
  const history = useQuery(queries.transform(sha))
  const original = history.data?.original
  const [natural, setNatural] = useState<Size | null>(null)
  const [edited, setEdit] = useState<Edit | null>(null)
  const [corner, setCorner] = useState<CornerDrag | null>(null)
  const stageRef = useRef<HTMLDivElement>(null)
  const stage = useElementSize(stageRef, FALLBACK_STAGE)
  const ref = useDialog(onCancel)
  const pointers = useRef(new Map<number, { x: number; y: number }>())

  const apply = useMutation({
    mutationFn: ({ ops }: { ops: Parameters<typeof api.transform>[1]; out: Size }) =>
      api.transform(original ?? sha, ops),
    onSuccess: (image, { out }) => {
      onApply(image, out)
    },
  })

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
  if (!edit && natural && history.data) {
    const { rot, flip, crop, resized } = parseOps(history.data.ops)
    const img = rotated(natural, rot)
    if (crop) {
      const match = !free && ratioMatches(crop.w / crop.h, targetRatio)
      edit = { rot, flip, aspect: match ? 'match' : 'free', crop, resize: resized }
    } else {
      edit = { rot, flip, resize: true, ...initial(img) }
    }
  }

  if (history.error) {
    return (
      <div
        ref={ref}
        className="editor"
        role="dialog"
        aria-modal="true"
        aria-label="Crop"
        tabIndex={-1}
      >
        <div className="editor-bar">
          <button type="button" className="btn quiet small" onClick={onCancel}>
            Cancel
          </button>
          <h2>Crop</h2>
        </div>
        <p className="editor-loading" role="alert">
          Couldn’t open this image for cropping: {history.error.message}
        </p>
      </div>
    )
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
  const pan = (dx: number, dy: number) => {
    moveView((v) => ({ ...v, tx: v.tx + dx, ty: v.ty + dy }))
  }
  const zoom = (factor: number, cx: number, cy: number) => {
    moveView((v) => zoomAt(v, factor, cx, cy))
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

  const point = (e: ReactPointerEvent) => stagePoint(stageRef.current, e)

  const onPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    const handle = (e.target as HTMLElement).dataset.corner as Corner | undefined
    e.currentTarget.setPointerCapture(e.pointerId)
    const p = point(e)
    if (handle && edit?.aspect === 'free' && frame && view) {
      setCorner({ corner: handle, ...p, frame, view })
      return
    }
    pointers.current.set(e.pointerId, p)
  }

  const onPointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    const p = point(e)
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
      pan(p.x - prev.x, p.y - prev.y)
      return
    }
    moveView((v) => pinch(v, prev, p, other))
  }

  const onPointerUp = (e: ReactPointerEvent<HTMLDivElement>) => {
    pointers.current.delete(e.pointerId)
    if (corner && edit) {
      // The new shape settles back into the middle of the stage.
      setEdit({ ...edit, crop: exactCrop(corner.view, corner.frame) })
      setCorner(null)
    }
  }

  const onKeyDown = (e: ReactKeyboardEvent) => {
    if (!frame) return
    const step = e.shiftKey ? 40 : 10
    const cx = frame.x + frame.w / 2
    const cy = frame.y + frame.h / 2
    const moves: Record<string, () => void> = {
      ArrowLeft: () => {
        pan(step, 0)
      },
      ArrowRight: () => {
        pan(-step, 0)
      },
      ArrowUp: () => {
        pan(0, step)
      },
      ArrowDown: () => {
        pan(0, -step)
      },
      '+': () => {
        zoom(1.1, cx, cy)
      },
      '=': () => {
        zoom(1.1, cx, cy)
      },
      '-': () => {
        zoom(1 / 1.1, cx, cy)
      },
    }
    const move = moves[e.key]
    if (move) {
      e.preventDefault()
      move()
    }
  }

  const placed: CSSProperties | undefined =
    view && img
      ? { left: view.tx, top: view.ty, width: img.w * view.s, height: img.h * view.s }
      : undefined
  const turned: CSSProperties | undefined =
    natural && edit && view
      ? {
          width: natural.w * view.s,
          height: natural.h * view.s,
          transform: `translate(-50%, -50%) scaleX(${edit.flip ? '-1' : '1'}) rotate(${String(edit.rot)}deg)`,
        }
      : undefined

  return (
    <div
      ref={ref}
      className="editor"
      role="dialog"
      aria-modal="true"
      aria-label="Crop"
      tabIndex={-1}
    >
      <div className="editor-bar">
        <button type="button" className="btn quiet small" onClick={onCancel}>
          Cancel
        </button>
        <h2>Crop</h2>
        <button
          type="button"
          className="btn small"
          disabled={!edit || !crop || !out || !img || apply.isPending}
          onClick={() => {
            if (edit && crop && out && img) {
              const to = edit.resize ? out : crop
              apply.mutate({ ops: buildOps(edit.rot, edit.flip, crop, img, to), out })
            }
          }}
        >
          {apply.isPending ? 'Applying…' : 'Apply'}
        </button>
      </div>

      <div
        ref={stageRef}
        className="crop-stage"
        role="application"
        aria-label="Image under the crop frame. Drag to move it; pinch, or press + and −, to zoom."
        tabIndex={0}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onKeyDown={onKeyDown}
        onWheel={(e) => {
          zoom(Math.exp(-e.deltaY / 400), e.nativeEvent.offsetX, e.nativeEvent.offsetY)
        }}
      >
        {original && (
          <div className="crop-image" style={placed}>
            <img
              src={blobUrl(original)}
              alt=""
              draggable={false}
              style={turned}
              onLoad={(e) => {
                const { naturalWidth: w, naturalHeight: h } = e.currentTarget
                if (w && h) setNatural({ w, h })
              }}
            />
          </div>
        )}
        {frame && edit && (
          <div
            className={edit.aspect === 'free' ? 'crop-frame free' : 'crop-frame'}
            style={{ left: frame.x, top: frame.y, width: frame.w, height: frame.h }}
          >
            {CORNERS.map((c) => (
              <span key={c} className={`crop-corner ${c}`} data-corner={c} aria-hidden />
            ))}
          </div>
        )}
        {!edit && <p className="editor-loading">Loading the image…</p>}
      </div>

      <div className="editor-controls">
        <CropTools
          aspects={(free ? FREE_ASPECTS : ASPECTS).map((a) =>
            a.id === 'match' ? { ...a, label: `Match ${formatSize(target.w, target.h)}` } : a,
          )}
          aspect={edit?.aspect}
          disabled={!edit}
          flipped={edit?.flip ?? false}
          resize={free ? undefined : (edit?.resize ?? true)}
          onAspect={setAspect}
          onRotate={rotate}
          onFlip={flip}
          onResize={(resize) => {
            if (edit) setEdit({ ...edit, resize })
          }}
          onReset={reset}
        />
        {out && crop && natural && edit && (
          <p className={scale > UPSCALE_WARN ? 'readout warn' : 'readout'} aria-live="polite">
            {edit.resize ? formatSize(out.w, out.h) : formatSize(crop.w, crop.h)} from{' '}
            {formatSize(natural.w, natural.h)}
            {!edit.resize && !free && (out.w !== crop.w || out.h !== crop.h) && (
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
                {free ? 'The model scales it up' : 'Scaled up'} {scale.toFixed(1)}×, so fine detail
                will be soft.
              </>
            )}
          </p>
        )}
        {apply.error && <p role="alert">{apply.error.message}</p>}
      </div>
    </div>
  )
}
