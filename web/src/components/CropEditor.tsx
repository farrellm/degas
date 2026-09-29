import { useMutation, useQuery } from '@tanstack/react-query'
import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from 'react'
import { api, blobUrl, type BlobInfo } from '../api'
import {
  ASPECTS,
  buildOps,
  centered,
  clampView,
  cropOf,
  dragCorner,
  exactCrop,
  frameFor,
  FREE_ASPECTS,
  frameOf,
  intersect,
  mirror,
  modelUpscale,
  outputSize,
  parseOps,
  ratio,
  reshape,
  rotated,
  upscale,
  UPSCALE_WARN,
  viewFor,
  zoomAt,
  type Aspect,
  type Constraints,
  type Rect,
  type Rotation,
  type Size,
  type View,
} from '../crop'
import { size } from '../format'

interface Props {
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
  onApply: (image: BlobInfo) => void
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
export function CropEditor({ sha, target, constraints, free = false, onApply, onCancel }: Props) {
  const history = useQuery({
    queryKey: ['transform', sha],
    queryFn: () => api.getTransform(sha),
    staleTime: Infinity,
  })
  const original = history.data?.original
  const [natural, setNatural] = useState<Size | null>(null)
  const [stage, setStage] = useState<Size>(FALLBACK_STAGE)
  const [edited, setEdit] = useState<Edit | null>(null)
  const [corner, setCorner] = useState<CornerDrag | null>(null)
  const stageRef = useRef<HTMLDivElement>(null)
  const ref = useRef<HTMLDivElement>(null)
  const pointers = useRef(new Map<number, { x: number; y: number }>())

  const apply = useMutation({
    mutationFn: (ops: Parameters<typeof api.transform>[1]) => api.transform(original ?? sha, ops),
    onSuccess: onApply,
  })

  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    ref.current?.focus()
    const overflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCancel()
    }
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = overflow
      opener?.focus()
    }
  }, [onCancel])

  useLayoutEffect(() => {
    const el = stageRef.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(([entry]) => {
      const r = entry?.contentRect
      if (r && r.width > 0 && r.height > 0) setStage({ w: r.width, h: r.height })
    })
    observer.observe(el)
    return () => {
      observer.disconnect()
    }
  }, [])

  const targetRatio = target.w / target.h

  /** Where a crop starts: all of a free crop's image, else the target's shape, centred. */
  const initial = (image: Size): Pick<Edit, 'aspect' | 'crop'> =>
    free
      ? { aspect: 'free', crop: { x: 0, y: 0, ...image } }
      : { aspect: 'match', crop: centered(image, targetRatio) }

  // Until edited: the earlier crop of a derived image, else where a crop starts.
  let edit = edited
  if (!edit && natural && history.data) {
    const { rot, flip, crop } = parseOps(history.data.ops)
    const img = rotated(natural, rot)
    if (crop) {
      const match = !free && Math.abs(crop.w / crop.h / targetRatio - 1) < 0.01
      edit = { rot, flip, aspect: match ? 'match' : 'free', crop }
    } else {
      edit = { rot, flip, ...initial(img) }
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
  const scale = crop && out ? (free ? modelUpscale(crop, target) : upscale(crop, out)) : 1

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
    setEdit({ rot: 0, flip: false, ...initial(natural) })
  }

  const point = (e: ReactPointerEvent) => {
    const r = stageRef.current?.getBoundingClientRect()
    return { x: e.clientX - (r?.left ?? 0), y: e.clientY - (r?.top ?? 0) }
  }

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
    // Pinch: zoom about the midpoint by the change in spread, and follow the midpoint.
    const before = Math.hypot(prev.x - other.x, prev.y - other.y)
    const after = Math.hypot(p.x - other.x, p.y - other.y)
    const mid = { x: (p.x + other.x) / 2, y: (p.y + other.y) / 2 }
    moveView((v) => {
      const z = before > 0 ? zoomAt(v, after / before, mid.x, mid.y) : v
      return { ...z, tx: z.tx + (p.x - prev.x) / 2, ty: z.ty + (p.y - prev.y) / 2 }
    })
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
              apply.mutate(buildOps(edit.rot, edit.flip, crop, img, out))
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
        <div className="aspect-chips" role="group" aria-label="Shape">
          {(free ? FREE_ASPECTS : ASPECTS).map((a) => (
            <button
              key={a.id}
              type="button"
              aria-pressed={edit?.aspect === a.id}
              disabled={!edit}
              onClick={() => {
                setAspect(a.id)
              }}
            >
              {a.id === 'match' ? `Match ${size(target.w, target.h)}` : a.label}
            </button>
          ))}
        </div>
        <div className="editor-tools">
          <button type="button" className="tool" disabled={!edit} onClick={rotate}>
            <svg viewBox="0 0 20 20" aria-hidden>
              <path d="M6 4.5 3.5 7 6 9.5" />
              <path d="M3.8 7H12a4.5 4.5 0 0 1 0 9H8" />
            </svg>
            Rotate
          </button>
          <button
            type="button"
            className="tool"
            aria-pressed={edit?.flip ?? false}
            disabled={!edit}
            onClick={flip}
          >
            <svg viewBox="0 0 20 20" aria-hidden>
              <path d="M10 2.5v15" strokeDasharray="2 2" />
              <path d="M7.5 5 3 14.5h4.5z" />
              <path d="M12.5 5 17 14.5h-4.5z" />
            </svg>
            Flip
          </button>
          <button type="button" className="tool" disabled={!edit} onClick={reset}>
            Reset
          </button>
        </div>
        {out && natural && (
          <p className={scale > UPSCALE_WARN ? 'readout warn' : 'readout'} aria-live="polite">
            {size(out.w, out.h)} from {size(natural.w, natural.h)}
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
