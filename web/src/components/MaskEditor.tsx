import { useMutation, useQuery } from '@tanstack/react-query'
import {
  useCallback,
  useDeferredValue,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from 'react'
import { api, blobUrl, isActive, type BlobInfo, type SelectPoint, type Selection } from '../api'
import { useAssets } from '../assets'
import { zoomAt, type Size, type View } from '../crop'
import {
  binarizeAlpha,
  clampView,
  emptyHistory,
  fitView,
  growSteps,
  hasAlpha,
  lumaToAlpha,
  push,
  redo,
  ringOffsets,
  toImage,
  undo,
  workingSize,
  type History,
} from '../mask'

export const SAM_ASSET = 'preprocessors/sam3'

interface Props {
  /** The image the mask is painted over, and its pixel size. */
  source: { sha: string; width: number; height: number }
  /** The current mask, to keep editing it. */
  mask: string | null
  /** The form's mask blur, in image pixels, for the blur preview (0 hides it). */
  blur: number
  /** A picture the same size as `source` to paint over and select in instead (a
   * ControlNet area is painted over the photo its trace came from). */
  underlay?: string
  /** "Mask", or "Area" for a ControlNet. */
  title?: string
  /** The stored mask, or null when nothing is painted. */
  onDone: (mask: BlobInfo | null) => void
  onCancel: () => void
}

type Tool = 'brush' | 'erase' | 'select'
type Combine = 'add' | 'subtract' | 'replace'

interface Stroke {
  kind: 'stroke'
  erase: boolean
  size: number
  points: [number, number][]
}

type Action =
  | Stroke
  | { kind: 'invert' }
  | { kind: 'clear' }
  | { kind: 'selection'; op: Combine; layer: HTMLCanvasElement }

// Used until the stage has been measured (and in tests, which have no layout).
const FALLBACK_STAGE: Size = { w: 360, h: 480 }
const HATCH_GAP = 7 // screen px between hatching strokes at the fitted zoom, like a sketch tile
const TAP_SLOP = 8
const LONG_PRESS_MS = 500
const MAX_GROW = 128 // image px of margin a selection can be grown by

function canvas(size: Size): HTMLCanvasElement {
  const c = document.createElement('canvas')
  c.width = size.w
  c.height = size.h
  return c
}

const context = (c: HTMLCanvasElement | null) => c?.getContext('2d') ?? null

/** Rose hatching as a canvas pattern; strokes drawn with it join without seams. */
function hatchPattern(ctx: CanvasRenderingContext2D, gap: number): CanvasPattern | string {
  const color =
    getComputedStyle(document.documentElement).getPropertyValue('--hatch').trim() || '#efa3b5'
  const t = Math.max(3, Math.round(gap))
  const tile = canvas({ w: t, h: t })
  const tc = context(tile)
  if (!tc) return color
  tc.strokeStyle = color
  tc.lineWidth = Math.max(1, t / 4.5)
  tc.lineCap = 'square'
  tc.beginPath()
  for (const o of [-t, 0, t]) {
    tc.moveTo(o, t)
    tc.lineTo(o + t, 0)
  }
  tc.stroke()
  return ctx.createPattern(tile, 'repeat') ?? color
}

function drawStroke(
  ctx: CanvasRenderingContext2D,
  stroke: Stroke,
  style: CanvasPattern | string,
  from = 0,
) {
  const pts = stroke.points.slice(Math.max(0, from - 1))
  const first = pts[0]
  if (!first) return
  ctx.save()
  ctx.globalCompositeOperation = stroke.erase ? 'destination-out' : 'source-over'
  ctx.strokeStyle = style
  ctx.fillStyle = style
  ctx.lineWidth = stroke.size
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  ctx.beginPath()
  if (pts.length === 1) {
    ctx.arc(first[0], first[1], stroke.size / 2, 0, 2 * Math.PI)
    ctx.fill()
  } else {
    ctx.moveTo(first[0], first[1])
    for (const [x, y] of pts.slice(1)) ctx.lineTo(x, y)
    ctx.stroke()
  }
  ctx.restore()
}

/** Apply one action to the mask (white where it will be redrawn, in the alpha channel). */
function apply(ctx: CanvasRenderingContext2D, action: Action, size: Size) {
  switch (action.kind) {
    case 'stroke':
      drawStroke(ctx, action, '#fff')
      break
    case 'clear':
      ctx.clearRect(0, 0, size.w, size.h)
      break
    case 'invert':
      ctx.save()
      ctx.globalCompositeOperation = 'xor'
      ctx.fillStyle = '#fff'
      ctx.fillRect(0, 0, size.w, size.h)
      ctx.restore()
      break
    case 'selection':
      ctx.save()
      if (action.op === 'replace') ctx.clearRect(0, 0, size.w, size.h)
      ctx.globalCompositeOperation = action.op === 'subtract' ? 'destination-out' : 'source-over'
      ctx.drawImage(action.layer, 0, 0)
      ctx.restore()
  }
}

/** A stored grey mask as an alpha layer at the working size. */
async function loadLayer(sha: string, size: Size): Promise<HTMLCanvasElement> {
  const img = new Image()
  img.src = blobUrl(sha)
  await img.decode()
  const layer = canvas(size)
  const ctx = context(layer)
  if (ctx) {
    ctx.drawImage(img, 0, 0, size.w, size.h)
    const data = ctx.getImageData(0, 0, size.w, size.h)
    lumaToAlpha(data.data)
    ctx.putImageData(data, 0, 0)
  }
  return layer
}

/** A layer grown by `r` px, to give a selection a margin to blend into. */
function grow(layer: HTMLCanvasElement, r: number): HTMLCanvasElement {
  const steps = growSteps(r)
  if (!steps.length) return layer
  const size = { w: layer.width, h: layer.height }
  let from = canvas(size)
  const first = context(from)
  if (!first) return layer
  first.drawImage(layer, 0, 0)
  const data = first.getImageData(0, 0, size.w, size.h)
  binarizeAlpha(data.data)
  first.putImageData(data, 0, 0)
  for (const step of steps) {
    const out = canvas(size)
    const ctx = context(out)
    if (!ctx) return from
    ctx.imageSmoothingEnabled = false
    ctx.drawImage(from, 0, 0)
    for (const o of ringOffsets(step)) ctx.drawImage(from, o.x, o.y)
    from = out
  }
  return from
}

/** The edge of a layer, `width` px thick, for showing a selection before it's used. */
function outline(layer: HTMLCanvasElement, width: number, style: string): HTMLCanvasElement {
  const size = { w: layer.width, h: layer.height }
  const inner = canvas(size)
  const ic = context(inner)
  const out = canvas(size)
  const oc = context(out)
  if (!ic || !oc) return out
  ic.drawImage(layer, 0, 0)
  ic.globalCompositeOperation = 'destination-in'
  for (const [dx, dy] of [
    [width, 0],
    [-width, 0],
    [0, width],
    [0, -width],
  ] as const) {
    ic.drawImage(layer, dx, dy)
  }
  oc.drawImage(layer, 0, 0)
  oc.globalCompositeOperation = 'destination-out'
  oc.drawImage(inner, 0, 0)
  oc.globalCompositeOperation = 'source-in'
  oc.fillStyle = style
  oc.fillRect(0, 0, size.w, size.h)
  return out
}

/** Full-screen mask painting over the source (design §8.2, ux.md Phase 6). */
export function MaskEditor({
  source,
  mask,
  blur,
  underlay,
  title = 'Mask',
  onDone,
  onCancel,
}: Props) {
  const picture = underlay ?? source.sha
  const work = useMemo(
    () => workingSize({ w: source.width, h: source.height }),
    [source.width, source.height],
  )
  const toWork = work.w / source.width // working px per image px
  const [stage, setStage] = useState<Size>(FALLBACK_STAGE)
  const [view, setView] = useState<View | null>(null)
  const [tool, setTool] = useState<Tool>('brush')
  const [brush, setBrush] = useState(() => Math.round(Math.max(work.w, work.h) / 24))
  const [history, setHistory] = useState<History<Action>>(emptyHistory)
  const [ready, setReady] = useState(!mask)
  const [imageOpacity, setImageOpacity] = useState(1)
  const [showBlur, setShowBlur] = useState(false)
  const [ring, setRing] = useState<{ x: number; y: number } | null>(null)
  const [points, setPoints] = useState<SelectPoint[]>([])
  const [exclude, setExclude] = useState(false)
  const [text, setText] = useState('')
  const [selection, setSelection] = useState<Selection | null>(null)
  const [shown, setShown] = useState(0)
  const [growBy, setGrowBy] = useState(8)
  const [layers, setLayers] = useState<Record<string, HTMLCanvasElement>>({})

  const ref = useRef<HTMLDivElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)
  const displayRef = useRef<HTMLCanvasElement>(null)
  const outlineRef = useRef<HTMLCanvasElement>(null)
  const maskCanvas = useRef<HTMLCanvasElement | null>(null)
  const base = useRef<HTMLCanvasElement | null>(null)
  const pattern = useRef<CanvasPattern | string>('#efa3b5')
  const stroke = useRef<Stroke | null>(null)
  const pointers = useRef(new Map<number, { x: number; y: number }>())
  const tap = useRef<{ x: number; y: number; long: boolean; timer: number } | null>(null)

  const session = useQuery({ queryKey: ['session'], queryFn: api.session })
  const assets = useAssets()
  const gpuReady = ['ready', 'busy'].includes(session.data?.session?.state ?? '')
  const samIndexed = assets.data?.some((a) => a.path === SAM_ASSET) ?? false
  const selectNote = !samIndexed
    ? `Put SAM 3 in Drive under degas/${SAM_ASSET}/ and rescan to use Select.`
    : !gpuReady
      ? isActive(session.data)
        ? 'Select works once the GPU session is ready.'
        : 'Start a session to use Select.'
      : null

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

  const maxBrush = Math.round(Math.max(work.w, work.h) / 3)
  const fitted = fitView(stage, work)
  const v = view ? clampView(view, stage, work) : fitted

  // The mask itself lives off screen; the display is the mask filled with hatching.
  const recomposite = useCallback(() => {
    const m = maskCanvas.current
    const ctx = context(displayRef.current)
    if (!m || !ctx) return
    ctx.save()
    ctx.clearRect(0, 0, work.w, work.h)
    ctx.drawImage(m, 0, 0)
    ctx.globalCompositeOperation = 'source-in'
    ctx.fillStyle = pattern.current
    ctx.fillRect(0, 0, work.w, work.h)
    ctx.restore()
  }, [work.w, work.h])

  const render = useCallback(
    (actions: Action[]) => {
      maskCanvas.current ??= canvas(work)
      const ctx = context(maskCanvas.current)
      if (!ctx) return
      ctx.clearRect(0, 0, work.w, work.h)
      if (base.current) ctx.drawImage(base.current, 0, 0)
      for (const a of actions) apply(ctx, a, work)
      recomposite()
    },
    [work, recomposite],
  )

  // The hatching is sized for the fitted zoom, so it reads like the feed's sketch tiles.
  useEffect(() => {
    const ctx = context(displayRef.current)
    if (ctx) pattern.current = hatchPattern(ctx, HATCH_GAP / fitted.s)
    recomposite()
  }, [fitted.s, recomposite])

  useEffect(() => {
    if (!mask) return
    let live = true
    loadLayer(mask, work)
      .then((layer) => {
        if (!live) return
        base.current = layer
        setReady(true)
      })
      .catch(() => {
        if (live) setReady(true) // start from an empty mask
      })
    return () => {
      live = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- load once, at open
  }, [])

  useEffect(() => {
    if (ready) render(history.done)
  }, [ready, history, render])

  // SAM ----------------------------------------------------------------------------------

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
    ctx.drawImage(outline(grown, Math.max(1, 2 / v.s), accent), 0, 0)
  }, [grown, v.s, work.w, work.h])

  const addPoint = (x: number, y: number, include: boolean) => {
    const next = [...points, { x: x / toWork, y: y / toWork, include }]
    setPoints(next)
    select.mutate({ points: next, text: text.trim() })
  }

  const clearSelection = () => {
    setPoints([])
    setSelection(null)
    select.reset()
  }

  const combine = (op: Combine) => {
    if (!layer || !grown) return
    const added = previewGrow === growBy ? grown : grow(layer, growBy * toWork)
    setHistory((h) => push(h, { kind: 'selection', op, layer: added }))
    clearSelection()
  }

  // Pointers -------------------------------------------------------------------------------

  const paint = (s: Stroke, from: number) => {
    const m = context(maskCanvas.current)
    const d = context(displayRef.current)
    if (m) drawStroke(m, s, '#fff', from)
    if (d) drawStroke(d, s, pattern.current, from)
  }

  const point = (e: { clientX: number; clientY: number }) => {
    const r = stageRef.current?.getBoundingClientRect()
    return { x: e.clientX - (r?.left ?? 0), y: e.clientY - (r?.top ?? 0) }
  }

  const moveView = (change: (view: View) => View) => {
    setView((prev) => clampView(change(prev ? clampView(prev, stage, work) : fitted), stage, work))
  }

  const cancelTap = () => {
    if (tap.current) window.clearTimeout(tap.current.timer)
    tap.current = null
  }

  const onPointerDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (!ready || (e.pointerType === 'mouse' && e.button !== 0)) return
    e.currentTarget.setPointerCapture(e.pointerId)
    const p = point(e)
    pointers.current.set(e.pointerId, p)
    if (pointers.current.size > 1) {
      // A second finger: this is a pinch, not paint.
      cancelTap()
      if (stroke.current) {
        stroke.current = null
        render(history.done)
      }
      return
    }
    if (tool === 'select') {
      const timer = window.setTimeout(() => {
        if (tap.current) tap.current.long = true
      }, LONG_PRESS_MS)
      tap.current = { ...p, long: false, timer }
      return
    }
    const at = toImage(v, p.x, p.y)
    stroke.current = {
      kind: 'stroke',
      erase: tool === 'erase',
      size: brush,
      points: [[at.x, at.y]],
    }
    paint(stroke.current, 0)
  }

  const onPointerMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    const p = point(e)
    if (tool !== 'select') setRing(p)
    const prev = pointers.current.get(e.pointerId)
    if (!prev) return
    const other = [...pointers.current].find(([id]) => id !== e.pointerId)?.[1]
    pointers.current.set(e.pointerId, p)
    if (other) {
      const before = Math.hypot(prev.x - other.x, prev.y - other.y)
      const after = Math.hypot(p.x - other.x, p.y - other.y)
      const mid = { x: (p.x + other.x) / 2, y: (p.y + other.y) / 2 }
      moveView((view) => {
        const z = before > 0 ? zoomAt(view, after / before, mid.x, mid.y) : view
        return { ...z, tx: z.tx + (p.x - prev.x) / 2, ty: z.ty + (p.y - prev.y) / 2 }
      })
      return
    }
    if (tap.current && Math.hypot(p.x - tap.current.x, p.y - tap.current.y) > TAP_SLOP) {
      cancelTap()
    }
    const s = stroke.current
    if (s) {
      const at = toImage(v, p.x, p.y)
      s.points.push([at.x, at.y])
      paint(s, s.points.length - 1)
    }
  }

  const onPointerUp = (e: ReactPointerEvent<HTMLDivElement>) => {
    pointers.current.delete(e.pointerId)
    const s = stroke.current
    if (s) {
      stroke.current = null
      setHistory((h) => push(h, s))
    }
    const t = tap.current
    if (t && tool === 'select' && gpuReady && samIndexed && !select.isPending) {
      const at = toImage(v, t.x, t.y)
      if (at.x >= 0 && at.y >= 0 && at.x <= work.w && at.y <= work.h) {
        addPoint(at.x, at.y, !(exclude || t.long))
      }
    }
    cancelTap()
  }

  const onKeyDown = (e: ReactKeyboardEvent) => {
    const cx = stage.w / 2
    const cy = stage.h / 2
    const mod = e.metaKey || e.ctrlKey
    const keys: Record<string, () => void> = {
      b: () => {
        setTool('brush')
      },
      e: () => {
        setTool('erase')
      },
      '[': () => {
        setBrush((b) => Math.max(2, Math.round(b / 1.25)))
      },
      ']': () => {
        setBrush((b) => Math.min(maxBrush, Math.round(b * 1.25)))
      },
      '+': () => {
        moveView((view) => zoomAt(view, 1.25, cx, cy))
      },
      '=': () => {
        moveView((view) => zoomAt(view, 1.25, cx, cy))
      },
      '-': () => {
        moveView((view) => zoomAt(view, 1 / 1.25, cx, cy))
      },
      ArrowLeft: () => {
        moveView((view) => ({ ...view, tx: view.tx + 40 }))
      },
      ArrowRight: () => {
        moveView((view) => ({ ...view, tx: view.tx - 40 }))
      },
      ArrowUp: () => {
        moveView((view) => ({ ...view, ty: view.ty + 40 }))
      },
      ArrowDown: () => {
        moveView((view) => ({ ...view, ty: view.ty - 40 }))
      },
    }
    if (mod && e.key.toLowerCase() === 'z') {
      e.preventDefault()
      setHistory(e.shiftKey ? redo : undo)
      return
    }
    const action = mod ? undefined : keys[e.key]
    if (action) {
      e.preventDefault()
      action()
    }
  }

  // Done -------------------------------------------------------------------------------------

  const save = useMutation({
    mutationFn: async (): Promise<BlobInfo | null> => {
      const m = maskCanvas.current
      const ctx = context(m)
      if (!m || !ctx) throw new Error('This browser can’t paint masks.')
      if (!hasAlpha(ctx.getImageData(0, 0, work.w, work.h).data)) return null
      const png = await new Promise<Blob | null>((resolve) => {
        m.toBlob(resolve, 'image/png')
      })
      if (!png) throw new Error('Couldn’t read the painted mask.')
      return api.uploadMask(source.sha, png)
    },
    onSuccess: onDone,
  })

  const layerStyle = {
    left: v.tx,
    top: v.ty,
    width: work.w * v.s,
    height: work.h * v.s,
  }
  const blurPx = (blur * toWork * v.s).toFixed(1)
  const onStage = (x: number, y: number) => ({ left: v.tx + x * v.s, top: v.ty + y * v.s })

  return (
    <div
      ref={ref}
      className="editor mask-editor"
      role="dialog"
      aria-modal="true"
      aria-label={title}
      tabIndex={-1}
    >
      <div className="editor-bar">
        <button type="button" className="btn quiet small" onClick={onCancel}>
          Cancel
        </button>
        <h2>{title}</h2>
        <button
          type="button"
          className="btn small"
          disabled={!ready || save.isPending}
          onClick={() => {
            save.mutate()
          }}
        >
          {save.isPending ? 'Saving…' : 'Done'}
        </button>
      </div>

      <div
        ref={stageRef}
        className={`crop-stage mask-stage tool-${tool}`}
        role="application"
        aria-label={
          tool === 'select'
            ? 'Image. Tap what to select; long-press to leave something out. Pinch to zoom.'
            : `Image. Drag to ${tool === 'erase' ? 'erase' : 'paint'} the mask; pinch, or press + and −, to zoom.`
        }
        tabIndex={0}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onPointerLeave={() => {
          setRing(null)
        }}
        onKeyDown={onKeyDown}
        onWheel={(e) => {
          const p = point(e)
          moveView((view) => zoomAt(view, Math.exp(-e.deltaY / 400), p.x, p.y))
        }}
      >
        <div className="mask-layer" style={layerStyle}>
          <img src={blobUrl(picture)} alt="" draggable={false} style={{ opacity: imageOpacity }} />
          <canvas
            ref={displayRef}
            width={work.w}
            height={work.h}
            style={showBlur ? { filter: `blur(${blurPx}px)` } : undefined}
          />
          <canvas ref={outlineRef} width={work.w} height={work.h} />
        </div>
        {points.map((p, i) => (
          <span
            key={i}
            className={p.include ? 'sam-point' : 'sam-point exclude'}
            style={onStage(p.x * toWork, p.y * toWork)}
            aria-hidden
          />
        ))}
        {ring && tool !== 'select' && (
          <span
            className="brush-ring"
            style={{ left: ring.x, top: ring.y, width: brush * v.s, height: brush * v.s }}
            aria-hidden
          />
        )}
        {!ready && <p className="editor-loading">Loading the mask…</p>}
      </div>

      <div className="editor-controls">
        <div className="segmented tabs-3" role="group" aria-label="Tool">
          {(['brush', 'erase', 'select'] as const).map((t) => (
            <button
              key={t}
              type="button"
              aria-pressed={tool === t}
              onClick={() => {
                setTool(t)
              }}
            >
              {t === 'brush' ? 'Brush' : t === 'erase' ? 'Erase' : 'Select'}
            </button>
          ))}
        </div>

        {tool === 'select' ? (
          <SelectControls
            note={selectNote}
            points={points}
            exclude={exclude}
            onExclude={setExclude}
            text={text}
            onText={setText}
            onFind={() => {
              select.mutate({ points, text: text.trim() })
            }}
            pending={select.isPending}
            error={select.error?.message ?? null}
            selection={selection}
            shown={shown}
            onShown={setShown}
            growBy={growBy}
            onGrow={setGrowBy}
            onCombine={combine}
            onClear={clearSelection}
          />
        ) : (
          <div className="mask-row">
            <label htmlFor="brush-size">Size</label>
            <input
              id="brush-size"
              type="range"
              min={2}
              max={maxBrush}
              value={brush}
              onChange={(e) => {
                setBrush(Number(e.target.value))
              }}
            />
            <output htmlFor="brush-size">{Math.round(brush / toWork)} px</output>
          </div>
        )}

        <div className="editor-tools mask-tools">
          <button
            type="button"
            className="tool"
            aria-label="Undo"
            disabled={!history.done.length}
            onClick={() => {
              setHistory(undo)
            }}
          >
            <svg viewBox="0 0 20 20" aria-hidden>
              <path d="M7 4.5 4 7.5l3 3" />
              <path d="M4.5 7.5H12a4 4 0 0 1 0 8H9" />
            </svg>
          </button>
          <button
            type="button"
            className="tool"
            aria-label="Redo"
            disabled={!history.undone.length}
            onClick={() => {
              setHistory(redo)
            }}
          >
            <svg viewBox="0 0 20 20" aria-hidden>
              <path d="m13 4.5 3 3-3 3" />
              <path d="M15.5 7.5H8a4 4 0 0 0 0 8h3" />
            </svg>
          </button>
          <button
            type="button"
            className="tool"
            disabled={!ready}
            onClick={() => {
              setHistory((h) => push(h, { kind: 'invert' }))
            }}
          >
            Invert
          </button>
          <button
            type="button"
            className="tool"
            disabled={!ready}
            onClick={() => {
              setHistory((h) => push(h, { kind: 'clear' }))
            }}
          >
            Clear
          </button>
          {blur > 0 && (
            <button
              type="button"
              className="tool"
              aria-pressed={showBlur}
              onClick={() => {
                setShowBlur(!showBlur)
              }}
            >
              Blur
            </button>
          )}
        </div>
        <div className="mask-row">
          <label htmlFor="image-opacity">Image</label>
          <input
            id="image-opacity"
            type="range"
            min={0.15}
            max={1}
            step={0.05}
            value={imageOpacity}
            onChange={(e) => {
              setImageOpacity(Number(e.target.value))
            }}
          />
          <output htmlFor="image-opacity">{Math.round(imageOpacity * 100)}%</output>
        </div>
        {save.error && <p role="alert">{save.error.message}</p>}
      </div>
    </div>
  )
}

interface SelectProps {
  note: string | null
  points: SelectPoint[]
  exclude: boolean
  onExclude: (exclude: boolean) => void
  text: string
  onText: (text: string) => void
  onFind: () => void
  pending: boolean
  error: string | null
  selection: Selection | null
  shown: number
  onShown: (i: number) => void
  growBy: number
  onGrow: (px: number) => void
  onCombine: (op: Combine) => void
  onClear: () => void
}

/** SAM 3: taps and a description make a selection, which then adds to the mask. */
function SelectControls(p: SelectProps) {
  if (p.note) return <p className="row-note mask-note">{p.note}</p>
  const count = p.selection?.candidates.length ?? 0
  return (
    <>
      {/* Not a <form>: the editor renders inside the Create form, and a nested submit would
          bubble up and queue a generation. */}
      <div className="mask-row describe">
        <label htmlFor="select-text" className="visually-hidden">
          Describe what to select
        </label>
        <input
          id="select-text"
          type="text"
          placeholder="Describe it, or tap the image"
          value={p.text}
          enterKeyHint="search"
          onChange={(e) => {
            p.onText(e.target.value)
          }}
          onKeyDown={(e) => {
            if (e.key !== 'Enter') return
            e.preventDefault()
            if (p.text.trim() && !p.pending) p.onFind()
          }}
        />
        <button
          type="button"
          className="btn quiet small"
          disabled={!p.text.trim() || p.pending}
          onClick={p.onFind}
        >
          Find
        </button>
      </div>
      <div className="mask-row">
        <div className="seed-modes" role="group" aria-label="Taps">
          <button
            type="button"
            aria-pressed={!p.exclude}
            onClick={() => {
              p.onExclude(false)
            }}
          >
            Include
          </button>
          <button
            type="button"
            aria-pressed={p.exclude}
            onClick={() => {
              p.onExclude(true)
            }}
          >
            Exclude
          </button>
        </div>
        {(p.points.length > 0 || p.selection) && (
          <button type="button" className="btn quiet small" onClick={p.onClear}>
            Start over
          </button>
        )}
      </div>
      <p className="row-note mask-note" aria-live="polite">
        {p.pending
          ? 'Selecting… the first time on a session copies SAM 3 to the GPU.'
          : p.error
            ? null
            : p.selection && count === 0
              ? 'Nothing matched. Tap it instead, or describe it differently.'
              : p.selection
                ? 'Shown as an outline. Add it to the mask, or tap to refine.'
                : 'Tap what to select. Long-press, or choose Exclude, to leave something out.'}
      </p>
      {p.error && <p role="alert">{p.error}</p>}
      <div className="mask-row">
        <label htmlFor="grow">Grow</label>
        <input
          id="grow"
          type="range"
          min={0}
          max={MAX_GROW}
          value={p.growBy}
          onChange={(e) => {
            p.onGrow(Number(e.target.value))
          }}
        />
        <output htmlFor="grow">{p.growBy} px</output>
      </div>
      {count > 0 && (
        <>
          <div className="mask-row">
            <button
              type="button"
              className="btn quiet small"
              disabled={p.shown <= 0}
              onClick={() => {
                p.onShown(p.shown - 1)
              }}
            >
              Smaller
            </button>
            <button
              type="button"
              className="btn quiet small"
              disabled={p.shown >= count - 1}
              onClick={() => {
                p.onShown(p.shown + 1)
              }}
            >
              Bigger
            </button>
          </div>
          <div className="mask-row combine">
            <button
              type="button"
              className="btn small"
              onClick={() => {
                p.onCombine('add')
              }}
            >
              Add
            </button>
            <button
              type="button"
              className="btn quiet small"
              onClick={() => {
                p.onCombine('subtract')
              }}
            >
              Subtract
            </button>
            <button
              type="button"
              className="btn quiet small"
              onClick={() => {
                p.onCombine('replace')
              }}
            >
              Replace
            </button>
          </div>
        </>
      )}
    </>
  )
}
