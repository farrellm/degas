import { useMutation, useQuery } from '@tanstack/react-query'
import {
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  useMemo,
  useRef,
  useState,
} from 'react'

import { api } from '@/api/client'
import { queries } from '@/api/queries'
import type { BlobInfo } from '@/api/types'
import { blobUrl } from '@/api/urls'
import { zoomAt } from '@/features/editors/crop/crop'
import { pinch, stagePoint } from '@/features/editors/gestures'
import { useAssets } from '@/hooks/useAssets'
import { useDialog } from '@/hooks/useDialog'
import { useElementSize } from '@/hooks/useElementSize'
import type { Size, View } from '@/lib/geometry'
import type { Source } from '@/lib/image'
import { isActive, isGpuReady } from '@/lib/session'

import type { Action, Stroke } from './canvas'
import {
  clampView,
  emptyHistory,
  fitView,
  type History,
  push,
  redo,
  toImage,
  undo,
  workingSize,
} from './mask'
import { MaskTools } from './MaskTools'
import { SelectControls } from './SelectControls'
import { useMaskCanvas } from './useMaskCanvas'
import { useSelection } from './useSelection'

export const SAM_ASSET = 'preprocessors/sam3'

export interface MaskEditorProps {
  /** The image the mask is painted over, and its pixel size. */
  source: Source
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

// Used until the stage has been measured (and in tests, which have no layout).
const FALLBACK_STAGE: Size = { w: 360, h: 480 }
const HATCH_GAP = 7 // screen px between hatching strokes at the fitted zoom, like a sketch tile
const TAP_SLOP = 8
const LONG_PRESS_MS = 500

/** Full-screen mask painting over the source (design §8.2, ux.md Phase 6). */
export function MaskEditor({
  source,
  mask,
  blur,
  underlay,
  title = 'Mask',
  onDone,
  onCancel,
}: MaskEditorProps) {
  const picture = underlay ?? source.sha
  const work = useMemo(
    () => workingSize({ w: source.width, h: source.height }),
    [source.width, source.height],
  )
  const toWork = work.w / source.width // working px per image px
  const [view, setView] = useState<View | null>(null)
  const [tool, setTool] = useState<Tool>('brush')
  const [brush, setBrush] = useState(() => Math.round(Math.max(work.w, work.h) / 24))
  const [history, setHistory] = useState<History<Action>>(emptyHistory)
  const [imageOpacity, setImageOpacity] = useState(1)
  const [showBlur, setShowBlur] = useState(false)
  const [ring, setRing] = useState<{ x: number; y: number } | null>(null)

  const ref = useDialog(onCancel)
  const stageRef = useRef<HTMLDivElement>(null)
  const stage = useElementSize(stageRef, FALLBACK_STAGE)
  const stroke = useRef<Stroke | null>(null)
  const pointers = useRef(new Map<number, { x: number; y: number }>())
  const tap = useRef<{ x: number; y: number; long: boolean; timer: number } | null>(null)

  const session = useQuery(queries.session())
  const assets = useAssets()
  const gpuReady = isGpuReady(session.data)
  const samIndexed = assets.data?.some((a) => a.path === SAM_ASSET) ?? false
  const selectNote = !samIndexed
    ? `Put SAM 3 in Drive under degas/${SAM_ASSET}/ and rescan to use Select.`
    : !gpuReady
      ? isActive(session.data)
        ? 'Select works once the GPU session is ready.'
        : 'Start a session to use Select.'
      : null

  const maxBrush = Math.round(Math.max(work.w, work.h) / 3)
  const fitted = fitView(stage, work)
  const v = view ? clampView(view, stage, work) : fitted

  const displayRef = useRef<HTMLCanvasElement>(null)
  const outlineRef = useRef<HTMLCanvasElement>(null)
  const { ready, render, paint, toPng } = useMaskCanvas({
    mask,
    displayRef,
    work,
    history,
    // The hatching is sized for the fitted zoom, so it reads like the feed's sketch tiles.
    hatchGap: HATCH_GAP / fitted.s,
  })

  const sam = useSelection({
    picture,
    outlineRef,
    work,
    toWork,
    zoom: v.s,
    onCombine: (op, layer) => {
      setHistory((h) => push(h, { kind: 'selection', op, layer }))
    },
  })

  // Pointers -------------------------------------------------------------------------------

  const point = (e: { clientX: number; clientY: number }) => stagePoint(stageRef.current, e)

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
      moveView((view) => pinch(view, prev, p, other))
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
    if (t && tool === 'select' && gpuReady && samIndexed && !sam.pending) {
      const at = toImage(v, t.x, t.y)
      if (at.x >= 0 && at.y >= 0 && at.x <= work.w && at.y <= work.h) {
        sam.addPoint(at.x, at.y, !(sam.exclude || t.long))
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
      const png = await toPng()
      return png && api.uploadMask(source.sha, png)
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
        {sam.points.map((p, i) => (
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
            points={sam.points}
            exclude={sam.exclude}
            onExclude={sam.setExclude}
            text={sam.text}
            onText={sam.setText}
            onFind={sam.find}
            pending={sam.pending}
            error={sam.error}
            selection={sam.selection}
            shown={sam.shown}
            onShown={sam.setShown}
            growBy={sam.growBy}
            onGrow={sam.setGrowBy}
            onCombine={sam.combine}
            onClear={sam.clear}
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

        <MaskTools
          canUndo={history.done.length > 0}
          canRedo={history.undone.length > 0}
          ready={ready}
          showBlur={blur > 0 ? showBlur : undefined}
          onUndo={() => {
            setHistory(undo)
          }}
          onRedo={() => {
            setHistory(redo)
          }}
          onInvert={() => {
            setHistory((h) => push(h, { kind: 'invert' }))
          }}
          onClear={() => {
            setHistory((h) => push(h, { kind: 'clear' }))
          }}
          onShowBlur={setShowBlur}
        />
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
