import './MaskEditor.css'

import { useMutation } from '@tanstack/react-query'
import { type KeyboardEvent as ReactKeyboardEvent, useMemo, useRef, useState } from 'react'

import { api } from '@/api/client'
import type { BlobInfo } from '@/api/types'
import { blobUrl } from '@/api/urls'
import { ChoiceChips } from '@/components/ChoiceChips'
import { zoomAt } from '@/features/editors/crop/crop'
import { EditorDialog } from '@/features/editors/EditorDialog'
import { keyMove, stagePoint } from '@/features/editors/gestures'
import { useElementSize } from '@/hooks/useElementSize'
import type { Size, View } from '@/lib/geometry'
import type { Source } from '@/lib/image'

import type { Action } from './canvas'
import {
  clampView,
  emptyHistory,
  fitView,
  type History,
  push,
  redo,
  undo,
  workingSize,
} from './mask'
import { MaskTools } from './MaskTools'
import { RangeRow } from './RangeRow'
import { SelectControls } from './SelectControls'
import { useMaskCanvas } from './useMaskCanvas'
import { useMaskPointers } from './useMaskPointers'
import { useSelection } from './useSelection'
import { useSelectNote } from './useSelectNote'

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

const TOOLS = [
  { id: 'brush', label: 'Brush' },
  { id: 'erase', label: 'Erase' },
  { id: 'select', label: 'Select' },
] as const
type Tool = (typeof TOOLS)[number]['id']

// Used until the stage has been measured (and in tests, which have no layout).
const FALLBACK_STAGE: Size = { w: 360, h: 480 }
const HATCH_GAP = 7 // screen px between hatching strokes at the fitted zoom, like a sketch tile

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
  const stageRef = useRef<HTMLDivElement>(null)
  const stage = useElementSize(stageRef, FALLBACK_STAGE)

  const selectNote = useSelectNote()

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
    onCombine: (op, layer) => setHistory((h) => push(h, { kind: 'selection', op, layer })),
  })

  const moveView = (change: (view: View) => View) =>
    setView((prev) => clampView(change(prev ? clampView(prev, stage, work) : fitted), stage, work))

  const pointers = useMaskPointers({
    stageRef,
    ready,
    selecting: tool === 'select',
    erase: tool === 'erase',
    brush,
    view: v,
    moveView,
    paint,
    onAbandon: () => render(history.done),
    onStroke: (stroke) => setHistory((h) => push(h, stroke)),
    onTap: (at, long) => {
      if (selectNote !== null || sam.pending) return
      if (at.x >= 0 && at.y >= 0 && at.x <= work.w && at.y <= work.h) {
        sam.addPoint(at.x, at.y, !(sam.exclude || long))
      }
    },
  })

  const onKeyDown = (e: ReactKeyboardEvent) => {
    const mod = e.metaKey || e.ctrlKey
    if (mod && e.key.toLowerCase() === 'z') {
      e.preventDefault()
      setHistory(e.shiftKey ? redo : undo)
      return
    }
    if (mod) return
    const move = keyMove(e.key, 40, 1.25, { x: stage.w / 2, y: stage.h / 2 })
    const keys: Record<string, (() => void) | undefined> = {
      b: () => setTool('brush'),
      e: () => setTool('erase'),
      '[': () => setBrush((b) => Math.max(2, Math.round(b / 1.25))),
      ']': () => setBrush((b) => Math.min(maxBrush, Math.round(b * 1.25))),
    }
    const action = move ? () => moveView(move) : keys[e.key]
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
    <EditorDialog
      title={title}
      className="mask-editor"
      onCancel={onCancel}
      action={
        <button
          type="button"
          className="btn small"
          disabled={!ready || save.isPending}
          onClick={() => save.mutate()}
        >
          {save.isPending ? 'Saving…' : 'Done'}
        </button>
      }
    >
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
        onPointerDown={pointers.onPointerDown}
        onPointerMove={pointers.onPointerMove}
        onPointerUp={pointers.onPointerUp}
        onPointerCancel={pointers.onPointerCancel}
        onPointerLeave={pointers.onPointerLeave}
        onKeyDown={onKeyDown}
        onWheel={(e) => {
          const p = stagePoint(stageRef.current, e)
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
        {pointers.ring && (
          <span
            className="brush-ring"
            style={{
              left: pointers.ring.x,
              top: pointers.ring.y,
              width: brush * v.s,
              height: brush * v.s,
            }}
            aria-hidden
          />
        )}
        {!ready && <p className="editor-loading">Loading the mask…</p>}
      </div>

      <div className="editor-controls">
        <ChoiceChips
          label="Tool"
          className="segmented tabs-3"
          options={TOOLS}
          value={tool}
          onChoose={setTool}
        />

        {tool === 'select' ? (
          <SelectControls note={selectNote} sam={sam} />
        ) : (
          <RangeRow
            id="brush-size"
            label="Size"
            min={2}
            max={maxBrush}
            value={brush}
            output={`${Math.round(brush / toWork)} px`}
            onChange={setBrush}
          />
        )}

        <MaskTools
          canUndo={history.done.length > 0}
          canRedo={history.undone.length > 0}
          ready={ready}
          showBlur={blur > 0 ? showBlur : undefined}
          onUndo={() => setHistory(undo)}
          onRedo={() => setHistory(redo)}
          onInvert={() => setHistory((h) => push(h, { kind: 'invert' }))}
          onClear={() => setHistory((h) => push(h, { kind: 'clear' }))}
          onShowBlur={setShowBlur}
        />
        <RangeRow
          id="image-opacity"
          label="Image"
          min={0.15}
          max={1}
          step={0.05}
          value={imageOpacity}
          output={`${Math.round(imageOpacity * 100)}%`}
          onChange={setImageOpacity}
        />
        {save.error && <p role="alert">{save.error.message}</p>}
      </div>
    </EditorDialog>
  )
}
