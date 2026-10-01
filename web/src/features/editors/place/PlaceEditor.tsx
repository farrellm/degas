import { useRef, type KeyboardEvent, type PointerEvent } from 'react'
import { thumbUrl } from '../api'
import type { Size } from '../crop'
import { align, clampPlace, margins, rescale, scaleOf, type Edge, type Place } from '../place'

interface Props {
  source: { sha: string; width: number; height: number }
  /** The canvas: the form's Size. */
  canvas: Size
  place: Place
  onChange: (place: Place) => void
}

const EDGES: { id: Edge | 'centre'; label: string }[] = [
  { id: 'left', label: 'Left' },
  { id: 'centre', label: 'Centre' },
  { id: 'right', label: 'Right' },
  { id: 'top', label: 'Top' },
  { id: 'bottom', label: 'Bottom' },
]

/**
 * Outpaint: the source on its canvas. The margins are hatched like a sketch tile, since
 * that's where new pixels will be drawn. Drag the image to move it.
 */
export function PlaceEditor({ source, canvas, place, onChange }: Props) {
  const box = useRef<HTMLDivElement>(null)
  const drag = useRef<{ x: number; y: number; from: Place } | null>(null)
  const shape = { w: source.width, h: source.height }
  const pct = (v: number, of: number) => `${String((v / of) * 100)}%`

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId)
    drag.current = { x: e.clientX, y: e.clientY, from: place }
  }
  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    const d = drag.current
    const width = box.current?.clientWidth
    if (!d || !width) return
    const k = canvas.w / width // canvas px per screen px
    onChange(
      clampPlace(
        { ...d.from, x: d.from.x + (e.clientX - d.x) * k, y: d.from.y + (e.clientY - d.y) * k },
        canvas,
      ),
    )
  }
  const onKeyDown = (e: KeyboardEvent) => {
    const step = e.shiftKey ? 64 : 8
    const moves: Record<string, [number, number]> = {
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
      ArrowUp: [0, -step],
      ArrowDown: [0, step],
    }
    const move = moves[e.key]
    if (!move) return
    e.preventDefault()
    onChange(clampPlace({ ...place, x: place.x + move[0], y: place.y + move[1] }, canvas))
  }

  return (
    <div className="place">
      <div
        ref={box}
        className="place-canvas"
        style={{
          aspectRatio: `${String(canvas.w)} / ${String(canvas.h)}`,
          width: `min(100%, calc(50vh * ${String(canvas.w / canvas.h)}))`,
        }}
      >
        <div
          className="place-image"
          role="slider"
          tabIndex={0}
          aria-label="Image on the canvas. Drag, or use the arrow keys, to move it."
          aria-valuetext={margins(place, canvas)}
          style={{
            left: pct(place.x, canvas.w),
            top: pct(place.y, canvas.h),
            width: pct(place.w, canvas.w),
            height: pct(place.h, canvas.h),
          }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={() => {
            drag.current = null
          }}
          onPointerCancel={() => {
            drag.current = null
          }}
          onKeyDown={onKeyDown}
        >
          <img src={thumbUrl(source.sha)} alt="" draggable={false} />
        </div>
      </div>
      <p className="readout" aria-live="polite">
        {margins(place, canvas)}
      </p>
      <div className="setting">
        <label className="setting-label" htmlFor="place-scale">
          Image size
        </label>
        <div className="slider-control">
          <input
            id="place-scale"
            type="range"
            min={0.1}
            max={1}
            step={0.01}
            value={scaleOf(place, shape, canvas)}
            onChange={(e) => {
              onChange(rescale(place, shape, canvas, Number(e.target.value)))
            }}
          />
          <output htmlFor="place-scale">{Math.round(scaleOf(place, shape, canvas) * 100)}%</output>
        </div>
      </div>
      <div className="aspect-chips place-align" role="group" aria-label="Move the image">
        {EDGES.map((edge) => {
          const to = align(place, canvas, edge.id)
          return (
            <button
              key={edge.id}
              type="button"
              disabled={to.x === place.x && to.y === place.y}
              onClick={() => {
                onChange(to)
              }}
            >
              {edge.label}
            </button>
          )
        })}
      </div>
    </div>
  )
}
