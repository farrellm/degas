import type { TraceId } from '@/api/types'
import { SliderRow } from '@/components/SliderRow'

import { type ControlUnit, photoOf, TRACES } from './control'

export interface TracePickerProps {
  unit: ControlUnit
  /** A trace is being made on the GPU. */
  busy: boolean
  /** The GPU can trace now; if not, whether a session is at least on its way. */
  gpuReady: boolean
  sessionActive: boolean
  error: string | null
  /** What became of the area when the picture or its trace last changed. */
  note: string | null
  /** The edge trace's Detail (0..1), shown while the unit has an edge trace. */
  detail: number
  onDetail: (detail: number) => void
  /** Use the picture itself as the control image. */
  onAsIs: () => void
  onTrace: (id: TraceId) => void
}

/** How a unit's picture becomes its control image: as it is, or traced for depth, pose or edges. */
export function TracePicker({
  unit,
  busy,
  gpuReady,
  sessionActive,
  error,
  note,
  detail,
  onDetail,
  onAsIs,
  onTrace,
}: TracePickerProps) {
  const photo = photoOf(unit)
  const chosen = unit.trace?.id ?? 'as-is'

  return (
    <>
      <div className="control-traces" role="group" aria-label="Trace">
        <button
          type="button"
          aria-pressed={!!unit.image && chosen === 'as-is'}
          disabled={!unit.image || busy}
          onClick={onAsIs}
        >
          As is
        </button>
        {TRACES.map((t) => (
          <button
            key={t.id}
            type="button"
            aria-pressed={chosen === t.id}
            disabled={!photo || busy || !gpuReady}
            onClick={() => {
              onTrace(t.id)
            }}
          >
            {t.label}
          </button>
        ))}
      </div>
      {busy && (
        <p className="row-note" role="status">
          Tracing on the GPU…
        </p>
      )}
      {!gpuReady && photo && (
        <p className="row-note">
          {sessionActive
            ? 'Tracing works once the GPU session is ready.'
            : 'Tracing runs on the GPU. Start a session to trace depth, poses or edges.'}
        </p>
      )}
      {error && (
        <p className="row-warning" role="alert">
          {error}
        </p>
      )}
      {note && <p className="row-note">{note}</p>}
      {unit.trace?.id === 'canny' && (
        <SliderRow
          id="edge-detail"
          label="Detail"
          min={0}
          max={1}
          value={detail}
          format={(v) => `${String(Math.round(v * 100))}%`}
          onChange={onDetail}
        />
      )}
    </>
  )
}
