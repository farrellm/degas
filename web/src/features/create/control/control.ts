import { unref, type Asset, type ControlSpec, type Fit, type Params, type TraceId } from './api'
import type { Source } from './draft'

/** A ControlNet unit in the Create form. */
export interface ControlUnit {
  /** Stable within the form, for React keys. */
  key: string
  /** The ControlNet's Drive path. */
  model: string
  /** The control image the ControlNet reads: a trace, or a picture used as it is. */
  image: Source | null
  /** How `image` was traced, and the picture it was traced from. */
  trace: { id: TraceId; params: Params; from: Source } | null
  /** The area the unit is limited to: a mask painted over `image`. */
  area: string | null
  scale: number
  start: number
  end: number
  fit: Fit
}

export const MAX_UNITS = 3

const TRACE_INFO: Record<TraceId, { label: string; image: string; model: string }> = {
  depth: { label: 'Depth', image: 'a depth map', model: 'a depth model' },
  pose: { label: 'Pose', image: 'a pose', model: 'a pose model' },
  canny: { label: 'Edges', image: 'an edge map', model: 'an edges model' },
}

export const TRACES = (['depth', 'pose', 'canny'] as const).map((id) => ({
  id,
  ...TRACE_INFO[id],
}))

export const traceInfo = (id: TraceId) => TRACE_INFO[id]

export function newUnit(): ControlUnit {
  return {
    key: Math.random().toString(36).slice(2),
    model: '',
    image: null,
    trace: null,
    area: null,
    scale: 0.7,
    start: 0,
    end: 1,
    fit: 'crop',
  }
}

/** A unit from a saved draft, which may predate fields added since. */
export const restoreUnit = (saved: Partial<ControlUnit>): ControlUnit => ({
  ...newUnit(),
  ...saved,
})

/**
 * The steps (1-based, inclusive) a unit guides out of `steps`. diffusers runs a unit on step
 * i (0-based) when i / n ≥ start and (i + 1) / n ≤ end; null when that's no step at all.
 */
export function stepSpan(
  start: number,
  end: number,
  steps: number,
): { first: number; last: number } | null {
  const first = Math.ceil(start * steps - 1e-9) + 1
  const last = Math.floor(end * steps + 1e-9)
  return steps > 0 && last >= first ? { first, last } : null
}

/** "Steps 1–24 of 30". */
export function stepsLabel(start: number, end: number, steps: number): string {
  const span = stepSpan(start, end, steps)
  if (!span) return 'No steps. Widen the range.'
  if (span.first === span.last) return `Step ${String(span.first)} of ${String(steps)}`
  return `Steps ${String(span.first)}–${String(span.last)} of ${String(steps)}`
}

/** The kind of control image a ControlNet reads: from its sidecar, else its file name. */
export function controlKind(asset: Asset | undefined, path = asset?.path ?? ''): TraceId | null {
  if (asset?.sidecar?.control) return asset.sidecar.control
  const name = (path.split('/').pop() ?? '').toLowerCase()
  if (name.includes('depth')) return 'depth'
  if (name.includes('pose')) return 'pose'
  if (/canny|edge/.test(name)) return 'canny'
  return null
}

/** The first ControlNet that reads this kind of trace. */
export function matchingModel(assets: Asset[], id: TraceId): Asset | undefined {
  return assets.find((a) => controlKind(a) === id)
}

/** A warning when the model and the trace look mismatched, else null. */
export function mismatch(model: Asset | undefined, path: string, trace: TraceId | null) {
  const kind = controlKind(model, path)
  if (!kind || !trace || kind === trace) return null
  return `This looks like ${traceInfo(kind).model}; the image is ${traceInfo(trace).image}.`
}

// Edges: one Detail value from 0 (only strong edges) to 1 (every edge). Canny's low
// threshold is half its high one.
export function edgeParams(detail: number): { low: number; high: number } {
  const high = Math.round(250 - 200 * detail)
  return { low: Math.round(high / 2), high }
}

export function edgeDetail(params: Params): number {
  const high = Number(params.high ?? 200)
  return Math.min(1, Math.max(0, Math.round(((250 - high) / 200) * 20) / 20))
}

export const DEFAULT_EDGES = edgeParams(0.25)

/** The unit's row in Create: what it reads, its weight, and its steps when not all. */
export function unitSummary(unit: ControlUnit, steps: number): string {
  const what = unit.trace ? traceInfo(unit.trace.id).label : unit.image ? 'Image' : 'No image yet'
  const parts = [what, `weight ${unit.scale.toFixed(2)}`]
  const span = stepSpan(unit.start, unit.end, steps)
  if (span && (span.first !== 1 || span.last !== steps))
    parts.push(`steps ${String(span.first)}–${String(span.last)}`)
  else if (!span) parts.push('no steps')
  if (unit.area) parts.push('in an area')
  return parts.join(', ')
}

/** Whether a unit is complete enough to submit. */
export const unitReady = (unit: ControlUnit) =>
  !!unit.model && !!unit.image && unit.start < unit.end

/** The picture shown under the control image when painting its area. */
export function underlay(unit: ControlUnit): string | null {
  const { image, trace } = unit
  if (!image) return null
  if (trace?.from.width === image.width && trace.from.height === image.height) return trace.from.sha
  return image.sha
}

export function unitSpec(unit: ControlUnit): ControlSpec {
  if (!unit.image) throw new Error('Choose a control image.')
  return {
    controlnet: { path: unit.model },
    image: `sha256:${unit.image.sha}`,
    fit: unit.fit,
    scale: unit.scale,
    start: unit.start,
    end: unit.end,
    ...(unit.area && { mask: `sha256:${unit.area}` }),
    ...(unit.trace && {
      preprocessor: {
        id: unit.trace.id,
        source: `sha256:${unit.trace.from.sha}`,
        params: unit.trace.params,
      },
    }),
  }
}

/**
 * A unit from a spec, for Remix. The spec's control image was fitted to the output size, so
 * that's its size; the size of the picture it was traced from isn't recorded (0 × 0).
 */
export function unitFromSpec(c: ControlSpec, size: { w: number; h: number }): ControlUnit {
  return {
    ...newUnit(),
    model: c.controlnet.path,
    image: { sha: unref(c.image), width: size.w, height: size.h },
    trace: c.preprocessor
      ? {
          id: c.preprocessor.id,
          params: c.preprocessor.params,
          from: { sha: unref(c.preprocessor.source), width: 0, height: 0 },
        }
      : null,
    area: c.mask ? unref(c.mask) : null,
    scale: c.scale,
    start: c.start,
    end: c.end,
  }
}
