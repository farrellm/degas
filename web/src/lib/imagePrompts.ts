import type { AdapterKind, Asset, ImagePromptSpec, Purpose } from '@/api/types'

import { stepSpan } from './steps'

// The words for an image prompt (IP-Adapter, Redux), shared by Create's sheet and the
// viewer's wall label.

/**
 * What to take from an image prompt's pictures, as its chips say. Each picks the blocks the
 * adapter acts in (its purpose) and the kind of model that suits it.
 */
export type Take = Purpose | 'face'

export interface TakeInfo {
  id: Take
  label: string
  /** What the choice does, under the chips. */
  note: string
  weight: number
}

// Weights from docs/ip-adapter.md §2: the whole UNet follows the pictures closely, so it
// starts lower; one or two blocks start at full strength.
const EVERYTHING: TakeInfo = {
  id: 'all',
  label: 'Everything',
  note: 'Its subject, colours and style; the prompt still counts.',
  weight: 0.6,
}

export const TAKES: TakeInfo[] = [
  EVERYTHING,
  {
    id: 'style',
    label: 'Style',
    note: 'Its colour, texture and strokes; the prompt decides what’s in the picture.',
    weight: 1,
  },
  { id: 'layout', label: 'Layout', note: 'Where things are, not how they look.', weight: 1 },
  {
    id: 'style_layout',
    label: 'Style and layout',
    note: 'How it looks and where things are; the prompt says what they are.',
    weight: 1,
  },
  { id: 'face', label: 'Face', note: 'A likeness of the face. Crop close to it.', weight: 0.6 },
]

export const takeInfo = (id: Take): TakeInfo => TAKES.find((t) => t.id === id) ?? EVERYTHING

interface Detail {
  downsample: number
  label: string
  note: string
}

/** Redux: how closely to follow the picture. Its 27 × 27 tokens are shrunk by `downsample`
 * (to 27, 13, 9 or 5 a side), as ComfyUI's Redux Advanced does. */
export const DETAILS: Detail[] = [
  {
    downsample: 1,
    label: 'Closely',
    note: 'Close variations of the picture; the prompt has little say.',
  },
  { downsample: 2, label: 'Somewhat', note: 'Near the picture, with some room for the prompt.' },
  {
    downsample: 3,
    label: 'Loosely',
    note: 'Its look and subject, leaving the prompt room to change them.',
  },
  { downsample: 5, label: 'Just the gist', note: 'Its overall colour and feel; the prompt leads.' },
]

export const detailInfo = (downsample: number): Detail =>
  DETAILS.find((d) => d.downsample === downsample) ?? {
    downsample,
    label: `Shrunk ${String(downsample)}×`,
    note: '',
  }

/** What a model was trained to carry: from its sidecar, else its file name. */
export function adapterKind(asset: Asset | undefined, path = asset?.path ?? ''): AdapterKind {
  if (asset?.sidecar?.purpose) return asset.sidecar.purpose
  const name = (path.split('/').pop() ?? '').toLowerCase()
  if (name.includes('faceid')) return 'faceid'
  if (name.includes('face')) return 'face'
  if (name.includes('composition')) return 'composition'
  return 'subject'
}

/** The choice a spec's unit was made with: an everything-unit with a face model was Face. */
export function takeOf(p: ImagePromptSpec): Take {
  const kind = adapterKind(undefined, p.adapter.path)
  return p.purpose === 'all' && (kind === 'face' || kind === 'faceid') ? 'face' : p.purpose
}

/** "Style, weight 1.00, steps 1–24, in an area": what a unit takes, how much, when and where. */
export function summaryText(unit: {
  what: string
  weight: number
  start: number
  end: number
  steps: number
  area: boolean
  /** Counted when there are several and they aren't shown beside the line. */
  pictures?: number
}): string {
  const parts = [unit.what, `weight ${unit.weight.toFixed(2)}`]
  if (unit.pictures !== undefined && unit.pictures > 1)
    parts.push(`${String(unit.pictures)} pictures`)
  const span = stepSpan(unit.start, unit.end, unit.steps)
  if (span && (span.first !== 1 || span.last !== unit.steps))
    parts.push(`steps ${String(span.first)}–${String(span.last)}`)
  else if (!span) parts.push('no steps')
  if (unit.area) parts.push('in an area')
  return parts.join(', ')
}

/** A finished job's unit, for the wall label. Only Redux records `downsample`. */
export function specSummary(p: ImagePromptSpec, steps: number): string {
  return summaryText({
    what: p.downsample == null ? takeInfo(takeOf(p)).label : detailInfo(p.downsample).label,
    weight: p.weight,
    start: p.start,
    end: p.end,
    steps,
    area: !!p.mask,
  })
}
