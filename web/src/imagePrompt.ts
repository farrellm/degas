import {
  unref,
  type AdapterKind,
  type Asset,
  type Fit,
  type ImagePromptSpec,
  type Purpose,
} from './api'
import { stepSpan } from './control'
import type { Source } from './draft'

/**
 * What to take from an image prompt's pictures, as its chips say. Each picks the blocks the
 * adapter acts in (its purpose) and the kind of model that suits it.
 */
export type Take = Purpose | 'face'

/** An image prompt (IP-Adapter) in the Create form. */
export interface PromptUnit {
  /** Stable within the form, for React keys. */
  key: string
  /** The IP-Adapter's Drive path. */
  model: string
  take: Take
  pictures: Source[]
  /** Where in the output it acts: a mask painted over `over`, a picture of the output's shape. */
  area: { sha: string; over: Source } | null
  weight: number
  start: number
  end: number
  fit: Fit
}

export const MAX_PROMPTS = 2
export const MAX_PICTURES = 4

interface TakeInfo {
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

export const purposeOf = (take: Take): Purpose => (take === 'face' ? 'all' : take)

/** What a model was trained to carry: from its sidecar, else its file name. */
export function adapterKind(asset: Asset | undefined, path = asset?.path ?? ''): AdapterKind {
  if (asset?.sidecar?.purpose) return asset.sidecar.purpose
  const name = (path.split('/').pop() ?? '').toLowerCase()
  if (name.includes('face')) return 'face'
  if (name.includes('composition')) return 'composition'
  return 'subject'
}

/** The kind of model each choice wants, best first. */
const WANTS: Record<Take, AdapterKind[]> = {
  all: ['subject'],
  style: ['subject'],
  layout: ['composition', 'subject'],
  style_layout: ['subject'],
  face: ['face'],
}

/** The model a choice picks: the first of the kind it wants, preferring a "plus" model,
 * which reads the picture's detail rather than a summary of it. */
export function modelFor(assets: Asset[], take: Take): Asset | undefined {
  for (const kind of WANTS[take]) {
    const found = assets.filter((a) => adapterKind(a) === kind)
    const plus = found.find((a) => /plus/i.test(a.path.split('/').pop() ?? ''))
    if (plus ?? found[0]) return plus ?? found[0]
  }
  return undefined
}

/** A warning when the model doesn't suit the choice, else null. */
export function mismatch(model: Asset | undefined, path: string, take: Take): string | null {
  const kind = adapterKind(model, path)
  if (take === 'face' && kind !== 'face')
    return 'This isn’t a face model; Face works best with one.'
  if (take !== 'face' && kind === 'face')
    return `This is a face model; it reads faces, not ${takeInfo(take).label.toLowerCase()}.`
  if (kind === 'composition' && take !== 'layout')
    return 'This is a composition model; it carries layout, not looks.'
  return null
}

export function newPrompt(): PromptUnit {
  return {
    key: Math.random().toString(36).slice(2),
    model: '',
    take: 'all',
    pictures: [],
    area: null,
    weight: takeInfo('all').weight,
    start: 0,
    end: 1,
    fit: 'crop',
  }
}

/** The unit's row in Create: what it takes, its weight, and its steps when not all. */
export function promptSummary(unit: PromptUnit, steps: number): string {
  const n = unit.pictures.length
  const parts = [
    n === 0 ? 'No picture yet' : takeInfo(unit.take).label,
    `weight ${unit.weight.toFixed(2)}`,
  ]
  if (n > 1) parts.push(`${String(n)} pictures`)
  const span = stepSpan(unit.start, unit.end, steps)
  if (span && (span.first !== 1 || span.last !== steps))
    parts.push(`steps ${String(span.first)}–${String(span.last)}`)
  else if (!span) parts.push('no steps')
  if (unit.area) parts.push('in an area')
  return parts.join(', ')
}

/** Whether a unit is complete enough to submit. */
export const promptReady = (unit: PromptUnit) =>
  !!unit.model && unit.pictures.length > 0 && unit.start < unit.end

/** Whether any picture isn't square, so the Fit choice matters. */
export const anyOblong = (unit: PromptUnit) =>
  unit.pictures.some((p) => p.height > 0 && Math.abs(p.width / p.height - 1) > 0.01)

export function promptSpec(unit: PromptUnit): ImagePromptSpec {
  return {
    adapter: { path: unit.model },
    images: unit.pictures.map((p) => `sha256:${p.sha}`),
    fit: unit.fit,
    purpose: purposeOf(unit.take),
    weight: unit.weight,
    start: unit.start,
    end: unit.end,
    ...(unit.area && { mask: `sha256:${unit.area.sha}` }),
  }
}

/**
 * A unit from a spec, for Remix. Its pictures were fitted to squares and their sizes aren't
 * recorded (0 × 0), and its area was fitted to the output, which it's shown over. An
 * everything-unit with a face model was chosen as Face.
 */
export function promptFromSpec(p: ImagePromptSpec, output: { w: number; h: number }): PromptUnit {
  const face = p.purpose === 'all' && adapterKind(undefined, p.adapter.path) === 'face'
  const mask = p.mask ? unref(p.mask) : null
  return {
    ...newPrompt(),
    model: p.adapter.path,
    take: face ? 'face' : p.purpose,
    pictures: p.images.map((sha) => ({ sha: unref(sha), width: 0, height: 0 })),
    area: mask ? { sha: mask, over: { sha: mask, width: output.w, height: output.h } } : null,
    weight: p.weight,
    start: p.start,
    end: p.end,
    fit: p.fit ?? 'crop',
  }
}
