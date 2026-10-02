import type {
  AdapterKind,
  Asset,
  Fit,
  ImagePromptOptions,
  ImagePromptSpec,
  Purpose,
} from '@/api/types'
import { ratioDiffers, type Size } from '@/lib/geometry'
import { ref, type Source, unref } from '@/lib/image'
import {
  adapterKind,
  detailInfo,
  summaryText,
  type Take,
  type TakeInfo,
  takeInfo,
  takeOf,
  TAKES,
} from '@/lib/imagePrompts'

/** An image prompt (IP-Adapter) in the Create form. */
export interface ImagePromptUnit {
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
  /** FaceID: how much of CLIP's reading of the face Plus v2 adds, and its LoRA's weight. */
  structure: number
  loraWeight: number
  /** Redux: how much its token grid is shrunk, so the prompt keeps a say (1 to 5). */
  downsample: number
}

export const MAX_PROMPTS = 2
export const MAX_PICTURES = 4

/** SDXL's options, for a family that doesn't say. */
export const SDXL_OPTIONS: ImagePromptOptions = {
  purposes: ['all', 'style', 'layout', 'style_layout'],
  areas: true,
  steps: true,
  faces: true,
  detail: false,
}

/** The chips a family's image prompts offer; none when there's only one choice. */
export function takesFor(options: ImagePromptOptions): TakeInfo[] {
  const takes = TAKES.filter((t) =>
    t.id === 'face' ? options.faces : options.purposes.includes(t.id),
  )
  return takes.length > 1 ? takes : []
}

/** FaceID finds the face itself and reads who it is, so the note says so. */
export const FACEID_NOTE =
  'Who the person is. The model finds the face itself, so the picture needn’t be cropped.'

// FaceID starts a little stronger than a CLIP face model (docs/ip-adapter.md §2.2).
const FACEID_WEIGHT = 0.8

export const purposeOf = (take: Take): Purpose => (take === 'face' ? 'all' : take)

/** The kind of model each choice wants, best first. */
const WANTS: Record<Take, AdapterKind[]> = {
  all: ['subject'],
  style: ['subject'],
  layout: ['composition', 'subject'],
  style_layout: ['subject'],
  face: ['faceid', 'face'],
}

/** Whether a model reads InsightFace identities; by file name, as the server decides. */
export const isFaceid = (path: string) =>
  (path.split('/').pop() ?? '').toLowerCase().includes('faceid')

// Plus models read the picture's detail rather than a summary of it; v2 is FaceID's newest.
const rank = (a: Asset) => {
  const name = (a.path.split('/').pop() ?? '').toLowerCase()
  return name.includes('plusv2') ? 2 : name.includes('plus') ? 1 : 0
}

/** A choice's starting weight, which is higher for a FaceID model. */
export const weightFor = (take: Take, model: string) =>
  take === 'face' && isFaceid(model) ? FACEID_WEIGHT : takeInfo(take).weight

/** The model a choice picks: the best of the first kind it wants that's in Drive. */
export function modelFor(assets: Asset[], take: Take): Asset | undefined {
  for (const kind of WANTS[take]) {
    const found = assets.filter((a) => adapterKind(a) === kind).sort((a, b) => rank(b) - rank(a))
    if (found[0]) return found[0]
  }
  return undefined
}

/** A warning when the model doesn't suit the choice, else null. */
export function mismatch(model: Asset | undefined, path: string, take: Take): string | null {
  const kind = adapterKind(model, path)
  const face = kind === 'face' || kind === 'faceid'
  if (take === 'face' && !face) return 'This isn’t a face model; Face works best with one.'
  if (take !== 'face' && face)
    return `This is a face model; it reads faces, not ${takeInfo(take).label.toLowerCase()}.`
  if (kind === 'composition' && take !== 'layout')
    return 'This is a composition model; it carries layout, not looks.'
  return null
}

/** A new unit; Redux (a family with `detail`) starts at full weight. */
export function newPrompt(options: ImagePromptOptions = SDXL_OPTIONS): ImagePromptUnit {
  return {
    key: Math.random().toString(36).slice(2),
    model: '',
    take: 'all',
    pictures: [],
    area: null,
    weight: options.detail ? 1 : takeInfo('all').weight,
    start: 0,
    end: 1,
    fit: 'crop',
    structure: 1,
    loraWeight: 0.6,
    downsample: 3,
  }
}

/** A unit from a saved draft, which may predate fields added since (FaceID's, Redux's). */
export const restorePrompt = (saved: Partial<ImagePromptUnit>): ImagePromptUnit => ({
  ...newPrompt(),
  ...saved,
})

/** The unit's row in Create: what it takes, its weight, and its steps when not all. */
export function promptSummary(
  unit: ImagePromptUnit,
  steps: number,
  options: ImagePromptOptions = SDXL_OPTIONS,
): string {
  const n = unit.pictures.length
  const what = options.detail ? detailInfo(unit.downsample).label : takeInfo(unit.take).label
  return summaryText({
    what: n === 0 ? 'No picture yet' : what,
    weight: unit.weight,
    start: unit.start,
    end: unit.end,
    steps,
    area: !!unit.area,
    pictures: n,
  })
}

/** Whether a unit is complete enough to submit. */
export const promptReady = (unit: ImagePromptUnit) =>
  !!unit.model && unit.pictures.length > 0 && unit.start < unit.end

/** Whether any picture isn't square, so the Fit choice matters. */
export const anyOblong = (unit: ImagePromptUnit) =>
  unit.pictures.some((p) => p.height > 0 && ratioDiffers(p.width / p.height, 1))

export function promptSpec(
  unit: ImagePromptUnit,
  options: ImagePromptOptions = SDXL_OPTIONS,
): ImagePromptSpec {
  return {
    adapter: { path: unit.model },
    images: unit.pictures.map((p) => ref(p.sha)),
    fit: unit.fit,
    purpose: purposeOf(unit.take),
    weight: unit.weight,
    start: unit.start,
    end: unit.end,
    ...(unit.area && { mask: ref(unit.area.sha) }),
    ...(isFaceid(unit.model) && { structure: unit.structure, lora_weight: unit.loraWeight }),
    ...(options.detail && { downsample: unit.downsample }),
  }
}

/**
 * A unit from a spec, for Remix. Its pictures were fitted to squares and their sizes aren't
 * recorded (0 × 0), and its area was fitted to the output, which it's shown over. An
 * everything-unit with a face model was chosen as Face.
 */
export function promptFromSpec(p: ImagePromptSpec, output: Size): ImagePromptUnit {
  const mask = p.mask ? unref(p.mask) : null
  return {
    ...newPrompt(),
    model: p.adapter.path,
    take: takeOf(p),
    pictures: p.images.map((sha) => ({ sha: unref(sha), width: 0, height: 0 })),
    area: mask ? { sha: mask, over: { sha: mask, width: output.w, height: output.h } } : null,
    weight: p.weight,
    start: p.start,
    end: p.end,
    fit: p.fit ?? 'crop',
    structure: p.structure ?? 1,
    loraWeight: p.lora_weight ?? 0.6,
    downsample: p.downsample ?? 3,
  }
}
