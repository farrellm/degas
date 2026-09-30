import { useQuery } from '@tanstack/react-query'
import {
  api,
  isPair,
  unref,
  type BlobInfo,
  type Fit,
  type LoraEntry,
  type Params,
  type SavedPrompt,
  type SeedMode,
  type Spec,
  type Variant,
} from './api'
import { variantFor } from './assets'
import { unitFromSpec, type ControlUnit } from './control'
import type { Place } from './place'

// The Create form's draft, kept across visits and reloads.
const DRAFT_KEY = 'degas.create.draft'

/** A source image in the form, with its pixel size (for the fit hint and the crop editor). */
export interface Source {
  sha: string
  width: number
  height: number
}

/** An inpaint mask, and the source image it was painted over. */
export interface MaskRef {
  sha: string
  source: string
}

/** What Create remembers for one family, so switching Image ⇄ Video loses nothing. */
export interface FamilyDraft {
  model: string
  mode?: string
  loras: LoraEntry[]
  params: Params
  source?: Source | null
  fit?: Fit
  /** `sha256:…` of the clip this draft continues. */
  extends?: string | null
  mask?: MaskRef | null
  /** Outpaint: where the source sits on the canvas. */
  place?: Place | null
  /** ControlNet units (SDXL). */
  control?: ControlUnit[]
  /** Edit (and Qwen's inpaint): the images after the source, in the order the model reads them. */
  refs?: Source[]
}

export interface Draft {
  family: string
  families: Record<string, Partial<FamilyDraft>>
  batchCount: number
  /** How a batch's seeds are chosen. */
  seedMode: SeedMode
  /** Families by when Create last had them open, most recent first. */
  recent: string[]
}

type Stored = Partial<Draft> & Partial<FamilyDraft>

export function loadDraft(): Draft {
  let raw: Stored = {}
  try {
    raw = JSON.parse(localStorage.getItem(DRAFT_KEY) ?? '{}') as Stored
  } catch {
    // unreadable: start over
  }
  const family = raw.family ?? 'sdxl'
  const families = raw.families ?? {}
  // Before Phase 4 the draft held one family's settings at the top level.
  if (!raw.families && (raw.model !== undefined || raw.params || raw.loras)) {
    families[family] = { model: raw.model, loras: raw.loras, params: raw.params }
  }
  return {
    family,
    families,
    batchCount: raw.batchCount ?? 1,
    seedMode: raw.seedMode ?? 'random',
    recent: raw.recent ?? [family],
  }
}

function store(draft: Draft) {
  const recent = [draft.family, ...draft.recent.filter((f) => f !== draft.family)]
  try {
    localStorage.setItem(DRAFT_KEY, JSON.stringify({ ...draft, recent }))
  } catch {
    // storage unavailable (private mode): the draft just isn't kept
  }
}

/** Save the form for `family`, which becomes the family Create opens with. */
export function saveFamilyDraft(
  family: string,
  fd: FamilyDraft,
  batchCount: number,
  seedMode: SeedMode,
) {
  const draft = loadDraft()
  store({ ...draft, family, families: { ...draft.families, [family]: fd }, batchCount, seedMode })
}

function update(family: string, change: (fd: Partial<FamilyDraft>) => Partial<FamilyDraft>) {
  const draft = loadDraft()
  const fd = change(draft.families[family] ?? {})
  store({ ...draft, family, families: { ...draft.families, [family]: fd } })
}

/** What Create takes with it to another family. */
export interface Carry {
  prompt: string
  model?: string
  mode?: string
  /** Kept only if the other family's draft has no source (or references) of its own. */
  source?: Source | null
  refs?: Source[]
}

/**
 * Switch Create to another family, carrying the prompt (and source) over if that family has
 * none yet, and choosing `model` and `mode` if given.
 */
export function switchFamily(family: string, carry: Carry) {
  update(family, (fd) => {
    const next = {
      ...fd,
      ...(carry.model && { model: carry.model }),
      ...(carry.mode && { mode: carry.mode }),
    }
    if (carry.source && !next.source) {
      Object.assign(next, { source: carry.source, extends: null, mask: null, place: null })
    }
    if (carry.refs?.length && !next.refs?.length) next.refs = carry.refs
    return String(next.params?.prompt ?? '').trim() || !carry.prompt
      ? next
      : { ...next, params: { ...next.params, prompt: carry.prompt } }
  })
}

/** The source a spec was made from, at the size the server fitted it to. */
function specSource(spec: Spec): Source | null {
  const source = spec.inputs?.source
  if (!source) return null
  const place = spec.inputs?.place
  return {
    sha: unref(source),
    width: place ? place.w : Number(spec.params.width),
    height: place ? place.h : Number(spec.params.height),
  }
}

/**
 * Replace the family's draft with a spec's settings (its model, mode, LoRAs, source and
 * parameters), pinned to `seed`. `source` overrides the spec's (an extension's last frame).
 */
export function draftFromSpec(spec: Spec, seed: number | null, source?: BlobInfo) {
  const loras = (spec.loras ?? []).map((l) =>
    isPair(l)
      ? {
          ...(l.high && { high: { path: l.high.path, weight: l.high.weight } }),
          ...(l.low && { low: { path: l.low.path, weight: l.low.weight } }),
        }
      : { path: l.path, weight: l.weight },
  )
  const draft = loadDraft()
  store({
    ...draft,
    family: spec.family,
    families: {
      ...draft.families,
      [spec.family]: {
        model: spec.model.path,
        mode: spec.mode,
        loras,
        params: { ...spec.params, ...(seed === null ? {} : { seed }) },
        source: source
          ? { sha: source.sha256, width: source.width ?? 0, height: source.height ?? 0 }
          : specSource(spec),
        extends: spec.inputs?.extends ?? null,
        mask:
          spec.inputs?.mask && spec.inputs.source && !source
            ? { sha: unref(spec.inputs.mask), source: unref(spec.inputs.source) }
            : null,
        place: spec.inputs?.place ?? null,
        control: (spec.control ?? []).map((c) =>
          unitFromSpec(c, { w: Number(spec.params.width), h: Number(spec.params.height) }),
        ),
        // The spec doesn't record the references' sizes; the Images row leaves them out.
        refs: (spec.inputs?.refs ?? []).map((r) => ({ sha: unref(r), width: 0, height: 0 })),
      },
    },
    batchCount: 1,
  })
}

/** Put a saved prompt into the current family's draft, keeping the rest of the settings. */
export function draftWithPrompt(p: SavedPrompt) {
  const draft = loadDraft()
  const family = draft.family
  update(family, (fd) => ({
    ...fd,
    params: { ...fd.params, prompt: p.prompt, negative_prompt: p.negative_prompt },
  }))
}

/** Start from an image: `family`'s draft switches to `mode` with this source. */
export function draftWithSource(family: string, mode: string, source: Source) {
  update(family, (fd) => ({ ...fd, mode, source, extends: null, mask: null, place: null }))
}

/**
 * Where "Use as source" sends an image: the family Create has open (image-to-image for
 * pictures, image-to-video for clips), staying in its current image mode if it has one.
 * Families that can't start from an image are skipped.
 */
export function useSourceTarget(): { family: string; mode: string } | null {
  const families = useQuery({ queryKey: ['families'], queryFn: api.families })
  const draft = loadDraft()
  const list = families.data ?? []
  const ordered = [
    ...list.filter((f) => f.id === draft.family),
    ...list.filter((f) => f.id !== draft.family),
  ]
  for (const f of ordered) {
    const modes = [...new Set(f.variants.flatMap((v) => v.modes))].filter((m) =>
      SOURCE_MODES.has(m),
    )
    const current = draft.families[f.id]?.mode
    const mode = current && modes.includes(current) ? current : modes[0]
    if (mode) return { family: f.id, mode }
  }
  return null
}

/**
 * The model variant Create has open, the way Create resolves it: the chosen model's, else
 * the first that does the chosen mode, else the family's first. Undefined until families load.
 */
export function useDraftVariant(): Variant | undefined {
  const families = useQuery({ queryKey: ['families'], queryFn: api.families })
  const draft = loadDraft()
  const family = families.data?.find((f) => f.id === draft.family) ?? families.data?.[0]
  if (!family) return undefined
  const { model, mode } = draft.families[family.id] ?? {}
  return (
    (model ? variantFor(model, family) : undefined) ??
    family.variants.find((v) => !!mode && v.modes.includes(mode)) ??
    family.variants[0]
  )
}

export const SOURCE_MODES = new Set(['i2i', 'i2v', 'edit', 'inpaint', 'outpaint'])
