import type { BlobInfo, Fit, LoraEntry, Params, SavedPrompt, SeedMode, Spec } from '@/api/types'
import type { Place } from '@/features/editors/place/place'
import { type MaskRef, type Source, unref } from '@/lib/image'
import { isPair } from '@/lib/loras'
import { readStored, writeStored } from '@/lib/storage'

import { type ControlUnit, restoreUnit, unitFromSpec } from './control/control'
import { type ImagePromptUnit, promptFromSpec, restorePrompt } from './image-prompt/imagePrompt'

// The Create form's draft, kept across visits and reloads.
const DRAFT_KEY = 'degas.create.draft'

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
  /** First and last frame: the last frame. */
  end?: Source | null
  /** ControlNet units (SDXL). */
  control?: ControlUnit[]
  /** Edit (and Qwen's inpaint): the images after the source, in the order the model reads them. */
  refs?: Source[]
  /** Image prompts (SDXL). */
  prompts?: ImagePromptUnit[]
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
    raw = JSON.parse(readStored(DRAFT_KEY) ?? '{}') as Stored
  } catch {
    // unreadable: start over
  }
  const family = raw.family ?? 'sdxl'
  const families = raw.families ?? {}
  // Before Phase 4 the draft held one family's settings at the top level.
  if (!raw.families && (raw.model !== undefined || raw.params || raw.loras)) {
    families[family] = { model: raw.model, loras: raw.loras, params: raw.params }
  }
  // Fill in what a unit saved before a field was added lacks, so the editors can rely on it.
  for (const fd of Object.values(families)) {
    if (fd.control) fd.control = fd.control.map(restoreUnit)
    if (fd.prompts) fd.prompts = fd.prompts.map(restorePrompt)
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
  // Where storage is unavailable (private mode) the draft just isn't kept.
  writeStored(DRAFT_KEY, JSON.stringify({ ...draft, recent }))
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
  /** The negative prompt, carried with the prompt when `keep` is set. */
  negative?: string
  /**
   * Replace the other family's prompt and images rather than filling them only when empty:
   * changing the model keeps what's been typed and chosen.
   */
  keep?: boolean
  model?: string
  mode?: string
  /** Kept only if the other family's draft has no source (or references) of its own. */
  source?: Source | null
  refs?: Source[]
}

/**
 * Switch Create to another family, carrying the prompt and images over if that family has
 * none yet (always, with `keep`), and choosing `model` and `mode` if given.
 */
export function switchFamily(family: string, carry: Carry) {
  update(family, (fd) => {
    const next = {
      ...fd,
      ...(carry.model && { model: carry.model }),
      ...(carry.mode && { mode: carry.mode }),
    }
    const source = carry.keep ? (carry.source ?? null) : (next.source ?? carry.source)
    // What was drawn over another source doesn't belong to this one.
    if (source?.sha !== next.source?.sha) {
      Object.assign(next, { source, extends: null, mask: null, place: null })
    }
    if (carry.keep ? carry.refs : carry.refs?.length && !next.refs?.length) next.refs = carry.refs
    if (carry.keep) {
      return {
        ...next,
        params: {
          ...next.params,
          prompt: carry.prompt,
          ...(carry.negative !== undefined && { negative_prompt: carry.negative }),
        },
      }
    }
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
        // Fitted to the output size, like the source; an extension starts without one.
        end:
          spec.inputs?.end && !source
            ? {
                sha: unref(spec.inputs.end),
                width: Number(spec.params.width),
                height: Number(spec.params.height),
              }
            : null,
        control: (spec.control ?? []).map((c) =>
          unitFromSpec(c, { w: Number(spec.params.width), h: Number(spec.params.height) }),
        ),
        // The spec doesn't record the references' sizes; the Images row leaves them out.
        refs: (spec.inputs?.refs ?? []).map((r) => ({ sha: unref(r), width: 0, height: 0 })),
        prompts: (spec.image_prompts ?? []).map((p) =>
          promptFromSpec(p, { w: Number(spec.params.width), h: Number(spec.params.height) }),
        ),
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
