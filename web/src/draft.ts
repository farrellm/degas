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
  type Spec,
} from './api'

// The Create form's draft, kept across visits and reloads.
const DRAFT_KEY = 'degas.create.draft'

/** A source image in the form, with its pixel size (for the fit hint and the crop editor). */
export interface Source {
  sha: string
  width: number
  height: number
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
}

export interface Draft {
  family: string
  families: Record<string, Partial<FamilyDraft>>
  batchCount: number
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
  return { family, families, batchCount: raw.batchCount ?? 1 }
}

function store(draft: Draft) {
  try {
    localStorage.setItem(DRAFT_KEY, JSON.stringify(draft))
  } catch {
    // storage unavailable (private mode): the draft just isn't kept
  }
}

/** Save the form for `family`, which becomes the family Create opens with. */
export function saveFamilyDraft(family: string, fd: FamilyDraft, batchCount: number) {
  const draft = loadDraft()
  store({ family, families: { ...draft.families, [family]: fd }, batchCount })
}

function update(family: string, change: (fd: Partial<FamilyDraft>) => Partial<FamilyDraft>) {
  const draft = loadDraft()
  const fd = change(draft.families[family] ?? {})
  store({ ...draft, family, families: { ...draft.families, [family]: fd } })
}

/** Switch Create to another family, carrying the prompt over if that family has none yet. */
export function switchFamily(family: string, prompt: string) {
  update(family, (fd) =>
    String(fd.params?.prompt ?? '').trim() || !prompt
      ? fd
      : { ...fd, params: { ...fd.params, prompt } },
  )
}

/** The source a spec was made from, at the size the server fitted it to. */
function specSource(spec: Spec): Source | null {
  const source = spec.inputs?.source
  if (!source) return null
  return {
    sha: unref(source),
    width: Number(spec.params.width),
    height: Number(spec.params.height),
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

/** Start a clip from an image: `family`'s draft switches to its image mode with this source. */
export function draftWithSource(family: string, mode: string, source: Source) {
  update(family, (fd) => ({ ...fd, mode, source, extends: null }))
}

/** The first family and mode that start from a source image (Wan 2.2 i2v in Phase 4). */
export function useSourceTarget(): { family: string; mode: string } | null {
  const families = useQuery({ queryKey: ['families'], queryFn: api.families })
  for (const f of families.data ?? []) {
    for (const v of f.variants) {
      const mode = v.modes.find((m) => SOURCE_MODES.has(m))
      if (mode) return { family: f.id, mode }
    }
  }
  return null
}

export const SOURCE_MODES = new Set(['i2i', 'i2v', 'inpaint', 'outpaint'])
