import type { LoraRef, Params, Spec } from './api'

// The Create form's draft, kept across visits and reloads.
const DRAFT_KEY = 'degas.create.draft'

export interface Draft {
  family: string
  model: string
  loras: LoraRef[]
  params: Params
  batchCount: number
}

export function loadDraft(): Partial<Draft> {
  try {
    return JSON.parse(localStorage.getItem(DRAFT_KEY) ?? '{}') as Partial<Draft>
  } catch {
    return {}
  }
}

export function saveDraft(draft: Draft) {
  try {
    localStorage.setItem(DRAFT_KEY, JSON.stringify(draft))
  } catch {
    // storage unavailable (private mode): the draft just isn't kept
  }
}

/** Replace the draft with a result's settings, pinned to the seed it used. */
export function draftFromSpec(spec: Spec, seed: number | null) {
  saveDraft({
    family: spec.family,
    model: spec.model.path,
    loras: (spec.loras ?? []).map(({ path, weight }) => ({ path, weight })),
    params: { ...spec.params, ...(seed === null ? {} : { seed }) },
    batchCount: 1,
  })
}
