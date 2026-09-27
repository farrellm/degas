import type { LoraRef, Params, SavedPrompt, Spec } from './api'

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

/** Replace the draft with an image's settings (its model, LoRAs and parameters), pinned to its seed. */
export function draftFromSpec(spec: Spec, seed: number | null) {
  saveDraft({
    family: spec.family,
    model: spec.model.path,
    loras: (spec.loras ?? []).map(({ path, weight }) => ({ path, weight })),
    params: { ...spec.params, ...(seed === null ? {} : { seed }) },
    batchCount: 1,
  })
}

/** Put a saved prompt into the draft, keeping the rest of the settings. */
export function draftWithPrompt(p: SavedPrompt) {
  const draft = loadDraft()
  saveDraft({
    family: draft.family ?? p.family ?? 'sdxl',
    model: draft.model ?? '',
    loras: draft.loras ?? [],
    params: { ...draft.params, prompt: p.prompt, negative_prompt: p.negative_prompt },
    batchCount: draft.batchCount ?? 1,
  })
}
