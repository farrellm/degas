// The job a filled-in Create form asks for.

import type { Fit, ImagePromptOptions, LoraEntry, Params, SeedMode, Spec } from '@/api/types'
import type { Place } from '@/features/editors/place/place'
import { type MaskRef, ref, type Source } from '@/lib/image'

import { type ControlUnit, unitSpec } from './control/control'
import { type ImagePromptUnit, promptSpec } from './image-prompt/imagePrompt'

export interface FormState {
  family: string
  variant: string
  mode: string
  model: string
  loras: LoraEntry[]
  params: Params
  /** The source and what goes with it, when the mode starts from an image. */
  inputs: {
    source: Source
    fit: Fit
    extends: string | null
    mask: MaskRef | null
    place: Place | null
    /** The images after the source, when the mode reads any. */
    refs: Source[]
  } | null
  /** ControlNet units and image prompts, when the family takes them. */
  control: ControlUnit[]
  prompts: ImagePromptUnit[]
  promptOptions: ImagePromptOptions
  batchCount: number
  seedMode: SeedMode
}

/** The spec to submit, and how its batch gets its seeds. */
export function buildJob(form: FormState): { spec: Spec; seedMode: SeedMode } {
  const { mode, params, inputs } = form
  // One image uses the seed as set; a batch either counts up from it or ignores it.
  const randomSeeds = form.batchCount > 1 && form.seedMode === 'random' && 'seed' in params
  return {
    spec: {
      family: form.family,
      variant: form.variant,
      mode,
      model: { path: form.model },
      loras: form.loras,
      params: randomSeeds ? { ...params, seed: -1 } : params,
      ...(inputs && {
        inputs: {
          source: ref(inputs.source.sha),
          fit: inputs.fit,
          ...(inputs.extends && { extends: inputs.extends }),
          ...(mode === 'inpaint' && inputs.mask && { mask: ref(inputs.mask.sha) }),
          ...(mode === 'outpaint' && inputs.place && { place: inputs.place }),
          ...(inputs.refs.length > 0 && { refs: inputs.refs.map((r) => ref(r.sha)) }),
        },
      }),
      ...(form.control.length > 0 && { control: form.control.map(unitSpec) }),
      ...(form.prompts.length > 0 && {
        image_prompts: form.prompts.map((p) => promptSpec(p, form.promptOptions)),
      }),
    },
    seedMode: randomSeeds ? 'random' : 'increment',
  }
}
