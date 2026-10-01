// What Create is set to, worked out from what was chosen and what the server and Drive offer.

import type { Asset, Family, ImagePromptOptions, Variant } from '@/api/types'
import { variantFor } from '@/lib/assets'

import { SDXL_OPTIONS } from './image-prompt/imagePrompt'
import { MODE_ORDER, SOURCE_MODES } from './modes'

/**
 * The variant a family's form is on: the chosen model's, else the first that does the
 * chosen mode, else the family's first.
 */
export function variantOf(
  family: Family,
  model: string | undefined,
  mode: string | undefined,
): Variant | undefined {
  return (
    (model ? variantFor(model, family) : undefined) ??
    family.variants.find((v) => !!mode && v.modes.includes(mode)) ??
    family.variants[0]
  )
}

/** The first model in `assets` that is `family`'s and can do `mode`. */
export function modelForMode(
  assets: Asset[] | undefined,
  family: Family,
  mode: string,
): Asset | undefined {
  return assets?.find(
    (a) =>
      a.family === family.id &&
      a.kind === 'model' &&
      !!variantFor(a.path, family)?.modes.includes(mode),
  )
}

export interface Selection {
  family: Family | undefined
  /** The families making the same media, this one included. */
  siblings: Family[]
  /**
   * Every mode that a family making the same media offers. Choosing one this family can't
   * do switches to a family that can.
   */
  allModes: string[]
  mode: string | undefined
  /** This family's models in Drive (undefined until the index loads). */
  models: Asset[] | undefined
  /**
   * What the model picker offers: every model of the same media that can do the mode.
   * Choosing another family's model switches to it.
   */
  pickable: Asset[] | undefined
  /** The model's path, or '' when there is none to use. */
  model: string
  /** The model isn't in Drive (any more). */
  modelMissing: boolean
  variant: Variant | undefined
  needsSource: boolean
  /** How many images an edit reads after its source. */
  maxRefs: number
  takesRefs: boolean
  loras: Asset[] | undefined
  controlnets: Asset[] | undefined
  adapters: Asset[] | undefined
  withControl: boolean
  withPrompts: boolean
  promptOptions: ImagePromptOptions
}

export function resolveSelection({
  families,
  assets,
  familyId,
  chosenModel,
  chosenMode,
}: {
  families: Family[] | undefined
  assets: Asset[] | undefined
  familyId: string
  chosenModel: string
  chosenMode: string | undefined
}): Selection {
  const family = families?.find((f) => f.id === familyId) ?? families?.[0]
  const models = assets?.filter(
    (a) => !!family && a.family === family.id && a.kind === 'model' && !!variantFor(a.path, family),
  )
  const siblings = families?.filter((f) => f.media === family?.media) ?? []
  const siblingModes = new Set(siblings.flatMap((f) => f.variants.flatMap((v) => v.modes)))
  const allModes = [
    ...MODE_ORDER.filter((m) => siblingModes.has(m)),
    ...[...siblingModes].filter((m) => !MODE_ORDER.includes(m)),
  ]
  const familyModes = new Set(family?.variants.flatMap((v) => v.modes))
  // The mode picks the model: without one chosen, it's the chosen model's first.
  const firstModel = chosenModel || (models?.[0]?.path ?? '')
  const mode =
    chosenMode && familyModes.has(chosenMode)
      ? chosenMode
      : ((family && firstModel ? variantFor(firstModel, family) : undefined)?.modes[0] ??
        family?.variants[0]?.modes[0])
  const fits = (path: string, f: Family | undefined) =>
    !!f && !!mode && !!variantFor(path, f)?.modes.includes(mode)
  const pickable = assets?.filter((a) => {
    const f = siblings.find((s) => s.id === a.family)
    return a.kind === 'model' && fits(a.path, f)
  })
  // A model remixed from an older image may have left Drive: keep it, flagged, rather
  // than silently swapping in another.
  const model =
    chosenModel && fits(chosenModel, family)
      ? chosenModel
      : (models?.find((a) => fits(a.path, family))?.path ?? '')
  const modelMissing = !!model && !!models && !models.some((m) => m.path === model)
  const variant = family && variantOf(family, model, mode)
  // Edits can read more images after the source; so does Qwen's inpaint, which is an edit.
  const maxRefs = variant?.max_refs ?? 0
  const takesRefs =
    maxRefs > 0 && (mode === 'edit' || (mode === 'inpaint' && !!variant?.modes.includes('edit')))
  const ofKind = (kind: string) => assets?.filter((a) => a.family === family?.id && a.kind === kind)

  return {
    family,
    siblings,
    allModes,
    mode,
    models,
    pickable,
    model,
    modelMissing,
    variant,
    needsSource: !!mode && SOURCE_MODES.has(mode),
    maxRefs,
    takesRefs,
    loras: ofKind('lora'),
    controlnets: ofKind('controlnet'),
    adapters: ofKind('ip_adapter'),
    withControl: !!family?.supports_control,
    withPrompts: !!family?.supports_image_prompts,
    promptOptions: family?.image_prompt_options ?? SDXL_OPTIONS,
  }
}
