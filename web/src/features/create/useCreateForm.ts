import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { queries } from '@/api/queries'
import type { Asset, BlobInfo, Family, Params, SavedPrompt, SeedMode } from '@/api/types'
import { defaultPlace, validPlace } from '@/features/editors/place/place'
import { useAssets } from '@/hooks/useAssets'
import { variantFor } from '@/lib/assets'
import { ratioDiffers, type Size } from '@/lib/geometry'
import { belowGpu } from '@/lib/gpu'
import { initialParams } from '@/lib/schema'
import { isActive } from '@/lib/session'

import { unitReady } from './control/control'
import { type Carry, loadDraft, switchFamily } from './draft'
import { promptReady } from './image-prompt/imagePrompt'
import { insertWord } from './prompt'
import { familyForMode, resolveSelection } from './selection'
import type { FormState } from './spec'
import { useLastFrame } from './useLastFrame'
import { useLoras } from './useLoras'
import { useSaveDraft } from './useSaveDraft'
import { useSourceInput } from './useSourceInput'
import { useUnitList } from './useUnitList'

/**
 * One family's Create form: what's been chosen and typed (kept as that family's draft), what
 * follows from it, and the job it asks for (`useSubmitJob` queues it). `onFamily` switches
 * Create to another family, whose own form then takes over.
 */
export function useCreateForm(familyId: string, onFamily: (id: string) => void) {
  const [draft] = useState(() => {
    const d = loadDraft()
    return { ...d.families[familyId], batchCount: d.batchCount, seedMode: d.seedMode }
  })
  const [chosenModel, setModel] = useState(draft.model ?? '')
  const [chosenMode, setMode] = useState(draft.mode)
  const [editedParams, setParams] = useState<Params | null>(null)
  const lora = useLoras(draft.loras)
  const { loras } = lora
  const input = useSourceInput(draft)
  const { source, mask, maskFits } = input
  const [refs, setRefs] = useState(draft.refs ?? [])
  const lastFrame = useLastFrame(draft.end)
  const end = lastFrame.image
  const control = useUnitList(draft.control ?? [])
  const prompts = useUnitList(draft.prompts ?? [])
  const [batchCount, setBatchCount] = useState(draft.batchCount)
  const [seedMode, setSeedMode] = useState<SeedMode>(draft.seedMode)

  const families = useQuery(queries.families())
  const session = useQuery(queries.session())
  const assets = useAssets()
  const selection = resolveSelection({
    families: families.data,
    assets: assets.data,
    familyId,
    chosenModel,
    chosenMode,
  })
  const { family, siblings, mode, model, modelMissing, variant } = selection
  const { needsSource, takesRefs, withControl, withPrompts, promptOptions } = selection

  const schema = useQuery(queries.schema(family?.id, variant?.id, mode))
  // The form always fits the current variant's schema: defaults, then what was typed.
  const params = schema.data ? initialParams(schema.data, editedParams ?? draft.params) : null

  useSaveDraft(
    familyId,
    params && mode
      ? {
          model,
          mode,
          loras,
          params,
          source,
          fit: input.fit,
          extends: input.extendsClip,
          mask,
          place: input.place,
          end,
          control: control.units,
          refs,
          prompts: prompts.units,
        }
      : null,
    batchCount,
    seedMode,
  )

  const canvas = params ? { w: Number(params.width), h: Number(params.height) } : null
  const place =
    mode === 'outpaint' && source && canvas
      ? validPlace(input.place, { w: source.width, h: source.height }, canvas)
        ? input.place
        : defaultPlace({ w: source.width, h: source.height }, canvas)
      : null

  if (families.isPending || schema.isPending || !params)
    return { ready: false, error: null } as const
  if (families.error || schema.error) {
    return { ready: false, error: families.error ?? schema.error } as const
  }

  /** The job the form asks for, as it stands. */
  const job: FormState | null =
    family && variant && mode
      ? {
          family: family.id,
          variant: variant.id,
          mode,
          model,
          loras,
          params,
          inputs:
            needsSource && source
              ? {
                  source,
                  fit: input.fit,
                  extends: input.extendsClip,
                  mask,
                  place,
                  end: mode === 'flf2v' ? end : null,
                  refs: takesRefs ? refs : [],
                }
              : null,
          control: withControl ? control.units : [],
          prompts: withPrompts ? prompts.units : [],
          promptOptions,
          batchCount,
          seedMode,
        }
      : null

  const target = { w: Number(params.width), h: Number(params.height) }
  const sessionGpu = isActive(session.data) ? session.data?.session?.gpu : undefined
  const misfit = !!source && ratioDiffers(source.width / source.height, target.w / target.h)
  const sourceUsable = !!source && !input.gone

  /** Hand Create over to another family's form, taking `carry` along. */
  const leave = (to: string, carry: Carry) => {
    switchFamily(to, carry)
    onFamily(to)
  }

  // Changing the model keeps the prompt and images, even over ones the other family had.
  const carry = (m: string, next?: Asset): Carry => ({
    prompt: String(params.prompt ?? ''),
    ...('negative_prompt' in params && { negative: String(params.negative_prompt ?? '') }),
    keep: true,
    model: next?.path,
    mode: m,
    source: input.gone ? null : source,
    refs,
  })

  const pickModel = (path: string) => {
    const next = family && variantFor(path, family)
    if (next && next.lora_format !== variant?.lora_format) lora.setLoras([])
    setModel(path)
  }

  return {
    ready: true as const,
    schema: schema.data,
    params,
    setParams,
    selection,
    /** Whether Drive's index has loaded. */
    indexed: !!assets.data,
    input,
    /** Where an outpaint's source sits: as placed, or the default while that doesn't fit. */
    place,
    refs,
    setRefs,
    lastFrame,
    loras,
    setLoras: lora.setLoras,
    control,
    prompts,
    batchCount,
    setBatchCount,
    seedMode,
    setSeedMode,
    job,

    /** Every kind of media the server has a family for. */
    media: [...new Set(families.data.map((f) => f.media))],
    /** The output size and step count the form is set to. */
    target,
    steps: Number(params.steps ?? 30),
    /** The source's shape isn't the output's, so it needs fitting. */
    misfit,
    /** There is a source and the server still has it. */
    sourceUsable,
    /** The running session's GPU, when it is below what the model needs. */
    underpowered:
      !!sessionGpu && !!variant && belowGpu(sessionGpu, variant.min_gpu)
        ? { needs: variant.min_gpu, has: sessionGpu }
        : null,
    /** No session is running, so jobs will wait. */
    noGpu: session.data !== undefined && !isActive(session.data),
    /** What keeps the form from being submitted. */
    blocked:
      !model ||
      modelMissing ||
      String(params.prompt ?? '').trim() === '' ||
      (needsSource && !sourceUsable) ||
      (mode === 'inpaint' && !maskFits) ||
      (mode === 'flf2v' && (!end || lastFrame.gone)) ||
      (withControl && !control.units.every(unitReady)) ||
      (withPrompts && !prompts.units.every(promptReady)),

    /** Put a LoRA's trigger word into the prompt at `at` (its end when not given). */
    insertTrigger: (word: string, at?: number) => {
      const text = String(params.prompt ?? '')
      setParams({ ...params, prompt: insertWord(text, at ?? text.length, word) })
    },

    /** Swap in a saved prompt, and its negative where the model takes one. */
    applySavedPrompt: (p: SavedPrompt) => {
      const next: Params = { ...params, prompt: p.prompt }
      if ('negative_prompt' in params) next.negative_prompt = p.negative_prompt
      setParams(next)
    },

    addLora: lora.add,
    dropLoras: lora.drop,

    /** Image ⇄ Video: the first family making the other media takes the prompt along. */
    chooseMedia: (media: Family['media']) => {
      const next = families.data.find((f) => f.media === media)
      if (!next || next.id === familyId) return
      leave(next.id, { prompt: String(params.prompt ?? '') })
    },

    /**
     * A mode this model can't do takes the first model that can: from this family if it has
     * one, else from the family used most recently that does.
     */
    chooseMode: (m: string) => {
      if (!family) return
      const recent = loadDraft().recent
      const to = familyForMode({ family, siblings, assets: assets.data, recent, mode: m })
      if (!to) return
      if (to.family.id !== family.id) {
        leave(to.family.id, carry(m, to.model))
        return
      }
      setMode(m)
      const current = model ? variantFor(model, family) : undefined
      if (!current?.modes.includes(m) && to.model) pickModel(to.model.path)
    },

    /** Letterboxing → outpaint: draw what's beyond the image instead of bars. */
    outpaintInstead: () => {
      setMode('outpaint')
      input.setPlace(null)
    },

    /** Choose a model; another family's switches Create to that family. */
    chooseModel: (asset: Asset) => {
      if (family && asset.family && asset.family !== family.id) {
        leave(asset.family, carry(mode ?? '', asset))
        return
      }
      pickModel(asset.path)
    },

    /** `meant` is the size a crop was made for, which a crop kept at its own size isn't. */
    takeSource: (image: BlobInfo, fromCrop: boolean, meant?: Size) => {
      const taken = input.take(image, fromCrop, target)
      const out = meant ?? { w: taken.width, h: taken.height }
      // A crop to another shape sets the size: that's what the crop was for. An outpaint's
      // size is its canvas, which the source sits inside, so it stays.
      if (fromCrop && mode !== 'outpaint' && (out.w !== target.w || out.h !== target.h))
        setParams({ ...params, width: out.w, height: out.h })
    },
  }
}
