import { useMutation, useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'

import { api } from '@/api/client'
import { queries } from '@/api/queries'
import type { Asset, BlobInfo, Family, LoraEntry, Params, SavedPrompt, SeedMode } from '@/api/types'
import { defaultPlace, validPlace } from '@/features/editors/place/place'
import { useAssets } from '@/hooks/useAssets'
import { useAutoDismiss } from '@/hooks/useAutoDismiss'
import { variantFor } from '@/lib/assets'
import { ratioDiffers, type Size } from '@/lib/geometry'
import { belowGpu } from '@/lib/gpu'
import type { Source } from '@/lib/image'
import { loraPaths, sameLora } from '@/lib/loras'
import { initialParams } from '@/lib/schema'
import { isActive } from '@/lib/session'

import { unitReady } from './control/control'
import { loadDraft, saveFamilyDraft, switchFamily } from './draft'
import { promptReady } from './image-prompt/imagePrompt'
import { insertWord } from './prompt'
import { modelForMode, resolveSelection } from './selection'
import { buildJob } from './spec'
import { useSourceInput } from './useSourceInput'
import { useUnitList } from './useUnitList'

// How long "Queued 4 images." stays up.
const QUEUED_MS = 5000

/**
 * One family's Create form: what's been chosen and typed (kept as that family's draft), what
 * follows from it, and the job it submits. `onFamily` switches Create to another family,
 * whose own form then takes over.
 */
export function useCreateForm(familyId: string, onFamily: (id: string) => void) {
  const [draft] = useState(() => {
    const d = loadDraft()
    return { ...d.families[familyId], batchCount: d.batchCount, seedMode: d.seedMode }
  })
  const [chosenModel, setModel] = useState(draft.model ?? '')
  const [chosenMode, setMode] = useState(draft.mode)
  const [editedParams, setParams] = useState<Params | null>(null)
  const [loras, setLoras] = useState<LoraEntry[]>(draft.loras ?? [])
  const input = useSourceInput(draft)
  const { source, mask, maskFits } = input
  const [refs, setRefs] = useState(draft.refs ?? [])
  const [end, setEnd] = useState<Source | null>(draft.end ?? null)
  // The server no longer stores the last frame (its thumbnail failed to load).
  const [endGone, setEndGone] = useState(false)
  const control = useUnitList(draft.control ?? [])
  const prompts = useUnitList(draft.prompts ?? [])
  const [batchCount, setBatchCount] = useState(draft.batchCount)
  const [seedMode, setSeedMode] = useState<SeedMode>(draft.seedMode)
  const [queued, setQueued] = useState<number | null>(null)

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

  useEffect(() => {
    if (!params || !mode) return
    saveFamilyDraft(
      familyId,
      {
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
      },
      batchCount,
      seedMode,
    )
  }, [
    familyId,
    model,
    mode,
    loras,
    params,
    source,
    input.fit,
    input.extendsClip,
    mask,
    input.place,
    end,
    control.units,
    refs,
    prompts.units,
    batchCount,
    seedMode,
  ])

  useAutoDismiss(queued, QUEUED_MS, () => setQueued(null))

  const canvas = params ? { w: Number(params.width), h: Number(params.height) } : null
  const place =
    mode === 'outpaint' && source && canvas
      ? validPlace(input.place, { w: source.width, h: source.height }, canvas)
        ? input.place
        : defaultPlace({ w: source.width, h: source.height }, canvas)
      : null

  const submit = useMutation({
    mutationFn: () => {
      if (!family || !variant || !mode || !params) throw new Error('The form is still loading.')
      const job = buildJob({
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
      })
      return api.submitJob(job.spec, batchCount, job.seedMode)
    },
    onSuccess: () => setQueued(batchCount),
  })

  if (families.isPending || schema.isPending || !params)
    return { ready: false, error: null } as const
  if (families.error || schema.error) {
    return { ready: false, error: families.error ?? schema.error } as const
  }

  const target = { w: Number(params.width), h: Number(params.height) }
  const sessionGpu = isActive(session.data) ? session.data?.session?.gpu : undefined
  const misfit = !!source && ratioDiffers(source.width / source.height, target.w / target.h)
  const sourceUsable = !!source && !input.gone

  // Changing the model keeps the prompt and images, even over ones the other family had.
  const carry = (m: string, next?: Asset) => ({
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
    if (next && next.lora_format !== variant?.lora_format) setLoras([])
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
    /** First and last frame: the last frame, which is fitted to the output like the source. */
    lastFrame: {
      image: end,
      gone: endGone,
      take: (image: BlobInfo) => {
        setEnd({
          sha: image.sha256,
          width: image.width ?? target.w,
          height: image.height ?? target.h,
        })
        setEndGone(false)
      },
      remove: () => {
        setEnd(null)
        setEndGone(false)
      },
      markGone: () => setEndGone(true),
    },
    loras,
    setLoras,
    control,
    prompts,
    batchCount,
    setBatchCount,
    seedMode,
    setSeedMode,
    /** How many were just queued, while the confirmation shows. */
    queued,
    submit,

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
      (mode === 'flf2v' && (!end || endGone)) ||
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

    /** Adding a LoRA that's already in the form does nothing. */
    addLora: (entry: LoraEntry) =>
      setLoras((ls) => (ls.some((l) => sameLora(l, entry)) ? ls : [...ls, entry])),

    /** Drop the LoRAs whose files were deleted from Drive. */
    dropLoras: (paths: string[]) =>
      setLoras((ls) => ls.filter((l) => !loraPaths(l).some((p) => paths.includes(p)))),

    /** Image ⇄ Video: the first family making the other media takes the prompt along. */
    chooseMedia: (media: Family['media']) => {
      const next = families.data.find((f) => f.media === media)
      if (!next || next.id === familyId) return
      switchFamily(next.id, { prompt: String(params.prompt ?? '') })
      onFamily(next.id)
    },

    /**
     * A mode this model can't do takes the first model that can: from this family if it has
     * one, else from the family used most recently that does.
     */
    chooseMode: (m: string) => {
      if (!family) return
      const recent = loadDraft().recent
      const rank = (f: Family) => (f.id === family.id ? -1 : recent.indexOf(f.id) + 1 || Infinity)
      const candidates = siblings
        .filter((f) => f.variants.some((v) => v.modes.includes(m)))
        .sort((a, b) => rank(a) - rank(b))
      const withModel = candidates.find((f) => modelForMode(assets.data, f, m))
      const to = withModel ?? candidates[0]
      if (!to) return
      if (to.id !== family.id) {
        switchFamily(to.id, carry(m, withModel && modelForMode(assets.data, withModel, m)))
        onFamily(to.id)
        return
      }
      setMode(m)
      const current = model ? variantFor(model, family) : undefined
      const next = modelForMode(assets.data, family, m)
      if (!current?.modes.includes(m) && next) pickModel(next.path)
    },

    /** Letterboxing → outpaint: draw what's beyond the image instead of bars. */
    outpaintInstead: () => {
      setMode('outpaint')
      input.setPlace(null)
    },

    /** Choose a model; another family's switches Create to that family. */
    chooseModel: (asset: Asset) => {
      if (family && asset.family && asset.family !== family.id) {
        switchFamily(asset.family, carry(mode ?? '', asset))
        onFamily(asset.family)
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
