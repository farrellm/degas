// The shape of a stored draft, checked as it's read back: localStorage holds whatever an
// older version of the app wrote. A field that's missing or of the wrong type is left out,
// and the reader's defaults fill it in, so one bad field doesn't cost the rest of the draft.

import * as v from 'valibot'

import type { LoraEntry, Params, SeedMode } from '@/api/types'
import { FIT_OPTIONS } from '@/lib/fit'
import { TAKES } from '@/lib/imagePrompts'

import { type ControlUnit, restoreUnit, TRACES } from './control/control'
import { type ImagePromptUnit, restorePrompt } from './image-prompt/imagePrompt'

/** Left out when missing or invalid. */
const kept = <S extends v.GenericSchema>(schema: S) => v.fallback(v.optional(schema), undefined)

const source = v.object({ sha: v.string(), width: v.number(), height: v.number() })
const rect = v.object({ x: v.number(), y: v.number(), w: v.number(), h: v.number() })
const fit = v.picklist(FIT_OPTIONS.map((f) => f.id))
const params: v.GenericSchema<unknown, Params> = v.record(
  v.string(),
  v.union([v.string(), v.number(), v.boolean(), v.null()]),
)

const loraRef = v.object({ path: v.string(), weight: v.number(), size: v.nullish(v.number()) })
const lora: v.GenericSchema<unknown, LoraEntry> = v.union([
  loraRef,
  v.object({ high: v.optional(loraRef), low: v.optional(loraRef) }),
])

// A unit saved before a field was added lacks it; the defaults fill it in.
const controlUnit: v.GenericSchema<unknown, ControlUnit> = v.pipe(
  v.object({
    key: kept(v.string()),
    model: kept(v.string()),
    image: kept(v.nullable(source)),
    trace: kept(
      v.nullable(v.object({ id: v.picklist(TRACES.map((t) => t.id)), params, from: source })),
    ),
    area: kept(v.nullable(v.string())),
    scale: kept(v.number()),
    start: kept(v.number()),
    end: kept(v.number()),
    fit: kept(fit),
  }),
  v.transform(restoreUnit),
)

const imagePromptUnit: v.GenericSchema<unknown, ImagePromptUnit> = v.pipe(
  v.object({
    key: kept(v.string()),
    model: kept(v.string()),
    take: kept(v.picklist(TAKES.map((t) => t.id))),
    pictures: kept(v.array(source)),
    area: kept(v.nullable(v.object({ sha: v.string(), over: source }))),
    weight: kept(v.number()),
    start: kept(v.number()),
    end: kept(v.number()),
    fit: kept(fit),
    structure: kept(v.number()),
    loraWeight: kept(v.number()),
    downsample: kept(v.number()),
  }),
  v.transform(restorePrompt),
)

export const familyDraftSchema = v.object({
  model: kept(v.string()),
  mode: kept(v.string()),
  loras: kept(v.array(lora)),
  params: kept(params),
  source: kept(v.nullable(source)),
  fit: kept(fit),
  extends: kept(v.nullable(v.string())),
  mask: kept(v.nullable(v.object({ sha: v.string(), source: v.string() }))),
  place: kept(v.nullable(rect)),
  end: kept(v.nullable(source)),
  control: kept(v.array(controlUnit)),
  refs: kept(v.array(source)),
  prompts: kept(v.array(imagePromptUnit)),
})

const seedModes: SeedMode[] = ['increment', 'random']

/** A stored draft: every field optional, and the pre-Phase 4 single-family fields beside them. */
export const storedSchema = v.fallback(
  v.object({
    family: kept(v.string()),
    families: kept(v.record(v.string(), v.fallback(familyDraftSchema, {}))),
    batchCount: kept(v.pipe(v.number(), v.integer(), v.minValue(1))),
    seedMode: kept(v.picklist(seedModes)),
    recent: kept(v.array(v.string())),
    model: kept(v.string()),
    loras: kept(v.array(lora)),
    params: kept(params),
  }),
  {},
)
