import type { ParamProp, ParamSchema, Params } from './api'

/** Default values for every parameter in a schema. */
export function schemaDefaults(schema: ParamSchema): Params {
  const out: Params = {}
  for (const [name, prop] of Object.entries(schema.properties)) {
    out[name] = prop.default ?? (prop.type === 'string' ? '' : null)
  }
  return out
}

/**
 * Defaults for a schema, overridden by any matching values saved from an earlier visit. A saved
 * choice that is no longer on offer (a renamed sampler) falls back to the default.
 */
export function initialParams(schema: ParamSchema, saved: Params | undefined): Params {
  const defaults = schemaDefaults(schema)
  const out: Params = {}
  for (const [k, v] of Object.entries(defaults)) {
    const choices = schema.properties[k]?.enum
    const keep = saved && k in saved && (!choices || choices.includes(String(saved[k])))
    out[k] = keep ? (saved[k] ?? null) : v
  }
  return out
}

/** The display label of an enum parameter's value (the value itself when it has none). */
export function enumLabel(
  prop: ParamProp | undefined,
  value: Params[string] | undefined,
): string | null {
  if (value == null) return null
  const i = prop?.enum?.indexOf(String(value)) ?? -1
  return prop?.['x-enum-labels']?.[i] ?? String(value)
}
