import type { ParamSchema, Params } from './api'

/** Default values for every parameter in a schema. */
export function schemaDefaults(schema: ParamSchema): Params {
  const out: Params = {}
  for (const [name, prop] of Object.entries(schema.properties)) {
    out[name] = prop.default ?? (prop.type === 'string' ? '' : null)
  }
  return out
}

/** Defaults for a schema, overridden by any matching values saved from an earlier visit. */
export function initialParams(schema: ParamSchema, saved: Params | undefined): Params {
  const defaults = schemaDefaults(schema)
  const out: Params = {}
  for (const [k, v] of Object.entries(defaults)) {
    out[k] = saved && k in saved ? (saved[k] ?? null) : v
  }
  return out
}
