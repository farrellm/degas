import type { CSSProperties } from 'react'

import type { ParamProp, SeedMode } from '@/api/types'

import { BatchSeeds } from './BatchSeeds'
import { FieldLabel } from './FieldLabel'

/** A batch's seeds: random, or counting up from the seed field (design §7). */
export interface SeedChoice {
  batch: boolean
  mode: SeedMode
  onMode: (mode: SeedMode) => void
}

export interface FieldProps {
  name: string
  prop: ParamProp
  value: string | number | boolean | null
  set: (name: string, value: string | number | boolean | null) => void
  seeds?: SeedChoice
}

export function Field({ name, prop, value, set, seeds }: FieldProps) {
  const label = prop.title ?? name
  const id = `param-${name}`
  const widget = prop['x-widget']
  const numeric = prop.type === 'integer' || prop.type === 'number'
  const reset = () => {
    if (prop.default !== undefined) set(name, prop.default)
  }

  if (prop.enum) {
    const labels = prop['x-enum-labels'] ?? prop.enum
    return (
      <div className="setting">
        <FieldLabel id={id} label={label} prop={prop} value={value} reset={reset} />
        <select
          id={id}
          value={String(value ?? '')}
          onChange={(e) => {
            set(name, e.target.value)
          }}
        >
          {prop.enum.map((v, i) => (
            <option key={v} value={v}>
              {labels[i] ?? v}
            </option>
          ))}
        </select>
        {prop.description && <small className="setting-note">{prop.description}</small>}
      </div>
    )
  }

  if (prop.type === 'boolean') {
    return (
      <label className="setting check" htmlFor={id}>
        <span className="setting-label">
          {label}
          {prop.description && <small>{prop.description}</small>}
        </span>
        <input
          id={id}
          type="checkbox"
          checked={value === true}
          onChange={(e) => {
            set(name, e.target.checked)
          }}
        />
      </label>
    )
  }

  if (widget === 'seed' && seeds?.batch) {
    return (
      <BatchSeeds
        id={id}
        label={label}
        value={value}
        set={(v) => {
          set(name, v)
        }}
        {...seeds}
      />
    )
  }

  if (widget === 'seed') {
    const random = value === -1 || value === null
    return (
      <div className="setting">
        <label className="setting-label" htmlFor={id}>
          {label}
        </label>
        <div className="seed-control">
          <input
            id={id}
            type="number"
            inputMode="numeric"
            placeholder="Random"
            value={random ? '' : String(value)}
            onChange={(e) => {
              set(name, e.target.value === '' ? -1 : Number(e.target.value))
            }}
          />
          {!random && (
            <button
              type="button"
              className="btn quiet small"
              onClick={() => {
                set(name, -1)
              }}
            >
              Randomize
            </button>
          )}
        </div>
      </div>
    )
  }

  if (numeric && widget === 'slider') {
    const at = notch(prop)
    return (
      <div className="setting">
        <FieldLabel id={id} label={label} prop={prop} value={value} reset={reset} />
        <div className="slider-control">
          <span
            className={at === null ? 'slider-track' : 'slider-track notched'}
            style={at === null ? undefined : ({ '--default': String(at) } as CSSProperties)}
          >
            <input
              id={id}
              type="range"
              min={prop.minimum}
              max={prop.maximum}
              step={prop['x-step'] ?? prop.multipleOf ?? (prop.type === 'integer' ? 1 : 0.1)}
              value={Number(value)}
              onChange={(e) => {
                set(name, Number(e.target.value))
              }}
            />
          </span>
          <output htmlFor={id}>{String(value)}</output>
        </div>
      </div>
    )
  }

  return (
    <div className="setting">
      <FieldLabel id={id} label={label} prop={prop} value={value} reset={reset} />
      <input
        id={id}
        type={numeric ? 'number' : 'text'}
        inputMode={numeric ? 'decimal' : undefined}
        min={prop.minimum}
        max={prop.maximum}
        step={prop.multipleOf}
        value={String(value ?? '')}
        onChange={(e) => {
          set(name, numeric ? Number(e.target.value) : e.target.value)
        }}
      />
    </div>
  )
}

/** Where a slider's default sits along its track, for the notch drawn under it. */
function notch(prop: ParamProp): number | null {
  const { minimum: min, maximum: max } = prop
  if (typeof prop.default !== 'number' || min === undefined || max === undefined || max <= min) {
    return null
  }
  return Math.min(1, Math.max(0, (prop.default - min) / (max - min)))
}
