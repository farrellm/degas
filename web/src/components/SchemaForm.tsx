import { useState } from 'react'
import type { ParamProp, ParamSchema, Params } from '../api'

interface Props {
  schema: ParamSchema
  values: Params
  onChange: (values: Params) => void
  presets: [number, number][]
}

/** Thin renderer for a family's param schema (design §8.1). */
export function SchemaForm({ schema, values, onChange, presets }: Props) {
  const [showAdvanced, setShowAdvanced] = useState(false)
  const set = (name: string, value: string | number | null) => {
    onChange({ ...values, [name]: value })
  }
  const entries = Object.entries(schema.properties)
  const basic = entries.filter(([, p]) => !p['x-advanced'])
  const advanced = entries.filter(([, p]) => p['x-advanced'])
  const hasAspect = entries.some(([, p]) => p['x-widget'] === 'aspect')

  const render = ([name, prop]: [string, ParamProp]) =>
    prop['x-widget'] === 'aspect' ? null : (
      <Field key={name} name={name} prop={prop} value={values[name] ?? null} set={set} />
    )

  return (
    <div className="form">
      {basic.map(render)}
      {hasAspect && (
        <AspectPicker
          width={Number(values.width)}
          height={Number(values.height)}
          presets={presets}
          onChange={(w, h) => {
            onChange({ ...values, width: w, height: h })
          }}
        />
      )}
      {advanced.length > 0 && (
        <details
          open={showAdvanced}
          onToggle={(e) => {
            setShowAdvanced(e.currentTarget.open)
          }}
        >
          <summary>Advanced</summary>
          {advanced.map(render)}
        </details>
      )}
    </div>
  )
}

interface FieldProps {
  name: string
  prop: ParamProp
  value: string | number | null
  set: (name: string, value: string | number | null) => void
}

function Field({ name, prop, value, set }: FieldProps) {
  const label = prop.title ?? name
  const id = `param-${name}`
  const widget = prop['x-widget']

  if (widget === 'prompt') {
    return (
      <label className="field" htmlFor={id}>
        <span>{label}</span>
        <textarea
          id={id}
          rows={name === 'prompt' ? 4 : 2}
          value={String(value ?? '')}
          onChange={(e) => {
            set(name, e.target.value)
          }}
        />
      </label>
    )
  }

  if (prop.enum) {
    const labels = prop['x-enum-labels'] ?? prop.enum
    return (
      <label className="field" htmlFor={id}>
        <span>{label}</span>
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
      </label>
    )
  }

  if (widget === 'seed') {
    const random = value === -1 || value === null
    return (
      <label className="field" htmlFor={id}>
        <span>{label}</span>
        <div className="row">
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
          <button
            type="button"
            className="secondary"
            onClick={() => {
              set(name, -1)
            }}
            disabled={random}
          >
            Random
          </button>
        </div>
      </label>
    )
  }

  const numeric = prop.type === 'integer' || prop.type === 'number'
  if (numeric && widget === 'slider') {
    return (
      <label className="field" htmlFor={id}>
        <span>
          {label} <output>{String(value)}</output>
        </span>
        <input
          id={id}
          type="range"
          min={prop.minimum}
          max={prop.maximum}
          step={prop.multipleOf ?? (prop.type === 'integer' ? 1 : 0.1)}
          value={Number(value)}
          onChange={(e) => {
            set(name, Number(e.target.value))
          }}
        />
      </label>
    )
  }

  return (
    <label className="field" htmlFor={id}>
      <span>{label}</span>
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
    </label>
  )
}

function AspectPicker({
  width,
  height,
  presets,
  onChange,
}: {
  width: number
  height: number
  presets: [number, number][]
  onChange: (w: number, h: number) => void
}) {
  return (
    <div className="field">
      <span>
        Size{' '}
        <output>
          {width}×{height}
        </output>
      </span>
      <div className="aspects">
        {presets.map(([w, h]) => (
          <button
            key={`${String(w)}x${String(h)}`}
            type="button"
            className={w === width && h === height ? 'aspect selected' : 'aspect'}
            aria-label={`${String(w)}×${String(h)}`}
            aria-pressed={w === width && h === height}
            onClick={() => {
              onChange(w, h)
            }}
          >
            <span
              className="aspect-box"
              style={{ aspectRatio: `${String(w)} / ${String(h)}` }}
              aria-hidden
            />
          </button>
        ))}
      </div>
    </div>
  )
}
