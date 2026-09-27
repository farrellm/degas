import { Fragment, type ReactNode, type Ref } from 'react'
import type { ParamProp, ParamSchema, Params } from '../api'
import { size } from '../format'

interface Props {
  schema: ParamSchema
  values: Params
  onChange: (values: Params) => void
  presets: [number, number][]
  /** Rows shown first in the settings list (e.g. the model picker). */
  leadingRows?: ReactNode
  /** The main prompt field, e.g. to insert trigger words at the cursor. */
  promptRef?: Ref<HTMLTextAreaElement>
  onPromptFocus?: () => void
}

/**
 * Thin renderer for a family's param schema (design §8.1): prompt widgets
 * go in the prompt block, everything else in the settings list, with
 * advanced params under "More settings".
 */
export function SchemaForm({
  schema,
  values,
  onChange,
  presets,
  leadingRows,
  promptRef,
  onPromptFocus,
}: Props) {
  const set = (name: string, value: string | number | null) => {
    onChange({ ...values, [name]: value })
  }
  const entries = Object.entries(schema.properties)
  const prompts = entries.filter(([, p]) => p['x-widget'] === 'prompt')
  const rest = entries.filter(([, p]) => p['x-widget'] !== 'prompt' && p['x-widget'] !== 'aspect')
  const basic = rest.filter(([, p]) => !p['x-advanced'])
  const advanced = rest.filter(([, p]) => p['x-advanced'])
  const hasAspect = entries.some(([, p]) => p['x-widget'] === 'aspect')

  const render = ([name, prop]: [string, ParamProp]) => (
    <Field key={name} name={name} prop={prop} value={values[name] ?? null} set={set} />
  )

  return (
    <>
      {prompts.length > 0 && (
        <div className="prompt-block">
          {prompts.map(([name, prop], i) => (
            <Fragment key={name}>
              {i > 0 && <hr />}
              <label htmlFor={`param-${name}`}>{prop.title ?? name}</label>
              <textarea
                ref={i === 0 ? promptRef : undefined}
                onFocus={i === 0 ? onPromptFocus : undefined}
                id={`param-${name}`}
                className={i > 0 ? 'secondary-prompt' : undefined}
                rows={i === 0 ? 3 : 1}
                placeholder={i === 0 ? 'Describe the picture' : undefined}
                value={String(values[name] ?? '')}
                onChange={(e) => {
                  set(name, e.target.value)
                }}
              />
            </Fragment>
          ))}
        </div>
      )}

      <div className="settings">
        {leadingRows}
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
        {basic.map(render)}
      </div>

      {advanced.length > 0 && (
        <details className="more">
          <summary>More settings</summary>
          <div className="settings">{advanced.map(render)}</div>
        </details>
      )}
    </>
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
  const numeric = prop.type === 'integer' || prop.type === 'number'

  if (prop.enum) {
    const labels = prop['x-enum-labels'] ?? prop.enum
    return (
      <div className="setting">
        <label className="setting-label" htmlFor={id}>
          {label}
        </label>
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
      </div>
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
    return (
      <div className="setting">
        <label className="setting-label" htmlFor={id}>
          {label}
        </label>
        <div className="slider-control">
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
          <output htmlFor={id}>{String(value)}</output>
        </div>
      </div>
    )
  }

  return (
    <div className="setting">
      <label className="setting-label" htmlFor={id}>
        {label}
      </label>
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
    <div className="setting stacked" role="group" aria-labelledby="aspect-label">
      <span className="setting-label" id="aspect-label">
        Size
      </span>
      <span className="setting-value">{size(width, height)}</span>
      <div className="setting-control aspects">
        {presets.map(([w, h]) => {
          const on = w === width && h === height
          return (
            <button
              key={`${String(w)}x${String(h)}`}
              type="button"
              className={w > h ? 'aspect wide' : 'aspect'}
              aria-label={`${String(w)}×${String(h)}`}
              aria-pressed={on}
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
          )
        })}
      </div>
    </div>
  )
}
