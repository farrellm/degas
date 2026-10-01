import { Fragment, type ReactNode, type Ref } from 'react'

import type { ParamProp, Params, ParamSchema } from '@/api/types'

import { AspectPicker } from './AspectPicker'
import { Field, type SeedChoice } from './Field'

export interface SchemaFormProps {
  schema: ParamSchema
  values: Params
  onChange: (values: Params) => void
  presets: [number, number][]
  /** Rows shown first in the settings list (e.g. the model picker). */
  leadingRows?: ReactNode
  /** The main prompt field, e.g. to insert trigger words at the cursor. */
  promptRef?: Ref<HTMLTextAreaElement>
  onPromptFocus?: () => void
  /** Shown beside the first prompt's label (e.g. saved prompts). */
  promptAside?: ReactNode
  promptPlaceholder?: string
  seeds?: SeedChoice
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
  promptAside,
  promptPlaceholder = 'Describe the picture',
  seeds,
}: SchemaFormProps) {
  const set = (name: string, value: string | number | boolean | null) => {
    onChange({ ...values, [name]: value })
  }
  const entries = Object.entries(schema.properties)
  const prompts = entries.filter(([, p]) => p['x-widget'] === 'prompt')
  const rest = entries.filter(([, p]) => p['x-widget'] !== 'prompt' && p['x-widget'] !== 'aspect')
  const basic = rest.filter(([, p]) => !p['x-advanced'])
  const advanced = rest.filter(([, p]) => p['x-advanced'])
  const hasAspect = entries.some(([, p]) => p['x-widget'] === 'aspect')

  const render = ([name, prop]: [string, ParamProp]) => (
    <Field
      key={name}
      name={name}
      prop={prop}
      value={values[name] ?? null}
      set={set}
      seeds={seeds}
    />
  )

  return (
    <>
      {prompts.length > 0 && (
        <div className="prompt-block">
          {prompts.map(([name, prop], i) => (
            <Fragment key={name}>
              {i > 0 && <hr />}
              {i === 0 && promptAside ? (
                <div className="prompt-head">
                  <label htmlFor={`param-${name}`}>{prop.title ?? name}</label>
                  {promptAside}
                </div>
              ) : (
                <label htmlFor={`param-${name}`}>{prop.title ?? name}</label>
              )}
              <textarea
                ref={i === 0 ? promptRef : undefined}
                onFocus={i === 0 ? onPromptFocus : undefined}
                id={`param-${name}`}
                className={i > 0 ? 'secondary-prompt' : undefined}
                rows={i === 0 ? 3 : 1}
                placeholder={i === 0 ? promptPlaceholder : undefined}
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
            defaultSize={[
              Number(schema.properties.width?.default),
              Number(schema.properties.height?.default),
            ]}
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
