import { useMutation, useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { api, type Params } from '../api'
import { SchemaForm } from '../components/SchemaForm'
import { SessionPrompt } from '../components/SessionPrompt'
import { initialParams } from '../schema'

const DRAFT_KEY = 'degas.create.draft'

interface Draft {
  family: string
  model: string
  params: Params
  batchCount: number
}

function loadDraft(): Partial<Draft> {
  try {
    return JSON.parse(localStorage.getItem(DRAFT_KEY) ?? '{}') as Partial<Draft>
  } catch {
    return {}
  }
}

function saveDraft(draft: Draft) {
  try {
    localStorage.setItem(DRAFT_KEY, JSON.stringify(draft))
  } catch {
    // storage unavailable (private mode): the draft just isn't kept
  }
}

export function CreateScreen({ onQueued }: { onQueued: () => void }) {
  const [draft] = useState(loadDraft)
  const [familyId, setFamilyId] = useState(draft.family ?? 'sdxl')
  const [chosenModel, setModel] = useState(draft.model ?? '')
  const [editedParams, setParams] = useState<Params | null>(null)
  const [batchCount, setBatchCount] = useState(draft.batchCount ?? 1)

  const families = useQuery({ queryKey: ['families'], queryFn: api.families })
  const family = families.data?.find((f) => f.id === familyId) ?? families.data?.[0]
  const variant = family?.variants[0]
  const mode = variant?.modes[0]

  const schema = useQuery({
    queryKey: ['schema', family?.id, variant?.id, mode],
    queryFn: () => api.schema(family?.id ?? '', variant?.id ?? '', mode ?? ''),
    enabled: !!family && !!variant && !!mode,
    staleTime: Infinity,
  })
  const models = useQuery({
    queryKey: ['assets', family?.id, 'model'],
    queryFn: () => api.assets(family?.id ?? '', 'model'),
    enabled: !!family,
  })

  // Until edited, the form shows the saved draft (or the schema defaults).
  const params = editedParams ?? (schema.data ? initialParams(schema.data, draft.params) : null)
  const model = models.data?.some((m) => m.path === chosenModel)
    ? chosenModel
    : (models.data?.[0]?.path ?? '')

  useEffect(() => {
    if (params) saveDraft({ family: familyId, model, params, batchCount })
  }, [familyId, model, params, batchCount])

  const submit = useMutation({
    mutationFn: () => {
      if (!family || !variant || !mode || !params) throw new Error('Form not ready')
      return api.submitJob(
        { family: family.id, variant: variant.id, mode, model: { path: model }, params },
        batchCount,
        'increment',
      )
    },
    onSuccess: onQueued,
  })

  if (families.isPending || schema.isPending || !params) {
    return <p className="muted">Loading…</p>
  }
  if (families.error || schema.error) {
    return <p role="alert">{(families.error ?? schema.error)?.message}</p>
  }

  return (
    <form
      className="create"
      onSubmit={(e) => {
        e.preventDefault()
        submit.mutate()
      }}
    >
      <SessionPrompt />

      {families.data.length > 1 && (
        <label className="field">
          <span>Family</span>
          <select
            value={familyId}
            onChange={(e) => {
              setFamilyId(e.target.value)
              setParams(null)
            }}
          >
            {families.data.map((f) => (
              <option key={f.id} value={f.id}>
                {f.label}
              </option>
            ))}
          </select>
        </label>
      )}

      <label className="field">
        <span>Model</span>
        {models.data?.length ? (
          <select
            value={model}
            onChange={(e) => {
              setModel(e.target.value)
            }}
          >
            {models.data.map((m) => (
              <option key={m.path} value={m.path}>
                {m.path.split('/').pop()}
              </option>
            ))}
          </select>
        ) : (
          <span className="muted">
            No models indexed. Put checkpoints in Drive under{' '}
            <code>degas/models/{family?.id}/</code> and rescan from the Session tab.
          </span>
        )}
      </label>

      <SchemaForm
        schema={schema.data}
        values={params}
        onChange={setParams}
        presets={variant?.size_constraints.presets ?? []}
      />

      <label className="field">
        <span>
          Batch <output>{batchCount}</output>
        </span>
        <input
          type="range"
          min={1}
          max={8}
          value={batchCount}
          onChange={(e) => {
            setBatchCount(Number(e.target.value))
          }}
        />
      </label>

      {submit.error && <p role="alert">{submit.error.message}</p>}
      <div className="sticky-action">
        <button type="submit" disabled={submit.isPending || !model}>
          {submit.isPending ? 'Queuing…' : 'Generate'}
        </button>
      </div>
    </form>
  )
}
