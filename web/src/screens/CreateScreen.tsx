import { useMutation, useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { api, isActive, type Params } from '../api'
import { SchemaForm } from '../components/SchemaForm'
import { loadDraft, saveDraft } from '../draft'
import { modelName } from '../format'
import { initialParams } from '../schema'

const MAX_BATCH = 8

interface Props {
  onOpenSession: () => void
  onShowResults: () => void
}

export function CreateScreen({ onOpenSession, onShowResults }: Props) {
  const [draft] = useState(loadDraft)
  const [familyId, setFamilyId] = useState(draft.family ?? 'sdxl')
  const [chosenModel, setModel] = useState(draft.model ?? '')
  const [editedParams, setParams] = useState<Params | null>(null)
  const [batchCount, setBatchCount] = useState(draft.batchCount ?? 1)
  const [queued, setQueued] = useState<number | null>(null)

  const families = useQuery({ queryKey: ['families'], queryFn: api.families })
  const session = useQuery({ queryKey: ['session'], queryFn: api.session })
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

  useEffect(() => {
    if (queued === null) return
    const id = setTimeout(() => {
      setQueued(null)
    }, 5000)
    return () => {
      clearTimeout(id)
    }
  }, [queued])

  const submit = useMutation({
    mutationFn: () => {
      if (!family || !variant || !mode || !params) throw new Error('The form is still loading.')
      return api.submitJob(
        { family: family.id, variant: variant.id, mode, model: { path: model }, params },
        batchCount,
        'increment',
      )
    },
    onSuccess: () => {
      setQueued(batchCount)
    },
  })

  if (families.isPending || schema.isPending || !params) {
    return <p className="loading">Loading…</p>
  }
  if (families.error || schema.error) {
    return <p role="alert">{(families.error ?? schema.error)?.message}</p>
  }

  const hasPrompt = String(params.prompt ?? '').trim() !== ''
  const noGpu = session.data !== undefined && !isActive(session.data)
  const items = batchCount === 1 ? 'image' : 'images'

  const modelRow = (
    <div className="setting">
      <label className="setting-label" htmlFor="model">
        Model
      </label>
      {models.data?.length ? (
        <select
          id="model"
          value={model}
          onChange={(e) => {
            setModel(e.target.value)
          }}
        >
          {models.data.map((m) => (
            <option key={m.path} value={m.path}>
              {modelName(m.path)}
            </option>
          ))}
        </select>
      ) : (
        <p className="empty-models" id="model">
          No models found. Put checkpoints in Drive under <code>degas/models/{family?.id}/</code>,
          then rescan from the GPU menu.
        </p>
      )}
    </div>
  )

  return (
    <form
      className="create"
      onSubmit={(e) => {
        e.preventDefault()
        submit.mutate()
      }}
    >
      {families.data.length > 1 && (
        <div className="settings">
          <div className="setting">
            <label className="setting-label" htmlFor="family">
              Family
            </label>
            <select
              id="family"
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
          </div>
        </div>
      )}

      <SchemaForm
        schema={schema.data}
        values={params}
        onChange={setParams}
        presets={variant?.size_constraints.presets ?? []}
        leadingRows={modelRow}
      />

      <div className="generate-bar">
        <div className="generate-bar-inner">
          {submit.error ? (
            <p className="bar-note error" role="alert">
              {submit.error.message}
            </p>
          ) : queued !== null ? (
            <p className="toast" role="status">
              <span>
                Queued {queued} {queued === 1 ? 'image' : 'images'}.
              </span>
              <button type="button" className="link" onClick={onShowResults}>
                See results
              </button>
            </p>
          ) : noGpu ? (
            <p className="bar-note">
              No GPU is running, so jobs will wait.{' '}
              <button type="button" className="link" onClick={onOpenSession}>
                Start a session
              </button>
            </p>
          ) : null}
          <div className="generate-row">
            <div className="stepper" role="group" aria-label="Batch size">
              <button
                type="button"
                aria-label="Fewer images"
                disabled={batchCount <= 1}
                onClick={() => {
                  setBatchCount((n) => Math.max(1, n - 1))
                }}
              >
                −
              </button>
              <output aria-label="Images per job">{batchCount}</output>
              <button
                type="button"
                aria-label="More images"
                disabled={batchCount >= MAX_BATCH}
                onClick={() => {
                  setBatchCount((n) => Math.min(MAX_BATCH, n + 1))
                }}
              >
                +
              </button>
            </div>
            <button
              type="submit"
              className="btn"
              disabled={submit.isPending || !model || !hasPrompt}
            >
              {submit.isPending
                ? 'Queuing…'
                : batchCount === 1
                  ? 'Generate'
                  : `Generate ${String(batchCount)} ${items}`}
            </button>
          </div>
        </div>
      </div>
    </form>
  )
}
