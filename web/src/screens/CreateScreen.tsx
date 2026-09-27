import { useMutation, useQuery } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { api, isActive, type LoraRef, type Params } from '../api'
import { assetLabel, useAssets } from '../assets'
import { AssetPicker } from '../components/AssetPicker'
import { LoraList } from '../components/LoraList'
import { SchemaForm } from '../components/SchemaForm'
import { loadDraft, saveDraft } from '../draft'
import { insertWord } from '../prompt'
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
  const [loras, setLoras] = useState<LoraRef[]>(draft.loras ?? [])
  const [batchCount, setBatchCount] = useState(draft.batchCount ?? 1)
  const [queued, setQueued] = useState<number | null>(null)
  const [picker, setPicker] = useState<'model' | 'lora' | null>(null)
  const promptRef = useRef<HTMLTextAreaElement>(null)
  const promptFocused = useRef(false)

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
  const assets = useAssets()
  const ofKind = (kind: string) =>
    assets.data?.filter((a) => a.family === family?.id && a.kind === kind)
  const models = ofKind('model')
  const loraIndex = ofKind('lora')

  // Until edited, the form shows the saved draft (or the schema defaults).
  const params = editedParams ?? (schema.data ? initialParams(schema.data, draft.params) : null)
  const model = models?.some((m) => m.path === chosenModel)
    ? chosenModel
    : (models?.[0]?.path ?? '')

  useEffect(() => {
    if (params) saveDraft({ family: familyId, model, loras, params, batchCount })
  }, [familyId, model, loras, params, batchCount])

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
        {
          family: family.id,
          variant: variant.id,
          mode,
          model: { path: model },
          loras: loras.map(({ path, weight }) => ({ path, weight })),
          params,
        },
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

  const insertTrigger = (word: string) => {
    const text = String(params.prompt ?? '')
    const el = promptRef.current
    // Before the prompt has been touched there's no cursor to honour: append.
    const at = promptFocused.current && el ? el.selectionEnd : text.length
    setParams({ ...params, prompt: insertWord(text, at, word) })
  }

  const leadingRows = (
    <>
      <button
        type="button"
        className="setting setting-button"
        onClick={() => {
          setPicker('model')
        }}
      >
        <span className="setting-label">Model</span>{' '}
        <span className={model ? 'setting-value' : 'setting-value none'}>
          {model ? assetLabel(model, models) : models ? 'None found' : 'Loading…'}
        </span>
      </button>
      {family?.lora_format === 'single' && (
        <LoraList
          loras={loras}
          index={loraIndex}
          onChange={setLoras}
          onAdd={() => {
            setPicker('lora')
          }}
          onTrigger={insertTrigger}
        />
      )}
    </>
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
        leadingRows={leadingRows}
        promptRef={promptRef}
        onPromptFocus={() => {
          promptFocused.current = true
        }}
      />

      {picker === 'model' && (
        <AssetPicker
          title="Model"
          noun="models"
          assets={models ?? []}
          selected={new Set([model])}
          empty={
            <p>
              No models found. Put checkpoints in Drive under{' '}
              <code>degas/models/{family?.id}/</code>, then rescan.
            </p>
          }
          onPick={(a) => {
            setModel(a.path)
            setPicker(null)
          }}
          onClose={() => {
            setPicker(null)
          }}
        />
      )}
      {picker === 'lora' && (
        <AssetPicker
          title="Add LoRA"
          noun="LoRAs"
          assets={loraIndex ?? []}
          selected={new Set(loras.map((l) => l.path))}
          thumbs
          empty={
            <p>
              No LoRAs found. Put them in Drive under <code>degas/loras/{family?.id}/</code>, then
              rescan.
            </p>
          }
          onPick={(a) => {
            if (!loras.some((l) => l.path === a.path)) {
              setLoras([...loras, { path: a.path, weight: a.sidecar?.default_weight ?? 1 }])
            }
            setPicker(null)
          }}
          onClose={() => {
            setPicker(null)
          }}
        />
      )}

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
