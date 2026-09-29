import { useMutation, useQuery } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import {
  api,
  isActive,
  isPair,
  thumbUrl,
  type BlobInfo,
  type Fit,
  type LoraEntry,
  type Params,
  type SeedMode,
} from '../api'
import { assetLabel, loraChoices, useAssets, variantFor } from '../assets'
import { AssetPicker } from '../components/AssetPicker'
import { ControlEditor } from '../components/ControlEditor'
import { ControlList } from '../components/ControlList'
import { newUnit, unitReady, unitSpec, type ControlUnit } from '../control'
import { CropEditor } from '../components/CropEditor'
import { ImagePicker } from '../components/ImagePicker'
import { LoraList } from '../components/LoraList'
import { MaskEditor } from '../components/MaskEditor'
import { MaskThumb } from '../components/MaskThumb'
import { PlaceEditor } from '../components/PlaceEditor'
import { PromptSheet } from '../components/PromptSheet'
import { RefList } from '../components/RefList'
import { SchemaForm } from '../components/SchemaForm'
import {
  loadDraft,
  saveFamilyDraft,
  SOURCE_MODES,
  switchFamily,
  type MaskRef,
  type Source,
} from '../draft'
import { size } from '../format'
import { defaultPlace, validPlace, type Place } from '../place'
import { insertWord } from '../prompt'
import { initialParams } from '../schema'

const MAX_BATCH = 8
const GPUS = ['T4', 'L4', 'A100', 'H100']

const MODE_LABELS: Record<string, string> = {
  t2i: 'From text',
  t2v: 'From text',
  i2i: 'From image',
  i2v: 'From image',
  edit: 'Edit',
  inpaint: 'Inpaint',
  outpaint: 'Outpaint',
}

const MEDIA_LABELS = { image: 'Image', video: 'Video' } as const

const FITS: { id: Fit; label: string }[] = [
  { id: 'crop', label: 'Crop to fit' },
  { id: 'pad', label: 'Letterbox' },
  { id: 'stretch', label: 'Stretch' },
]

interface Props {
  onOpenSession: () => void
  onShowResults: () => void
}

/** Create: one form per family, each keeping its own draft. */
export function CreateScreen(props: Props) {
  const [familyId, setFamilyId] = useState(() => loadDraft().family)
  return <CreateForm key={familyId} familyId={familyId} onFamily={setFamilyId} {...props} />
}

function CreateForm({
  familyId,
  onFamily,
  onOpenSession,
  onShowResults,
}: Props & { familyId: string; onFamily: (id: string) => void }) {
  const [draft] = useState(() => {
    const d = loadDraft()
    return { ...d.families[familyId], batchCount: d.batchCount, seedMode: d.seedMode }
  })
  const [chosenModel, setModel] = useState(draft.model ?? '')
  const [chosenMode, setMode] = useState(draft.mode)
  const [editedParams, setParams] = useState<Params | null>(null)
  const [loras, setLoras] = useState<LoraEntry[]>(draft.loras ?? [])
  const [source, setSource] = useState<Source | null>(draft.source ?? null)
  const [fit, setFit] = useState<Fit>(draft.fit ?? 'crop')
  const [extendsClip, setExtends] = useState(draft.extends ?? null)
  const [mask, setMask] = useState<MaskRef | null>(draft.mask ?? null)
  const [maskNote, setMaskNote] = useState<string | null>(null)
  const [painting, setPainting] = useState(false)
  const [chosenPlace, setPlace] = useState<Place | null>(draft.place ?? null)
  const [control, setControl] = useState<ControlUnit[]>(draft.control ?? [])
  const [refs, setRefs] = useState<Source[]>(draft.refs ?? [])
  const [editingUnit, setEditingUnit] = useState<string | null>(null)
  const [batchCount, setBatchCount] = useState(draft.batchCount)
  const [seedMode, setSeedMode] = useState<SeedMode>(draft.seedMode)
  const [queued, setQueued] = useState<number | null>(null)
  const [picker, setPicker] = useState<'model' | 'lora' | 'prompts' | 'image' | 'ref' | null>(null)
  // The image being cropped, and for a reference, its place in the Images row (one past the
  // end adds it).
  const [cropping, setCropping] = useState<{ sha: string; ref?: number } | null>(null)
  const [sourceGone, setSourceGone] = useState(false)
  const promptRef = useRef<HTMLTextAreaElement>(null)
  const promptFocused = useRef(false)

  const families = useQuery({ queryKey: ['families'], queryFn: api.families })
  const session = useQuery({ queryKey: ['session'], queryFn: api.session })
  const family = families.data?.find((f) => f.id === familyId) ?? families.data?.[0]
  const assets = useAssets()
  const models = assets.data?.filter(
    (a) => !!family && a.family === family.id && a.kind === 'model' && !!variantFor(a.path, family),
  )
  // The picker offers every family that makes the same media: choosing another family's model
  // switches to it.
  const siblings = families.data?.filter((f) => f.media === family?.media) ?? []
  const pickable = assets.data?.filter((a) => {
    const f = siblings.find((s) => s.id === a.family)
    return a.kind === 'model' && !!f && !!variantFor(a.path, f)
  })
  // A model remixed from an older image may have left Drive: keep it, flagged, rather
  // than silently swapping in another.
  const model = chosenModel || (models?.[0]?.path ?? '')
  const modelMissing = !!model && !!models && !models.some((m) => m.path === model)
  const variant = (family && model ? variantFor(model, family) : undefined) ?? family?.variants[0]
  const mode = chosenMode && variant?.modes.includes(chosenMode) ? chosenMode : variant?.modes[0]
  const needsSource = !!mode && SOURCE_MODES.has(mode)
  // Edits can read more images after the source; so does Qwen's inpaint, which is an edit.
  const maxRefs = variant?.max_refs ?? 0
  const takesRefs =
    maxRefs > 0 && (mode === 'edit' || (mode === 'inpaint' && !!variant?.modes.includes('edit')))

  const schema = useQuery({
    queryKey: ['schema', family?.id, variant?.id, mode],
    queryFn: () => api.schema(family?.id ?? '', variant?.id ?? '', mode ?? ''),
    enabled: !!family && !!variant && !!mode,
    staleTime: Infinity,
  })
  const loraIndex = assets.data?.filter((a) => a.family === family?.id && a.kind === 'lora')
  const choices = loraChoices(loraIndex, variant)
  const controlnets = assets.data?.filter((a) => a.family === family?.id && a.kind === 'controlnet')
  const withControl = !!family?.supports_control

  // The form always fits the current variant's schema: defaults, then what was typed.
  const params = schema.data ? initialParams(schema.data, editedParams ?? draft.params) : null

  useEffect(() => {
    if (!params || !mode) return
    saveFamilyDraft(
      familyId,
      {
        model,
        mode,
        loras,
        params,
        source,
        fit,
        extends: extendsClip,
        mask,
        place: chosenPlace,
        control,
        refs,
      },
      batchCount,
      seedMode,
    )
  }, [
    familyId,
    model,
    mode,
    loras,
    params,
    source,
    fit,
    extendsClip,
    mask,
    chosenPlace,
    control,
    refs,
    batchCount,
    seedMode,
  ])

  useEffect(() => {
    if (queued === null) return
    const id = setTimeout(() => {
      setQueued(null)
    }, 5000)
    return () => {
      clearTimeout(id)
    }
  }, [queued])

  const canvas = params ? { w: Number(params.width), h: Number(params.height) } : null
  const place =
    mode === 'outpaint' && source && canvas
      ? validPlace(chosenPlace, { w: source.width, h: source.height }, canvas)
        ? chosenPlace
        : defaultPlace({ w: source.width, h: source.height }, canvas)
      : null
  const maskFits = !!mask && !!source && mask.source === source.sha

  const submit = useMutation({
    mutationFn: () => {
      if (!family || !variant || !mode || !params) throw new Error('The form is still loading.')
      // One image uses the seed as set; a batch either counts up from it or ignores it.
      const randomSeeds = batchCount > 1 && seedMode === 'random' && 'seed' in params
      return api.submitJob(
        {
          family: family.id,
          variant: variant.id,
          mode,
          model: { path: model },
          loras,
          params: randomSeeds ? { ...params, seed: -1 } : params,
          ...(needsSource &&
            source && {
              inputs: {
                source: `sha256:${source.sha}`,
                fit,
                ...(extendsClip && { extends: extendsClip }),
                ...(mode === 'inpaint' && mask && { mask: `sha256:${mask.sha}` }),
                ...(mode === 'outpaint' && place && { place }),
                ...(takesRefs && refs.length > 0 && { refs: refs.map((r) => `sha256:${r.sha}`) }),
              },
            }),
          ...(withControl && control.length > 0 && { control: control.map(unitSpec) }),
        },
        batchCount,
        randomSeeds ? 'random' : 'increment',
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

  const video = family?.media === 'video'
  const hasPrompt = String(params.prompt ?? '').trim() !== ''
  const noGpu = session.data !== undefined && !isActive(session.data)
  const noun = (n: number) => (video ? (n === 1 ? 'clip' : 'clips') : n === 1 ? 'image' : 'images')
  const target = { w: Number(params.width), h: Number(params.height) }
  const sessionGpu = isActive(session.data) ? session.data?.session?.gpu : undefined
  const underpowered =
    !!sessionGpu && !!variant && GPUS.indexOf(sessionGpu) < GPUS.indexOf(variant.min_gpu)
  const sourceIgnored = !!source && !!variant && !variant.modes.some((m) => SOURCE_MODES.has(m))
  const misfit =
    !!source && Math.abs(source.width / source.height / (target.w / target.h) - 1) > 0.01
  const media = [...new Set(families.data.map((f) => f.media))]

  const insertTrigger = (word: string) => {
    const text = String(params.prompt ?? '')
    const el = promptRef.current
    // Before the prompt has been touched there's no cursor to honour: append.
    const at = promptFocused.current && el ? el.selectionEnd : text.length
    setParams({ ...params, prompt: insertWord(text, at, word) })
  }

  const takeSource = (image: BlobInfo, fromCrop: boolean) => {
    const w = image.width ?? target.w
    const h = image.height ?? target.h
    const previous = source
    setSource({ sha: image.sha256, width: w, height: h })
    setSourceGone(false)
    setExtends(null)
    setPlace(null)
    setMaskNote(null)
    // A crop to another shape sets the size: that's what the crop was for. An outpaint's
    // size is its canvas, which the source sits inside, so it stays.
    if (fromCrop && mode !== 'outpaint' && (w !== target.w || h !== target.h))
      setParams({ ...params, width: w, height: h })
    // A new crop of the same image carries the mask with it; another image drops it.
    if (!mask || image.sha256 === mask.source) return
    if (!fromCrop || previous?.sha !== mask.source) {
      setMask(null)
      return
    }
    api
      .remapMask(mask.sha, mask.source, image.sha256)
      .then((moved) => {
        if (moved.empty) {
          setMask(null)
          setMaskNote('The crop left out the whole mask.')
        } else {
          setMask({ sha: moved.sha256, source: image.sha256 })
          setMaskNote('The mask moved with the crop.')
        }
      })
      .catch(() => {
        setMask(null)
        setMaskNote('The mask couldn’t follow the crop. Paint it again.')
      })
  }

  const sourceRow = needsSource && (
    <div className={sourceGone ? 'source-row missing' : 'source-row'}>
      <button
        type="button"
        className="setting setting-button"
        onClick={() => {
          setPicker('image')
        }}
      >
        <span className="setting-label">{takesRefs ? 'Image 1' : 'Source'}</span>{' '}
        <span className={source ? 'setting-value' : 'setting-value none'}>
          {source ? (
            <>
              <img
                className="source-thumb"
                src={thumbUrl(source.sha)}
                alt=""
                onError={() => {
                  setSourceGone(true)
                }}
              />
              {size(source.width, source.height)}
            </>
          ) : (
            'Choose an image'
          )}
        </span>
      </button>
      {sourceGone && <p className="row-warning">This image is no longer stored. Choose another.</p>}
      {extendsClip && source && !sourceGone && (
        <p className="row-note">Continues a clip from its last frame.</p>
      )}
      {source && (
        <div className="row-buttons source-actions">
          <button
            type="button"
            className="btn quiet small"
            disabled={sourceGone}
            onClick={() => {
              setCropping({ sha: source.sha })
            }}
          >
            Crop
          </button>
          <button
            type="button"
            className="btn quiet small"
            onClick={() => {
              setSource(null)
              setExtends(null)
              setSourceGone(false)
            }}
          >
            Remove
          </button>
        </div>
      )}
    </div>
  )

  const leadingRows = (
    <>
      {sourceRow}
      {needsSource && source && misfit && mode !== 'outpaint' && (
        <div className="fit-row">
          <div className="setting">
            <label className="setting-label" htmlFor="fit">
              Fit
            </label>
            <select
              id="fit"
              value={fit}
              onChange={(e) => {
                setFit(e.target.value as Fit)
              }}
            >
              {FITS.map((f) => (
                <option key={f.id} value={f.id}>
                  {f.label}
                </option>
              ))}
            </select>
          </div>
          {fit === 'pad' && variant?.modes.includes('outpaint') && (
            <p className="row-note">
              <button
                type="button"
                className="link"
                onClick={() => {
                  setMode('outpaint')
                  setPlace(null)
                }}
              >
                Outpaint the bars
              </button>{' '}
              to draw what’s beyond the image instead.
            </p>
          )}
        </div>
      )}
      {mode === 'inpaint' && source && !sourceGone && (
        <div className="source-row">
          <button
            type="button"
            className="setting setting-button"
            onClick={() => {
              setPainting(true)
            }}
          >
            <span className="setting-label">Mask</span>{' '}
            <span className={maskFits ? 'setting-value' : 'setting-value none'}>
              {maskFits ? (
                <>
                  <MaskThumb source={source.sha} mask={mask.sha} />
                  Edit mask
                </>
              ) : (
                'Paint the area to redraw'
              )}
            </span>
          </button>
          {maskNote && <p className="row-note">{maskNote}</p>}
          {maskFits && (
            <div className="row-buttons source-actions">
              <button
                type="button"
                className="btn quiet small"
                onClick={() => {
                  setMask(null)
                  setMaskNote(null)
                }}
              >
                Clear
              </button>
            </div>
          )}
        </div>
      )}
      {mode === 'outpaint' && source && !sourceGone && place && (
        <PlaceEditor source={source} canvas={target} place={place} onChange={setPlace} />
      )}
      {takesRefs && (
        <RefList
          refs={refs}
          max={maxRefs}
          onChange={setRefs}
          onAdd={() => {
            setPicker('ref')
          }}
          onCrop={(i) => {
            const ref = refs[i]
            if (ref) setCropping({ sha: ref.sha, ref: i })
          }}
        />
      )}
      <div className={modelMissing ? 'model-row missing' : 'model-row'}>
        <button
          type="button"
          className="setting setting-button"
          aria-describedby={modelMissing ? 'model-missing' : undefined}
          onClick={() => {
            setPicker('model')
          }}
        >
          <span className="setting-label">Model</span>{' '}
          <span className={model ? 'setting-value' : 'setting-value none'}>
            {model ? assetLabel(model, models) : models ? 'None found' : 'Loading…'}
          </span>
        </button>
        {modelMissing && (
          <p className="row-warning" id="model-missing">
            Not found in Drive. Pick another model.
          </p>
        )}
        {variant?.model_dir &&
          model &&
          !modelMissing &&
          assetLabel(model, models) !== variant.label && (
            <p className="row-note">{variant.label}</p>
          )}
        {underpowered && (
          <p className="row-warning">
            Needs an {variant.min_gpu}; this {sessionGpu} session may run it slowly.
          </p>
        )}
        {sourceIgnored && (
          <p className="row-warning">
            This model can’t start from an image. Pick an image-to-video model to use the source.
          </p>
        )}
      </div>
      <LoraList
        loras={loras}
        index={loraIndex}
        onChange={setLoras}
        onAdd={() => {
          setPicker('lora')
        }}
        onTrigger={insertTrigger}
      />
      {withControl && (
        <ControlList
          units={control}
          index={controlnets}
          steps={Number(params.steps ?? 30)}
          onOpen={setEditingUnit}
          onAdd={() => {
            const unit = newUnit()
            setControl([...control, unit])
            setEditingUnit(unit.key)
          }}
        />
      )}
    </>
  )

  const modeChips = variant && variant.modes.length > 1 && (
    <div className="mode-chips" role="group" aria-label="Start from">
      {variant.modes.map((m) => (
        <button
          key={m}
          type="button"
          aria-pressed={m === mode}
          onClick={() => {
            setMode(m)
          }}
        >
          {MODE_LABELS[m] ?? m}
        </button>
      ))}
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
      {media.length > 1 && (
        <div className="segmented" role="group" aria-label="Make">
          {media.map((m) => (
            <button
              key={m}
              type="button"
              aria-pressed={m === family?.media}
              onClick={() => {
                const next = families.data.find((f) => f.media === m)
                if (!next || next.id === familyId) return
                switchFamily(next.id, String(params.prompt ?? ''))
                onFamily(next.id)
              }}
            >
              {MEDIA_LABELS[m]}
            </button>
          ))}
        </div>
      )}

      <SchemaForm
        schema={schema.data}
        values={params}
        onChange={setParams}
        presets={variant?.size_constraints.presets ?? []}
        leadingRows={leadingRows}
        afterPrompt={modeChips}
        promptPlaceholder={video ? 'Describe the clip' : 'Describe the picture'}
        promptRef={promptRef}
        seeds={{ batch: batchCount > 1, mode: seedMode, onMode: setSeedMode }}
        onPromptFocus={() => {
          promptFocused.current = true
        }}
        promptAside={
          <button
            type="button"
            className="prompt-aside"
            onClick={() => {
              setPicker('prompts')
            }}
          >
            Prompts
          </button>
        }
      />

      {picker === 'prompts' && (
        <PromptSheet
          prompt={String(params.prompt ?? '')}
          negativePrompt={String(params.negative_prompt ?? '')}
          family={familyId}
          onUse={(p) => {
            const next: Params = { ...params, prompt: p.prompt }
            if ('negative_prompt' in params) next.negative_prompt = p.negative_prompt
            setParams(next)
            setPicker(null)
          }}
          onClose={() => {
            setPicker(null)
          }}
        />
      )}

      {picker === 'model' && family && (
        <AssetPicker
          title="Model"
          noun="models"
          assets={pickable ?? []}
          selected={new Set([model])}
          describe={
            siblings.length > 1 || family.variants.length > 1
              ? (a) => {
                  const f = siblings.find((s) => s.id === a.family)
                  const v = f && variantFor(a.path, f)
                  return v?.model_dir ? v.label : siblings.length > 1 ? (f?.label ?? null) : null
                }
              : undefined
          }
          empty={
            <p>
              No models found. Put {video ? 'diffusers model folders' : 'checkpoints'} in Drive
              under{' '}
              <code>
                degas/models/{family.id}/{family.variants.length > 1 ? '<variant>/' : ''}
              </code>
              , then rescan.
            </p>
          }
          onPick={(a) => {
            if (a.family && a.family !== family.id) {
              switchFamily(a.family, String(params.prompt ?? ''), a.path)
              onFamily(a.family)
              return
            }
            const next = variantFor(a.path, family)
            if (next && next.lora_format !== variant?.lora_format) setLoras([])
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
          assets={choices.rows}
          selected={
            new Set(
              choices.rows
                .filter((row) => {
                  const entry = choices.entryFor(row)
                  const key = JSON.stringify(stripWeights(entry))
                  return loras.some((l) => JSON.stringify(stripWeights(l)) === key)
                })
                .map((row) => row.path),
            )
          }
          thumbs
          empty={
            <p>
              No LoRAs for this model. Put them in Drive under{' '}
              <code>degas/loras/{family?.id}/</code>, then rescan.
              {variant?.lora_format === 'paired_hi_lo' &&
                ' A14B LoRAs come in pairs named …_high_noise and …_low_noise.'}
            </p>
          }
          onPick={(a) => {
            const entry = choices.entryFor(a)
            const key = JSON.stringify(stripWeights(entry))
            if (!loras.some((l) => JSON.stringify(stripWeights(l)) === key)) {
              setLoras([...loras, entry])
            }
            setPicker(null)
          }}
          onClose={() => {
            setPicker(null)
          }}
        />
      )}
      {picker === 'image' && (
        <ImagePicker
          onUse={(image) => {
            takeSource(image, false)
            setPicker(null)
          }}
          onCrop={(image) => {
            setPicker(null)
            setCropping({ sha: image.sha256 })
          }}
          onClose={() => {
            setPicker(null)
          }}
        />
      )}
      {picker === 'ref' && (
        <ImagePicker
          onUse={(image) => {
            setRefs([
              ...refs,
              { sha: image.sha256, width: image.width ?? 0, height: image.height ?? 0 },
            ])
            setPicker(null)
          }}
          onCrop={(image) => {
            setPicker(null)
            setCropping({ sha: image.sha256, ref: refs.length })
          }}
          onClose={() => {
            setPicker(null)
          }}
        />
      )}
      {painting && source && (
        <MaskEditor
          source={source}
          mask={maskFits ? mask.sha : null}
          blur={Number(params.mask_blur ?? 0)}
          onDone={(painted) => {
            setMask(painted ? { sha: painted.sha256, source: source.sha } : null)
            setMaskNote(null)
            setPainting(false)
          }}
          onCancel={() => {
            setPainting(false)
          }}
        />
      )}
      {editingUnit && variant && (
        <UnitEditor
          unit={control.find((u) => u.key === editingUnit)}
          controlnets={controlnets ?? []}
          familyId={familyId}
          source={needsSource && !sourceGone ? source : null}
          target={target}
          steps={Number(params.steps ?? 30)}
          constraints={variant.size_constraints}
          onChange={(update) => {
            setControl((units) => units.map((u) => (u.key === editingUnit ? update(u) : u)))
          }}
          onRemove={() => {
            setControl((units) => units.filter((u) => u.key !== editingUnit))
            setEditingUnit(null)
          }}
          onClose={() => {
            // A unit left without an image or a model isn't worth keeping.
            setControl((units) =>
              units.filter((u) => u.key !== editingUnit || !!u.image || !!u.model),
            )
            setEditingUnit(null)
          }}
        />
      )}
      {cropping && variant && (
        <CropEditor
          sha={cropping.sha}
          target={target}
          constraints={variant.size_constraints}
          free={cropping.ref !== undefined}
          refsKeepSize={variant.ref_max_pixels != null}
          onApply={(image) => {
            const at = cropping.ref
            if (at === undefined) {
              takeSource(image, true)
            } else {
              const cropped = {
                sha: image.sha256,
                width: image.width ?? 0,
                height: image.height ?? 0,
              }
              setRefs([...refs.slice(0, at), cropped, ...refs.slice(at + 1)])
            }
            setCropping(null)
          }}
          onCancel={() => {
            setCropping(null)
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
                Queued {queued} {noun(queued)}.
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
                aria-label={`Fewer ${noun(2)}`}
                disabled={batchCount <= 1}
                onClick={() => {
                  setBatchCount((n) => Math.max(1, n - 1))
                }}
              >
                −
              </button>
              <output aria-label={`${video ? 'Clips' : 'Images'} per job`}>{batchCount}</output>
              <button
                type="button"
                aria-label={`More ${noun(2)}`}
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
              disabled={
                submit.isPending ||
                !model ||
                modelMissing ||
                !hasPrompt ||
                (needsSource && (!source || sourceGone)) ||
                (mode === 'inpaint' && !maskFits) ||
                (withControl && !control.every(unitReady))
              }
            >
              {submit.isPending
                ? 'Queuing…'
                : batchCount === 1
                  ? 'Generate'
                  : `Generate ${String(batchCount)} ${noun(batchCount)}`}
            </button>
          </div>
        </div>
      </div>
    </form>
  )
}

/** The control editor for a unit that's still in the list. */
function UnitEditor({
  unit,
  ...props
}: Omit<Parameters<typeof ControlEditor>[0], 'unit'> & { unit: ControlUnit | undefined }) {
  return unit ? <ControlEditor unit={unit} {...props} /> : null
}

/** A LoRA's files, without weights, to tell whether it's already in the list. */
function stripWeights(l: LoraEntry): unknown {
  return isPair(l) ? [l.high?.path, l.low?.path] : l.path
}
