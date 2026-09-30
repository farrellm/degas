import { useMutation, useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import {
  api,
  thumbUrl,
  type Asset,
  type BlobInfo,
  type Fit,
  type ImagePromptOptions,
  type Variant,
} from '../api'
import { assetLabel } from '../assets'
import type { Source } from '../draft'
import { size } from '../format'
import {
  DETAILS,
  FACEID_NOTE,
  MAX_PICTURES,
  detailInfo,
  adapterKind,
  anyOblong,
  isFaceid,
  mismatch,
  modelFor,
  takeInfo,
  takesFor,
  weightFor,
  type PromptUnit,
  type Take,
} from '../imagePrompt'
import { AssetPicker } from './AssetPicker'
import { StepRange } from './ControlEditor'
import { CropEditor } from './CropEditor'
import { ImagePicker } from './ImagePicker'
import { MaskEditor } from './MaskEditor'
import { MaskThumb } from './MaskThumb'
import { Sheet } from './Sheet'

const FITS: { id: Fit; label: string }[] = [
  { id: 'crop', label: 'The middle square' },
  { id: 'pad', label: 'All of it, letterboxed' },
]

const KIND_LABELS = {
  subject: 'Reads the whole picture',
  face: 'Reads faces',
  faceid: 'Reads who a face is (FaceID)',
  composition: 'Reads layout',
} as const

type Overlay = 'image' | 'crop' | 'area' | 'model' | null

interface Props {
  unit: PromptUnit
  /** The family's image prompt models in the Drive index. */
  adapters: Asset[]
  familyId: string
  /** A picture of the output's shape to paint the area over (Create's source), if any. */
  canvasOver: Source | null
  /** The form's output size and step count. */
  target: { w: number; h: number }
  steps: number
  constraints: Variant['size_constraints']
  /** What the family's image prompts can do. */
  options: ImagePromptOptions
  onChange: (update: (unit: PromptUnit) => PromptUnit) => void
  onRemove: () => void
  onClose: () => void
}

const toSource = (b: BlobInfo): Source => ({
  sha: b.sha256,
  width: b.width ?? 0,
  height: b.height ?? 0,
})

const round = (x: number) => Math.round(x * 1000) / 1000

/** A plain grey picture of the output's shape, to paint an area over when there's no source. */
async function blankCanvas(target: { w: number; h: number }): Promise<Source> {
  const canvas = document.createElement('canvas')
  canvas.width = target.w
  canvas.height = target.h
  const g = canvas.getContext('2d')
  if (g) {
    g.fillStyle = '#80868c'
    g.fillRect(0, 0, target.w, target.h)
  }
  const blob = await new Promise<Blob | null>((resolve) => {
    canvas.toBlob(resolve, 'image/png')
  })
  if (!blob) throw new Error('Couldn’t make a canvas to paint the area on.')
  return toSource(await api.upload(blob))
}

/**
 * One image prompt (docs/ip-adapter.md §4.6): its pictures, what to take from them, the
 * model, the weight, the steps it acts on, and the area of the output it's limited to. The
 * image picker, crop and area editors replace the sheet while they're open.
 */
export function PromptEditor({
  unit,
  adapters,
  familyId,
  canvasOver,
  target,
  steps,
  constraints,
  options,
  onChange,
  onRemove,
  onClose,
}: Props) {
  const takes = takesFor(options)
  const [overlay, setOverlay] = useState<Overlay>(null)
  // The picture being cropped, and its place in the row (one past the end adds it).
  const [cropping, setCropping] = useState<{ sha: string; at: number } | null>(null)
  const [areaOver, setAreaOver] = useState<Source | null>(null)
  const session = useQuery({ queryKey: ['session'], queryFn: api.session })
  const gpuReady = ['ready', 'busy'].includes(session.data?.session?.state ?? '')
  const faceid = isFaceid(unit.model)

  // The area is painted over a picture of the output's shape: the one it was painted over,
  // Create's source when it has that shape, else a blank canvas.
  const openArea = useMutation({
    mutationFn: async () => {
      const shaped = (s: Source | null | undefined): s is Source =>
        !!s && s.width === target.w && s.height === target.h && s.sha !== unit.area?.sha
      const over = unit.area?.over
      if (shaped(over)) return over
      if (shaped(canvasOver)) return canvasOver
      return blankCanvas(target)
    },
    onSuccess: (over) => {
      setAreaOver(over)
      setOverlay('area')
    },
  })

  const putPicture = (picture: Source, at: number) => {
    onChange((u) => ({
      ...u,
      pictures: [...u.pictures.slice(0, at), picture, ...u.pictures.slice(at + 1)],
    }))
  }

  const choose = (take: Take) => {
    onChange((u) => {
      const current = adapters.find((a) => a.path === u.model)
      const keep = !!u.model && !!current && !mismatch(current, u.model, take)
      const model = keep ? u.model : (modelFor(adapters, take)?.path ?? u.model)
      return { ...u, take, model, weight: weightFor(take, model) }
    })
  }

  if (overlay === 'image') {
    return (
      <ImagePicker
        onUse={(image) => {
          setOverlay(null)
          putPicture(toSource(image), unit.pictures.length)
        }}
        onCrop={(image) => {
          setCropping({ sha: image.sha256, at: unit.pictures.length })
          setOverlay('crop')
        }}
        onClose={() => {
          setOverlay(null)
        }}
      />
    )
  }
  if (overlay === 'crop' && cropping) {
    return (
      <CropEditor
        sha={cropping.sha}
        target={target}
        constraints={constraints}
        square
        onApply={(image) => {
          putPicture(toSource(image), cropping.at)
          setOverlay(null)
        }}
        onCancel={() => {
          setOverlay(null)
        }}
      />
    )
  }
  if (overlay === 'area' && areaOver) {
    return (
      <MaskEditor
        source={areaOver}
        mask={
          // An area is kept when the picture it's painted over has the same size.
          unit.area?.over.width === areaOver.width && unit.area.over.height === areaOver.height
            ? unit.area.sha
            : null
        }
        blur={0}
        title="Area"
        onDone={(painted) => {
          onChange((u) => ({
            ...u,
            area: painted ? { sha: painted.sha256, over: areaOver } : null,
          }))
          setOverlay(null)
        }}
        onCancel={() => {
          setOverlay(null)
        }}
      />
    )
  }
  if (overlay === 'model') {
    return (
      <AssetPicker
        title="Image prompt model"
        noun="models"
        assets={adapters}
        selected={new Set([unit.model])}
        describe={(a) => KIND_LABELS[adapterKind(a)]}
        empty={
          <p>
            No image prompt models found. Put{' '}
            {options.detail ? 'the FLUX.1-Redux-dev diffusers folder' : 'IP-Adapters'} in Drive
            under <code>degas/ip_adapters/{familyId}/</code>, then rescan.
          </p>
        }
        onPick={(a) => {
          onChange((u) => ({ ...u, model: a.path }))
          setOverlay(null)
        }}
        onClose={() => {
          setOverlay(null)
        }}
      />
    )
  }

  const model = adapters.find((a) => a.path === unit.model)
  const modelMissing = !!unit.model && !model
  const warning = unit.model ? mismatch(model, unit.model, unit.take) : null
  const span = { a: Math.round(unit.start * steps), b: Math.round(unit.end * steps) }
  const setSpan = (a: number, b: number) => {
    onChange((u) => ({ ...u, start: round(a / steps), end: round(b / steps) }))
  }

  return (
    <Sheet title="Image prompt" onClose={onClose}>
      <div className="prompt-editor">
        <div className="prompt-pictures" role="group" aria-label="Pictures">
          {unit.pictures.map((picture, i) => {
            const n = String(i + 1)
            return (
              <figure key={`${picture.sha}-${n}`} className="prompt-picture">
                {/* Square, because the model sees the middle square. */}
                {faceid ? (
                  <FaceTile sha={picture.sha} n={n} gpuReady={gpuReady} />
                ) : (
                  <img src={thumbUrl(picture.sha)} alt={`Picture ${n}`} />
                )}
                <figcaption className="row-buttons">
                  <button
                    type="button"
                    className="btn quiet small"
                    aria-label={`Crop picture ${n}`}
                    onClick={() => {
                      setCropping({ sha: picture.sha, at: i })
                      setOverlay('crop')
                    }}
                  >
                    Crop
                  </button>
                  <button
                    type="button"
                    className="lora-remove"
                    aria-label={`Remove picture ${n}`}
                    onClick={() => {
                      onChange((u) => ({ ...u, pictures: u.pictures.filter((_, j) => j !== i) }))
                    }}
                  >
                    <svg viewBox="0 0 12 12" aria-hidden>
                      <path d="M2 2l8 8M10 2l-8 8" />
                    </svg>
                  </button>
                </figcaption>
              </figure>
            )
          })}
          {unit.pictures.length < MAX_PICTURES && (
            <button
              type="button"
              className="prompt-add"
              onClick={() => {
                setOverlay('image')
              }}
            >
              {unit.pictures.length === 0 ? 'Choose a picture' : 'Add picture'}
            </button>
          )}
        </div>
        {faceid && unit.pictures.length > 0 && !gpuReady && (
          <p className="row-note">The face in each picture is found on the GPU when it runs.</p>
        )}
        <p className="row-note">
          {faceid && unit.pictures.length > 0
            ? 'The model reads who the face is, not the rest of the picture.'
            : unit.pictures.length === 0
              ? 'The image will take after this picture, the way it takes after the prompt.'
              : unit.pictures.length > 1
                ? 'The model reads them together.'
                : 'The model sees the middle square, at low resolution: fine detail and text don’t carry.'}
        </p>

        {takes.length > 0 && (
          <div className="control-traces" role="group" aria-label="Take from it">
            {takes.map((t) => (
              <button
                key={t.id}
                type="button"
                aria-pressed={unit.take === t.id}
                onClick={() => {
                  choose(t.id)
                }}
              >
                {t.label}
              </button>
            ))}
          </div>
        )}
        {takes.length > 0 && (
          <p className="row-note">
            {unit.take === 'face' && faceid ? FACEID_NOTE : takeInfo(unit.take).note}
          </p>
        )}
        {options.detail && (
          <>
            <div className="control-traces" role="group" aria-label="How closely">
              {DETAILS.map((d) => (
                <button
                  key={d.downsample}
                  type="button"
                  aria-pressed={unit.downsample === d.downsample}
                  onClick={() => {
                    onChange((u) => ({ ...u, downsample: d.downsample }))
                  }}
                >
                  {d.label}
                </button>
              ))}
            </div>
            <p className="row-note">{detailInfo(unit.downsample).note}</p>
          </>
        )}

        <div className={modelMissing ? 'model-row missing' : 'model-row'}>
          <button
            type="button"
            className="setting setting-button"
            onClick={() => {
              setOverlay('model')
            }}
          >
            <span className="setting-label">Model</span>{' '}
            <span className={unit.model ? 'setting-value' : 'setting-value none'}>
              {unit.model ? assetLabel(unit.model, adapters) : 'Choose a model'}
            </span>
          </button>
          {modelMissing && <p className="row-warning">Not found in Drive. Pick another model.</p>}
          {warning && !modelMissing && <p className="row-warning">{warning}</p>}
        </div>

        <div className="setting">
          <label className="setting-label" htmlFor="prompt-weight">
            Weight
          </label>
          <div className="slider-control">
            <input
              id="prompt-weight"
              type="range"
              min={0}
              max={2}
              step={0.05}
              value={unit.weight}
              onChange={(e) => {
                const weight = Number(e.target.value)
                onChange((u) => ({ ...u, weight }))
              }}
            />
            <output htmlFor="prompt-weight">{unit.weight.toFixed(2)}</output>
          </div>
        </div>

        {faceid && (
          <>
            <Slider
              id="prompt-structure"
              label="Face structure"
              min={0}
              max={2}
              value={unit.structure}
              note="How much of the face’s shape comes from the picture, on top of who it is."
              onChange={(structure) => {
                onChange((u) => ({ ...u, structure }))
              }}
            />
            <Slider
              id="prompt-lora"
              label="Face LoRA"
              min={0}
              max={1.5}
              value={unit.loraWeight}
              note="The LoRA trained with the model. Lower lets the checkpoint’s own look through."
              onChange={(loraWeight) => {
                onChange((u) => ({ ...u, loraWeight }))
              }}
            />
          </>
        )}

        {options.steps && (
          <StepRange
            steps={steps}
            a={span.a}
            b={span.b}
            onChange={setSpan}
            note="Ending early keeps its influence on the big shapes and leaves the details to the prompt."
          />
        )}

        {options.areas && (
          <div className="source-row">
            <button
              type="button"
              className="setting setting-button"
              disabled={openArea.isPending}
              onClick={() => {
                openArea.mutate()
              }}
            >
              <span className="setting-label">Area</span>{' '}
              <span className={unit.area ? 'setting-value' : 'setting-value none'}>
                {unit.area ? (
                  <>
                    <MaskThumb source={unit.area.over.sha} mask={unit.area.sha} />
                    Edit area
                  </>
                ) : (
                  'Limit to an area'
                )}
              </span>
            </button>
            {openArea.error && (
              <p className="row-warning" role="alert">
                {openArea.error.message}
              </p>
            )}
            {unit.area && (
              <div className="row-buttons source-actions">
                <button
                  type="button"
                  className="btn quiet small"
                  onClick={() => {
                    onChange((u) => ({ ...u, area: null }))
                  }}
                >
                  Clear
                </button>
              </div>
            )}
            {unit.area &&
              (unit.area.over.width !== target.w || unit.area.over.height !== target.h) && (
                <p className="row-note">
                  Painted at {size(unit.area.over.width, unit.area.over.height)}; it will be fitted
                  to {size(target.w, target.h)}.
                </p>
              )}
          </div>
        )}

        {anyOblong(unit) && (
          <div className="setting">
            <label className="setting-label" htmlFor="prompt-fit">
              Show it
            </label>
            <select
              id="prompt-fit"
              value={unit.fit}
              onChange={(e) => {
                const fit = e.target.value as Fit
                onChange((u) => ({ ...u, fit }))
              }}
            >
              {FITS.map((f) => (
                <option key={f.id} value={f.id}>
                  {f.label}
                </option>
              ))}
            </select>
          </div>
        )}

        <div className="row-buttons control-remove">
          <button type="button" className="btn danger small" onClick={onRemove}>
            Remove this image prompt
          </button>
        </div>
      </div>
    </Sheet>
  )
}

/**
 * A FaceID picture: once the GPU has found its face, the aligned crop the model reads, which
 * shows whether it found the right one.
 */
function FaceTile({ sha, n, gpuReady }: { sha: string; n: string; gpuReady: boolean }) {
  const face = useQuery({
    queryKey: ['face', sha],
    queryFn: () => api.findFace(sha),
    enabled: gpuReady,
    retry: false,
    staleTime: Infinity,
  })
  if (face.data) {
    const others = face.data.faces - 1
    return (
      <>
        <img src={thumbUrl(face.data.image.sha256)} alt={`The face in picture ${n}`} />
        {others > 0 && (
          <span className="prompt-picture-note">
            The biggest of {String(face.data.faces)} faces
          </span>
        )}
      </>
    )
  }
  return (
    <>
      <img
        src={thumbUrl(sha)}
        alt={`Picture ${n}`}
        className={face.isFetching ? 'finding' : undefined}
      />
      {face.error && (
        <span className="prompt-picture-note warn" role="alert">
          {face.error.message}
        </span>
      )}
    </>
  )
}

function Slider({
  id,
  label,
  min,
  max,
  value,
  note,
  onChange,
}: {
  id: string
  label: string
  min: number
  max: number
  value: number
  note: string
  onChange: (value: number) => void
}) {
  return (
    <>
      <div className="setting">
        <label className="setting-label" htmlFor={id}>
          {label}
        </label>
        <div className="slider-control">
          <input
            id={id}
            type="range"
            min={min}
            max={max}
            step={0.05}
            value={value}
            onChange={(e) => {
              onChange(Number(e.target.value))
            }}
          />
          <output htmlFor={id}>{value.toFixed(2)}</output>
        </div>
      </div>
      <p className="row-note slider-note">{note}</p>
    </>
  )
}
