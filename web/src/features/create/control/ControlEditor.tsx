import { useMutation, useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import {
  api,
  blobUrl,
  isActive,
  type Asset,
  type BlobInfo,
  type Fit,
  type Params,
  type TraceId,
  type Variant,
} from '../api'
import { assetLabel } from '../assets'
import {
  DEFAULT_EDGES,
  TRACES,
  controlKind,
  edgeDetail,
  edgeParams,
  matchingModel,
  mismatch,
  stepSpan,
  stepsLabel,
  traceInfo,
  underlay,
  type ControlUnit,
} from '../control'
import type { Source } from '../draft'
import { AssetPicker } from './AssetPicker'
import { CropEditor } from './CropEditor'
import { ImagePicker } from './ImagePicker'
import { MaskEditor } from './MaskEditor'
import { MaskThumb } from './MaskThumb'
import { Sheet } from './Sheet'

const FITS: { id: Fit; label: string }[] = [
  { id: 'crop', label: 'Crop to fit' },
  { id: 'pad', label: 'Letterbox' },
  { id: 'stretch', label: 'Stretch' },
]

// How long the Detail slider rests before the edges are traced again.
const RETRACE_MS = 400

type Overlay = 'image' | 'crop' | 'area' | 'model' | null

interface Props {
  unit: ControlUnit
  /** The family's ControlNets in the Drive index. */
  controlnets: Asset[]
  familyId: string
  /** Create's source image, offered as the picture to trace. */
  source: Source | null
  /** The form's output size and step count. */
  target: { w: number; h: number }
  steps: number
  constraints: Variant['size_constraints']
  onChange: (update: (unit: ControlUnit) => ControlUnit) => void
  onRemove: () => void
  onClose: () => void
}

const toSource = (b: BlobInfo): Source => ({
  sha: b.sha256,
  width: b.width ?? 0,
  height: b.height ?? 0,
})

const sameSize = (a: Source | null, b: Source | null) =>
  !!a && !!b && a.width === b.width && a.height === b.height

/** The picture a unit's image came from: what it was traced from, or itself. */
const photoOf = (unit: ControlUnit) => unit.trace?.from ?? unit.image

/**
 * One ControlNet unit (design §8.2, ux.md Phase 7): the picture and its trace, the model,
 * the weight, the steps it guides, and the area it's limited to. The image picker, crop and
 * area editors replace the sheet while they're open.
 */
export function ControlEditor({
  unit,
  controlnets,
  familyId,
  source,
  target,
  steps,
  constraints,
  onChange,
  onRemove,
  onClose,
}: Props) {
  const [overlay, setOverlay] = useState<Overlay>(null)
  const [cropping, setCropping] = useState<string | null>(null)
  const [note, setNote] = useState<string | null>(null)
  const [overPhoto, setOverPhoto] = useState(false)
  const [detail, setDetail] = useState(() =>
    unit.trace?.id === 'canny' ? edgeDetail(unit.trace.params) : edgeDetail(DEFAULT_EDGES),
  )
  const session = useQuery({ queryKey: ['session'], queryFn: api.session })
  const gpuReady = ['ready', 'busy'].includes(session.data?.session?.state ?? '')

  // Trace the unit's picture. The area stays when the trace is the size it was painted at.
  const trace = useMutation({
    mutationFn: async (v: { id: TraceId; params: Params; from: Source }) => {
      const out = await api.trace(v.id, v.from.sha, v.params)
      return { ...v, image: toSource(out.image) }
    },
    onMutate: () => {
      setNote(null)
    },
    onSuccess: (r) => {
      if (unit.area && !sameSize(unit.image, r.image))
        setNote('The new trace is another size, so the area was cleared.')
      onChange((u) => ({
        ...u,
        image: r.image,
        trace: { id: r.id, params: r.params, from: r.from },
        area: sameSize(u.image, r.image) ? u.area : null,
        model: u.model || (matchingModel(controlnets, r.id)?.path ?? ''),
      }))
    },
  })

  // A new picture (picked, cropped, or Create's source): the area follows a crop of the
  // same picture, and an existing trace is made again from the new picture.
  const repicture = useMutation({
    mutationFn: async (v: { photo: Source; before: ControlUnit; cropped: boolean }) => {
      const { photo, before, cropped } = v
      let area: string | null = null
      let message: string | null = null
      const old = photoOf(before)
      const areaOnPhoto = !!old && underlay(before) === old.sha
      if (before.area && cropped && old && areaOnPhoto) {
        try {
          const moved = await api.remapMask(before.area, old.sha, photo.sha)
          if (moved.empty) message = 'The crop left out the whole area.'
          else {
            area = moved.sha256
            message = 'The area moved with the crop.'
          }
        } catch {
          message = 'The area couldn’t follow the crop. Paint it again.'
        }
      } else if (before.area) {
        message = 'A new picture clears the area.'
      }
      if (!before.trace) return { image: photo, trace: null, area, message }
      try {
        const out = await api.trace(before.trace.id, photo.sha, before.trace.params)
        return {
          image: toSource(out.image),
          trace: { ...before.trace, from: photo },
          area,
          message,
        }
      } catch (e) {
        const why = e instanceof Error ? e.message : String(e)
        return { image: photo, trace: null, area, message: `Using the picture as it is. ${why}` }
      }
    },
    onMutate: () => {
      setNote(null)
    },
    onSuccess: (r) => {
      onChange((u) => ({ ...u, image: r.image, trace: r.trace, area: r.area }))
      setNote(r.message)
    },
  })

  const busy = trace.isPending || repicture.isPending
  const error = trace.error ?? repicture.error

  // Edges: trace again once the Detail slider rests.
  const cannyTrace = unit.trace?.id === 'canny' ? unit.trace : null
  useEffect(() => {
    if (!cannyTrace || edgeDetail(cannyTrace.params) === detail) return
    const id = setTimeout(() => {
      trace.mutate({ id: 'canny', params: edgeParams(detail), from: cannyTrace.from })
    }, RETRACE_MS)
    return () => {
      clearTimeout(id)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- the mutation object changes each render
  }, [detail, cannyTrace])

  const takePicture = (photo: Source, cropped: boolean) => {
    repicture.mutate({ photo, before: unit, cropped })
  }

  if (overlay === 'image') {
    return (
      <ImagePicker
        onUse={(image) => {
          setOverlay(null)
          takePicture(toSource(image), false)
        }}
        onCrop={(image) => {
          setCropping(image.sha256)
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
        sha={cropping}
        target={target}
        constraints={constraints}
        onApply={(image) => {
          const photo = photoOf(unit)
          setOverlay(null)
          takePicture(toSource(image), cropping === photo?.sha)
        }}
        onCancel={() => {
          setOverlay(null)
        }}
      />
    )
  }
  if (overlay === 'area' && unit.image) {
    const under = underlay(unit)
    return (
      <MaskEditor
        source={unit.image}
        mask={unit.area}
        blur={0}
        underlay={under && under !== unit.image.sha ? under : undefined}
        title="Area"
        onDone={(painted) => {
          onChange((u) => ({ ...u, area: painted?.sha256 ?? null }))
          setNote(null)
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
        title="ControlNet"
        noun="ControlNets"
        assets={controlnets}
        selected={new Set([unit.model])}
        describe={(a) => {
          const kind = controlKind(a)
          return kind ? `Reads ${traceInfo(kind).image}` : null
        }}
        empty={
          <p>
            No ControlNets found. Put them in Drive under <code>degas/controlnets/{familyId}/</code>
            , then rescan.
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

  const photo = photoOf(unit)
  const model = controlnets.find((a) => a.path === unit.model)
  const modelMissing = !!unit.model && !model
  const warning = unit.image && mismatch(model, unit.model, unit.trace?.id ?? null)
  const misfit =
    !!unit.image &&
    unit.image.height > 0 &&
    Math.abs(unit.image.width / unit.image.height / (target.w / target.h) - 1) > 0.01
  const photoUnder = underlay(unit)
  const canOverlay = !!unit.trace && !!photoUnder && photoUnder !== unit.image?.sha
  const chosen = unit.trace?.id ?? 'as-is'
  const span = { a: Math.round(unit.start * steps), b: Math.round(unit.end * steps) }
  const setSpan = (a: number, b: number) => {
    onChange((u) => ({ ...u, start: round(a / steps), end: round(b / steps) }))
  }

  return (
    <Sheet title="ControlNet" onClose={onClose}>
      <div className="control-editor">
        {unit.image ? (
          <div className="control-preview">
            <div
              className={busy ? 'control-picture sketch indeterminate' : 'control-picture'}
              style={{
                aspectRatio: `${String(unit.image.width || 1)} / ${String(unit.image.height || 1)}`,
              }}
            >
              {overPhoto && canOverlay && photoUnder && (
                <img src={blobUrl(photoUnder)} alt="" className="control-photo" />
              )}
              <img
                src={blobUrl(unit.image.sha)}
                alt={unit.trace ? `${traceInfo(unit.trace.id).label} trace` : 'Control image'}
                className={overPhoto && canOverlay ? 'control-trace over' : 'control-trace'}
              />
            </div>
            <div className="row-buttons control-picture-actions">
              {canOverlay && (
                <button
                  type="button"
                  className="btn quiet small"
                  aria-pressed={overPhoto}
                  onClick={() => {
                    setOverPhoto(!overPhoto)
                  }}
                >
                  Over the picture
                </button>
              )}
              <span className="spacer" />
              <button
                type="button"
                className="btn quiet small"
                disabled={busy || !photo}
                onClick={() => {
                  if (!photo) return
                  setCropping(photo.sha)
                  setOverlay('crop')
                }}
              >
                Crop
              </button>
              <button
                type="button"
                className="btn quiet small"
                disabled={busy}
                onClick={() => {
                  setOverlay('image')
                }}
              >
                Change
              </button>
            </div>
          </div>
        ) : (
          <div className="control-empty">
            <p>Choose the picture whose layout the image should follow.</p>
            <div className="row-buttons">
              {source && (
                <button
                  type="button"
                  className="btn small"
                  disabled={busy}
                  onClick={() => {
                    takePicture(source, false)
                  }}
                >
                  Use the source
                </button>
              )}
              <button
                type="button"
                className={source ? 'btn quiet small' : 'btn small'}
                disabled={busy}
                onClick={() => {
                  setOverlay('image')
                }}
              >
                Choose image
              </button>
            </div>
          </div>
        )}
        {unit.image && source && photo?.sha !== source.sha && (
          <p className="row-note">
            <button
              type="button"
              className="link"
              disabled={busy}
              onClick={() => {
                takePicture(source, false)
              }}
            >
              Use the source
            </button>{' '}
            instead of this picture.
          </p>
        )}

        <div className="control-traces" role="group" aria-label="Trace">
          <button
            type="button"
            aria-pressed={!!unit.image && chosen === 'as-is'}
            disabled={!unit.image || busy}
            onClick={() => {
              if (!unit.trace) return
              const from = unit.trace.from
              onChange((u) => ({
                ...u,
                image: from,
                trace: null,
                area: sameSize(u.image, from) ? u.area : null,
              }))
            }}
          >
            As is
          </button>
          {TRACES.map((t) => (
            <button
              key={t.id}
              type="button"
              aria-pressed={chosen === t.id}
              disabled={!photo || busy || !gpuReady}
              onClick={() => {
                if (!photo) return
                const params = t.id === 'canny' ? edgeParams(detail) : {}
                trace.mutate({ id: t.id, params, from: photo })
              }}
            >
              {t.label}
            </button>
          ))}
        </div>
        {busy && (
          <p className="row-note" role="status">
            Tracing on the GPU…
          </p>
        )}
        {!gpuReady && photo && (
          <p className="row-note">
            {isActive(session.data)
              ? 'Tracing works once the GPU session is ready.'
              : 'Tracing runs on the GPU. Start a session to trace depth, poses or edges.'}
          </p>
        )}
        {error && (
          <p className="row-warning" role="alert">
            {error.message}
          </p>
        )}
        {note && <p className="row-note">{note}</p>}
        {cannyTrace && (
          <div className="setting">
            <label className="setting-label" htmlFor="edge-detail">
              Detail
            </label>
            <div className="slider-control">
              <input
                id="edge-detail"
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={detail}
                onChange={(e) => {
                  setDetail(Number(e.target.value))
                }}
              />
              <output htmlFor="edge-detail">{Math.round(detail * 100)}%</output>
            </div>
          </div>
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
              {unit.model ? assetLabel(unit.model, controlnets) : 'Choose a ControlNet'}
            </span>
          </button>
          {modelMissing && (
            <p className="row-warning">Not found in Drive. Pick another ControlNet.</p>
          )}
          {warning && !modelMissing && <p className="row-warning">{warning}</p>}
        </div>

        <div className="setting">
          <label className="setting-label" htmlFor="control-weight">
            Weight
          </label>
          <div className="slider-control">
            <input
              id="control-weight"
              type="range"
              min={0}
              max={2}
              step={0.05}
              value={unit.scale}
              onChange={(e) => {
                const scale = Number(e.target.value)
                onChange((u) => ({ ...u, scale }))
              }}
            />
            <output htmlFor="control-weight">{unit.scale.toFixed(2)}</output>
          </div>
        </div>

        <StepRange steps={steps} a={span.a} b={span.b} onChange={setSpan} />

        {unit.image && (
          <div className="source-row">
            <button
              type="button"
              className="setting setting-button"
              onClick={() => {
                setOverlay('area')
              }}
            >
              <span className="setting-label">Area</span>{' '}
              <span className={unit.area ? 'setting-value' : 'setting-value none'}>
                {unit.area ? (
                  <>
                    <MaskThumb source={photoUnder ?? unit.image.sha} mask={unit.area} />
                    Edit area
                  </>
                ) : (
                  'Limit to an area'
                )}
              </span>
            </button>
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
          </div>
        )}

        {misfit && (
          <div className="setting">
            <label className="setting-label" htmlFor="control-fit">
              Fit
            </label>
            <select
              id="control-fit"
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
            Remove this ControlNet
          </button>
        </div>
      </div>
    </Sheet>
  )
}

const round = (x: number) => Math.round(x * 1000) / 1000

/**
 * The steps a unit guides, as two thumbs on one track counted in the form's steps. Early
 * steps set the layout, so ending early leaves the details free.
 */
export function StepRange({
  steps,
  a,
  b,
  onChange,
  note = 'Early steps set the layout; ending early leaves the details free.',
}: {
  steps: number
  a: number
  b: number
  onChange: (a: number, b: number) => void
  /** What the range is for, under the track. */
  note?: string
}) {
  const n = Math.max(1, steps)
  const pct = (x: number) => `${String((x / n) * 100)}%`
  const span = stepSpan(a / n, b / n, n)
  return (
    <div className="setting stacked step-range" role="group" aria-labelledby="step-range-label">
      <span className="setting-label" id="step-range-label">
        Steps
      </span>
      <output className="readout" aria-live="polite">
        {stepsLabel(a / n, b / n, n).replace(/^Steps? /, '')}
      </output>
      <div className="setting-control">
        <div className="step-track">
          <span
            className={span ? 'step-span' : 'step-span none'}
            style={{ left: pct(a), right: `calc(100% - ${pct(b)})` }}
          />
          <input
            type="range"
            aria-label="First step"
            min={0}
            max={n}
            step={1}
            value={a}
            onChange={(e) => {
              onChange(Math.min(Number(e.target.value), b - 1), b)
            }}
          />
          <input
            type="range"
            aria-label="Last step"
            min={0}
            max={n}
            step={1}
            value={b}
            onChange={(e) => {
              onChange(a, Math.max(Number(e.target.value), a + 1))
            }}
          />
        </div>
        <p className="row-note">{note}</p>
      </div>
    </div>
  )
}
