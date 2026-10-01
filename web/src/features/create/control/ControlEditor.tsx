import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { queries } from '@/api/queries'
import type { Asset, Variant } from '@/api/types'
import { blobUrl } from '@/api/urls'
import { AssetPicker } from '@/components/AssetPicker/AssetPicker'
import { FitSelect } from '@/components/FitSelect'
import { ImagePicker } from '@/components/ImagePicker/ImagePicker'
import { Sheet } from '@/components/Sheet'
import { SliderRow } from '@/components/SliderRow'
import { StepRange } from '@/components/StepRange'
import { CropEditor } from '@/features/editors/crop/CropEditor'
import { MaskEditor } from '@/features/editors/mask/MaskEditor'
import { MaskThumb } from '@/features/editors/mask/MaskThumb'
import { assetLabel } from '@/lib/assets'
import { ratioDiffers, type Size } from '@/lib/geometry'
import { type Source, toSource } from '@/lib/image'
import { isActive, isGpuReady } from '@/lib/session'
import { stepFraction } from '@/lib/steps'

import {
  controlKind,
  type ControlUnit,
  edgeParams,
  mismatch,
  photoOf,
  sameSize,
  traceInfo,
  TRACES,
  underlay,
} from './control'
import { useControlTrace } from './useControlTrace'

type Overlay = 'image' | 'crop' | 'area' | 'model' | null

export interface ControlEditorProps {
  unit: ControlUnit
  /** The family's ControlNets in the Drive index. */
  controlnets: Asset[]
  familyId: string
  /** Create's source image, offered as the picture to trace. */
  source: Source | null
  /** The form's output size and step count. */
  target: Size
  steps: number
  constraints: Variant['size_constraints']
  onChange: (update: (unit: ControlUnit) => ControlUnit) => void
  onRemove: () => void
  onClose: () => void
}

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
}: ControlEditorProps) {
  const [overlay, setOverlay] = useState<Overlay>(null)
  const [cropping, setCropping] = useState<string | null>(null)
  const [overPhoto, setOverPhoto] = useState(false)
  const session = useQuery(queries.session())
  const gpuReady = isGpuReady(session.data)

  const { busy, error, note, clearNote, cannyTrace, detail, setDetail, trace, takePicture } =
    useControlTrace({ unit, controlnets, onChange })

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
          clearNote()
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
    ratioDiffers(unit.image.width / unit.image.height, target.w / target.h)
  const photoUnder = underlay(unit)
  const canOverlay = !!unit.trace && !!photoUnder && photoUnder !== unit.image?.sha
  const chosen = unit.trace?.id ?? 'as-is'
  const span = { a: Math.round(unit.start * steps), b: Math.round(unit.end * steps) }
  const setSpan = (a: number, b: number) => {
    onChange((u) => ({ ...u, start: stepFraction(a, steps), end: stepFraction(b, steps) }))
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
                trace({ id: t.id, params, from: photo })
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
          <SliderRow
            id="edge-detail"
            label="Detail"
            min={0}
            max={1}
            value={detail}
            format={(v) => `${String(Math.round(v * 100))}%`}
            onChange={setDetail}
          />
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

        <SliderRow
          id="control-weight"
          label="Weight"
          min={0}
          max={2}
          value={unit.scale}
          onChange={(scale) => {
            onChange((u) => ({ ...u, scale }))
          }}
        />

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
          <FitSelect
            id="control-fit"
            value={unit.fit}
            onChange={(fit) => {
              onChange((u) => ({ ...u, fit }))
            }}
          />
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
