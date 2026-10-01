import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { queries } from '@/api/queries'
import type { Asset, Variant } from '@/api/types'
import { FitSelect } from '@/components/FitSelect'
import { ImagePicker } from '@/components/ImagePicker/ImagePicker'
import { Sheet } from '@/components/Sheet'
import { SliderRow } from '@/components/SliderRow'
import { StepRange } from '@/components/StepRange'
import { AreaRow } from '@/features/create/rows/AreaRow'
import { UnitModelRow } from '@/features/create/rows/UnitModelRow'
import { CropEditor } from '@/features/editors/crop/CropEditor'
import { MaskEditor } from '@/features/editors/mask/MaskEditor'
import { assetLabel } from '@/lib/assets'
import { ratioDiffers, type Size } from '@/lib/geometry'
import { type Source, toSource } from '@/lib/image'
import { isActive, isGpuReady } from '@/lib/session'
import { stepFraction } from '@/lib/steps'

import { type ControlUnit, edgeParams, mismatch, photoOf, sameSize, underlay } from './control'
import { ControlNetPicker } from './ControlNetPicker'
import { ControlPicture } from './ControlPicture'
import { TracePicker } from './TracePicker'
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

  const { busy, error, note, clearNote, detail, setDetail, trace, takePicture } = useControlTrace({
    unit,
    controlnets,
    onChange,
  })

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
      <ControlNetPicker
        controlnets={controlnets}
        familyId={familyId}
        selected={unit.model}
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

  const model = controlnets.find((a) => a.path === unit.model)
  const warning = unit.image && mismatch(model, unit.model, unit.trace?.id ?? null)
  const misfit =
    !!unit.image &&
    unit.image.height > 0 &&
    ratioDiffers(unit.image.width / unit.image.height, target.w / target.h)
  const span = { a: Math.round(unit.start * steps), b: Math.round(unit.end * steps) }
  const setSpan = (a: number, b: number) => {
    onChange((u) => ({ ...u, start: stepFraction(a, steps), end: stepFraction(b, steps) }))
  }

  return (
    <Sheet title="ControlNet" onClose={onClose}>
      <div className="control-editor">
        <ControlPicture
          unit={unit}
          source={source}
          busy={busy}
          overPhoto={overPhoto}
          onOverPhoto={setOverPhoto}
          onCrop={(sha) => {
            setCropping(sha)
            setOverlay('crop')
          }}
          onChoose={() => {
            setOverlay('image')
          }}
          onUseSource={(picture) => {
            takePicture(picture, false)
          }}
        />

        <TracePicker
          unit={unit}
          busy={busy}
          gpuReady={gpuReady}
          sessionActive={isActive(session.data)}
          error={error?.message ?? null}
          note={note}
          detail={detail}
          onDetail={setDetail}
          onAsIs={() => {
            if (!unit.trace) return
            const from = unit.trace.from
            onChange((u) => ({
              ...u,
              image: from,
              trace: null,
              area: sameSize(u.image, from) ? u.area : null,
            }))
          }}
          onTrace={(id) => {
            const photo = photoOf(unit)
            if (photo) trace({ id, params: id === 'canny' ? edgeParams(detail) : {}, from: photo })
          }}
        />

        <UnitModelRow
          value={unit.model ? assetLabel(unit.model, controlnets) : null}
          placeholder="Choose a ControlNet"
          missing={unit.model && !model ? 'Not found in Drive. Pick another ControlNet.' : null}
          warning={warning}
          onPick={() => {
            setOverlay('model')
          }}
        />

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
          <AreaRow
            label="Area"
            area={unit.area ? { source: underlay(unit) ?? unit.image.sha, mask: unit.area } : null}
            editText="Edit area"
            emptyText="Limit to an area"
            onOpen={() => {
              setOverlay('area')
            }}
            onClear={() => {
              onChange((u) => ({ ...u, area: null }))
            }}
          />
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
