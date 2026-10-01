import { useMutation, useQuery } from '@tanstack/react-query'
import { useState } from 'react'

import { queries } from '@/api/queries'
import type { Asset, ImagePromptOptions, Variant } from '@/api/types'
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
import type { FitOption } from '@/lib/fit'
import { formatSize } from '@/lib/format'
import type { Size } from '@/lib/geometry'
import { type Source, toSource } from '@/lib/image'
import { isGpuReady } from '@/lib/session'
import { stepFraction } from '@/lib/steps'

import { AdapterPicker } from './AdapterPicker'
import { blankCanvas } from './blankCanvas'
import { FaceSliders } from './FaceSliders'
import {
  anyOblong,
  type ImagePromptUnit,
  isFaceid,
  mismatch,
  modelFor,
  type Take,
  weightFor,
} from './imagePrompt'
import { PromptPictures } from './PromptPictures'
import { TakePicker } from './TakePicker'

const FITS: FitOption[] = [
  { id: 'crop', label: 'The middle square' },
  { id: 'pad', label: 'All of it, letterboxed' },
]

type Overlay = 'image' | 'crop' | 'area' | 'model' | null

export interface ImagePromptEditorProps {
  unit: ImagePromptUnit
  /** The family's image prompt models in the Drive index. */
  adapters: Asset[]
  familyId: string
  /** A picture of the output's shape to paint the area over (Create's source), if any. */
  canvasOver: Source | null
  /** The form's output size and step count. */
  target: Size
  steps: number
  constraints: Variant['size_constraints']
  /** What the family's image prompts can do. */
  options: ImagePromptOptions
  onChange: (update: (unit: ImagePromptUnit) => ImagePromptUnit) => void
  onRemove: () => void
  onClose: () => void
}

/**
 * One image prompt (docs/ip-adapter.md §4.6): its pictures, what to take from them, the
 * model, the weight, the steps it acts on, and the area of the output it's limited to. The
 * image picker, crop and area editors replace the sheet while they're open.
 */
export function ImagePromptEditor({
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
}: ImagePromptEditorProps) {
  const [overlay, setOverlay] = useState<Overlay>(null)
  // The picture being cropped, and its place in the row (one past the end adds it).
  const [cropping, setCropping] = useState<{ sha: string; at: number } | null>(null)
  const [areaOver, setAreaOver] = useState<Source | null>(null)
  const session = useQuery(queries.session())
  const gpuReady = isGpuReady(session.data)
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
      <AdapterPicker
        adapters={adapters}
        familyId={familyId}
        selected={unit.model}
        redux={options.detail}
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
  const warning = unit.model ? mismatch(model, unit.model, unit.take) : null
  const span = { a: Math.round(unit.start * steps), b: Math.round(unit.end * steps) }
  const setSpan = (a: number, b: number) => {
    onChange((u) => ({ ...u, start: stepFraction(a, steps), end: stepFraction(b, steps) }))
  }

  return (
    <Sheet title="Image prompt" onClose={onClose}>
      <div className="prompt-editor">
        <PromptPictures
          pictures={unit.pictures}
          faceid={faceid}
          gpuReady={gpuReady}
          onCrop={(i) => {
            const picture = unit.pictures[i]
            if (!picture) return
            setCropping({ sha: picture.sha, at: i })
            setOverlay('crop')
          }}
          onRemove={(i) => {
            onChange((u) => ({ ...u, pictures: u.pictures.filter((_, j) => j !== i) }))
          }}
          onAdd={() => {
            setOverlay('image')
          }}
        />

        <TakePicker
          unit={unit}
          options={options}
          faceid={faceid}
          onTake={choose}
          onDownsample={(downsample) => {
            onChange((u) => ({ ...u, downsample }))
          }}
        />

        <UnitModelRow
          value={unit.model ? assetLabel(unit.model, adapters) : null}
          placeholder="Choose a model"
          missing={unit.model && !model ? 'Not found in Drive. Pick another model.' : null}
          warning={warning}
          onPick={() => {
            setOverlay('model')
          }}
        />

        <SliderRow
          id="prompt-weight"
          label="Weight"
          min={0}
          max={2}
          value={unit.weight}
          onChange={(weight) => {
            onChange((u) => ({ ...u, weight }))
          }}
        />

        {faceid && (
          <FaceSliders
            structure={unit.structure}
            loraWeight={unit.loraWeight}
            onStructure={(structure) => {
              onChange((u) => ({ ...u, structure }))
            }}
            onLoraWeight={(loraWeight) => {
              onChange((u) => ({ ...u, loraWeight }))
            }}
          />
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
          <AreaRow
            label="Area"
            area={unit.area ? { source: unit.area.over.sha, mask: unit.area.sha } : null}
            editText="Edit area"
            emptyText="Limit to an area"
            disabled={openArea.isPending}
            before={
              openArea.error && (
                <p className="row-warning" role="alert">
                  {openArea.error.message}
                </p>
              )
            }
            after={
              unit.area &&
              (unit.area.over.width !== target.w || unit.area.over.height !== target.h) && (
                <p className="row-note">
                  Painted at {formatSize(unit.area.over.width, unit.area.over.height)}; it will be
                  fitted to {formatSize(target.w, target.h)}.
                </p>
              )
            }
            onOpen={() => {
              openArea.mutate()
            }}
            onClear={() => {
              onChange((u) => ({ ...u, area: null }))
            }}
          />
        )}

        {anyOblong(unit) && (
          <FitSelect
            id="prompt-fit"
            label="Show it"
            value={unit.fit}
            options={FITS}
            onChange={(fit) => {
              onChange((u) => ({ ...u, fit }))
            }}
          />
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
