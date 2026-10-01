import { useRef, useState } from 'react'

import { ImagePicker } from '@/components/ImagePicker/ImagePicker'
import { CropEditor } from '@/features/editors/crop/CropEditor'
import { MaskEditor } from '@/features/editors/mask/MaskEditor'
import { PlaceEditor } from '@/features/editors/place/PlaceEditor'
import { assetLabel } from '@/lib/assets'
import { toSource } from '@/lib/image'

import { newUnit } from './control/control'
import { ControlEditor } from './control/ControlEditor'
import { ControlList } from './control/ControlList'
import type { CreateScreenProps } from './CreateScreen'
import { GenerateBar } from './GenerateBar'
import { modelFor, newPrompt } from './image-prompt/imagePrompt'
import { ImagePromptEditor } from './image-prompt/ImagePromptEditor'
import { ImagePromptList } from './image-prompt/ImagePromptList'
import { LoraPicker } from './LoraPicker'
import { MediaSwitch } from './MediaSwitch'
import { ModeChips } from './ModeChips'
import { ModelPicker } from './ModelPicker'
import { AreaRow } from './rows/AreaRow'
import { FitRow } from './rows/FitRow'
import { LoraList } from './rows/LoraList'
import { ModelRow } from './rows/ModelRow'
import { RefList } from './rows/RefList'
import { SourceRow } from './rows/SourceRow'
import { SavedPromptsSheet } from './SavedPromptsSheet'
import { SchemaForm } from './schema-form/SchemaForm'
import { useCreateForm } from './useCreateForm'

type Picker = 'model' | 'lora' | 'prompts' | 'image' | 'ref' | null

export interface CreateFormProps extends CreateScreenProps {
  familyId: string
  onFamily: (id: string) => void
}

/** One family's form. Remounted (by key) when the family changes, so it starts from that draft. */
export function CreateForm({ familyId, onFamily, onOpenSession, onShowResults }: CreateFormProps) {
  const form = useCreateForm(familyId, onFamily)
  const [picker, setPicker] = useState<Picker>(null)
  const [painting, setPainting] = useState(false)
  // The image being cropped, and for a reference, its place in the Images row (one past the
  // end adds it).
  const [cropping, setCropping] = useState<{ sha: string; ref?: number } | null>(null)
  const promptRef = useRef<HTMLTextAreaElement>(null)
  const promptFocused = useRef(false)

  if (!form.ready) {
    return form.error ? (
      <p role="alert">{form.error.message}</p>
    ) : (
      <p className="loading">Loading…</p>
    )
  }

  const { params, selection, input, refs, control, prompts, target, steps } = form
  const { family, mode, model, modelMissing, variant, needsSource, takesRefs } = selection
  const { source, mask, maskFits } = input
  const video = family?.media === 'video'
  const modelLabel = model ? assetLabel(model, selection.models) : null

  const closePicker = () => {
    setPicker(null)
  }

  const leadingRows = (
    <>
      {needsSource && (
        <SourceRow
          label={takesRefs ? 'Image 1' : 'Source'}
          source={source}
          gone={input.gone}
          continuesClip={!!input.extendsClip}
          onPick={() => {
            setPicker('image')
          }}
          onCrop={() => {
            if (source) setCropping({ sha: source.sha })
          }}
          onRemove={input.remove}
          onGone={input.markGone}
        />
      )}
      {needsSource && source && form.misfit && mode !== 'outpaint' && (
        <FitRow
          fit={input.fit}
          onFit={input.setFit}
          onOutpaint={variant?.modes.includes('outpaint') ? form.outpaintInstead : undefined}
        />
      )}
      {mode === 'inpaint' && source && !input.gone && (
        <AreaRow
          label="Mask"
          area={maskFits && mask ? { source: source.sha, mask: mask.sha } : null}
          editText="Edit mask"
          emptyText="Paint the area to redraw"
          before={input.maskNote && <p className="row-note">{input.maskNote}</p>}
          onOpen={() => {
            setPainting(true)
          }}
          onClear={() => {
            input.setMask(null)
          }}
        />
      )}
      {mode === 'outpaint' && source && !input.gone && form.place && (
        <PlaceEditor source={source} canvas={target} place={form.place} onChange={input.setPlace} />
      )}
      {takesRefs && (
        <RefList
          refs={refs}
          max={selection.maxRefs}
          onChange={form.setRefs}
          onAdd={() => {
            setPicker('ref')
          }}
          onCrop={(i) => {
            const ref = refs[i]
            if (ref) setCropping({ sha: ref.sha, ref: i })
          }}
        />
      )}
      <ModelRow
        value={modelLabel ?? (form.indexed ? 'None found' : 'Loading…')}
        chosen={!!model}
        missing={modelMissing}
        variantLabel={
          variant?.model_dir && modelLabel && !modelMissing && modelLabel !== variant.label
            ? variant.label
            : null
        }
        underpowered={form.underpowered}
        onPick={() => {
          setPicker('model')
        }}
      />
      <LoraList
        loras={form.loras}
        index={selection.loras}
        onChange={form.setLoras}
        onAdd={() => {
          setPicker('lora')
        }}
        onTrigger={(word) => {
          // Before the prompt has been touched there's no cursor to honour: append.
          const el = promptRef.current
          form.insertTrigger(word, promptFocused.current && el ? el.selectionEnd : undefined)
        }}
      />
      {selection.withControl && (
        <ControlList
          units={control.units}
          index={selection.controlnets}
          steps={steps}
          onOpen={control.open}
          onAdd={() => {
            control.add(newUnit())
          }}
        />
      )}
      {selection.withPrompts && (
        <ImagePromptList
          units={prompts.units}
          index={selection.adapters}
          steps={steps}
          options={selection.promptOptions}
          onOpen={prompts.open}
          onAdd={() => {
            prompts.add({
              ...newPrompt(selection.promptOptions),
              model: modelFor(selection.adapters ?? [], 'all')?.path ?? '',
            })
          }}
        />
      )}
    </>
  )

  return (
    <form
      className="create"
      onSubmit={(e) => {
        e.preventDefault()
        form.submit.mutate()
      }}
    >
      <MediaSwitch media={form.media} current={family?.media} onChoose={form.chooseMedia} />
      <ModeChips modes={selection.allModes} mode={mode} onChoose={form.chooseMode} />

      <SchemaForm
        schema={form.schema}
        values={params}
        onChange={form.setParams}
        presets={variant?.size_constraints.presets ?? []}
        leadingRows={leadingRows}
        promptPlaceholder={video ? 'Describe the clip' : 'Describe the picture'}
        promptRef={promptRef}
        seeds={{ batch: form.batchCount > 1, mode: form.seedMode, onMode: form.setSeedMode }}
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
        <SavedPromptsSheet
          prompt={String(params.prompt ?? '')}
          negativePrompt={String(params.negative_prompt ?? '')}
          family={familyId}
          onUse={(p) => {
            form.applySavedPrompt(p)
            closePicker()
          }}
          onClose={closePicker}
        />
      )}
      {picker === 'model' && family && (
        <ModelPicker
          family={family}
          siblings={selection.siblings}
          mode={mode}
          models={selection.pickable ?? []}
          selected={model}
          onPick={(asset) => {
            form.chooseModel(asset)
            // Another family's model switches to that family's form, which opens without a picker.
            if (!asset.family || asset.family === family.id) closePicker()
          }}
          onClose={closePicker}
        />
      )}
      {picker === 'lora' && (
        <LoraPicker
          family={family}
          variant={variant}
          index={selection.loras}
          loras={form.loras}
          onAdd={form.addLora}
          onDeleted={form.dropLoras}
          onClose={closePicker}
        />
      )}
      {picker === 'image' && (
        <ImagePicker
          onUse={(image) => {
            form.takeSource(image, false)
            closePicker()
          }}
          onCrop={(image) => {
            closePicker()
            setCropping({ sha: image.sha256 })
          }}
          onClose={closePicker}
        />
      )}
      {picker === 'ref' && (
        <ImagePicker
          onUse={(image) => {
            form.setRefs([...refs, toSource(image)])
            closePicker()
          }}
          onCrop={(image) => {
            closePicker()
            setCropping({ sha: image.sha256, ref: refs.length })
          }}
          onClose={closePicker}
        />
      )}
      {painting && source && (
        <MaskEditor
          source={source}
          mask={maskFits && mask ? mask.sha : null}
          blur={Number(params.mask_blur ?? 0)}
          onDone={(painted) => {
            input.setMask(painted)
            setPainting(false)
          }}
          onCancel={() => {
            setPainting(false)
          }}
        />
      )}
      {control.current && variant && (
        <ControlEditor
          unit={control.current}
          controlnets={selection.controlnets ?? []}
          familyId={familyId}
          source={needsSource && form.sourceUsable ? source : null}
          target={target}
          steps={steps}
          constraints={variant.size_constraints}
          onChange={control.update}
          onRemove={control.remove}
          onClose={() => {
            // A unit left without an image or a model isn't worth keeping.
            control.close((u) => !!u.image || !!u.model)
          }}
        />
      )}
      {prompts.current && variant && (
        <ImagePromptEditor
          unit={prompts.current}
          adapters={selection.adapters ?? []}
          familyId={familyId}
          canvasOver={needsSource && form.sourceUsable && !form.misfit ? source : null}
          target={target}
          steps={steps}
          constraints={variant.size_constraints}
          options={selection.promptOptions}
          onChange={prompts.update}
          onRemove={prompts.remove}
          onClose={() => {
            // A unit left without a picture isn't worth keeping.
            prompts.close((u) => u.pictures.length > 0)
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
          onApply={(image, out) => {
            const at = cropping.ref
            if (at === undefined) form.takeSource(image, true, out)
            else form.setRefs([...refs.slice(0, at), toSource(image), ...refs.slice(at + 1)])
            setCropping(null)
          }}
          onCancel={() => {
            setCropping(null)
          }}
        />
      )}

      <GenerateBar
        video={video}
        batchCount={form.batchCount}
        onBatchCount={form.setBatchCount}
        pending={form.submit.isPending}
        blocked={form.blocked}
        error={form.submit.error?.message ?? null}
        queued={form.queued}
        noGpu={form.noGpu}
        onShowResults={onShowResults}
        onOpenSession={onOpenSession}
      />
    </form>
  )
}
