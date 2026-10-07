import { act, fireEvent, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import {
  ASSETS,
  FAMILIES,
  PROMPT,
  RESULT,
  RUNNING,
  SCHEMA,
  SPEC,
  WAN,
  WAN_ASSETS,
} from '@/test/fixtures'
import { FakeEventSource, mockApi, VIDEO_ROUTES } from '@/test/mockApi'
import { renderApp } from '@/test/render'

describe('Create', () => {
  it('renders the generation form from the family schema', async () => {
    mockApi()
    renderApp()
    expect(await screen.findByLabelText('Prompt')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Model Studio XL v10' })).toBeInTheDocument()
    expect(screen.getByRole('slider', { name: /Steps/ })).toHaveValue('30')
    expect(screen.getByRole('button', { name: '832×1216' })).toBeInTheDocument()
    expect(screen.getByText(/No GPU is running/)).toBeInTheDocument()
  })

  it('marks the defaults and resets a setting to its default', async () => {
    mockApi()
    const user = userEvent.setup()
    renderApp()
    const steps = await screen.findByRole('slider', { name: /Steps/ })
    expect(screen.getByRole('button', { name: '1024×1024, default' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /^Reset / })).not.toBeInTheDocument()

    fireEvent.change(steps, { target: { value: '42' } })
    const reset = screen.getByRole('button', { name: 'Reset Steps to 30' })
    expect(reset).toHaveTextContent('Reset to 30')

    await user.click(reset)
    expect(steps).toHaveValue('30')
    expect(screen.queryByRole('button', { name: /^Reset / })).not.toBeInTheDocument()
  })

  it('submits a job and confirms it was queued', async () => {
    const submitted: unknown[] = []
    mockApi({
      'POST /api/jobs': (init) => {
        submitted.push(JSON.parse(init?.body as string))
        return { id: 'j1' }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.type(await screen.findByLabelText('Prompt'), 'a lighthouse')
    await user.click(screen.getByRole('button', { name: '832×1216' }))
    await user.click(screen.getByText('More settings'))
    await user.click(screen.getByRole('checkbox', { name: /Built-in VAE in float32/ }))
    await user.click(screen.getByRole('button', { name: 'Generate' }))

    expect(await screen.findByText('Queued 1 image.')).toBeInTheDocument()
    expect(submitted).toEqual([
      {
        spec: {
          family: 'sdxl',
          variant: 'base',
          mode: 't2i',
          model: { path: 'models/sdxl/studio.safetensors' },
          loras: [],
          params: {
            prompt: 'a lighthouse',
            width: 832,
            height: 1216,
            steps: 30,
            seed: -1,
            scheduler: 'euler',
            vae_fp32: true,
          },
        },
        batch_count: 1,
        seed_mode: 'increment',
      },
    ])
  })

  it('picks a model, showing which ones are already on the GPU', async () => {
    mockApi({ 'GET /api/session': () => RUNNING })
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: 'Model Studio XL v10' }))
    const sheet = screen.getByRole('dialog', { name: 'Model' })
    const current = within(sheet).getByRole('button', { name: /Studio XL v10/ })
    expect(current).toHaveAttribute('aria-pressed', 'true')
    expect(current).toHaveTextContent('6.9 GB, on the GPU')
    const other = within(sheet).getByRole('button', { name: /^base/ })
    expect(other).toHaveTextContent('6.9 GB, about 100 s to copy')
    expect(within(sheet).getByText('Indexed', { exact: false })).toBeInTheDocument()
    await user.click(other)
    expect(screen.queryByRole('dialog', { name: 'Model' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Model base' })).toBeInTheDocument()
  })

  it('adds a LoRA, weights it, and taps its trigger word into the prompt', async () => {
    const submitted: { spec: { loras: unknown; params: { prompt: string } } }[] = []
    mockApi({
      'POST /api/jobs': (init) => {
        submitted.push(JSON.parse(init?.body as string) as (typeof submitted)[number])
        return { id: 'j1' }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.type(await screen.findByLabelText('Prompt'), 'a lighthouse')
    await user.click(screen.getByRole('button', { name: 'Add LoRA' }))
    await user.click(
      within(screen.getByRole('dialog', { name: 'Add LoRA' })).getByRole('button', {
        name: /Film Grain v3/,
      }),
    )
    const weight = screen.getByRole('slider', { name: 'Film Grain v3 weight' })
    expect(weight).toHaveValue('0.8') // the sidecar's default weight
    fireEvent.change(weight, { target: { value: '0.55' } })
    expect(screen.getByText('0.55')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Add “filmgrain” to the prompt' }))
    expect(screen.getByLabelText('Prompt')).toHaveValue('a lighthouse, filmgrain')

    await user.click(screen.getByRole('button', { name: 'Generate' }))
    await screen.findByText('Queued 1 image.')
    expect(submitted[0]?.spec.loras).toEqual([
      { path: 'loras/sdxl/film.safetensors', weight: 0.55 },
    ])

    await user.click(screen.getByRole('button', { name: 'Remove Film Grain v3' }))
    expect(screen.queryByRole('slider', { name: 'Film Grain v3 weight' })).not.toBeInTheDocument()
  })

  it('deletes a LoRA from Drive and drops it from the form', async () => {
    let assets = ASSETS
    const deleted: string[] = []
    mockApi({
      'GET /api/assets': () => assets,
      'DELETE /api/assets': () => {
        const url = new URL(vi.mocked(fetch).mock.calls.at(-1)?.[0] as string, 'http://x')
        deleted.push(...url.searchParams.getAll('path'))
        assets = assets.filter((a) => !deleted.includes(a.path))
        return { deleted }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: 'Add LoRA' }))
    let sheet = screen.getByRole('dialog', { name: 'Add LoRA' })
    await user.click(within(sheet).getByRole('button', { name: /Film Grain v3/ }))
    expect(screen.getByRole('slider', { name: 'Film Grain v3 weight' })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Add LoRA' }))
    sheet = screen.getByRole('dialog', { name: 'Add LoRA' })
    await user.click(within(sheet).getByRole('button', { name: 'Delete LoRAs' }))
    sheet = screen.getByRole('dialog', { name: 'Delete LoRAs' })
    expect(within(sheet).queryByRole('button', { name: 'Import a LoRA' })).toBeNull()
    await user.click(within(sheet).getByRole('button', { name: /Film Grain v3/ }))
    const confirm = within(sheet).getByRole('group', { name: 'Confirm delete' })
    expect(confirm).toHaveTextContent('Delete Film Grain v3? Its files go to Drive’s trash.')
    await user.click(within(confirm).getByRole('button', { name: 'Delete' }))

    await within(sheet).findByText('Deleted Film Grain v3.')
    expect(deleted).toEqual(['loras/sdxl/film.safetensors'])
    expect(within(sheet).queryByRole('button', { name: /Film Grain v3/ })).toBeNull()
    expect(screen.queryByRole('slider', { name: 'Film Grain v3 weight' })).not.toBeInTheDocument()
  })

  it('imports a LoRA from Civitai and adds it to the form', async () => {
    let source: FakeEventSource | undefined
    vi.stubGlobal(
      'EventSource',
      class extends FakeEventSource {
        constructor() {
          super()
          // eslint-disable-next-line @typescript-eslint/no-this-alias -- the test drives it
          source = this
        }
      },
    )
    const imported = {
      path: 'loras/sdxl/test_style_xl_v1.0.safetensors',
      family: 'sdxl',
      kind: 'lora',
      size: 228_000_000,
      sidecar: { label: 'Test Style XL', default_weight: 0.8 },
      preview_thumb: null,
    }
    let assets: unknown[] = ASSETS
    const job = {
      id: 'i1',
      label: 'Test Style XL',
      family: 'sdxl',
      paths: [imported.path],
      state: 'copying',
      done: 0,
      total: 228_000_000,
      error: null,
      warnings: [],
    }
    const posted: unknown[] = []
    mockApi({
      'GET /api/assets': () => assets,
      'GET /api/civitai/import': () => null,
      'POST /api/civitai/plan': () => ({
        origin: 'civitai',
        model_name: 'Test Style XL',
        version_name: 'v1.0',
        base_model: 'SDXL 1.0',
        family: 'sdxl',
        label: 'Test Style XL',
        trigger_words: [],
        weight: 0.8,
        files: [
          { civitai_name: 'x.safetensors', path: imported.path, size: 228_000_000, half: null },
        ],
        warnings: [],
      }),
      'POST /api/civitai/import': (init) => {
        posted.push(JSON.parse(init?.body as string))
        return job
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: 'Add LoRA' }))
    const sheet = screen.getByRole('dialog', { name: 'Add LoRA' })
    await user.click(within(sheet).getByRole('button', { name: 'Import a LoRA' }))
    await user.type(
      within(sheet).getByLabelText('Civitai or Hugging Face link'),
      'https://civitai.com/models/100001',
    )
    await user.click(within(sheet).getByRole('button', { name: 'Check link' }))
    expect(await within(sheet).findByText('v1.0, SDXL 1.0, 228 MB')).toBeInTheDocument()
    await user.click(within(sheet).getByRole('button', { name: 'Import' }))
    expect(posted).toEqual([{ url: 'https://civitai.com/models/100001', hint: 'sdxl' }])
    expect(await within(sheet).findByText('Copying to Drive, 0% of 228 MB')).toBeInTheDocument()

    const send = (data: unknown) => {
      act(() => {
        source?.onmessage?.(new MessageEvent('message', { data: JSON.stringify(data) }))
      })
    }
    send({ type: 'import', import: { ...job, done: 114_000_000 } })
    expect(await within(sheet).findByText('Copying to Drive, 50% of 228 MB')).toBeInTheDocument()
    assets = [...ASSETS, imported]
    send({ type: 'import', import: { ...job, state: 'done', done: job.total } })
    expect(
      await within(sheet).findByText('Imported Test Style XL and added it.'),
    ).toBeInTheDocument()
    expect(screen.getByRole('slider', { name: 'Test Style XL weight' })).toHaveValue('0.8')
  })

  it('flags a LoRA from an old draft that is no longer in Drive', async () => {
    localStorage.setItem(
      'degas.create.draft',
      JSON.stringify({ loras: [{ path: 'loras/sdxl/gone.safetensors', weight: 1 }] }),
    )
    mockApi()
    renderApp()
    expect(await screen.findByText(/Not found in Drive/)).toBeInTheDocument()
    expect(screen.getByRole('slider', { name: 'gone weight' })).toBeInTheDocument()
  })

  it('keeps Generate disabled until there is a prompt', async () => {
    mockApi()
    const user = userEvent.setup()
    renderApp()
    const generate = await screen.findByRole('button', { name: 'Generate' })
    expect(generate).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'More images' }))
    await user.type(screen.getByLabelText('Prompt'), 'a lighthouse')
    expect(screen.getByRole('button', { name: 'Generate 2 images' })).toBeEnabled()
  })

  it('saves the current prompt and swaps in a saved one', async () => {
    const saved: unknown[] = []
    mockApi({
      'GET /api/prompts': () => [PROMPT],
      'POST /api/prompts': (init) => {
        saved.push(JSON.parse(init?.body as string))
        return { ...PROMPT, id: 'p2' }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.type(await screen.findByLabelText('Prompt'), 'a lighthouse')
    await user.click(screen.getByRole('button', { name: 'Prompts' }))
    const sheet = screen.getByRole('dialog', { name: 'Saved prompts' })
    await user.click(within(sheet).getByRole('button', { name: 'Save this prompt' }))
    expect(await within(sheet).findByRole('button', { name: 'Saved' })).toBeDisabled()
    expect(saved).toEqual([{ prompt: 'a lighthouse', negative_prompt: '', family: 'sdxl' }])

    await user.click(within(sheet).getByRole('button', { name: /Harbour/ }))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(screen.getByLabelText('Prompt')).toHaveValue('a harbour at dusk, oil painting')
  })

  it('outpaints around a placed image, and needs a mask to inpaint', async () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null)
    const submitted: { spec: { mode: string; inputs?: unknown } }[] = []
    const modes = ['t2i', 'i2i', 'inpaint', 'outpaint']
    const variant = FAMILIES[0]?.variants[0]
    mockApi({
      'GET /api/families': () => [{ ...FAMILIES[0], variants: [{ ...variant, modes }] }],
      'POST /api/jobs': (init) => {
        submitted.push(JSON.parse(init?.body as string) as (typeof submitted)[number])
        return { id: 'j2' }
      },
    })
    localStorage.setItem(
      'degas.create.draft',
      JSON.stringify({
        family: 'sdxl',
        families: {
          sdxl: {
            model: 'models/sdxl/studio.safetensors',
            mode: 'outpaint',
            loras: [],
            params: { prompt: 'a lighthouse', width: 1344, height: 768 },
            source: { sha: 'abc', width: 1024, height: 1024 },
          },
        },
      }),
    )
    const user = userEvent.setup()
    renderApp()
    expect(await screen.findByText('+288 px left, +288 px right')).toBeInTheDocument()
    expect(screen.queryByLabelText('Fit')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Left' }))
    expect(screen.getByText('+576 px right')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Generate' }))
    expect(await screen.findByText('Queued 1 image.')).toBeInTheDocument()
    expect(submitted[0]?.spec).toMatchObject({
      mode: 'outpaint',
      inputs: { source: 'sha256:abc', place: { x: 0, y: 0, w: 768, h: 768 } },
    })

    await user.click(screen.getByRole('button', { name: 'Inpaint' }))
    expect(screen.getByRole('button', { name: 'Generate' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: /Mask Paint the area to redraw/ }))
    const editor = await screen.findByRole('dialog', { name: 'Mask' })
    await user.click(within(editor).getByRole('button', { name: 'Select' }))
    expect(within(editor).getByText(/Put SAM 3 in Drive/)).toBeInTheDocument()
    await user.click(within(editor).getByRole('button', { name: 'Cancel' }))
    expect(screen.queryByRole('dialog', { name: 'Mask' })).not.toBeInTheDocument()
  })

  it('switches to Qwen from the model picker and edits with ordered images', async () => {
    const submitted: { spec: Record<string, unknown> }[] = []
    const transforms: { ops: unknown[] }[] = []
    const qwen = {
      id: 'qwen21',
      label: 'Qwen-Image 2.1',
      media: 'image',
      lora_format: 'single',
      variants: [
        {
          id: 'base',
          label: 'Qwen-Image 2.1',
          min_gpu: 'L4',
          modes: ['t2i', 'edit', 'inpaint'],
          model_dir: null,
          max_refs: 9,
          size_constraints: {
            multiple_of: 32,
            min_pixels: 262144,
            max_pixels: 4300800,
            presets: [[1024, 1024]],
          },
        },
      ],
    }
    const harbour = { ...RESULT, id: 'r2', blob_sha: 'def', width: 640, height: 480 }
    harbour.spec = { ...SPEC, params: { ...SPEC.params, prompt: 'a harbour' } }
    mockApi({
      'GET /api/families': () => [...FAMILIES, qwen],
      'GET /api/families/qwen21/schema': () => SCHEMA,
      'GET /api/assets': () => [
        ...ASSETS,
        {
          path: 'models/qwen21/Qwen-Image-2.1',
          family: 'qwen21',
          kind: 'model',
          size: 36e9,
          sidecar: null,
          preview_thumb: null,
        },
      ],
      'GET /api/results': () => ({ results: [RESULT, harbour], cursor: null }),
      'GET /api/blobs/def/transform': () => ({ original: 'def', ops: [] }),
      'POST /api/blobs/def/transform': (init) => {
        transforms.push(JSON.parse(init?.body as string) as (typeof transforms)[number])
        return { sha256: 'd2', media_type: 'image/png', width: 640, height: 480 }
      },
      'POST /api/jobs': (init) => {
        submitted.push(JSON.parse(init?.body as string) as (typeof submitted)[number])
        return { id: 'j2' }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: 'Model Studio XL v10' }))
    const sheet = screen.getByRole('dialog', { name: 'Model' })
    const row = within(sheet).getByRole('button', { name: /^Qwen-Image-2\.1/ })
    expect(row).toHaveTextContent('Qwen-Image 2.1')
    await user.click(row)
    expect(await screen.findByRole('button', { name: 'Edit' })).toBeInTheDocument()

    await user.type(screen.getByLabelText('Prompt'), 'the boat from image 2 in the bay')
    await user.click(screen.getByRole('button', { name: 'Edit' }))
    await user.click(screen.getByRole('button', { name: /Image 1 Choose an image/ }))
    let picker = screen.getByRole('dialog', { name: 'Choose image' })
    await user.click(await within(picker).findByRole('button', { name: 'Image: a lighthouse' }))
    await user.click(within(picker).getByRole('button', { name: 'Use image' }))

    // A reference crops freely, and keeps its own size: the model sizes it.
    await user.click(screen.getByRole('button', { name: 'Add image' }))
    picker = screen.getByRole('dialog', { name: 'Choose image' })
    await user.click(await within(picker).findByRole('button', { name: 'Image: a harbour' }))
    await user.click(within(picker).getByRole('button', { name: 'Crop' }))
    const editor = await screen.findByRole('dialog', { name: 'Crop' })
    const img = editor.querySelector('img')
    if (!img) throw new Error('no image in the editor')
    Object.defineProperty(img, 'naturalWidth', { value: 640 })
    Object.defineProperty(img, 'naturalHeight', { value: 480 })
    fireEvent.load(img)
    expect(within(editor).getByRole('button', { name: 'Free' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    expect(within(editor).queryByRole('button', { name: /^Match/ })).not.toBeInTheDocument()
    expect(await within(editor).findByText(/640 × 480 from 640 × 480/)).toBeInTheDocument()
    expect(within(editor).getByText(/The model scales it up 1\.8×/)).toBeInTheDocument()
    await user.click(within(editor).getByRole('button', { name: 'Apply' }))
    expect(transforms).toEqual([{ ops: [] }])

    await user.click(screen.getByRole('button', { name: 'Add image' }))
    picker = screen.getByRole('dialog', { name: 'Choose image' })
    await user.click(await within(picker).findByRole('button', { name: 'Image: a lighthouse' }))
    await user.click(within(picker).getByRole('button', { name: 'Use image' }))
    expect(screen.getByText('640 × 480')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Crop image 3' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Move image 2 earlier' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Move image 3 earlier' }))

    await user.click(screen.getByRole('button', { name: 'Generate' }))
    expect(await screen.findByText('Queued 1 image.')).toBeInTheDocument()
    expect(submitted[0]?.spec).toMatchObject({
      family: 'qwen21',
      mode: 'edit',
      model: { path: 'models/qwen21/Qwen-Image-2.1' },
      inputs: { source: 'sha256:abc', refs: ['sha256:abc', 'sha256:d2'] },
    })
  })

  it('offers every mode, and follows it with the model', async () => {
    const sizes = { multiple_of: 8, min_pixels: 262144, max_pixels: 4194304, presets: [] }
    const edit = (id: string, label: string, modes: string[]) => ({
      id,
      label,
      media: 'image',
      lora_format: 'single',
      variants: [
        {
          id: 'base',
          label,
          min_gpu: 'L4',
          modes,
          model_dir: null,
          max_refs: 3,
          size_constraints: sizes,
        },
      ],
    })
    const sdxl = {
      ...FAMILIES[0],
      variants: [
        { ...FAMILIES[0]?.variants[0], modes: ['t2i', 'i2i', 'inpaint', 'outpaint'] },
        {
          id: 'inpainting',
          label: 'SDXL Inpainting',
          min_gpu: 'T4',
          modes: ['inpaint', 'outpaint'],
          model_dir: 'models/sdxl/inpainting',
          size_constraints: sizes,
        },
      ],
    }
    const model = (path: string, family: string) => ({
      path,
      family,
      kind: 'model',
      size: 1e9,
      sidecar: null,
      preview_thumb: null,
    })
    mockApi({
      'GET /api/families': () => [
        sdxl,
        edit('qwen21', 'Qwen-Image 2.1', ['t2i', 'edit', 'inpaint']),
        edit('klein', 'FLUX.2 [klein]', ['edit']),
      ],
      'GET /api/families/klein/schema': () => SCHEMA,
      'GET /api/assets': () => [
        ...ASSETS,
        model('models/sdxl/inpainting/fill.safetensors', 'sdxl'),
        model('models/qwen21/Qwen-Image-2.1', 'qwen21'),
        model('models/klein/Klein-9B', 'klein'),
      ],
    })
    localStorage.setItem(
      'degas.create.draft',
      JSON.stringify({
        family: 'sdxl',
        recent: ['sdxl', 'klein', 'qwen21'],
        families: {
          sdxl: {
            model: 'models/sdxl/inpainting/fill.safetensors',
            mode: 'inpaint',
            loras: [],
            params: { prompt: 'a lighthouse', width: 1024, height: 1024 },
            source: { sha: 'abc', width: 1024, height: 1024 },
          },
        },
      }),
    )
    const user = userEvent.setup()
    renderApp()
    const chips = await screen.findByRole('group', { name: 'Start from' })
    expect(
      within(chips)
        .getAllByRole('button')
        .map((b) => b.textContent),
    ).toEqual(['From text', 'From image', 'Edit', 'Inpaint', 'Outpaint'])
    expect(within(chips).getByRole('button', { name: 'Inpaint' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    // The chips come before the prompt.
    expect(
      chips.compareDocumentPosition(screen.getByLabelText('Prompt')) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()

    // The picker lists only the models that can inpaint.
    await user.click(screen.getByRole('button', { name: /^Model fill/ }))
    let sheet = screen.getByRole('dialog', { name: 'Model' })
    expect(within(sheet).getByRole('button', { name: /^fill/ })).toBeInTheDocument()
    expect(within(sheet).getByRole('button', { name: /^Qwen-Image-2\.1/ })).toHaveTextContent(
      'Qwen-Image 2.1, needs an L4, 1.0 GB',
    )
    expect(within(sheet).getByRole('button', { name: /^fill/ })).not.toHaveTextContent('needs')
    expect(within(sheet).queryByRole('button', { name: /^Klein/ })).not.toBeInTheDocument()
    await user.click(within(sheet).getByRole('button', { name: 'Done' }))

    // An inpainting checkpoint can't start from text: the family's first model that can.
    await user.click(within(chips).getByRole('button', { name: 'From text' }))
    expect(screen.getByRole('button', { name: 'Model Studio XL v10' })).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Model Studio XL v10' }))
    sheet = screen.getByRole('dialog', { name: 'Model' })
    expect(within(sheet).queryByRole('button', { name: /^fill/ })).not.toBeInTheDocument()
    await user.click(within(sheet).getByRole('button', { name: 'Done' }))

    // SDXL can't edit: the family used most recently that can, with the prompt and source.
    await user.click(screen.getByRole('button', { name: 'Edit' }))
    expect(await screen.findByRole('button', { name: /^Model Klein/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Edit' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByLabelText('Prompt')).toHaveValue('a lighthouse')
    expect(screen.getByRole('button', { name: /^Image 1 1024 × 1024/ })).toBeInTheDocument()
  })

  it("stops adding images at the model's limit", async () => {
    const klein = {
      id: 'klein',
      label: 'FLUX.2 [klein]',
      media: 'image',
      lora_format: 'single',
      variants: [
        {
          id: '9b',
          label: 'FLUX.2 [klein] 9B',
          min_gpu: 'L4',
          modes: ['edit'],
          model_dir: null,
          max_refs: 3,
          size_constraints: {
            multiple_of: 16,
            min_pixels: 262144,
            max_pixels: 4194304,
            presets: [[1024, 1024]],
          },
        },
      ],
    }
    mockApi({
      'GET /api/families': () => [...FAMILIES, klein],
      'GET /api/families/klein/schema': () => SCHEMA,
      'GET /api/assets': () => [
        ...ASSETS,
        {
          path: 'models/klein/FLUX.2-klein-9B',
          family: 'klein',
          kind: 'model',
          size: 35e9,
          sidecar: null,
          preview_thumb: null,
        },
      ],
    })
    const image = (sha: string) => ({ sha, width: 640, height: 480 })
    localStorage.setItem(
      'degas.create.draft',
      JSON.stringify({
        family: 'klein',
        families: {
          klein: {
            model: 'models/klein/FLUX.2-klein-9B',
            mode: 'edit',
            loras: [],
            params: { prompt: 'make it night', width: 1024, height: 1024 },
            source: image('abc'),
            refs: [image('r1'), image('r2'), image('r3')],
          },
        },
      }),
    )
    renderApp()
    expect(await screen.findByRole('button', { name: 'Add image' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Crop image 4' })).toBeInTheDocument()
  })

  it('makes a video between a first and a last frame', async () => {
    const [fiveB, ...rest] = WAN.variants
    const wan = { ...WAN, variants: [{ ...fiveB, modes: ['t2v', 'i2v', 'flf2v'] }, ...rest] }
    mockApi({ ...VIDEO_ROUTES, 'GET /api/families': () => [...FAMILIES, wan] })
    localStorage.setItem(
      'degas.create.draft',
      JSON.stringify({
        family: 'wan22',
        families: {
          wan22: {
            model: 'models/wan22/ti2v-5b',
            mode: 'flf2v',
            params: { prompt: 'a door opens', width: 1280, height: 704 },
            source: { sha: 'aaa', width: 1280, height: 704 },
          },
        },
      }),
    )
    const user = userEvent.setup()
    renderApp()
    expect(await screen.findByRole('button', { name: 'First and last' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    expect(screen.getByRole('button', { name: /^First frame/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^Last frame Choose an image/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Generate' })).toBeDisabled()

    // The other image modes don't show the slot.
    await user.click(screen.getByRole('button', { name: 'From image' }))
    expect(screen.queryByRole('button', { name: /^Last frame/ })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /^Source/ })).toBeInTheDocument()
  })

  it('sends the last frame with a first-and-last-frame job', async () => {
    const [fiveB, ...rest] = WAN.variants
    const wan = { ...WAN, variants: [{ ...fiveB, modes: ['t2v', 'i2v', 'flf2v'] }, ...rest] }
    const submitted: { spec: { mode: string; inputs?: unknown } }[] = []
    mockApi({
      ...VIDEO_ROUTES,
      'GET /api/families': () => [...FAMILIES, wan],
      'POST /api/jobs': (init) => {
        submitted.push(JSON.parse(init?.body as string) as (typeof submitted)[number])
        return { id: 'j6' }
      },
    })
    const frame = { sha: 'aaa', width: 1280, height: 704 }
    localStorage.setItem(
      'degas.create.draft',
      JSON.stringify({
        family: 'wan22',
        families: {
          wan22: {
            model: 'models/wan22/ti2v-5b',
            mode: 'flf2v',
            params: { prompt: 'a door opens', width: 1280, height: 704 },
            source: frame,
            end: { ...frame, sha: 'zzz' },
          },
        },
      }),
    )
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: 'Generate' }))
    await screen.findByText('Queued 1 clip.')
    expect(submitted[0]?.spec).toMatchObject({
      mode: 'flf2v',
      inputs: { source: 'sha256:aaa', end: 'sha256:zzz' },
    })

    // The first frame's Remove, then the last frame's.
    const [, removeLast] = screen.getAllByRole('button', { name: 'Remove' })
    if (removeLast) await user.click(removeLast)
    expect(screen.getByRole('button', { name: /^Last frame Choose an image/ })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Generate' })).toBeDisabled()
  })

  it('pairs A14B LoRAs with a weight per expert, and warns about a small GPU', async () => {
    localStorage.setItem(
      'degas.create.draft',
      JSON.stringify({
        family: 'wan22',
        families: { wan22: { model: 'models/wan22/t2v-a14b/Wan2.2-T2V-A14B', params: {} } },
      }),
    )
    const submitted: { spec: { loras: unknown } }[] = []
    mockApi({
      ...VIDEO_ROUTES,
      'GET /api/session': () => RUNNING,
      'POST /api/jobs': (init) => {
        submitted.push(JSON.parse(init?.body as string) as (typeof submitted)[number])
        return { id: 'j4' }
      },
    })
    const user = userEvent.setup()
    renderApp()
    expect(
      await screen.findByText('Needs an A100; this L4 session may run it slowly.'),
    ).toBeInTheDocument()
    expect(screen.getByText('Wan 2.2 T2V A14B')).toBeInTheDocument()
    // Every video mode is offered; this model starts from text.
    expect(screen.getByRole('button', { name: 'From text' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    expect(screen.getByRole('button', { name: 'From image' })).toHaveAttribute(
      'aria-pressed',
      'false',
    )

    await user.click(screen.getByRole('button', { name: 'Add LoRA' }))
    const sheet = screen.getByRole('dialog', { name: 'Add LoRA' })
    expect(within(sheet).queryByRole('button', { name: /grain/ })).not.toBeInTheDocument()
    await user.click(within(sheet).getByRole('button', { name: /motion/ }))
    fireEvent.change(screen.getByRole('slider', { name: 'motion low-noise weight' }), {
      target: { value: '0.5' },
    })
    expect(screen.getByRole('slider', { name: 'motion high-noise weight' })).toHaveValue('1')

    await user.type(screen.getByLabelText('Prompt'), 'surf')
    await user.click(screen.getByRole('button', { name: 'Generate' }))
    await screen.findByText('Queued 1 clip.')
    expect(submitted[0]?.spec.loras).toEqual([
      {
        high: { path: 'loras/wan22/motion_high_noise.safetensors', weight: 1 },
        low: { path: 'loras/wan22/motion_low_noise.safetensors', weight: 0.5 },
      },
    ])
  })

  it('applies a single-file LoRA tagged for an A14B variant to both experts', async () => {
    localStorage.setItem(
      'degas.create.draft',
      JSON.stringify({
        family: 'wan22',
        families: { wan22: { model: 'models/wan22/t2v-a14b/Wan2.2-T2V-A14B', params: {} } },
      }),
    )
    const submitted: { spec: { loras: unknown } }[] = []
    const single = 'loras/wan22/wan21_style.safetensors'
    mockApi({
      ...VIDEO_ROUTES,
      'GET /api/assets': () => [
        ...ASSETS,
        ...WAN_ASSETS,
        {
          path: single,
          family: 'wan22',
          kind: 'lora',
          size: 300e6,
          sidecar: { variants: ['t2v-a14b'], default_weight: 0.8 },
          preview_thumb: null,
        },
      ],
      'POST /api/jobs': (init) => {
        submitted.push(JSON.parse(init?.body as string) as (typeof submitted)[number])
        return { id: 'j5' }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: 'Add LoRA' }))
    const sheet = screen.getByRole('dialog', { name: 'Add LoRA' })
    await user.click(within(sheet).getByRole('button', { name: /wan21 style/i }))
    fireEvent.change(screen.getByRole('slider', { name: /wan21 style low-noise weight/i }), {
      target: { value: '0.5' },
    })

    await user.type(screen.getByLabelText('Prompt'), 'surf')
    await user.click(screen.getByRole('button', { name: 'Generate' }))
    await screen.findByText('Queued 1 clip.')
    expect(submitted[0]?.spec.loras).toEqual([
      { high: { path: single, weight: 0.8 }, low: { path: single, weight: 0.5 } },
    ])
  })

  it('imports a source image from a link without queueing a job', async () => {
    const variant = FAMILIES[0]?.variants[0]
    const jobs: unknown[] = []
    const fetched: unknown[] = []
    mockApi({
      'GET /api/families': () => [
        { ...FAMILIES[0], variants: [{ ...variant, modes: ['t2i', 'i2i'] }] },
      ],
      'POST /api/blobs/from-url': (init) => {
        fetched.push(JSON.parse(init?.body as string))
        return {
          sha256: 'u1',
          media_type: 'image/jpeg',
          width: 1200,
          height: 1200,
        }
      },
      'POST /api/jobs': (init) => {
        jobs.push(init?.body)
        return { id: 'j2' }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.type(await screen.findByLabelText('Prompt'), 'a lighthouse')
    await user.click(screen.getByRole('button', { name: 'From image' }))
    await user.click(await screen.findByRole('button', { name: /Choose an image/ }))
    const picker = screen.getByRole('dialog', { name: 'Choose image' })
    await user.click(within(picker).getByRole('tab', { name: 'Link' }))
    const link = 'https://example.com/is/image/x?wid=1200&hei=1200'
    await user.type(within(picker).getByLabelText('Image link'), `${link}{Enter}`)
    expect(await within(picker).findByRole('button', { name: 'Use image' })).toBeInTheDocument()
    expect(fetched).toEqual([{ url: link }])
    expect(jobs).toEqual([])
  })

  it('chooses how a batch gets its seeds', async () => {
    const submitted: { spec: { params: { seed: number } }; seed_mode: string }[] = []
    mockApi({
      'POST /api/jobs': (init) => {
        submitted.push(JSON.parse(init?.body as string) as (typeof submitted)[number])
        return { id: 'j1' }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.type(await screen.findByLabelText('Prompt'), 'a lighthouse')
    await user.type(screen.getByLabelText('Seed'), '1234')
    await user.click(screen.getByRole('button', { name: 'More images' }))

    // A batch can't repeat one seed: random seeds, or counting up from the seed.
    const random = screen.getByRole('button', { name: 'Random seeds' })
    expect(random).toHaveAttribute('aria-pressed', 'true')
    expect(screen.queryByLabelText('from')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Generate 2 images' }))
    await screen.findByText('Queued 2 images.')
    expect(submitted[0]?.seed_mode).toBe('random')
    expect(submitted[0]?.spec.params.seed).toBe(-1)

    await user.click(screen.getByRole('button', { name: 'Count up' }))
    expect(screen.getByLabelText('from')).toHaveValue(1234)
    await user.click(screen.getByRole('button', { name: 'Generate 2 images' }))
    await vi.waitFor(() => {
      expect(submitted).toHaveLength(2)
    })
    expect(submitted[1]?.seed_mode).toBe('increment')
    expect(submitted[1]?.spec.params.seed).toBe(1234)
  })
})
