import { fireEvent, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import { ASSETS, FAMILIES, RESULT, RUNNING } from '@/test/fixtures'
import { mockApi } from '@/test/mockApi'
import { renderApp } from '@/test/render'

describe('Image prompts', () => {
  it('takes the style of a cropped picture as an image prompt', async () => {
    const transforms: unknown[] = []
    const submitted: { spec: { image_prompts?: unknown } }[] = []
    const adapter = (name: string, purpose?: string) => ({
      path: `ip_adapters/sdxl/${name}`,
      family: 'sdxl',
      kind: 'ip_adapter',
      size: 8.5e8,
      sidecar: purpose ? { purpose } : null,
      preview_thumb: null,
    })
    mockApi({
      'GET /api/families': () => [{ ...FAMILIES[0], supports_image_prompts: true }],
      'GET /api/assets': () => [
        ...ASSETS,
        adapter('ip-adapter_sdxl_vit-h.safetensors'),
        adapter('ip-adapter-plus_sdxl_vit-h.safetensors'),
        adapter('portrait.safetensors', 'face'),
      ],
      'GET /api/results': () => ({ results: [RESULT], cursor: null }),
      'GET /api/blobs/abc/transform': () => ({ original: 'abc', ops: [] }),
      'POST /api/blobs/abc/transform': (init) => {
        transforms.push(JSON.parse(init?.body as string))
        return { sha256: 'sq', media_type: 'image/png', width: 832, height: 832 }
      },
      'POST /api/jobs': (init) => {
        submitted.push(JSON.parse(init?.body as string) as (typeof submitted)[number])
        return { id: 'j4' }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.type(await screen.findByLabelText('Prompt'), 'a lighthouse at dusk')
    await user.click(await screen.findByRole('button', { name: 'Add image prompt' }))
    const sheet = await screen.findByRole('dialog', { name: 'Image prompt' })
    // A new unit starts with the plus model, which reads the picture's detail.
    expect(within(sheet).getByRole('button', { name: /Model ip-adapter-plus/ })).toBeInTheDocument()
    await user.click(within(sheet).getByRole('button', { name: 'Choose a picture' }))

    // The picture is cropped square, as the model sees it.
    const picker = screen.getByRole('dialog', { name: 'Choose image' })
    await user.click(await within(picker).findByRole('button', { name: 'Image: a lighthouse' }))
    await user.click(within(picker).getByRole('button', { name: 'Crop' }))
    const editor = await screen.findByRole('dialog', { name: 'Crop' })
    const img = editor.querySelector('img')
    if (!img) throw new Error('no image in the editor')
    Object.defineProperty(img, 'naturalWidth', { value: 832 })
    Object.defineProperty(img, 'naturalHeight', { value: 1216 })
    fireEvent.load(img)
    expect(within(editor).getByRole('button', { name: '1:1' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    expect(await within(editor).findByText(/832 × 832 from 832 × 1216/)).toBeInTheDocument()
    expect(within(editor).queryByText(/The model sees the middle square/)).not.toBeInTheDocument()
    await user.click(within(editor).getByRole('button', { name: 'Apply' }))
    expect(transforms).toEqual([{ ops: [{ op: 'crop', x: 0, y: 192, w: 832, h: 832 }] }])

    const unit = await screen.findByRole('dialog', { name: 'Image prompt' })
    expect(within(unit).getByRole('img', { name: 'Picture 1' })).toBeInTheDocument()
    // Face picks a face model; Style goes back to the plus model at full weight.
    await user.click(within(unit).getByRole('button', { name: 'Face' }))
    expect(within(unit).getByRole('button', { name: /Model portrait/ })).toBeInTheDocument()
    await user.click(within(unit).getByRole('button', { name: 'Style' }))
    expect(within(unit).getByRole('button', { name: /Model ip-adapter-plus/ })).toBeInTheDocument()
    expect(within(unit).getByText(/the prompt decides what’s in the picture/)).toBeInTheDocument()
    fireEvent.change(within(unit).getByLabelText('Last step'), { target: { value: '24' } })
    await user.click(within(unit).getByRole('button', { name: 'Done' }))

    expect(screen.getByText('Style, weight 1.00, steps 1–24')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Generate' }))
    expect(await screen.findByText('Queued 1 image.')).toBeInTheDocument()
    expect(submitted[0]?.spec.image_prompts).toEqual([
      {
        adapter: { path: 'ip_adapters/sdxl/ip-adapter-plus_sdxl_vit-h.safetensors' },
        images: ['sha256:sq'],
        fit: 'crop',
        purpose: 'style',
        weight: 1,
        start: 0,
        end: 0.8,
      },
    ])
  })

  it('reads who a face is with FaceID, showing the face the GPU found', async () => {
    const faces: unknown[] = []
    const submitted: { spec: { image_prompts?: unknown } }[] = []
    const adapter = (name: string) => ({
      path: `ip_adapters/sdxl/${name}`,
      family: 'sdxl',
      kind: 'ip_adapter',
      size: 1.5e9,
      sidecar: null,
      preview_thumb: null,
    })
    mockApi({
      'GET /api/families': () => [{ ...FAMILIES[0], supports_image_prompts: true }],
      'GET /api/assets': () => [
        ...ASSETS,
        adapter('ip-adapter-plus_sdxl_vit-h.safetensors'),
        adapter('ip-adapter-plus-face_sdxl_vit-h.safetensors'),
        adapter('ip-adapter-faceid-plusv2_sdxl.bin'),
      ],
      'GET /api/session': () => RUNNING,
      'GET /api/results': () => ({ results: [RESULT], cursor: null }),
      'POST /api/preprocess': (init) => {
        faces.push(JSON.parse(init?.body as string))
        return {
          image: { sha256: 'crop', media_type: 'image/png', width: 224, height: 224 },
          faces: 2,
        }
      },
      'POST /api/jobs': (init) => {
        submitted.push(JSON.parse(init?.body as string) as (typeof submitted)[number])
        return { id: 'j5' }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.type(await screen.findByLabelText('Prompt'), 'a portrait in oils')
    await user.click(await screen.findByRole('button', { name: 'Add image prompt' }))
    const sheet = await screen.findByRole('dialog', { name: 'Image prompt' })
    await user.click(within(sheet).getByRole('button', { name: 'Face' }))
    // Face prefers FaceID over the CLIP face model, and starts it at 0.8.
    expect(
      within(sheet).getByRole('button', { name: /Model ip-adapter-faceid/ }),
    ).toBeInTheDocument()
    expect(within(sheet).getByText('0.80')).toBeInTheDocument()
    expect(within(sheet).getByText(/needn’t be cropped/)).toBeInTheDocument()
    fireEvent.change(within(sheet).getByLabelText('Face LoRA'), { target: { value: '0.5' } })

    await user.click(within(sheet).getByRole('button', { name: 'Choose a picture' }))
    const picker = screen.getByRole('dialog', { name: 'Choose image' })
    await user.click(await within(picker).findByRole('button', { name: 'Image: a lighthouse' }))
    await user.click(within(picker).getByRole('button', { name: 'Use image' }))
    const unit = await screen.findByRole('dialog', { name: 'Image prompt' })
    expect(
      await within(unit).findByRole('img', { name: 'The face in picture 1' }),
    ).toBeInTheDocument()
    expect(within(unit).getByText('The biggest of 2 faces')).toBeInTheDocument()
    expect(faces).toEqual([{ id: 'face', image: 'abc', params: {} }])
    await user.click(within(unit).getByRole('button', { name: 'Done' }))

    await user.click(screen.getByRole('button', { name: 'Generate' }))
    expect(await screen.findByText('Queued 1 image.')).toBeInTheDocument()
    expect(submitted[0]?.spec.image_prompts).toEqual([
      {
        adapter: { path: 'ip_adapters/sdxl/ip-adapter-faceid-plusv2_sdxl.bin' },
        images: ['sha256:abc'],
        fit: 'crop',
        purpose: 'all',
        weight: 0.8,
        start: 0,
        end: 1,
        structure: 1,
        lora_weight: 0.5,
      },
    ])
  })

  it('makes an image after a picture with FLUX.1 Redux, loosely by default', async () => {
    const submitted: { spec: { image_prompts?: unknown } }[] = []
    mockApi({
      'GET /api/families': () => [
        {
          ...FAMILIES[0],
          supports_image_prompts: true,
          image_prompt_options: {
            purposes: ['all'],
            areas: false,
            steps: false,
            faces: false,
            detail: true,
          },
        },
      ],
      'GET /api/assets': () => [
        ...ASSETS,
        {
          path: 'ip_adapters/sdxl/FLUX.1-Redux-dev',
          family: 'sdxl',
          kind: 'ip_adapter',
          size: 9.9e8,
          sidecar: null,
          preview_thumb: null,
        },
      ],
      'GET /api/results': () => ({ results: [RESULT], cursor: null }),
      'POST /api/jobs': (init) => {
        submitted.push(JSON.parse(init?.body as string) as (typeof submitted)[number])
        return { id: 'j6' }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.type(await screen.findByLabelText('Prompt'), 'the same harbour in winter')
    await user.click(await screen.findByRole('button', { name: 'Add image prompt' }))
    const sheet = await screen.findByRole('dialog', { name: 'Image prompt' })
    // Redux has no blocks to choose, no area and no step range: only how closely to follow.
    expect(within(sheet).queryByRole('button', { name: 'Style' })).not.toBeInTheDocument()
    expect(within(sheet).queryByRole('group', { name: 'Steps' })).not.toBeInTheDocument()
    expect(within(sheet).queryByText('Limit to an area')).not.toBeInTheDocument()
    expect(within(sheet).getByRole('button', { name: 'Loosely' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    expect(within(sheet).getByText('1.00')).toBeInTheDocument()
    await user.click(within(sheet).getByRole('button', { name: 'Just the gist' }))
    expect(within(sheet).getByText(/the prompt leads/)).toBeInTheDocument()
    await user.click(within(sheet).getByRole('button', { name: 'Choose a picture' }))
    const picker = screen.getByRole('dialog', { name: 'Choose image' })
    await user.click(await within(picker).findByRole('button', { name: 'Image: a lighthouse' }))
    await user.click(within(picker).getByRole('button', { name: 'Use image' }))
    await user.click(
      within(await screen.findByRole('dialog', { name: 'Image prompt' })).getByRole('button', {
        name: 'Done',
      }),
    )

    expect(screen.getByText('Just the gist, weight 1.00')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Generate' }))
    expect(await screen.findByText('Queued 1 image.')).toBeInTheDocument()
    expect(submitted[0]?.spec.image_prompts).toEqual([
      {
        adapter: { path: 'ip_adapters/sdxl/FLUX.1-Redux-dev' },
        images: ['sha256:abc'],
        fit: 'crop',
        purpose: 'all',
        weight: 1,
        start: 0,
        end: 1,
        downsample: 5,
      },
    ])
  })
})
