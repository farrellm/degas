import { fireEvent, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { ASSETS, FAMILIES, RUNNING } from '@/test/fixtures'
import { mockApi } from '@/test/mockApi'
import { renderApp } from '@/test/render'

describe('ControlNet', () => {
  it('guides an image with a depth trace of its source', async () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null)
    const traces: unknown[] = []
    const submitted: { spec: { control?: unknown } }[] = []
    const variant = FAMILIES[0]?.variants[0]
    mockApi({
      'GET /api/families': () => [
        {
          ...FAMILIES[0],
          supports_control: true,
          variants: [{ ...variant, modes: ['t2i', 'i2i'] }],
        },
      ],
      'GET /api/assets': () => [
        ...ASSETS,
        {
          path: 'controlnets/sdxl/canny.safetensors',
          family: 'sdxl',
          kind: 'controlnet',
          size: 2.5e9,
          sidecar: null,
          preview_thumb: null,
        },
        {
          path: 'controlnets/sdxl/diffusers-xl',
          family: 'sdxl',
          kind: 'controlnet',
          size: 2.5e9,
          sidecar: { label: 'Depth XL', control: 'depth' },
          preview_thumb: null,
        },
      ],
      'GET /api/session': () => RUNNING,
      'POST /api/preprocess': (init) => {
        traces.push(JSON.parse(init?.body as string))
        return { image: { sha256: 'dep', media_type: 'image/png', width: 1024, height: 1024 } }
      },
      'POST /api/jobs': (init) => {
        submitted.push(JSON.parse(init?.body as string) as (typeof submitted)[number])
        return { id: 'j3' }
      },
    })
    localStorage.setItem(
      'degas.create.draft',
      JSON.stringify({
        family: 'sdxl',
        families: {
          sdxl: {
            model: 'models/sdxl/juggernaut.safetensors',
            mode: 'i2i',
            loras: [],
            params: { prompt: 'a dancer', width: 1024, height: 1024, steps: 30 },
            source: { sha: 'abc', width: 1024, height: 1024 },
          },
        },
      }),
    )
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: 'Add ControlNet' }))
    const sheet = await screen.findByRole('dialog', { name: 'ControlNet' })
    expect(within(sheet).getByRole('button', { name: 'Depth' })).toBeDisabled()
    await user.click(within(sheet).getByRole('button', { name: 'Use the source' }))
    expect(within(sheet).getByRole('button', { name: 'As is' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    await user.click(within(sheet).getByRole('button', { name: 'Depth' }))
    expect(await within(sheet).findByRole('img', { name: 'Depth trace' })).toBeInTheDocument()
    expect(traces[0]).toEqual({ id: 'depth', image: 'abc', params: {} })
    // The trace picks the ControlNet that reads depth maps.
    expect(within(sheet).getByRole('button', { name: /Model Depth XL/ })).toBeInTheDocument()
    fireEvent.change(within(sheet).getByLabelText('Last step'), { target: { value: '24' } })
    expect(within(sheet).getByText('1–24 of 30')).toBeInTheDocument()
    await user.click(within(sheet).getByRole('button', { name: 'Done' }))

    expect(screen.getByText('Depth, weight 0.70, steps 1–24')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Generate' }))
    expect(await screen.findByText('Queued 1 image.')).toBeInTheDocument()
    expect(submitted[0]?.spec.control).toEqual([
      {
        controlnet: { path: 'controlnets/sdxl/diffusers-xl' },
        image: 'sha256:dep',
        fit: 'crop',
        scale: 0.7,
        start: 0,
        end: 0.8,
        preprocessor: { id: 'depth', source: 'sha256:abc', params: {} },
      },
    ])
  })
})
