import { fireEvent, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import { RESULT } from '@/test/fixtures'
import { mockApi, VIDEO_ROUTES } from '@/test/mockApi'
import { renderApp } from '@/test/render'

describe('Crop editor', () => {
  it('makes a clip from a cropped image', async () => {
    const transforms: unknown[] = []
    const submitted: { spec: Record<string, unknown> }[] = []
    mockApi({
      ...VIDEO_ROUTES,
      'GET /api/results': () => ({ results: [RESULT], cursor: null }),
      'GET /api/blobs/abc/transform': () => ({ original: 'abc', ops: [] }),
      'POST /api/blobs/abc/transform': (init) => {
        transforms.push(JSON.parse(init?.body as string))
        return { sha256: 'd1', media_type: 'image/png', width: 1280, height: 704 }
      },
      'POST /api/jobs': (init) => {
        submitted.push(JSON.parse(init?.body as string) as (typeof submitted)[number])
        return { id: 'j2' }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: 'Video' }))
    expect(await screen.findByRole('button', { name: 'Model TI2V 5B' })).toBeInTheDocument()
    await user.type(screen.getByLabelText('Prompt'), 'the lighthouse beam sweeps')
    expect(screen.getByRole('button', { name: 'Generate' })).toBeEnabled()

    await user.click(screen.getByRole('button', { name: 'From image' }))
    expect(screen.getByRole('button', { name: 'Generate' })).toBeDisabled() // needs a source
    await user.click(screen.getByRole('button', { name: /Source Choose an image/ }))
    const picker = screen.getByRole('dialog', { name: 'Choose image' })
    await user.click(await within(picker).findByRole('button', { name: 'Image: a lighthouse' }))
    expect(within(picker).getByText('832 × 1216')).toBeInTheDocument()
    await user.click(within(picker).getByRole('button', { name: 'Crop' }))

    const editor = await screen.findByRole('dialog', { name: 'Crop' })
    const img = editor.querySelector('img')
    if (!img) throw new Error('no image in the editor')
    Object.defineProperty(img, 'naturalWidth', { value: 832 })
    Object.defineProperty(img, 'naturalHeight', { value: 1216 })
    fireEvent.load(img)
    expect(within(editor).getByRole('button', { name: 'Match 1280 × 704' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    expect(await within(editor).findByText(/1280 × 704 from 832 × 1216/)).toBeInTheDocument()
    expect(within(editor).getByText(/Scaled up 1\.5×|Scaled up 1\.6×/)).toBeInTheDocument()
    await user.click(within(editor).getByRole('button', { name: 'Apply' }))

    expect(await screen.findByRole('button', { name: 'Source 1280 × 704' })).toBeInTheDocument()
    const [{ ops }] = transforms as [{ ops: { op: string }[] }]
    expect(ops.map((o) => o.op)).toEqual(['crop', 'resize'])
    expect(ops[1]).toEqual({ op: 'resize', w: 1280, h: 704 })

    await user.click(screen.getByRole('button', { name: 'Generate' }))
    expect(await screen.findByText('Queued 1 clip.')).toBeInTheDocument()
    expect(submitted[0]?.spec).toMatchObject({
      family: 'wan22',
      variant: 'ti2v-5b',
      mode: 'i2v',
      model: { path: 'models/wan22/ti2v-5b' },
      inputs: { source: 'sha256:d1', fit: 'crop' },
    })
  })

  it('keeps a crop at its own size when Resize is off', async () => {
    const transforms: { ops: { op: string }[] }[] = []
    mockApi({
      ...VIDEO_ROUTES,
      'GET /api/results': () => ({ results: [RESULT], cursor: null }),
      'GET /api/blobs/abc/transform': () => ({ original: 'abc', ops: [] }),
      'POST /api/blobs/abc/transform': (init) => {
        transforms.push(JSON.parse(init?.body as string) as (typeof transforms)[number])
        return { sha256: 'd1', media_type: 'image/png', width: 832, height: 458 }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: 'Video' }))
    await user.click(await screen.findByRole('button', { name: 'From image' }))
    await user.click(screen.getByRole('button', { name: /Source Choose an image/ }))
    const picker = screen.getByRole('dialog', { name: 'Choose image' })
    await user.click(await within(picker).findByRole('button', { name: 'Image: a lighthouse' }))
    await user.click(within(picker).getByRole('button', { name: 'Crop' }))

    const editor = await screen.findByRole('dialog', { name: 'Crop' })
    const img = editor.querySelector('img')
    if (!img) throw new Error('no image in the editor')
    Object.defineProperty(img, 'naturalWidth', { value: 832 })
    Object.defineProperty(img, 'naturalHeight', { value: 1216 })
    fireEvent.load(img)
    const resize = await within(editor).findByRole('button', { name: 'Resize' })
    expect(resize).toHaveAttribute('aria-pressed', 'true')
    await user.click(resize)
    expect(resize).toHaveAttribute('aria-pressed', 'false')
    expect(within(editor).getByText(/fitted to 1280 × 704 when it’s used/)).toBeInTheDocument()
    await user.click(within(editor).getByRole('button', { name: 'Apply' }))

    expect(await screen.findByRole('button', { name: 'Source 832 × 458' })).toBeInTheDocument()
    expect(transforms[0]?.ops.map((o) => o.op)).toEqual(['crop'])
    // The form keeps the size the crop was made for, not the crop's own.
    const sizeRow = screen.getByRole('group', { name: 'Size' })
    expect(within(sizeRow).getByText('1280 × 704')).toBeInTheDocument()
  })
})
