import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { ASSETS, FAMILIES, RUNNING } from '@/test/fixtures'
import { mockApi } from '@/test/mockApi'
import { renderApp } from '@/test/render'

describe('Mask editor', () => {
  it('finds a described selection without submitting the Create form', async () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null)
    const selects: unknown[] = []
    const jobs: unknown[] = []
    const modes = ['t2i', 'i2i', 'inpaint', 'outpaint']
    const variant = FAMILIES[0]?.variants[0]
    mockApi({
      'GET /api/families': () => [{ ...FAMILIES[0], variants: [{ ...variant, modes }] }],
      'GET /api/assets': () => [
        ...ASSETS,
        { path: 'preprocessors/sam3', family: null, kind: 'preprocessor', size: 1, sidecar: null },
      ],
      'GET /api/session': () => RUNNING,
      'POST /api/preprocess': (init) => {
        selects.push(JSON.parse(init?.body as string))
        return { candidates: [], chosen: null }
      },
      'POST /api/jobs': (init) => {
        jobs.push(JSON.parse(init?.body as string))
        return { id: 'j9' }
      },
    })
    localStorage.setItem(
      'degas.create.draft',
      JSON.stringify({
        family: 'sdxl',
        families: {
          sdxl: {
            model: 'models/sdxl/studio.safetensors',
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
    await user.click(await screen.findByRole('button', { name: /Mask Paint the area to redraw/ }))
    const editor = await screen.findByRole('dialog', { name: 'Mask' })
    await user.click(within(editor).getByRole('button', { name: 'Select' }))
    const describe = await within(editor).findByLabelText('Describe what to select')
    await user.type(describe, 'the lighthouse{Enter}')
    await waitFor(() => {
      expect(selects).toHaveLength(1)
    })
    await user.click(within(editor).getByRole('button', { name: 'Find' }))
    await waitFor(() => {
      expect(selects).toHaveLength(2)
    })
    expect(selects[0]).toMatchObject({ id: 'sam', params: { text: 'the lighthouse' } })
    expect(screen.getByRole('dialog', { name: 'Mask' })).toBeInTheDocument()
    expect(jobs).toHaveLength(0)
  })
})
