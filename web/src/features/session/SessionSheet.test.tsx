import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import { RUNNING, SESSION } from '@/test/fixtures'
import { mockApi, VIDEO_ROUTES } from '@/test/mockApi'
import { renderApp } from '@/test/render'

describe('Session', () => {
  it('lists what is on the GPU in the session sheet', async () => {
    mockApi({ 'GET /api/session': () => RUNNING })
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: /L4 session ready/ }))
    const section = screen.getByRole('region', { name: 'On the GPU' })
    expect(within(section).getByText('Studio XL v10')).toBeInTheDocument()
    expect(within(section).getByText(/6.9 GB of 150 GB used/)).toBeInTheDocument()
  })

  it('starts a GPU session from the header', async () => {
    const started: unknown[] = []
    mockApi({
      'POST /api/session': (init) => {
        started.push(JSON.parse(init?.body as string))
        return SESSION
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: 'No GPU' }))
    await user.click(screen.getByRole('button', { name: /T4/ }))
    await user.click(screen.getByRole('button', { name: 'Start T4 session' }))
    expect(started).toEqual([{ gpu: 'T4', high_mem: false }])
  })

  it("dims GPUs below the Create model's minimum and starts on the first that meets it", async () => {
    const started: unknown[] = []
    mockApi({
      ...VIDEO_ROUTES,
      'GET /api/session': () => ({ ...SESSION, gpus: ['T4', 'L4', 'A100', 'H100'] }),
      'POST /api/session': (init) => {
        started.push(JSON.parse(init?.body as string))
        return SESSION
      },
    })
    localStorage.setItem('degas.session.gpu', 'L4')
    localStorage.setItem(
      'degas.create.draft',
      JSON.stringify({
        family: 'wan22',
        families: { wan22: { model: 'models/wan22/t2v-a14b/Wan2.2-T2V-A14B', loras: [] } },
      }),
    )
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: 'No GPU' }))
    const sheet = await screen.findByRole('dialog', { name: 'GPU session' })
    expect(within(sheet).getByText('Wan 2.2 T2V A14B needs an A100 or better.')).toBeInTheDocument()
    expect(within(sheet).getByRole('button', { name: /^L4/ })).toHaveAccessibleDescription(
      'Wan 2.2 T2V A14B needs an A100 or better.',
    )
    expect(within(sheet).getByRole('button', { name: /^H100/ })).not.toHaveAccessibleDescription()
    // The last-used L4 is too small, so the default rises to the A100.
    expect(within(sheet).getByRole('button', { name: 'Start A100 session' })).toBeInTheDocument()

    // Choosing a smaller one is allowed, with a warning.
    await user.click(within(sheet).getByRole('button', { name: /^L4/ }))
    expect(
      within(sheet).getByText('Wan 2.2 T2V A14B needs an A100; an L4 may run it slowly.'),
    ).toBeInTheDocument()
    await user.click(within(sheet).getByRole('button', { name: /^A100/ }))
    await user.click(within(sheet).getByRole('button', { name: 'Start A100 session' }))
    expect(started).toEqual([{ gpu: 'A100', high_mem: true }])
  })

  it('explains how to get notifications on a phone', async () => {
    mockApi()
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: /No GPU/ }))
    const section = await screen.findByRole('region', { name: 'Notifications' })
    expect(section).toHaveTextContent('add Degas to the Home Screen')
  })
})
