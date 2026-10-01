import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { LIBRARY_ITEM, PROMPT } from '@/test/fixtures'
import { mockApi } from '@/test/mockApi'
import { renderApp } from '@/test/render'

describe('Library', () => {
  it('remixes a kept image from the library, flagging a model that left Drive', async () => {
    const searches: string[] = []
    mockApi({
      'GET /api/library': () => ({ items: [LIBRARY_ITEM], cursor: null }),
    })
    const fetchMock = vi.mocked(fetch)
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: 'Library' }))
    expect(await screen.findByRole('region', { name: 'Today' })).toBeInTheDocument()
    await user.type(screen.getByLabelText('Search the library'), 'dusk')
    await vi.waitFor(() => {
      for (const [input] of fetchMock.mock.calls) {
        const url = new URL(
          input instanceof Request ? input.url : input.toString(),
          'http://localhost',
        )
        if (url.pathname === '/api/library') searches.push(url.searchParams.get('q') ?? '')
      }
      expect(searches).toContain('dusk')
    })

    await user.click(screen.getByRole('button', { name: 'Open a harbour at dusk' }))
    expect(screen.getByLabelText('Tags')).toHaveValue('sea')
    await user.click(screen.getByRole('button', { name: 'Remix' }))

    expect(await screen.findByLabelText('Prompt')).toHaveValue('a harbour at dusk')
    expect(screen.getByLabelText('Seed')).toHaveValue(77)
    expect(screen.getByRole('slider', { name: 'Film Grain v3 weight' })).toHaveValue('0.6')
    expect(screen.getByText('Not found in Drive. Pick another model.')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Generate' })).toBeDisabled()
  })

  it('opens a kept image without React warnings', async () => {
    const errors = vi.spyOn(console, 'error')
    mockApi({
      'GET /api/library': () => ({ items: [LIBRARY_ITEM], cursor: null }),
    })
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: 'Library' }))
    await user.click(await screen.findByRole('button', { name: 'Open a harbour at dusk' }))
    expect(screen.getByLabelText('Tags')).toBeInTheDocument()
    expect(errors).not.toHaveBeenCalled()
  })

  it('deletes a kept image after confirming', async () => {
    const deleted: string[] = []
    mockApi({
      'GET /api/library': () => ({ items: deleted.length ? [] : [LIBRARY_ITEM], cursor: null }),
      'DELETE /api/library/k1': () => {
        deleted.push('k1')
        return { deleted: true }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: 'Library' }))
    await user.click(await screen.findByRole('button', { name: 'Open a harbour at dusk' }))
    await user.click(screen.getByRole('button', { name: 'Delete' }))
    expect(deleted).toEqual([])
    await user.click(
      within(screen.getByRole('group', { name: 'Confirm delete' })).getByRole('button', {
        name: 'Delete',
      }),
    )
    expect(await screen.findByText('Nothing kept yet.')).toBeInTheDocument()
    expect(deleted).toEqual(['k1'])
  })

  it('renames and uses a saved prompt from the library', async () => {
    const edits: unknown[] = []
    mockApi({
      'GET /api/prompts': () => [PROMPT],
      'PATCH /api/prompts/p1': (init) => {
        edits.push(JSON.parse(init?.body as string))
        return { ...PROMPT, name: 'Dusk' }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: 'Library' }))
    await user.click(screen.getByRole('button', { name: 'Prompts' }))
    await user.click(await screen.findByRole('button', { name: 'Rename Harbour' }))
    const name = screen.getByLabelText('Prompt name')
    await user.clear(name)
    await user.type(name, 'Dusk{Enter}')
    expect(edits).toEqual([{ name: 'Dusk' }])

    await user.click(await screen.findByRole('button', { name: 'Use' }))
    expect(await screen.findByLabelText('Prompt')).toHaveValue('a harbour at dusk, oil painting')
  })

  it('opens a saved prompt to its full text on a tap', async () => {
    mockApi({ 'GET /api/prompts': () => [PROMPT] })
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: 'Library' }))
    await user.click(screen.getByRole('button', { name: 'Prompts' }))
    const text = await screen.findByRole('button', { name: /a harbour at dusk/, expanded: false })
    await user.click(text)
    expect(text).toHaveAttribute('aria-expanded', 'true')
    expect(text).toHaveClass('expanded')
    await user.click(text)
    expect(text).toHaveAttribute('aria-expanded', 'false')
  })
})
