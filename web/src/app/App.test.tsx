import { screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { mockApi } from '@/test/mockApi'
import { renderApp } from '@/test/render'

describe('App', () => {
  it('opens where a notification points', async () => {
    history.replaceState(null, '', '/?sheet=session')
    mockApi()
    renderApp()
    expect(await screen.findByRole('dialog', { name: 'GPU session' })).toBeInTheDocument()
    expect(location.search).toBe('')
  })

  it('reports an unreachable server', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.resolve(new Response(null, { status: 502 }))),
    )
    renderApp()
    expect((await screen.findAllByRole('alert'))[0]).toHaveTextContent('HTTP 502')
  })
})
