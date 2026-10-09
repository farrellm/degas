import { screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { JOB_DONE } from '@/test/fixtures'
import { mockApi, serverEvents } from '@/test/mockApi'
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

  it('keeps a server event it can’t read out of the cache, and refetches instead', async () => {
    const { send } = serverEvents()
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined)
    const fetch = mockApi({ 'GET /api/jobs': () => [{ ...JOB_DONE, status: 'queued' }] })
    const jobReads = () =>
      fetch.mock.calls.filter(
        ([url, init]) => url === '/api/jobs' && (init?.method ?? 'GET') === 'GET',
      ).length
    renderApp()
    expect(await screen.findByLabelText('1 in progress')).toBeInTheDocument()
    const before = jobReads()

    send({ type: 'job', job: { id: JOB_DONE.id, status: 'paused' } })
    await waitFor(() => expect(jobReads()).toBeGreaterThan(before))
    expect(screen.getByLabelText('1 in progress')).toBeInTheDocument()
    expect(warn).toHaveBeenCalledWith('unreadable server event', expect.any(String))
  })
})
