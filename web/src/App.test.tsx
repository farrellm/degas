import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'

describe('App', () => {
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('shows the server version', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.resolve(Response.json({ status: 'ok', version: '0.1.0' }))),
    )
    render(<App />)
    expect(await screen.findByText('Server 0.1.0')).toBeInTheDocument()
  })

  it('reports an unreachable server', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.resolve(new Response(null, { status: 502 }))),
    )
    render(<App />)
    expect(await screen.findByRole('alert')).toHaveTextContent('HTTP 502')
  })
})
