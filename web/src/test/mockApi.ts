import { vi } from 'vitest'

import { ASSETS, FAMILIES, SCHEMA, SESSION, WAN, WAN_ASSETS, WAN_SCHEMA } from './fixtures'

export class FakeEventSource {
  onmessage: ((e: MessageEvent<string>) => void) | null = null
  close() {
    // nothing to close
  }
}

export type Handler = (init?: RequestInit) => unknown

export function mockApi(overrides: Record<string, Handler> = {}) {
  const routes: Record<string, Handler> = {
    'GET /api/families': () => FAMILIES,
    'GET /api/families/sdxl/schema': () => SCHEMA,
    'GET /api/assets': () => ASSETS,
    'GET /api/drive': () => ({
      configured: true,
      authorized: true,
      error: null,
      indexed_at: '2026-09-27T12:00:00Z',
    }),
    'GET /api/session': () => SESSION,
    'GET /api/jobs': () => [],
    'GET /api/results': () => ({ results: [], cursor: null }),
    ...overrides,
  }
  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const href = input instanceof Request ? input.url : input.toString()
    const url = new URL(href, 'http://localhost')
    // Decoded, as the server routes it.
    const key = `${init?.method ?? 'GET'} ${decodeURIComponent(url.pathname)}`
    const handler = routes[key]
    if (!handler) return Promise.resolve(new Response(null, { status: 404 }))
    return Promise.resolve(Response.json(handler(init)))
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

export const VIDEO_ROUTES: Record<string, Handler> = {
  'GET /api/families': () => [...FAMILIES, WAN],
  'GET /api/families/wan22/schema': () => WAN_SCHEMA,
  'GET /api/assets': () => [...ASSETS, ...WAN_ASSETS],
}
