import '@testing-library/jest-dom/vitest'

import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach, vi } from 'vitest'

import { setDiscretion } from '@/lib/discretion'

import { FakeEventSource } from './mockApi'

// Every test starts with nothing remembered and no live event stream.
beforeEach(() => {
  localStorage.clear()
  vi.stubGlobal('EventSource', FakeEventSource)
})

afterEach(() => cleanup())

// Registered last, so it runs first: before the tree is unmounted, as when it was each suite's own.
afterEach(() => {
  setDiscretion(false)
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})
