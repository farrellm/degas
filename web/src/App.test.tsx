import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'

const FAMILIES = [
  {
    id: 'sdxl',
    label: 'Stable Diffusion XL',
    media: 'image',
    variants: [
      {
        id: 'base',
        label: 'SDXL',
        min_gpu: 'T4',
        modes: ['t2i'],
        size_constraints: {
          multiple_of: 8,
          min_pixels: 262144,
          max_pixels: 2359296,
          presets: [
            [1024, 1024],
            [832, 1216],
          ],
        },
      },
    ],
  },
]

const SCHEMA = {
  type: 'object',
  required: ['prompt'],
  properties: {
    prompt: { type: 'string', title: 'Prompt', 'x-widget': 'prompt' },
    width: { type: 'integer', title: 'Width', default: 1024, 'x-widget': 'aspect' },
    height: { type: 'integer', title: 'Height', default: 1024, 'x-widget': 'aspect' },
    steps: {
      type: 'integer',
      title: 'Steps',
      default: 30,
      minimum: 1,
      maximum: 100,
      'x-widget': 'slider',
    },
    seed: { type: 'integer', title: 'Seed', default: -1, 'x-widget': 'seed' },
    scheduler: {
      type: 'string',
      title: 'Sampler',
      default: 'euler',
      enum: ['euler', 'ddim'],
      'x-enum-labels': ['Euler', 'DDIM'],
      'x-widget': 'select',
      'x-advanced': true,
    },
  },
}

const SESSION = {
  session: null,
  step: null,
  worker: null,
  idle_deadline: null,
  idle_timeout_min: 15,
  drive: { configured: true, authorized: true, error: null, push_error: null },
  gpus: ['T4', 'L4'],
}

const MODELS = [
  { path: 'models/sdxl/studio.safetensors', family: 'sdxl', kind: 'model', size: 1 },
]

const SPEC = {
  family: 'sdxl',
  variant: 'base',
  mode: 't2i',
  model: { path: 'models/sdxl/studio.safetensors' },
  params: {
    prompt: 'a lighthouse',
    width: 832,
    height: 1216,
    steps: 30,
    seed: -1,
    scheduler: 'euler',
  },
}

class FakeEventSource {
  onmessage: ((e: MessageEvent<string>) => void) | null = null
  close() {
    // nothing to close
  }
}

type Handler = (init?: RequestInit) => unknown

function mockApi(overrides: Record<string, Handler> = {}) {
  const routes: Record<string, Handler> = {
    'GET /api/families': () => FAMILIES,
    'GET /api/families/sdxl/schema': () => SCHEMA,
    'GET /api/assets': () => MODELS,
    'GET /api/session': () => SESSION,
    'GET /api/jobs': () => [],
    'GET /api/results': () => ({ results: [], cursor: null }),
    ...overrides,
  }
  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const href = input instanceof Request ? input.url : input.toString()
    const url = new URL(href, 'http://localhost')
    const key = `${init?.method ?? 'GET'} ${url.pathname}`
    const handler = routes[key]
    if (!handler) return Promise.resolve(new Response(null, { status: 404 }))
    return Promise.resolve(Response.json(handler(init)))
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

function renderApp() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>,
  )
}

describe('App', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.stubGlobal('EventSource', FakeEventSource)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('renders the generation form from the family schema', async () => {
    mockApi()
    renderApp()
    expect(await screen.findByLabelText('Prompt')).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: 'Model' })).toHaveValue(
      'models/sdxl/studio.safetensors',
    )
    expect(screen.getByRole('slider', { name: /Steps/ })).toHaveValue('30')
    expect(screen.getByRole('button', { name: '832×1216' })).toBeInTheDocument()
    expect(screen.getByText(/No GPU is running/)).toBeInTheDocument()
  })

  it('submits a job and confirms it was queued', async () => {
    const submitted: unknown[] = []
    mockApi({
      'POST /api/jobs': (init) => {
        submitted.push(JSON.parse(init?.body as string))
        return { id: 'j1' }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.type(await screen.findByLabelText('Prompt'), 'a lighthouse')
    await user.click(screen.getByRole('button', { name: '832×1216' }))
    await user.click(screen.getByRole('button', { name: 'Generate' }))

    expect(await screen.findByText('Queued 1 image.')).toBeInTheDocument()
    expect(submitted).toEqual([
      {
        spec: {
          family: 'sdxl',
          variant: 'base',
          mode: 't2i',
          model: { path: 'models/sdxl/studio.safetensors' },
          params: {
            prompt: 'a lighthouse',
            width: 832,
            height: 1216,
            steps: 30,
            seed: -1,
            scheduler: 'euler',
          },
        },
        batch_count: 1,
        seed_mode: 'increment',
      },
    ])
  })

  it('keeps Generate disabled until there is a prompt', async () => {
    mockApi()
    const user = userEvent.setup()
    renderApp()
    const generate = await screen.findByRole('button', { name: 'Generate' })
    expect(generate).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'More images' }))
    await user.type(screen.getByLabelText('Prompt'), 'a lighthouse')
    expect(screen.getByRole('button', { name: 'Generate 2 images' })).toBeEnabled()
  })

  it('shows job progress in the results feed', async () => {
    mockApi({
      'GET /api/jobs': () => [
        {
          id: 'j1',
          status: 'running',
          spec: SPEC,
          seeds: [1, 2],
          progress: { job: 'j1', item: 1, phase: 'denoise', step: 12, steps: 30 },
          error: null,
        },
      ],
    })
    const user = userEvent.setup()
    renderApp()
    await user.click(screen.getByRole('button', { name: /Results/ }))
    expect(await screen.findByText('Denoising 12/30')).toBeInTheDocument()
    expect(screen.getByText('a lighthouse')).toBeInTheDocument()
    expect(screen.getByRole('img', { name: 'Image 2: Denoising 12/30' })).toBeInTheDocument()
  })

  it("reuses a result's settings in the form", async () => {
    mockApi({
      'GET /api/results': () => ({
        results: [
          {
            id: 'r1',
            job_id: 'j1',
            item_index: 0,
            blob_sha: 'abc',
            media_type: 'image/png',
            seed: 1234,
            width: 832,
            height: 1216,
            created_at: '2026-09-27T12:00:00Z',
            expires_at: null,
            spec: SPEC,
          },
        ],
        cursor: null,
      }),
    })
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: /Results/ }))
    await user.click(await screen.findByRole('button', { name: /Open image 1, seed 1234/ }))
    await user.click(screen.getByRole('button', { name: 'Reuse settings' }))
    expect(await screen.findByLabelText('Prompt')).toHaveValue('a lighthouse')
    expect(screen.getByLabelText('Seed')).toHaveValue(1234)
    expect(screen.getByRole('button', { name: '832×1216' })).toHaveAttribute('aria-pressed', 'true')
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

  it('reports an unreachable server', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.resolve(new Response(null, { status: 502 }))),
    )
    renderApp()
    expect((await screen.findAllByRole('alert'))[0]).toHaveTextContent('HTTP 502')
  })
})
