import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'

const FAMILIES = [
  {
    id: 'sdxl',
    label: 'Stable Diffusion XL',
    media: 'image',
    lora_format: 'single',
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

function SESSION_BASE() {
  return {
    session: null,
    step: null,
    worker: null,
    idle_deadline: null,
    idle_timeout_min: 15,
    drive: { configured: true, authorized: true, error: null, push_error: null },
    gpus: ['T4', 'L4'],
  }
}

const SESSION = SESSION_BASE()

const ASSETS = [
  {
    path: 'models/sdxl/juggernaut.safetensors',
    family: 'sdxl',
    kind: 'model',
    size: 6_938_040_682,
    sidecar: { label: 'Juggernaut XL v10' },
    preview_thumb: null,
  },
  {
    path: 'models/sdxl/base.safetensors',
    family: 'sdxl',
    kind: 'model',
    size: 6_938_040_682,
    sidecar: null,
    preview_thumb: null,
  },
  {
    path: 'loras/sdxl/film.safetensors',
    family: 'sdxl',
    kind: 'lora',
    size: 144_000_000,
    sidecar: { label: 'Film Grain v3', trigger_words: ['filmgrain'], default_weight: 0.8 },
    preview_thumb: 'abc',
  },
]

const RUNNING = {
  ...SESSION_BASE(),
  session: {
    id: 's1',
    gpu: 'L4',
    high_mem: false,
    state: 'ready',
    started_at: '2026-09-27T12:00:00Z',
    ended_at: null,
    last_activity_at: '2026-09-27T12:00:00Z',
    error: null,
  },
  worker: {
    gpu: 'NVIDIA L4',
    vram_free: 20e9,
    vram_total: 24e9,
    disk_free: 150e9,
    cache: {
      used: 6_938_040_682,
      budget: 150e9,
      files: [{ path: 'models/sdxl/juggernaut.safetensors', size: 6_938_040_682, last_used: 1 }],
    },
  },
}

const SPEC = {
  family: 'sdxl',
  variant: 'base',
  mode: 't2i',
  model: { path: 'models/sdxl/juggernaut.safetensors' },
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
    expect(screen.getByRole('button', { name: 'Model Juggernaut XL v10' })).toBeInTheDocument()
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
          model: { path: 'models/sdxl/juggernaut.safetensors' },
          loras: [],
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

  it('picks a model, showing which ones are already on the GPU', async () => {
    mockApi({ 'GET /api/session': () => RUNNING })
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: 'Model Juggernaut XL v10' }))
    const sheet = screen.getByRole('dialog', { name: 'Model' })
    const current = within(sheet).getByRole('button', { name: /Juggernaut XL v10/ })
    expect(current).toHaveAttribute('aria-pressed', 'true')
    expect(current).toHaveTextContent('6.9 GB, on the GPU')
    const other = within(sheet).getByRole('button', { name: /^base/ })
    expect(other).toHaveTextContent('6.9 GB, about 100 s to copy')
    expect(within(sheet).getByText('Indexed', { exact: false })).toBeInTheDocument()
    await user.click(other)
    expect(screen.queryByRole('dialog', { name: 'Model' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Model base' })).toBeInTheDocument()
  })

  it('adds a LoRA, weights it, and taps its trigger word into the prompt', async () => {
    const submitted: { spec: { loras: unknown; params: { prompt: string } } }[] = []
    mockApi({
      'POST /api/jobs': (init) => {
        submitted.push(JSON.parse(init?.body as string) as (typeof submitted)[number])
        return { id: 'j1' }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.type(await screen.findByLabelText('Prompt'), 'a lighthouse')
    await user.click(screen.getByRole('button', { name: 'Add LoRA' }))
    await user.click(
      within(screen.getByRole('dialog', { name: 'Add LoRA' })).getByRole('button', {
        name: /Film Grain v3/,
      }),
    )
    const weight = screen.getByRole('slider', { name: 'Film Grain v3 weight' })
    expect(weight).toHaveValue('0.8') // the sidecar's default weight
    fireEvent.change(weight, { target: { value: '0.55' } })
    expect(screen.getByText('0.55')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Add “filmgrain” to the prompt' }))
    expect(screen.getByLabelText('Prompt')).toHaveValue('a lighthouse, filmgrain')

    await user.click(screen.getByRole('button', { name: 'Generate' }))
    await screen.findByText('Queued 1 image.')
    expect(submitted[0]?.spec.loras).toEqual([
      { path: 'loras/sdxl/film.safetensors', weight: 0.55 },
    ])

    await user.click(screen.getByRole('button', { name: 'Remove Film Grain v3' }))
    expect(screen.queryByRole('slider', { name: 'Film Grain v3 weight' })).not.toBeInTheDocument()
  })

  it('flags a LoRA from an old draft that is no longer in Drive', async () => {
    localStorage.setItem(
      'degas.create.draft',
      JSON.stringify({ loras: [{ path: 'loras/sdxl/gone.safetensors', weight: 1 }] }),
    )
    mockApi()
    renderApp()
    expect(await screen.findByText(/Not found in Drive/)).toBeInTheDocument()
    expect(screen.getByRole('slider', { name: 'gone weight' })).toBeInTheDocument()
  })

  it('lists what is on the GPU in the session sheet', async () => {
    mockApi({ 'GET /api/session': () => RUNNING })
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: /L4 session ready/ }))
    const section = screen.getByRole('region', { name: 'On the GPU' })
    expect(within(section).getByText('Juggernaut XL v10')).toBeInTheDocument()
    expect(within(section).getByText(/6.9 GB of 150 GB used/)).toBeInTheDocument()
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
