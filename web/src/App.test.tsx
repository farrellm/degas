import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { StrictMode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { installShield, setDiscretion } from './discretion'

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
    vae_fp32: {
      type: 'boolean',
      title: 'Built-in VAE in float32',
      description: "The checkpoint's own VAE instead of the fp16 fix. Slower.",
      default: false,
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
    path: 'models/sdxl/studio.safetensors',
    family: 'sdxl',
    kind: 'model',
    size: 6_938_040_682,
    sidecar: { label: 'Studio XL v10' },
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
      files: [{ path: 'models/sdxl/studio.safetensors', size: 6_938_040_682, last_used: 1 }],
    },
  },
}

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

const JOB_DONE = {
  id: 'j1',
  status: 'done',
  spec: SPEC,
  seeds: [1234],
  created_at: '2026-09-27T12:00:00Z',
  progress: null,
  error: null,
}

const RESULT = {
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
  library_id: null,
  spec: SPEC,
}

const LIBRARY_ITEM = {
  id: 'k1',
  kind: 'image',
  blob_sha: 'abc',
  media_type: 'image/png',
  width: 832,
  height: 1216,
  config: {
    ...SPEC,
    degas_version: 1,
    model: { path: 'models/sdxl/retired.safetensors' },
    loras: [{ path: 'loras/sdxl/film.safetensors', weight: 0.6 }],
    params: { ...SPEC.params, prompt: 'a harbour at dusk', seed: 77 },
  },
  title: null,
  tags: ['sea'],
  created_at: new Date().toISOString(),
  source_result_id: null,
}

const PROMPT = {
  id: 'p1',
  name: 'Harbour',
  prompt: 'a harbour at dusk, oil painting',
  negative_prompt: '',
  family: 'sdxl',
  tags: [],
  created_at: '2026-09-27T12:00:00Z',
}

const WAN = {
  id: 'wan22',
  label: 'Wan 2.2',
  media: 'video',
  lora_format: 'paired_hi_lo',
  variants: [
    {
      id: 'ti2v-5b',
      label: 'Wan 2.2 TI2V 5B',
      min_gpu: 'L4',
      modes: ['t2v', 'i2v'],
      model_dir: 'models/wan22/ti2v-5b',
      lora_format: 'single',
      size_constraints: {
        multiple_of: 32,
        min_pixels: 399360,
        max_pixels: 942080,
        presets: [
          [1280, 704],
          [704, 1280],
        ],
      },
    },
    {
      id: 't2v-a14b',
      label: 'Wan 2.2 T2V A14B',
      min_gpu: 'A100',
      modes: ['t2v'],
      model_dir: 'models/wan22/t2v-a14b',
      lora_format: 'paired_hi_lo',
      size_constraints: {
        multiple_of: 16,
        min_pixels: 230400,
        max_pixels: 921600,
        presets: [
          [1280, 720],
          [832, 480],
        ],
      },
    },
  ],
}

const WAN_SCHEMA = {
  type: 'object',
  required: ['prompt'],
  properties: {
    prompt: { type: 'string', title: 'Prompt', 'x-widget': 'prompt' },
    width: { type: 'integer', title: 'Width', default: 1280, 'x-widget': 'aspect' },
    height: { type: 'integer', title: 'Height', default: 704, 'x-widget': 'aspect' },
    num_frames: {
      type: 'integer',
      title: 'Frames',
      default: 121,
      minimum: 17,
      maximum: 121,
      'x-step': 4,
      'x-widget': 'slider',
    },
    fps: { type: 'integer', title: 'Frame rate', default: 24, 'x-widget': 'slider' },
    steps: { type: 'integer', title: 'Steps', default: 50, 'x-widget': 'slider' },
    cfg: { type: 'number', title: 'CFG', default: 5, 'x-widget': 'slider' },
    seed: { type: 'integer', title: 'Seed', default: -1, 'x-widget': 'seed' },
  },
}

const WAN_ASSETS = [
  {
    path: 'models/wan22/ti2v-5b',
    family: 'wan22',
    kind: 'model',
    size: 32e9,
    sidecar: { label: 'TI2V 5B' },
    preview_thumb: null,
  },
  {
    path: 'models/wan22/t2v-a14b/Wan2.2-T2V-A14B',
    family: 'wan22',
    kind: 'model',
    size: 120e9,
    sidecar: null,
    preview_thumb: null,
  },
  ...['motion_high_noise', 'motion_low_noise', 'grain'].map((name) => ({
    path: `loras/wan22/${name}.safetensors`,
    family: 'wan22',
    kind: 'lora',
    size: 300e6,
    sidecar: null,
    preview_thumb: null,
  })),
]

const WAN_SPEC = {
  family: 'wan22',
  variant: 'ti2v-5b',
  mode: 't2v',
  model: { path: 'models/wan22/ti2v-5b' },
  params: {
    prompt: 'waves on the harbour wall',
    width: 1280,
    height: 704,
    num_frames: 121,
    fps: 24,
    steps: 50,
    cfg: 5,
    seed: 5,
  },
}

const VIDEO_RESULT = {
  ...RESULT,
  id: 'v1',
  job_id: 'j9',
  blob_sha: 'vid',
  media_type: 'video/mp4',
  seed: 5,
  width: 1280,
  height: 704,
  duration: 5.04,
  segments: null,
  spec: WAN_SPEC,
}

const VIDEO_ROUTES: Record<string, Handler> = {
  'GET /api/families': () => [...FAMILIES, WAN],
  'GET /api/families/wan22/schema': () => WAN_SCHEMA,
  'GET /api/assets': () => [...ASSETS, ...WAN_ASSETS],
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
  // StrictMode as in main.tsx, so effects that don't survive a remount show up here.
  return render(
    <StrictMode>
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>
    </StrictMode>,
  )
}

describe('App', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.stubGlobal('EventSource', FakeEventSource)
  })

  afterEach(() => {
    setDiscretion(false)
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('renders the generation form from the family schema', async () => {
    mockApi()
    renderApp()
    expect(await screen.findByLabelText('Prompt')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Model Studio XL v10' })).toBeInTheDocument()
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
    await user.click(screen.getByText('More settings'))
    await user.click(screen.getByRole('checkbox', { name: /Built-in VAE in float32/ }))
    await user.click(screen.getByRole('button', { name: 'Generate' }))

    expect(await screen.findByText('Queued 1 image.')).toBeInTheDocument()
    expect(submitted).toEqual([
      {
        spec: {
          family: 'sdxl',
          variant: 'base',
          mode: 't2i',
          model: { path: 'models/sdxl/studio.safetensors' },
          loras: [],
          params: {
            prompt: 'a lighthouse',
            width: 832,
            height: 1216,
            steps: 30,
            seed: -1,
            scheduler: 'euler',
            vae_fp32: true,
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
    await user.click(await screen.findByRole('button', { name: 'Model Studio XL v10' }))
    const sheet = screen.getByRole('dialog', { name: 'Model' })
    const current = within(sheet).getByRole('button', { name: /Studio XL v10/ })
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
    expect(within(section).getByText('Studio XL v10')).toBeInTheDocument()
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

  it("remixes a result's settings in the form", async () => {
    mockApi({ 'GET /api/results': () => ({ results: [RESULT], cursor: null }) })
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: /Results/ }))
    await user.click(await screen.findByRole('button', { name: /Open image 1, seed 1234/ }))
    await user.click(screen.getByRole('button', { name: 'Remix' }))
    expect(await screen.findByLabelText('Prompt')).toHaveValue('a lighthouse')
    expect(screen.getByLabelText('Seed')).toHaveValue(1234)
    expect(screen.getByRole('button', { name: '832×1216' })).toHaveAttribute('aria-pressed', 'true')
  })

  it('keeps a result, marking its tile and group', async () => {
    let kept = false
    const now = Date.now()
    const expiring = { ...RESULT, expires_at: new Date(now + 3_600_000).toISOString() }
    mockApi({
      'GET /api/jobs': () => [{ ...JOB_DONE }],
      'GET /api/results': () => ({
        results: [{ ...expiring, library_id: kept ? 'k1' : null }],
        cursor: null,
      }),
      'POST /api/results/r1/save': () => {
        kept = true
        return LIBRARY_ITEM
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: /Results/ }))
    expect(await screen.findByText('60 min left')).toHaveClass('soon')
    await user.click(screen.getByRole('button', { name: /Open image 1/ }))
    await user.click(screen.getByRole('button', { name: 'Keep' }))
    const keptButton = await screen.findByRole('button', { name: 'Kept' })
    expect(keptButton).toHaveAttribute('aria-pressed', 'true')
    await user.click(screen.getByRole('button', { name: 'Close' }))
    expect(screen.getByRole('button', { name: /Open image 1, seed 1234, kept/ })).toHaveClass(
      'kept',
    )
    expect(screen.getByText('Kept')).toBeInTheDocument()
  })

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

  it('saves the current prompt and swaps in a saved one', async () => {
    const saved: unknown[] = []
    mockApi({
      'GET /api/prompts': () => [PROMPT],
      'POST /api/prompts': (init) => {
        saved.push(JSON.parse(init?.body as string))
        return { ...PROMPT, id: 'p2' }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.type(await screen.findByLabelText('Prompt'), 'a lighthouse')
    await user.click(screen.getByRole('button', { name: 'Prompts' }))
    const sheet = screen.getByRole('dialog', { name: 'Saved prompts' })
    await user.click(within(sheet).getByRole('button', { name: 'Save this prompt' }))
    expect(await within(sheet).findByRole('button', { name: 'Saved' })).toBeDisabled()
    expect(saved).toEqual([{ prompt: 'a lighthouse', negative_prompt: '', family: 'sdxl' }])

    await user.click(within(sheet).getByRole('button', { name: /Harbour/ }))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(screen.getByLabelText('Prompt')).toHaveValue('a harbour at dusk, oil painting')
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

  it('makes a clip from a cropped image', async () => {
    const transforms: unknown[] = []
    const submitted: { spec: Record<string, unknown> }[] = []
    mockApi({
      ...VIDEO_ROUTES,
      'GET /api/results': () => ({ results: [RESULT], cursor: null }),
      'GET /api/blobs/abc/transform': () => ({ original: 'abc', ops: [] }),
      'POST /api/blobs/abc/transform': (init) => {
        transforms.push(JSON.parse(init?.body as string))
        return { sha256: 'd1', media_type: 'image/png', width: 1280, height: 704 }
      },
      'POST /api/jobs': (init) => {
        submitted.push(JSON.parse(init?.body as string) as (typeof submitted)[number])
        return { id: 'j2' }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: 'Video' }))
    expect(await screen.findByRole('button', { name: 'Model TI2V 5B' })).toBeInTheDocument()
    await user.type(screen.getByLabelText('Prompt'), 'the lighthouse beam sweeps')
    expect(screen.getByRole('button', { name: 'Generate' })).toBeEnabled()

    await user.click(screen.getByRole('button', { name: 'From image' }))
    expect(screen.getByRole('button', { name: 'Generate' })).toBeDisabled() // needs a source
    await user.click(screen.getByRole('button', { name: /Source Choose an image/ }))
    const picker = screen.getByRole('dialog', { name: 'Choose image' })
    await user.click(await within(picker).findByRole('button', { name: 'Image: a lighthouse' }))
    expect(within(picker).getByText('832 × 1216')).toBeInTheDocument()
    await user.click(within(picker).getByRole('button', { name: 'Crop' }))

    const editor = await screen.findByRole('dialog', { name: 'Crop' })
    const img = editor.querySelector('img')
    if (!img) throw new Error('no image in the editor')
    Object.defineProperty(img, 'naturalWidth', { value: 832 })
    Object.defineProperty(img, 'naturalHeight', { value: 1216 })
    fireEvent.load(img)
    expect(within(editor).getByRole('button', { name: 'Match 1280 × 704' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    expect(await within(editor).findByText(/1280 × 704 from 832 × 1216/)).toBeInTheDocument()
    expect(within(editor).getByText(/Scaled up 1\.5×|Scaled up 1\.6×/)).toBeInTheDocument()
    await user.click(within(editor).getByRole('button', { name: 'Apply' }))

    expect(await screen.findByRole('button', { name: 'Source 1280 × 704' })).toBeInTheDocument()
    const [{ ops }] = transforms as [{ ops: { op: string }[] }]
    expect(ops.map((o) => o.op)).toEqual(['crop', 'resize'])
    expect(ops[1]).toEqual({ op: 'resize', w: 1280, h: 704 })

    await user.click(screen.getByRole('button', { name: 'Generate' }))
    expect(await screen.findByText('Queued 1 clip.')).toBeInTheDocument()
    expect(submitted[0]?.spec).toMatchObject({
      family: 'wan22',
      variant: 'ti2v-5b',
      mode: 'i2v',
      model: { path: 'models/wan22/ti2v-5b' },
      inputs: { source: 'sha256:d1', fit: 'crop' },
    })
  })

  it('outpaints around a placed image, and needs a mask to inpaint', async () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null)
    const submitted: { spec: { mode: string; inputs?: unknown } }[] = []
    const modes = ['t2i', 'i2i', 'inpaint', 'outpaint']
    const variant = FAMILIES[0]?.variants[0]
    mockApi({
      'GET /api/families': () => [{ ...FAMILIES[0], variants: [{ ...variant, modes }] }],
      'POST /api/jobs': (init) => {
        submitted.push(JSON.parse(init?.body as string) as (typeof submitted)[number])
        return { id: 'j2' }
      },
    })
    localStorage.setItem(
      'degas.create.draft',
      JSON.stringify({
        family: 'sdxl',
        families: {
          sdxl: {
            model: 'models/sdxl/studio.safetensors',
            mode: 'outpaint',
            loras: [],
            params: { prompt: 'a lighthouse', width: 1344, height: 768 },
            source: { sha: 'abc', width: 1024, height: 1024 },
          },
        },
      }),
    )
    const user = userEvent.setup()
    renderApp()
    expect(await screen.findByText('+288 px left, +288 px right')).toBeInTheDocument()
    expect(screen.queryByLabelText('Fit')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Left' }))
    expect(screen.getByText('+576 px right')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Generate' }))
    expect(await screen.findByText('Queued 1 image.')).toBeInTheDocument()
    expect(submitted[0]?.spec).toMatchObject({
      mode: 'outpaint',
      inputs: { source: 'sha256:abc', place: { x: 0, y: 0, w: 768, h: 768 } },
    })

    await user.click(screen.getByRole('button', { name: 'Inpaint' }))
    expect(screen.getByRole('button', { name: 'Generate' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: /Mask Paint the area to redraw/ }))
    const editor = await screen.findByRole('dialog', { name: 'Mask' })
    await user.click(within(editor).getByRole('button', { name: 'Select' }))
    expect(within(editor).getByText(/Put SAM 3 in Drive/)).toBeInTheDocument()
    await user.click(within(editor).getByRole('button', { name: 'Cancel' }))
    expect(screen.queryByRole('dialog', { name: 'Mask' })).not.toBeInTheDocument()
  })

  it('switches to Qwen from the model picker and edits with ordered images', async () => {
    const submitted: { spec: Record<string, unknown> }[] = []
    const transforms: { ops: unknown[] }[] = []
    const qwen = {
      id: 'qwen21',
      label: 'Qwen-Image 2.1',
      media: 'image',
      lora_format: 'single',
      variants: [
        {
          id: 'base',
          label: 'Qwen-Image 2.1',
          min_gpu: 'L4',
          modes: ['t2i', 'edit', 'inpaint'],
          model_dir: null,
          size_constraints: {
            multiple_of: 32,
            min_pixels: 262144,
            max_pixels: 4300800,
            presets: [[1024, 1024]],
          },
        },
      ],
    }
    const harbour = { ...RESULT, id: 'r2', blob_sha: 'def', width: 640, height: 480 }
    harbour.spec = { ...SPEC, params: { ...SPEC.params, prompt: 'a harbour' } }
    mockApi({
      'GET /api/families': () => [...FAMILIES, qwen],
      'GET /api/families/qwen21/schema': () => SCHEMA,
      'GET /api/assets': () => [
        ...ASSETS,
        {
          path: 'models/qwen21/Qwen-Image-2.1',
          family: 'qwen21',
          kind: 'model',
          size: 36e9,
          sidecar: null,
          preview_thumb: null,
        },
      ],
      'GET /api/results': () => ({ results: [RESULT, harbour], cursor: null }),
      'GET /api/blobs/def/transform': () => ({ original: 'def', ops: [] }),
      'POST /api/blobs/def/transform': (init) => {
        transforms.push(JSON.parse(init?.body as string) as (typeof transforms)[number])
        return { sha256: 'd2', media_type: 'image/png', width: 640, height: 480 }
      },
      'POST /api/jobs': (init) => {
        submitted.push(JSON.parse(init?.body as string) as (typeof submitted)[number])
        return { id: 'j2' }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: 'Model Studio XL v10' }))
    const sheet = screen.getByRole('dialog', { name: 'Model' })
    const row = within(sheet).getByRole('button', { name: /^Qwen-Image-2\.1/ })
    expect(row).toHaveTextContent('Qwen-Image 2.1')
    await user.click(row)
    expect(await screen.findByRole('button', { name: 'Edit' })).toBeInTheDocument()

    await user.type(screen.getByLabelText('Prompt'), 'the boat from image 2 in the bay')
    await user.click(screen.getByRole('button', { name: 'Edit' }))
    await user.click(screen.getByRole('button', { name: /Image 1 Choose an image/ }))
    let picker = screen.getByRole('dialog', { name: 'Choose image' })
    await user.click(await within(picker).findByRole('button', { name: 'Image: a lighthouse' }))
    await user.click(within(picker).getByRole('button', { name: 'Use image' }))

    // A reference crops freely, and keeps its own size: the model sizes it.
    await user.click(screen.getByRole('button', { name: 'Add image' }))
    picker = screen.getByRole('dialog', { name: 'Choose image' })
    await user.click(await within(picker).findByRole('button', { name: 'Image: a harbour' }))
    await user.click(within(picker).getByRole('button', { name: 'Crop' }))
    const editor = await screen.findByRole('dialog', { name: 'Crop' })
    const img = editor.querySelector('img')
    if (!img) throw new Error('no image in the editor')
    Object.defineProperty(img, 'naturalWidth', { value: 640 })
    Object.defineProperty(img, 'naturalHeight', { value: 480 })
    fireEvent.load(img)
    expect(within(editor).getByRole('button', { name: 'Free' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    expect(within(editor).queryByRole('button', { name: /^Match/ })).not.toBeInTheDocument()
    expect(await within(editor).findByText(/640 × 480 from 640 × 480/)).toBeInTheDocument()
    expect(within(editor).getByText(/The model scales it up 1\.8×/)).toBeInTheDocument()
    await user.click(within(editor).getByRole('button', { name: 'Apply' }))
    expect(transforms).toEqual([{ ops: [] }])

    await user.click(screen.getByRole('button', { name: 'Add image' }))
    picker = screen.getByRole('dialog', { name: 'Choose image' })
    await user.click(await within(picker).findByRole('button', { name: 'Image: a lighthouse' }))
    await user.click(within(picker).getByRole('button', { name: 'Use image' }))
    expect(screen.getByText('640 × 480')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Crop image 3' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Move image 2 earlier' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Move image 3 earlier' }))

    await user.click(screen.getByRole('button', { name: 'Generate' }))
    expect(await screen.findByText('Queued 1 image.')).toBeInTheDocument()
    expect(submitted[0]?.spec).toMatchObject({
      family: 'qwen21',
      mode: 'edit',
      model: { path: 'models/qwen21/Qwen-Image-2.1' },
      inputs: { source: 'sha256:abc', refs: ['sha256:abc', 'sha256:d2'] },
    })
  })

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

  it('guides an image with a depth trace of its source', async () => {
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null)
    const traces: unknown[] = []
    const submitted: { spec: { control?: unknown } }[] = []
    const variant = FAMILIES[0]?.variants[0]
    mockApi({
      'GET /api/families': () => [
        {
          ...FAMILIES[0],
          supports_control: true,
          variants: [{ ...variant, modes: ['t2i', 'i2i'] }],
        },
      ],
      'GET /api/assets': () => [
        ...ASSETS,
        {
          path: 'controlnets/sdxl/canny.safetensors',
          family: 'sdxl',
          kind: 'controlnet',
          size: 2.5e9,
          sidecar: null,
          preview_thumb: null,
        },
        {
          path: 'controlnets/sdxl/diffusers-xl',
          family: 'sdxl',
          kind: 'controlnet',
          size: 2.5e9,
          sidecar: { label: 'Depth XL', control: 'depth' },
          preview_thumb: null,
        },
      ],
      'GET /api/session': () => RUNNING,
      'POST /api/preprocess': (init) => {
        traces.push(JSON.parse(init?.body as string))
        return { image: { sha256: 'dep', media_type: 'image/png', width: 1024, height: 1024 } }
      },
      'POST /api/jobs': (init) => {
        submitted.push(JSON.parse(init?.body as string) as (typeof submitted)[number])
        return { id: 'j3' }
      },
    })
    localStorage.setItem(
      'degas.create.draft',
      JSON.stringify({
        family: 'sdxl',
        families: {
          sdxl: {
            model: 'models/sdxl/studio.safetensors',
            mode: 'i2i',
            loras: [],
            params: { prompt: 'a dancer', width: 1024, height: 1024, steps: 30 },
            source: { sha: 'abc', width: 1024, height: 1024 },
          },
        },
      }),
    )
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: 'Add ControlNet' }))
    const sheet = await screen.findByRole('dialog', { name: 'ControlNet' })
    expect(within(sheet).getByRole('button', { name: 'Depth' })).toBeDisabled()
    await user.click(within(sheet).getByRole('button', { name: 'Use the source' }))
    expect(within(sheet).getByRole('button', { name: 'As is' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    await user.click(within(sheet).getByRole('button', { name: 'Depth' }))
    expect(await within(sheet).findByRole('img', { name: 'Depth trace' })).toBeInTheDocument()
    expect(traces[0]).toEqual({ id: 'depth', image: 'abc', params: {} })
    // The trace picks the ControlNet that reads depth maps.
    expect(within(sheet).getByRole('button', { name: /Model Depth XL/ })).toBeInTheDocument()
    fireEvent.change(within(sheet).getByLabelText('Last step'), { target: { value: '24' } })
    expect(within(sheet).getByText('1–24 of 30')).toBeInTheDocument()
    await user.click(within(sheet).getByRole('button', { name: 'Done' }))

    expect(screen.getByText('Depth, weight 0.70, steps 1–24')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Generate' }))
    expect(await screen.findByText('Queued 1 image.')).toBeInTheDocument()
    expect(submitted[0]?.spec.control).toEqual([
      {
        controlnet: { path: 'controlnets/sdxl/diffusers-xl' },
        image: 'sha256:dep',
        fit: 'crop',
        scale: 0.7,
        start: 0,
        end: 0.8,
        preprocessor: { id: 'depth', source: 'sha256:abc', params: {} },
      },
    ])
  })

  it('plays a clip and extends it from its last frame', async () => {
    const submitted: { spec: { inputs?: unknown } }[] = []
    mockApi({
      ...VIDEO_ROUTES,
      'GET /api/results': () => ({ results: [VIDEO_RESULT], cursor: null }),
      'POST /api/results/v1/extend': () => ({
        spec: {
          ...WAN_SPEC,
          mode: 'i2v',
          params: { ...WAN_SPEC.params, seed: -1 },
          inputs: { source: 'sha256:f1', extends: 'sha256:vid' },
        },
        source: { sha256: 'f1', media_type: 'image/png', width: 1280, height: 704 },
      }),
      'POST /api/jobs': (init) => {
        submitted.push(JSON.parse(init?.body as string) as (typeof submitted)[number])
        return { id: 'j3' }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: /Results/ }))
    const tile = await screen.findByRole('button', { name: /Open clip 1, seed 5/ })
    expect(within(tile).getByText('0:05')).toBeInTheDocument()
    await user.click(tile)
    const viewer = screen.getByRole('dialog', { name: 'Video' })
    expect(viewer.querySelector('video')).toHaveAttribute('src', '/api/blobs/vid')
    expect(within(viewer).getByText('121 frames at 24 fps, 5.0 s')).toBeInTheDocument()
    await user.click(within(viewer).getByRole('button', { name: 'Extend' }))

    expect(await screen.findByText('Continues a clip from its last frame.')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Video' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: 'From image' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
    await user.click(screen.getByRole('button', { name: 'Generate' }))
    await screen.findByText('Queued 1 clip.')
    expect(submitted[0]?.spec.inputs).toEqual({
      source: 'sha256:f1',
      fit: 'crop',
      extends: 'sha256:vid',
    })
  })

  it('pairs A14B LoRAs with a weight per expert, and warns about a small GPU', async () => {
    localStorage.setItem(
      'degas.create.draft',
      JSON.stringify({
        family: 'wan22',
        families: { wan22: { model: 'models/wan22/t2v-a14b/Wan2.2-T2V-A14B', params: {} } },
      }),
    )
    const submitted: { spec: { loras: unknown } }[] = []
    mockApi({
      ...VIDEO_ROUTES,
      'GET /api/session': () => RUNNING,
      'POST /api/jobs': (init) => {
        submitted.push(JSON.parse(init?.body as string) as (typeof submitted)[number])
        return { id: 'j4' }
      },
    })
    const user = userEvent.setup()
    renderApp()
    expect(
      await screen.findByText('Needs an A100; this L4 session may run it slowly.'),
    ).toBeInTheDocument()
    expect(screen.getByText('Wan 2.2 T2V A14B')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'From image' })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Add LoRA' }))
    const sheet = screen.getByRole('dialog', { name: 'Add LoRA' })
    expect(within(sheet).queryByRole('button', { name: /grain/ })).not.toBeInTheDocument()
    await user.click(within(sheet).getByRole('button', { name: /motion/ }))
    fireEvent.change(screen.getByRole('slider', { name: 'motion low-noise weight' }), {
      target: { value: '0.5' },
    })
    expect(screen.getByRole('slider', { name: 'motion high-noise weight' })).toHaveValue('1')

    await user.type(screen.getByLabelText('Prompt'), 'surf')
    await user.click(screen.getByRole('button', { name: 'Generate' }))
    await screen.findByText('Queued 1 clip.')
    expect(submitted[0]?.spec.loras).toEqual([
      {
        high: { path: 'loras/wan22/motion_high_noise.safetensors', weight: 1 },
        low: { path: 'loras/wan22/motion_low_noise.safetensors', weight: 0.5 },
      },
    ])
  })

  it('chooses how a batch gets its seeds', async () => {
    const submitted: { spec: { params: { seed: number } }; seed_mode: string }[] = []
    mockApi({
      'POST /api/jobs': (init) => {
        submitted.push(JSON.parse(init?.body as string) as (typeof submitted)[number])
        return { id: 'j1' }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.type(await screen.findByLabelText('Prompt'), 'a lighthouse')
    await user.type(screen.getByLabelText('Seed'), '1234')
    await user.click(screen.getByRole('button', { name: 'More images' }))

    // A batch can't repeat one seed: random seeds, or counting up from the seed.
    const random = screen.getByRole('button', { name: 'Random seeds' })
    expect(random).toHaveAttribute('aria-pressed', 'true')
    expect(screen.queryByLabelText('from')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Generate 2 images' }))
    await screen.findByText('Queued 2 images.')
    expect(submitted[0]?.seed_mode).toBe('random')
    expect(submitted[0]?.spec.params.seed).toBe(-1)

    await user.click(screen.getByRole('button', { name: 'Count up' }))
    expect(screen.getByLabelText('from')).toHaveValue(1234)
    await user.click(screen.getByRole('button', { name: 'Generate 2 images' }))
    await vi.waitFor(() => {
      expect(submitted).toHaveLength(2)
    })
    expect(submitted[1]?.seed_mode).toBe('increment')
    expect(submitted[1]?.spec.params.seed).toBe(1234)
  })

  it('reorders the queue', async () => {
    const prompts: Record<string, string> = { a: 'first', b: 'second', c: 'third' }
    const order = ['a', 'b', 'c']
    const moves: { id: string; body: unknown }[] = []
    const patch = (id: string) => (init?: RequestInit) => {
      const body = JSON.parse(init?.body as string) as { position: number }
      moves.push({ id, body })
      order.splice(order.indexOf(id), 1)
      order.splice(body.position, 0, id)
      return {}
    }
    mockApi({
      'GET /api/jobs': () =>
        order.map((id, i) => ({
          ...JOB_DONE,
          id,
          status: 'queued',
          queue_position: i + 1,
          spec: { ...SPEC, params: { ...SPEC.params, prompt: prompts[id] } },
        })),
      'PATCH /api/jobs/b': patch('b'),
      'PATCH /api/jobs/c': patch('c'),
    })
    const user = userEvent.setup()
    renderApp()
    await user.click(screen.getByRole('button', { name: /Results/ }))
    const third = await screen.findByRole('region', { name: 'third' })
    const first = screen.getByRole('region', { name: 'first' })
    expect(within(first).queryByRole('button', { name: 'Move to top' })).not.toBeInTheDocument()

    await user.click(within(third).getByRole('button', { name: 'Move to top' }))
    expect(moves).toEqual([{ id: 'c', body: { position: 0 } }])
    const regions = () => screen.getAllByRole('region').map((r) => r.getAttribute('aria-label'))
    expect(regions()).toEqual(['third', 'first', 'second'])

    const handle = within(screen.getByRole('region', { name: 'second' })).getByRole('button', {
      name: 'Queue position 3 of 3',
    })
    handle.focus()
    await user.keyboard('{ArrowUp}')
    expect(moves[1]).toEqual({ id: 'b', body: { position: 1 } })
    await vi.waitFor(() => {
      expect(regions()).toEqual(['third', 'second', 'first'])
    })
    expect(document.activeElement).toHaveAccessibleName('Queue position 2 of 3')
  })

  it('clears finished results after asking', async () => {
    let cleared = false
    mockApi({
      'GET /api/jobs': () => (cleared ? [] : [JOB_DONE]),
      'GET /api/results': () => ({ results: cleared ? [] : [RESULT], cursor: null }),
      'DELETE /api/results': () => {
        cleared = true
        return { results: 1, jobs: 1 }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.click(screen.getByRole('button', { name: /Results/ }))
    await user.click(await screen.findByRole('button', { name: 'Clear results' }))
    const confirm = screen.getByRole('group', { name: 'Confirm clear' })
    expect(cleared).toBe(false)
    await user.click(within(confirm).getByRole('button', { name: 'Clear results' }))
    expect(await screen.findByText('Nothing here yet.')).toBeInTheDocument()
  })

  it('undoes cancelling a queued job', async () => {
    let status = 'queued'
    mockApi({
      'GET /api/jobs': () => [{ ...JOB_DONE, status, queue_position: 1 }],
      'DELETE /api/jobs/j1': () => {
        status = 'cancelled'
        return { cancelled: true }
      },
      'POST /api/jobs/j1/restore': () => {
        status = 'queued'
        return {}
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.click(screen.getByRole('button', { name: /Results/ }))
    await user.click(await screen.findByRole('button', { name: 'Cancel' }))
    const toast = await screen.findByRole('status')
    expect(toast).toHaveTextContent('Cancelled')
    expect(await screen.findByText('Nothing here yet.')).toBeInTheDocument()
    await user.click(within(toast).getByRole('button', { name: 'Undo' }))
    expect(await screen.findByRole('region', { name: 'a lighthouse' })).toBeInTheDocument()
    expect(screen.queryByText('Cancelled')).not.toBeInTheDocument()
  })

  it('opens where a notification points', async () => {
    history.replaceState(null, '', '/?sheet=session')
    mockApi()
    renderApp()
    expect(await screen.findByRole('dialog', { name: 'GPU session' })).toBeInTheDocument()
    expect(location.search).toBe('')
  })

  it('explains how to get notifications on a phone', async () => {
    mockApi()
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: /No GPU/ }))
    const section = await screen.findByRole('region', { name: 'Notifications' })
    expect(section).toHaveTextContent('add Degas to the Home Screen')
  })

  it('covers images and prompts until tapped, and remembers the choice', async () => {
    const second = { ...RESULT, id: 'r2', item_index: 1, blob_sha: 'def', seed: 1235 }
    mockApi({
      'GET /api/jobs': () => [{ ...JOB_DONE, seeds: [1234, 1235] }],
      'GET /api/results': () => ({ results: [RESULT, second], cursor: null }),
    })
    const user = userEvent.setup()
    renderApp()
    const toggle = screen.getByRole('button', { name: 'Cover images' })
    await user.click(toggle)
    expect(toggle).toHaveAttribute('aria-pressed', 'true')
    expect(document.documentElement).toHaveAttribute('data-discreet')
    expect(localStorage.getItem('degas.discretion')).toBe('1')

    await user.click(screen.getByRole('button', { name: /Results/ }))
    const tile = await screen.findByRole('button', { name: 'Show image 1' })
    expect(tile).toHaveClass('covered')
    expect(screen.getByRole('button', { name: 'Show image 2' })).toHaveClass('covered')
    await user.click(screen.getByRole('button', { name: 'Show prompt' }))
    expect(screen.getByText('a lighthouse')).not.toHaveClass('covered-text')

    // The first tap uncovers; the second opens.
    await user.click(tile)
    expect(tile).not.toHaveClass('covered')
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /Open image 1, seed 1234/ }))
    const viewer = screen.getByRole('dialog', { name: 'Image' })
    expect(within(viewer).queryByRole('button', { name: 'Show image' })).not.toBeInTheDocument()
    expect(within(viewer).getByRole('button', { name: 'Show prompt' })).toBeInTheDocument()

    // The next image is still covered.
    await user.click(within(viewer).getByRole('button', { name: 'Next image' }))
    await user.click(within(viewer).getByRole('button', { name: 'Show image' }))
    expect(within(viewer).queryByRole('button', { name: 'Show image' })).not.toBeInTheDocument()

    // Closing the viewer covers everything again.
    await user.click(within(viewer).getByRole('button', { name: 'Close' }))
    expect(screen.getByRole('button', { name: 'Show image 1' })).toHaveClass('covered')
    expect(screen.getByRole('button', { name: 'Show prompt' })).toBeInTheDocument()

    await user.click(toggle)
    expect(document.documentElement).not.toHaveAttribute('data-discreet')
    expect(screen.getByRole('button', { name: /Open image 1/ })).not.toHaveClass('covered')
  })

  it('shows a covered image only while it is held', async () => {
    mockApi({
      'GET /api/jobs': () => [{ ...JOB_DONE }],
      'GET /api/results': () => ({ results: [RESULT], cursor: null }),
    })
    setDiscretion(true)
    const user = userEvent.setup()
    renderApp()
    await user.click(screen.getByRole('button', { name: /Results/ }))
    const tile = await screen.findByRole('button', { name: 'Show image 1' })
    fireEvent.pointerDown(tile, { button: 0, clientX: 10, clientY: 10 })
    await waitFor(() => {
      expect(tile).toHaveClass('peek')
    })
    expect(tile).not.toHaveClass('covered')
    fireEvent.pointerUp(tile)
    fireEvent.click(tile)
    expect(tile).toHaveClass('covered')
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()

    // Moving first is a scroll, not a peek.
    fireEvent.pointerDown(tile, { button: 0, clientX: 10, clientY: 10 })
    fireEvent.pointerMove(tile, { clientX: 10, clientY: 60 })
    await new Promise((r) => setTimeout(r, 300))
    expect(tile).not.toHaveClass('peek')
  })

  it('covers the screen and every image when the app is left', async () => {
    mockApi({
      'GET /api/jobs': () => [{ ...JOB_DONE }],
      'GET /api/results': () => ({ results: [RESULT], cursor: null }),
    })
    installShield()
    setDiscretion(true)
    const user = userEvent.setup()
    renderApp()
    await user.click(screen.getByRole('button', { name: /Results/ }))
    await user.click(await screen.findByRole('button', { name: 'Show image 1' }))
    expect(screen.getByRole('button', { name: /Open image 1/ })).toBeInTheDocument()

    const root = document.documentElement
    fireEvent.blur(window)
    expect(root).toHaveAttribute('data-shield')
    fireEvent.focus(window)
    expect(root).not.toHaveAttribute('data-shield')
    // Losing focus alone (a share sheet) keeps what was uncovered.
    expect(screen.getByRole('button', { name: /Open image 1/ })).toBeInTheDocument()

    const state = vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('hidden')
    fireEvent(document, new Event('visibilitychange'))
    expect(root).toHaveAttribute('data-shield')
    expect(await screen.findByRole('button', { name: 'Show image 1' })).toHaveClass('covered')
    state.mockReturnValue('visible')
    fireEvent(document, new Event('visibilitychange'))
    expect(root).not.toHaveAttribute('data-shield')

    // A tab change covers again too.
    await user.click(screen.getByRole('button', { name: 'Show image 1' }))
    await user.click(screen.getByRole('button', { name: 'Create' }))
    await user.click(screen.getByRole('button', { name: /Results/ }))
    expect(await screen.findByRole('button', { name: 'Show image 1' })).toHaveClass('covered')
  })

  it('covers library images', async () => {
    mockApi({ 'GET /api/library': () => ({ items: [LIBRARY_ITEM], cursor: null }) })
    setDiscretion(true)
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: 'Library' }))
    const tile = await screen.findByRole('button', { name: 'Show image' })
    expect(tile).toHaveClass('covered')
    await user.click(tile)
    expect(screen.getByRole('button', { name: 'Open a harbour at dusk' })).not.toHaveClass(
      'covered',
    )
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
