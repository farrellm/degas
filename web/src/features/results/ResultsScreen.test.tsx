import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import {
  IMAGE_PROMPTS,
  JOB_DONE,
  LIBRARY_ITEM,
  RESULT,
  SPEC,
  VIDEO_RESULT,
  WAN_SPEC,
} from '@/test/fixtures'
import { mockApi, VIDEO_ROUTES } from '@/test/mockApi'
import { renderApp } from '@/test/render'

describe('Results', () => {
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
    expect(
      screen.getByRole('button', { name: 'Open image 2: Denoising 12/30' }),
    ).toBeInTheDocument()
  })

  it('opens a queued job in the viewer and remixes it', async () => {
    mockApi({ 'GET /api/jobs': () => [{ ...JOB_DONE, status: 'queued', seeds: [42] }] })
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: /Results/ }))
    const group = await screen.findByRole('region', { name: 'a lighthouse' })
    expect(within(group).queryByRole('button', { name: 'Remix' })).not.toBeInTheDocument()
    await user.click(within(group).getByRole('button', { name: 'Open image 1: Queued' }))
    const viewer = screen.getByRole('dialog')
    expect(within(viewer).getByRole('img', { name: 'Queued' })).toBeInTheDocument()
    expect(within(viewer).getByText(/seed 42/)).toBeInTheDocument()
    await user.click(within(viewer).getByRole('button', { name: 'Remix' }))
    expect(await screen.findByLabelText('Prompt')).toHaveValue('a lighthouse')
    expect(screen.getByLabelText('Seed')).toHaveValue(42)
  })

  it('stays on a sketch in the viewer as it finishes', async () => {
    let finished = false
    mockApi({
      'GET /api/jobs': () => [{ ...JOB_DONE, status: finished ? 'done' : 'running' }],
      'GET /api/results': () => ({ results: finished ? [RESULT] : [], cursor: null }),
    })
    const user = userEvent.setup()
    const { client } = renderApp()
    await user.click(await screen.findByRole('button', { name: /Results/ }))
    await user.click(await screen.findByRole('button', { name: /Open image 1:/ }))
    finished = true
    await client.invalidateQueries()
    const viewer = screen.getByRole('dialog')
    expect(await within(viewer).findByRole('img', { name: 'a lighthouse' })).toBeInTheDocument()
    expect(within(viewer).getByRole('button', { name: 'Keep' })).toBeInTheDocument()
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

  it("saves a result's prompt to the saved prompts", async () => {
    const saved: unknown[] = []
    mockApi({
      'GET /api/results': () => ({ results: [RESULT], cursor: null }),
      'GET /api/prompts': () => [],
      'POST /api/prompts': (init) => {
        saved.push(JSON.parse(init?.body as string))
        return { id: 'p2' }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: /Results/ }))
    await user.click(await screen.findByRole('button', { name: /Open image 1, seed 1234/ }))
    await user.click(screen.getByRole('button', { name: 'Save prompt' }))
    expect(await screen.findByRole('button', { name: 'Prompt saved' })).toBeDisabled()
    expect(saved).toEqual([{ prompt: 'a lighthouse', negative_prompt: '', family: 'sdxl' }])
  })

  it('shows a result’s image prompts in the viewer, and none where there are none', async () => {
    const prompted = { ...RESULT, spec: { ...SPEC, image_prompts: IMAGE_PROMPTS } }
    const plain = { ...RESULT, id: 'r2', item_index: 1, blob_sha: 'def', seed: 1235 }
    mockApi({ 'GET /api/results': () => ({ results: [prompted, plain], cursor: null }) })
    const user = userEvent.setup()
    renderApp()
    await user.click(await screen.findByRole('button', { name: /Results/ }))
    await user.click(await screen.findByRole('button', { name: /Open image 1, seed 1234/ }))
    const prompts = screen.getByRole('group', { name: 'Image prompts' })
    expect(prompts.querySelectorAll('img')).toHaveLength(3)
    expect(within(prompts).getByText('Style, weight 1.00, steps 1–24')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Next image' }))
    expect(screen.queryByRole('group', { name: 'Image prompts' })).not.toBeInTheDocument()
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

  it('deletes one group of results after asking', async () => {
    const second = { ...JOB_DONE, id: 'j2', spec: { ...SPEC, params: { prompt: 'second' } } }
    const deleted: string[] = []
    mockApi({
      'GET /api/jobs': () => [JOB_DONE, second].filter((j) => !deleted.includes(j.id)),
      'GET /api/results': () => ({
        results: [RESULT, { ...RESULT, id: 'r2', job_id: 'j2', spec: second.spec }].filter(
          (r) => !deleted.includes(r.job_id),
        ),
        cursor: null,
      }),
      'DELETE /api/jobs/j2/results': () => {
        deleted.push('j2')
        return { results: 1, jobs: 1 }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.click(screen.getByRole('button', { name: /Results/ }))
    const group = await screen.findByRole('region', { name: 'second' })
    await user.click(within(group).getByRole('button', { name: 'Delete' }))
    const confirm = within(group).getByRole('group', { name: 'Confirm delete' })
    expect(confirm).toHaveTextContent('Delete this image?')
    expect(deleted).toEqual([])
    await user.click(within(confirm).getByRole('button', { name: 'Delete' }))
    await vi.waitFor(() => {
      expect(screen.queryByRole('region', { name: 'second' })).not.toBeInTheDocument()
    })
    expect(deleted).toEqual(['j2'])
    expect(screen.getAllByRole('region')).toHaveLength(1)
  })

  it('retries a failed job in its place', async () => {
    let retried = false
    mockApi({
      'GET /api/jobs': () => [
        retried
          ? { ...JOB_DONE, id: 'j2', status: 'queued', queue_position: 2 }
          : { ...JOB_DONE, status: 'error', error: 'CUDA out of memory' },
      ],
      'POST /api/jobs/j1/retry': () => {
        retried = true
        return { ...JOB_DONE, id: 'j2', status: 'queued' }
      },
    })
    const user = userEvent.setup()
    renderApp()
    await user.click(screen.getByRole('button', { name: /Results/ }))
    const group = await screen.findByRole('region', { name: 'a lighthouse' })
    await user.click(within(group).getByRole('button', { name: 'Retry' }))
    expect(await screen.findByRole('button', { name: 'Cancel' })).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Retry' })).not.toBeInTheDocument()
    expect(retried).toBe(true)
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
})
