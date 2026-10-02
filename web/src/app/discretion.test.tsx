import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { installShield, setDiscretion } from '@/lib/discretion'
import { IMAGE_PROMPTS, JOB_DONE, LIBRARY_ITEM, RESULT, SPEC } from '@/test/fixtures'
import { mockApi } from '@/test/mockApi'
import { renderApp } from '@/test/render'

describe('Discretion', () => {
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
    // An uncovered image shows its prompt too.
    expect(within(viewer).queryByRole('button', { name: 'Show image' })).not.toBeInTheDocument()
    expect(within(viewer).queryByRole('button', { name: 'Show prompt' })).not.toBeInTheDocument()

    // The next image is still covered, and so is its prompt, until the image is tapped.
    await user.click(within(viewer).getByRole('button', { name: 'Next image' }))
    expect(within(viewer).getByRole('button', { name: 'Show prompt' })).toBeInTheDocument()
    await user.click(within(viewer).getByRole('button', { name: 'Show image' }))
    expect(within(viewer).queryByRole('button', { name: 'Show image' })).not.toBeInTheDocument()
    expect(within(viewer).queryByRole('button', { name: 'Show prompt' })).not.toBeInTheDocument()

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

  it('covers model and LoRA names and errors', async () => {
    const withLora = { ...SPEC, loras: [{ path: 'loras/sdxl/film.safetensors', weight: 0.8 }] }
    const first = { ...RESULT, spec: withLora }
    const second = { ...first, id: 'r2', item_index: 1, blob_sha: 'def', seed: 1235 }
    mockApi({
      'GET /api/jobs': () => [
        { ...JOB_DONE, spec: withLora, seeds: [1234, 1235] },
        {
          ...JOB_DONE,
          id: 'j2',
          status: 'error',
          error: 'Could not load LoRA loras/sdxl/film.safetensors',
        },
      ],
      'GET /api/results': () => ({ results: [first, second], cursor: null }),
    })
    setDiscretion(true)
    const user = userEvent.setup()
    renderApp()
    await user.click(screen.getByRole('button', { name: /Results/ }))
    await user.click(await screen.findByRole('button', { name: 'Show error' }))
    expect(screen.getByText(/Could not load LoRA/)).not.toHaveClass('covered-text')

    // The feed's model line uncovers with the prompt.
    const group = screen.getByText('Studio XL v10 + 1 LoRA,').closest('section')
    if (!group) throw new Error('no group')
    expect(within(group).getByRole('button', { name: 'Show model' })).toBeInTheDocument()
    await user.click(within(group).getByRole('button', { name: 'Show prompt' }))
    expect(within(group).queryByRole('button', { name: 'Show model' })).not.toBeInTheDocument()

    // An uncovered image names its model and LoRAs; a covered one hides them until it's tapped.
    await user.click(screen.getByRole('button', { name: 'Show image 1' }))
    await user.click(screen.getByRole('button', { name: /Open image 1/ }))
    const viewer = screen.getByRole('dialog', { name: 'Image' })
    expect(within(viewer).getByText(/Studio XL v10, with Film Grain v3 at 0.8/)).toBeInTheDocument()
    await user.click(within(viewer).getByRole('button', { name: 'Next image' }))
    expect(within(viewer).getByRole('button', { name: 'Show model' })).toBeInTheDocument()
    await user.click(within(viewer).getByRole('button', { name: 'Show image' }))
    expect(within(viewer).queryByRole('button', { name: 'Show model' })).not.toBeInTheDocument()
  })

  it('covers an image prompt’s pictures and model until its image or prompt is shown', async () => {
    const spec = { ...SPEC, image_prompts: IMAGE_PROMPTS }
    const first = { ...RESULT, spec }
    const second = { ...first, id: 'r2', item_index: 1, blob_sha: 'def', seed: 1235 }
    const third = { ...first, id: 'r3', item_index: 2, blob_sha: 'ghi', seed: 1236 }
    mockApi({
      'GET /api/jobs': () => [{ ...JOB_DONE, spec, seeds: [1234, 1235, 1236] }],
      'GET /api/results': () => ({ results: [first, second, third], cursor: null }),
    })
    setDiscretion(true)
    const user = userEvent.setup()
    renderApp()
    await user.click(screen.getByRole('button', { name: /Results/ }))
    await user.click(await screen.findByRole('button', { name: 'Show image 1' }))
    await user.click(screen.getByRole('button', { name: /Open image 1/ }))
    const viewer = screen.getByRole('dialog', { name: 'Image' })
    const prompts = () => within(viewer).getByRole('group', { name: 'Image prompts' })
    expect(prompts()).not.toHaveClass('covered')
    expect(within(prompts()).getByText('Face, weight 0.80')).toBeInTheDocument()

    // The next image's are covered until the image is tapped.
    await user.click(within(viewer).getByRole('button', { name: 'Next image' }))
    expect(prompts()).toHaveClass('covered')
    expect(within(prompts()).getAllByRole('button', { name: 'Show image prompt' })).toHaveLength(2)
    await user.click(within(viewer).getByRole('button', { name: 'Show image' }))
    expect(prompts()).not.toHaveClass('covered')

    // Or until the prompt is: the pictures go with the words.
    await user.click(within(viewer).getByRole('button', { name: 'Next image' }))
    expect(prompts()).toHaveClass('covered')
    await user.click(within(viewer).getByRole('button', { name: 'Show prompt' }))
    expect(prompts()).not.toHaveClass('covered')
    expect(within(viewer).getByRole('button', { name: 'Show image' })).toBeInTheDocument()
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
})
