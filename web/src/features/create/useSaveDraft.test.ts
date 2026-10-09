import { renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { type FamilyDraft, loadDraft } from './draft'
import { useSaveDraft } from './useSaveDraft'

const fd = (prompt: string): FamilyDraft => ({ model: 'm', loras: [], params: { prompt } })

describe('useSaveDraft', () => {
  it('saves when the draft changes, not at every render', () => {
    const write = vi.spyOn(Storage.prototype, 'setItem')
    const { rerender } = renderHook(({ draft }) => useSaveDraft('sdxl', draft, 1, 'random'), {
      initialProps: { draft: fd('a cat') },
    })
    expect(write).toHaveBeenCalledTimes(1)

    // A new object with the same contents, as the form builds at each render.
    rerender({ draft: fd('a cat') })
    expect(write).toHaveBeenCalledTimes(1)

    rerender({ draft: fd('a dog') })
    expect(write).toHaveBeenCalledTimes(2)
    expect(loadDraft().families.sdxl?.params?.prompt).toBe('a dog')
  })

  it('saves nothing while the form loads', () => {
    renderHook(() => useSaveDraft('sdxl', null, 1, 'random'))
    expect(loadDraft().families).toEqual({})
  })
})
