import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { api } from '@/api/client'
import { queries, queryKeys } from '@/api/queries'

import type { ViewerItem } from './Viewer'

/** Add the item's prompt to the saved prompts; shows Prompt saved once it's there. */
export function SavePrompt({ item }: { item: ViewerItem }) {
  const qc = useQueryClient()
  const prompts = useQuery(queries.prompts())
  const params = item.spec?.params ?? {}
  const prompt = String(params.prompt ?? '')
  const negativePrompt = String(params.negative_prompt ?? '')
  const save = useMutation({
    mutationFn: () =>
      api.savePrompt({
        prompt,
        negative_prompt: negativePrompt,
        family: item.spec?.family ?? null,
      }),
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.prompts }),
  })
  const isSaved =
    save.isSuccess ||
    !!prompts.data?.some((p) => p.prompt === prompt && p.negative_prompt === negativePrompt)

  return (
    <>
      <button
        type="button"
        className="btn quiet"
        disabled={!prompt.trim() || isSaved || save.isPending}
        onClick={() => save.mutate()}
      >
        {isSaved && prompt.trim() ? 'Prompt saved' : 'Save prompt'}
      </button>
      {save.error && (
        <p className="viewer-note" role="alert">
          {save.error.message}
        </p>
      )}
    </>
  )
}
