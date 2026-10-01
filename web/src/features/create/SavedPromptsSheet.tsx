import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, type SavedPrompt } from '../api'
import { Sheet } from './Sheet'

interface Props {
  prompt: string
  negativePrompt: string
  family: string
  onUse: (p: SavedPrompt) => void
  onClose: () => void
}

/** Saved prompts, opened from the prompt block: save the current one or swap one in. */
export function PromptSheet({ prompt, negativePrompt, family, onUse, onClose }: Props) {
  const qc = useQueryClient()
  const prompts = useQuery({ queryKey: ['prompts', ''], queryFn: () => api.prompts() })
  const save = useMutation({
    mutationFn: () => api.savePrompt({ prompt, negative_prompt: negativePrompt, family }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['prompts'] }),
  })
  const isSaved =
    save.isSuccess ||
    !!prompts.data?.some((p) => p.prompt === prompt && p.negative_prompt === negativePrompt)

  return (
    <Sheet title="Saved prompts" onClose={onClose}>
      <div className="prompt-save">
        <button
          type="button"
          className="btn quiet"
          disabled={!prompt.trim() || isSaved || save.isPending}
          onClick={() => {
            save.mutate()
          }}
        >
          {isSaved && prompt.trim() ? 'Saved' : 'Save this prompt'}
        </button>
        {save.error && <p role="alert">{save.error.message}</p>}
      </div>
      {prompts.error && <p role="alert">{prompts.error.message}</p>}
      {prompts.data?.length === 0 && (
        <p className="asset-empty">
          No saved prompts yet. Write one, then choose Save this prompt to use it again later.
        </p>
      )}
      <ul className="prompt-list">
        {prompts.data?.map((p) => (
          <li key={p.id}>
            <button
              type="button"
              className="prompt-row"
              onClick={() => {
                onUse(p)
              }}
            >
              <PromptText p={p} />
            </button>
          </li>
        ))}
      </ul>
    </Sheet>
  )
}

/** A saved prompt's name, text and negative, as listed in the sheet and the Library. */
export function PromptText({ p }: { p: SavedPrompt }) {
  return (
    <>
      <span className="prompt-name">{p.name}</span>
      <span className="prompt-text">{p.prompt}</span>
      {p.negative_prompt && <span className="prompt-negative">Negative: {p.negative_prompt}</span>}
    </>
  )
}
