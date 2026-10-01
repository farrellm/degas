import type { SavedPrompt } from '@/api/types'

/** A saved prompt's name, text and negative, as listed in the sheet and the Library. */
export function SavedPromptText({ p }: { p: SavedPrompt }) {
  return (
    <>
      <span className="prompt-name">{p.name}</span>
      <span className="prompt-text">{p.prompt}</span>
      {p.negative_prompt && <span className="prompt-negative">Negative: {p.negative_prompt}</span>}
    </>
  )
}
