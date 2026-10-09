import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

import { api } from '@/api/client'
import { queries, queryKeys } from '@/api/queries'
import type { SavedPrompt } from '@/api/types'
import { SavedPromptText } from '@/components/SavedPromptText'
import { draftWithPrompt } from '@/features/create/draft'

export function SavedPrompts({ q, onUse }: { q: string; onUse: () => void }) {
  const prompts = useQuery(queries.prompts(q))

  if (prompts.isPending) return <p className="loading">Loading…</p>
  if (prompts.error) return <p role="alert">{prompts.error.message}</p>
  if (prompts.data.length === 0) {
    return q ? (
      <p className="library-empty">No saved prompts match “{q}”.</p>
    ) : (
      <div className="feed-empty">
        <p className="lede">No saved prompts.</p>
        <p>In Create, open Prompts and choose Save this prompt.</p>
      </div>
    )
  }
  return (
    <ul className="prompt-list library-prompts">
      {prompts.data.map((p) => (
        <PromptRow
          key={p.id}
          p={p}
          onUse={() => {
            draftWithPrompt(p)
            onUse()
          }}
        />
      ))}
    </ul>
  )
}

function PromptRow({ p, onUse }: { p: SavedPrompt; onUse: () => void }) {
  const qc = useQueryClient()
  const [mode, setMode] = useState<'view' | 'rename' | 'delete'>('view')
  const [name, setName] = useState(p.name)
  const [expanded, setExpanded] = useState(false)
  const onSettled = () => qc.invalidateQueries({ queryKey: queryKeys.prompts })
  const rename = useMutation({
    mutationFn: () => api.editPrompt(p.id, { name: name.trim() }),
    onSuccess: () => setMode('view'),
    onSettled,
  })
  const remove = useMutation({ mutationFn: () => api.deletePrompt(p.id), onSettled })
  const error = rename.error ?? remove.error

  return (
    <li className="prompt-item">
      {mode === 'rename' ? (
        <form
          className="rename"
          onSubmit={(e) => {
            e.preventDefault()
            if (name.trim()) rename.mutate()
          }}
        >
          <input
            aria-label="Prompt name"
            value={name}
            autoFocus
            onChange={(e) => setName(e.target.value)}
          />
          <button type="submit" className="btn small" disabled={!name.trim() || rename.isPending}>
            Rename
          </button>
          <button
            type="button"
            className="btn quiet small"
            onClick={() => {
              setName(p.name)
              setMode('view')
            }}
          >
            Cancel
          </button>
        </form>
      ) : (
        <button
          type="button"
          className={expanded ? 'prompt-toggle expanded' : 'prompt-toggle'}
          aria-expanded={expanded}
          onClick={() => setExpanded(!expanded)}
        >
          <SavedPromptText p={p} />
        </button>
      )}
      {mode === 'delete' ? (
        <div className="row-buttons" role="group" aria-label="Confirm delete">
          <span className="confirm-text">Delete this prompt?</span>
          <button
            type="button"
            className="btn danger small"
            disabled={remove.isPending}
            onClick={() => remove.mutate()}
          >
            Delete
          </button>
          <button type="button" className="btn quiet small" onClick={() => setMode('view')}>
            Cancel
          </button>
        </div>
      ) : mode === 'view' ? (
        <div className="row-buttons">
          <button type="button" className="btn quiet small" onClick={onUse}>
            Use
          </button>
          <button
            type="button"
            className="btn quiet small"
            aria-label={`Rename ${p.name}`}
            onClick={() => setMode('rename')}
          >
            Rename
          </button>
          <button
            type="button"
            className="btn quiet small"
            aria-label={`Delete ${p.name}`}
            onClick={() => setMode('delete')}
          >
            Delete
          </button>
        </div>
      ) : null}
      {error && <p role="alert">{error.message}</p>}
    </li>
  )
}
