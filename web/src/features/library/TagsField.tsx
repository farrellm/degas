import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

import { api } from '@/api/client'
import { queryKeys } from '@/api/queries'
import type { LibraryItem } from '@/api/types'

/** Comma-separated tags, saved when the field loses focus. */
export function TagsField({ item }: { item: LibraryItem }) {
  const qc = useQueryClient()
  const [text, setText] = useState(item.tags.join(', '))
  const edit = useMutation({
    mutationFn: (tags: string[]) => api.editLibraryItem(item.id, { tags }),
    onSuccess: (saved) => {
      setText(saved.tags.join(', '))
      return qc.invalidateQueries({ queryKey: queryKeys.library })
    },
  })
  const commit = () => {
    const tags = text
      .split(',')
      .map((t) => t.trim())
      .filter(Boolean)
    if (tags.join(',') !== item.tags.join(',')) edit.mutate(tags)
  }
  return (
    <div className="tags-field">
      <label htmlFor="tags">Tags</label>
      <input
        id="tags"
        type="text"
        placeholder="Add tags, separated by commas"
        autoCapitalize="none"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') e.currentTarget.blur()
        }}
      />
      {edit.error && <p role="alert">{edit.error.message}</p>}
    </div>
  )
}
