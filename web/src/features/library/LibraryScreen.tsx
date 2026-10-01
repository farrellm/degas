import { useDeferredValue, useState } from 'react'

import { LibraryImages } from './LibraryImages'
import { SavedPrompts } from './SavedPrompts'

type View = 'Images' | 'Prompts'
const VIEWS: View[] = ['Images', 'Prompts']

export function LibraryScreen({ onRemix }: { onRemix: () => void }) {
  const [view, setView] = useState<View>('Images')
  const [query, setQuery] = useState('')
  const q = useDeferredValue(query.trim())

  return (
    <div className="library">
      <div className="library-bar">
        <input
          type="search"
          className="library-search"
          aria-label="Search the library"
          placeholder="Search prompts and tags"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value)
          }}
        />
        <div className="segmented" role="group" aria-label="Show">
          {VIEWS.map((v) => (
            <button
              key={v}
              type="button"
              aria-pressed={v === view}
              onClick={() => {
                setView(v)
              }}
            >
              {v}
            </button>
          ))}
        </div>
      </div>
      {view === 'Images' ? (
        <LibraryImages q={q} onRemix={onRemix} />
      ) : (
        <SavedPrompts q={q} onUse={onRemix} />
      )}
    </div>
  )
}
