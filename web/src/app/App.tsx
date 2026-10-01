import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'

import { queries } from '@/api/queries'
import { CreateScreen } from '@/features/create/CreateScreen'
import { LibraryScreen } from '@/features/library/LibraryScreen'
import { ResultsScreen } from '@/features/results/ResultsScreen'
import { NotificationAsk } from '@/features/session/NotificationAsk'
import { SessionChip } from '@/features/session/SessionChip'
import { SessionSheet } from '@/features/session/SessionSheet'
import { coverAll } from '@/lib/discretion'

import { DiscretionToggle } from './DiscretionToggle'
import { linkTarget, type Tab, TABS } from './tabs'
import { useServerEvents } from './useServerEvents'

/** The app shell: the header, the three tabs, and the session sheet. */
export function App() {
  const [tab, setTab] = useState<Tab>(() => linkTarget(location.href).tab ?? 'Create')
  const [sessionOpen, setSessionOpen] = useState(() => linkTarget(location.href).session)
  useServerEvents()

  useEffect(() => {
    if (location.search) history.replaceState(null, '', location.pathname)
    // A tapped notification while the app is already open.
    const sw = 'serviceWorker' in navigator ? navigator.serviceWorker : null
    const onMessage = (e: MessageEvent<{ type?: string; url?: string }>) => {
      if (e.data.type !== 'open' || !e.data.url) return
      const target = linkTarget(e.data.url)
      if (target.tab) setTab(target.tab)
      if (target.session) setSessionOpen(true)
    }
    sw?.addEventListener('message', onMessage)
    return () => {
      sw?.removeEventListener('message', onMessage)
    }
  }, [])

  const session = useQuery(queries.session())
  const jobs = useQuery(queries.jobs())
  const pending = jobs.data?.filter((j) => j.status === 'queued' || j.status === 'running').length

  return (
    <div className="app">
      <header className="app-header">
        <span className="wordmark">Degas</span>
        <div className="header-actions">
          <DiscretionToggle />
          <SessionChip
            onOpen={() => {
              setSessionOpen(true)
            }}
          />
        </div>
      </header>
      {session.error && (
        <p className="server-down" role="alert">
          Can't reach the Degas server ({session.error.message}). Check that it's running and that
          this device is on the tailnet.
        </p>
      )}
      <NotificationAsk jobs={jobs.data} />
      <main className={tab === 'Create' ? 'app-main' : 'app-main results-main'}>
        {tab === 'Create' && (
          <CreateScreen
            onOpenSession={() => {
              setSessionOpen(true)
            }}
            onShowResults={() => {
              setTab('Results')
            }}
          />
        )}
        {tab === 'Results' && (
          <ResultsScreen
            onRemix={() => {
              setTab('Create')
            }}
            onCreate={() => {
              setTab('Create')
            }}
          />
        )}
        {tab === 'Library' && (
          <LibraryScreen
            onRemix={() => {
              setTab('Create')
            }}
          />
        )}
      </main>
      <nav className="tabbar" aria-label="Sections">
        {TABS.map((t) => (
          <button
            key={t}
            type="button"
            aria-current={t === tab ? 'page' : undefined}
            onClick={() => {
              setTab(t)
              coverAll()
            }}
          >
            {t}
            {t === 'Results' && pending ? (
              <span className="count" aria-label={`${String(pending)} in progress`}>
                {pending}
              </span>
            ) : null}
          </button>
        ))}
      </nav>
      {sessionOpen && (
        <SessionSheet
          onClose={() => {
            setSessionOpen(false)
          }}
        />
      )}
    </div>
  )
}
