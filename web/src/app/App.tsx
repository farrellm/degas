import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { api } from './api'
import { DiscretionToggle } from './components/DiscretionToggle'
import { NotificationAsk } from './components/NotificationAsk'
import { SessionChip } from './components/SessionChip'
import { SessionSheet } from './components/SessionSheet'
import { coverAll } from './discretion'
import { useServerEvents } from './events'
import { CreateScreen } from './screens/CreateScreen'
import { LibraryScreen } from './screens/LibraryScreen'
import { ResultsScreen } from './screens/ResultsScreen'

const TABS = ['Create', 'Results', 'Library'] as const
type Tab = (typeof TABS)[number]

/** Where a link into the app points: `/?tab=results`, `/?sheet=session` (notifications). */
function linkTarget(url: string): { tab?: Tab; session: boolean } {
  const params = new URL(url, location.origin).searchParams
  const tab = TABS.find((t) => t.toLowerCase() === params.get('tab'))
  return { tab, session: params.get('sheet') === 'session' }
}

function App() {
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

  const session = useQuery({ queryKey: ['session'], queryFn: api.session })
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: api.jobs })
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

export default App
