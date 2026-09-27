import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { api } from './api'
import { SessionChip } from './components/SessionChip'
import { SessionSheet } from './components/SessionSheet'
import { useServerEvents } from './events'
import { CreateScreen } from './screens/CreateScreen'
import { ResultsScreen } from './screens/ResultsScreen'

const TABS = ['Create', 'Results'] as const
type Tab = (typeof TABS)[number]

function App() {
  const [tab, setTab] = useState<Tab>('Create')
  const [sessionOpen, setSessionOpen] = useState(false)
  useServerEvents()
  const session = useQuery({ queryKey: ['session'], queryFn: api.session })
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: api.jobs })
  const pending = jobs.data?.filter((j) => j.status === 'queued' || j.status === 'running').length

  return (
    <div className="app">
      <header className="app-header">
        <span className="wordmark">Degas</span>
        <SessionChip
          onOpen={() => {
            setSessionOpen(true)
          }}
        />
      </header>
      {session.error && (
        <p className="server-down" role="alert">
          Can't reach the Degas server ({session.error.message}). Check that it's running and that
          this device is on the tailnet.
        </p>
      )}
      <main className={tab === 'Results' ? 'app-main results-main' : 'app-main'}>
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
            onReuse={() => {
              setTab('Create')
            }}
            onCreate={() => {
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
