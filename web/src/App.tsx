import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { api } from './api'
import { useServerEvents } from './events'
import { CreateScreen } from './screens/CreateScreen'
import { QueueScreen } from './screens/QueueScreen'
import { ResultsScreen } from './screens/ResultsScreen'
import { SessionScreen } from './screens/SessionScreen'

const TABS = ['Create', 'Queue', 'Results', 'Session'] as const
type Tab = (typeof TABS)[number]

function App() {
  const [tab, setTab] = useState<Tab>('Create')
  useServerEvents()
  const session = useQuery({ queryKey: ['session'], queryFn: api.session })
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: api.jobs })
  const state = session.data?.session?.state
  const pending = jobs.data?.filter((j) => j.status === 'queued' || j.status === 'running').length

  return (
    <div className="app">
      <header>
        <h1>Degas</h1>
        {session.error && <span role="alert">Server unreachable: {session.error.message}</span>}
      </header>
      <main>
        {tab === 'Create' && (
          <CreateScreen
            onQueued={() => {
              setTab('Queue')
            }}
          />
        )}
        {tab === 'Queue' && <QueueScreen />}
        {tab === 'Results' && <ResultsScreen />}
        {tab === 'Session' && <SessionScreen />}
      </main>
      <nav className="tabs">
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
            {t === 'Queue' && pending ? <span className="count">{pending}</span> : null}
            {t === 'Session' && state ? (
              <span className={`dot ${state}`} aria-label={`Session ${state}`} />
            ) : null}
          </button>
        ))}
      </nav>
    </div>
  )
}

export default App
