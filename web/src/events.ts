import { useQueryClient } from '@tanstack/react-query'
import { useEffect } from 'react'
import type { CivitaiImport, Job, Progress, SessionSnapshot } from './api'

type ServerEvent =
  | { type: 'hello' }
  | ({ type: 'session' } & SessionSnapshot)
  | { type: 'job'; job: Job }
  | ({ type: 'progress' } & Progress)
  | { type: 'result' }
  | { type: 'assets' }
  | { type: 'library' }
  | { type: 'prompts' }
  | { type: 'swept' }
  | { type: 'import'; import: CivitaiImport }

/** Follow `/api/events` and keep the query cache in sync with the server. */
export function useServerEvents() {
  const qc = useQueryClient()

  useEffect(() => {
    const source = new EventSource('/api/events')

    source.onmessage = (msg: MessageEvent<string>) => {
      const event = JSON.parse(msg.data) as ServerEvent
      switch (event.type) {
        case 'hello':
          // (Re)connected: anything may have changed while we were away.
          void qc.invalidateQueries()
          break
        case 'session': {
          const { type, ...snapshot } = event
          qc.setQueryData<SessionSnapshot>(['session'], snapshot)
          break
        }
        case 'job': {
          const jobs = qc.getQueryData<Job[]>(['jobs'])
          if (jobs?.some((j) => j.id === event.job.id)) {
            qc.setQueryData<Job[]>(['jobs'], (js) =>
              js?.map((j) => (j.id === event.job.id ? event.job : j)),
            )
          } else {
            void qc.invalidateQueries({ queryKey: ['jobs'] })
          }
          break
        }
        case 'progress': {
          const { type, ...progress } = event
          qc.setQueryData<Job[]>(['jobs'], (jobs) =>
            jobs?.map((j) => (j.id === progress.job ? { ...j, progress } : j)),
          )
          break
        }
        case 'result':
          void qc.invalidateQueries({ queryKey: ['results'] })
          break
        case 'library':
          void qc.invalidateQueries({ queryKey: ['library'] })
          void qc.invalidateQueries({ queryKey: ['results'] })
          break
        case 'prompts':
          void qc.invalidateQueries({ queryKey: ['prompts'] })
          break
        case 'swept':
          void qc.invalidateQueries({ queryKey: ['results'] })
          void qc.invalidateQueries({ queryKey: ['jobs'] })
          break
        case 'assets':
          void qc.invalidateQueries({ queryKey: ['assets'] })
          void qc.invalidateQueries({ queryKey: ['drive'] })
          break
        case 'import':
          qc.setQueryData<CivitaiImport>(['civitai-import'], event.import)
          break
      }
    }

    // iOS suspends background tabs; refetch when the app comes back.
    const onVisible = () => {
      if (document.visibilityState === 'visible') void qc.invalidateQueries()
    }
    document.addEventListener('visibilitychange', onVisible)
    return () => {
      source.close()
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [qc])
}
