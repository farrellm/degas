import { useQueryClient } from '@tanstack/react-query'
import { useEffect } from 'react'

import { parseEvent } from '@/api/events'
import { queries, queryKeys } from '@/api/queries'

/** Follow `/api/events` and keep the query cache in sync with the server. */
export function useServerEvents() {
  const qc = useQueryClient()

  useEffect(() => {
    const source = new EventSource('/api/events')

    source.onmessage = (msg: MessageEvent<string>) => {
      const event = parseEvent(msg.data)
      if (!event) {
        // Not an event this app can read (an older app, a newer server): don't let it into
        // the cache; fetch everything instead.
        console.warn('unreadable server event', msg.data)
        void qc.invalidateQueries()
        return
      }
      switch (event.type) {
        case 'hello':
          // (Re)connected: anything may have changed while we were away.
          void qc.invalidateQueries()
          break
        case 'session': {
          const { type, ...snapshot } = event
          qc.setQueryData(queries.session().queryKey, snapshot)
          break
        }
        case 'job': {
          const jobs = qc.getQueryData(queries.jobs().queryKey)
          if (jobs?.some((j) => j.id === event.job.id)) {
            qc.setQueryData(queries.jobs().queryKey, (js) =>
              js?.map((j) => (j.id === event.job.id ? event.job : j)),
            )
          } else {
            void qc.invalidateQueries({ queryKey: queryKeys.jobs })
          }
          break
        }
        case 'progress': {
          const { type, ...progress } = event
          qc.setQueryData(queries.jobs().queryKey, (jobs) =>
            jobs?.map((j) => (j.id === progress.job ? { ...j, progress } : j)),
          )
          break
        }
        case 'result':
          void qc.invalidateQueries({ queryKey: queryKeys.results })
          break
        case 'library':
          void qc.invalidateQueries({ queryKey: queryKeys.library })
          void qc.invalidateQueries({ queryKey: queryKeys.results })
          break
        case 'prompts':
          void qc.invalidateQueries({ queryKey: queryKeys.prompts })
          break
        case 'swept':
          void qc.invalidateQueries({ queryKey: queryKeys.results })
          void qc.invalidateQueries({ queryKey: queryKeys.jobs })
          break
        case 'assets':
          void qc.invalidateQueries({ queryKey: queryKeys.assets })
          void qc.invalidateQueries({ queryKey: queryKeys.drive })
          break
        case 'import':
          qc.setQueryData(queries.civitaiImport().queryKey, event.import)
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
