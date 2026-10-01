import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render } from '@testing-library/react'
import { StrictMode } from 'react'

import { App } from '@/app/App'

/** Render the whole app, as these tests do: they drive a feature the way a user reaches it. */
export function renderApp() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  // StrictMode as in main.tsx, so effects that don't survive a remount show up here.
  return render(
    <StrictMode>
      <QueryClientProvider client={client}>
        <App />
      </QueryClientProvider>
    </StrictMode>,
  )
}
