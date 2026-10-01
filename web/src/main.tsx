import '@fontsource-variable/newsreader/opsz.css'
import '@fontsource-variable/newsreader/opsz-italic.css'
import '@fontsource-variable/schibsted-grotesk'
import './styles/index.css'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'

import { App } from './app/App'
import { installShield } from './lib/discretion'

const root = document.getElementById('root')
if (!root) throw new Error('#root element missing')

// Server events keep the cache fresh, so window-focus refetches aren't needed.
const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 1 } },
})

// The service worker keeps the app shell offline and shows push notifications.
if (import.meta.env.PROD && 'serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').catch((e: unknown) => {
    console.warn('service worker registration failed', e)
  })
}

installShield()

createRoot(root).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
)
