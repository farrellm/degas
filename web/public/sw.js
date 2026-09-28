// Degas service worker: keeps the app shell available offline and shows Web Push
// notifications (design §8.1, §8.3). The API is never cached: it's live GPU state.

const CACHE = 'degas-shell-v1'
const SHELL = ['/', '/manifest.webmanifest', '/icon.svg', '/apple-touch-icon.png']

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(CACHE)
      .then((cache) => cache.addAll(SHELL))
      .then(() => self.skipWaiting()),
  )
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  )
})

self.addEventListener('fetch', (event) => {
  const request = event.request
  const url = new URL(request.url)
  if (request.method !== 'GET' || url.origin !== self.location.origin) return
  if (url.pathname.startsWith('/api/')) return

  // Built assets have content hashes in their names, so a cached copy is always right.
  if (url.pathname.startsWith('/assets/')) {
    event.respondWith(
      caches.match(request).then(
        (hit) =>
          hit ??
          fetch(request).then((response) => {
            if (response.ok) {
              const copy = response.clone()
              void caches.open(CACHE).then((cache) => cache.put(request, copy))
            }
            return response
          }),
      ),
    )
    return
  }

  // Everything else (the page itself, icons): the network first, so updates arrive.
  event.respondWith(
    fetch(request)
      .then((response) => {
        if (response.ok) {
          const copy = response.clone()
          const key = request.mode === 'navigate' ? '/' : request
          void caches.open(CACHE).then((cache) => cache.put(key, copy))
        }
        return response
      })
      .catch(() =>
        caches
          .match(request.mode === 'navigate' ? '/' : request)
          .then((hit) => hit ?? Response.error()),
      ),
  )
})

self.addEventListener('push', (event) => {
  let data
  try {
    data = event.data ? event.data.json() : {}
  } catch {
    data = { title: event.data ? event.data.text() : 'Degas' }
  }
  event.waitUntil(
    self.registration.showNotification(data.title || 'Degas', {
      body: data.body || undefined,
      tag: data.tag || undefined,
      icon: '/icon-192.png',
      data: { url: data.url || '/' },
    }),
  )
})

self.addEventListener('notificationclick', (event) => {
  event.notification.close()
  const url = (event.notification.data && event.notification.data.url) || '/'
  event.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then((windows) => {
      const open = windows[0]
      if (open) {
        // The app is running: tell it where to go rather than reloading it.
        open.postMessage({ type: 'open', url })
        return open.focus()
      }
      return self.clients.openWindow(url)
    }),
  )
})
