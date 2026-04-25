// Baseline service worker — minimal "always online with graceful fallback"
//
// Strategy:
//   • App shell (/, /index.html, manifest, icons) → cache-first, falls
//     back to network. This makes opening the App fast even on flaky
//     mobile data.
//   • API calls (/api/**)              → network-only (don't cache, the
//     data is always fresh and the server is the source of truth).
//   • Media (/api/clips/*/video, etc.) → network-only too — videos are
//     too big to cache, and we want range requests to work.

// Bump this when shipping shell changes — old PWAs will pick up the new
// version on their next page load (browser checks the SW script byte-for-byte
// every load, and a different cache name triggers a fresh shell fetch).
const CACHE = 'baseline-shell-v4';
const SHELL = [
  '/',
  '/index.html',
  '/manifest.webmanifest',
  '/icon-192.png',
  '/icon-512.png',
  '/apple-touch-icon.png',
  '/favicon.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting())
  );
});

// Allow the page to tell us "you're the new SW, take over now"
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(
      keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))
    )).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Never cache API or media — always go to network.
  if (url.pathname.startsWith('/api/')) {
    return; // let browser handle it directly
  }

  // App shell: cache-first
  if (event.request.method === 'GET') {
    event.respondWith(
      caches.match(event.request).then((cached) => {
        if (cached) return cached;
        return fetch(event.request).then((resp) => {
          // Opportunistic: cache successful shell-ish responses
          if (resp && resp.status === 200 && resp.type === 'basic') {
            const copy = resp.clone();
            caches.open(CACHE).then((c) => c.put(event.request, copy));
          }
          return resp;
        }).catch(() => caches.match('/'));
      })
    );
  }
});
