// Baseline service worker — network-first for the shell.
//
// Strategy change after a stuck-on-stale-shell incident:
//   • HTML / navigation requests → ALWAYS try network first, fall back
//     to cache only if truly offline. The app is tunnel-served, so
//     "online" is the normal state. Caching the shell aggressively
//     was leaving users on old code after deploys, even with
//     CACHE-name bumps.
//   • API calls (/api/**) → never cached (data is fresh per request).
//   • Static assets (icons, manifest) → cache-first for installability,
//     but tolerate misses.
//
// This means the cache name barely matters anymore — the new HTML always
// wins on the next page load.

const CACHE = 'baseline-shell-v8';
const STATIC_ASSETS = [
  '/manifest.webmanifest',
  '/icon-192.png',
  '/icon-512.png',
  '/apple-touch-icon.png',
  '/favicon.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE)
      .then((c) => c.addAll(STATIC_ASSETS).catch(() => {}))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  // Nuke ALL old caches — even ones we don't recognize. Free-tier hosts
  // sometimes leave orphan caches that pin stale assets.
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') self.skipWaiting();
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;
  const url = new URL(event.request.url);

  // API: bypass entirely
  if (url.pathname.startsWith('/api/')) return;

  // HTML / navigation: network first, cache as last-resort
  const isDoc =
    event.request.mode === 'navigate' ||
    event.request.destination === 'document' ||
    url.pathname === '/' ||
    url.pathname.endsWith('.html');

  if (isDoc) {
    event.respondWith(
      fetch(event.request)
        .then((resp) => {
          // Don't cache HTML — let the browser always fetch the latest.
          return resp;
        })
        .catch(() => caches.match(event.request).then((c) => c || caches.match('/')))
    );
    return;
  }

  // Static assets: cache first, network as fallback
  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request).then((resp) => {
        if (resp && resp.status === 200 && resp.type === 'basic') {
          const copy = resp.clone();
          caches.open(CACHE).then((c) => c.put(event.request, copy)).catch(() => {});
        }
        return resp;
      });
    })
  );
});
