/* Network-first service worker: always fetch fresh when online, fall back to the last good copy offline.
 * Bump CACHE to force-drop old entries. Audio never goes through here (it lives in IndexedDB). */
const CACHE = 'dictation-scenarios-v1';
const PRECACHE = [
  './',
  './index.html',
  './app.js',
  './styles.css',
  './sample-scenarios.json',
  '../recording/scenarios.json',
  'https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js',
];

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    // Individually, so one missing file (e.g. scenarios.json not committed yet) does not fail the install.
    await Promise.all(PRECACHE.map((u) => cache.add(u).catch(() => {})));
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    for (const k of await caches.keys()) if (k !== CACHE) await caches.delete(k);
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin && url.hostname !== 'cdnjs.cloudflare.com') return;
  event.respondWith((async () => {
    const cache = await caches.open(CACHE);
    try {
      const res = await fetch(req);
      if (res.ok || res.type === 'opaque') cache.put(req, res.clone()).catch(() => {});
      return res;
    } catch (e) {
      const hit = await cache.match(req, { ignoreSearch: true });
      if (hit) return hit;
      throw e;
    }
  })());
});
