const CACHE_NAME = 'weather-assistant-v1';
const ASSETS_TO_CACHE = [
  '/',
  '/static/index.html',
  '/static/icon.png',
  '/static/style.css',
  '/static/manifest.json',
  '/static/marked.min.js'
];

// Install event - cache static assets with preload
self.addEventListener('install', event => {
  event.waitUntil(
    Promise.all([
      caches.open(CACHE_NAME).then(cache => {
        // Preload and cache all assets
        return cache.addAll(ASSETS_TO_CACHE);
      }),
      // Preload marked.min.js specifically with highest priority
      caches.open(CACHE_NAME).then(cache => {
        return cache.add('/static/marked.min.js');
      })
    ]).then(() => self.skipWaiting())
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys()
      .then(cacheNames => {
        return Promise.all(
          cacheNames
            .filter(name => name !== CACHE_NAME)
            .map(name => caches.delete(name))
        );
      })
      .then(() => self.clients.claim())
  );
});

// Fetch event - cache first for static assets, network first for API calls
self.addEventListener('fetch', event => {
  // Skip cross-origin requests
  if (!event.request.url.startsWith(self.location.origin)) {
    return;
  }

  // Handle API requests differently
  if (event.request.url.includes('/get_weather_advice') || 
      event.request.url.includes('/ask_followup')) {
    return; // Let the browser handle API requests normally
  }

  // Cache-first strategy for marked.min.js
  if (event.request.url.includes('marked.min.js')) {
    event.respondWith(
      caches.match(event.request)
        .then(response => {
          return response || fetch(event.request)
            .then(fetchResponse => {
              const responseToCache = fetchResponse.clone();
              caches.open(CACHE_NAME)
                .then(cache => {
                  cache.put(event.request, responseToCache);
                });
              return fetchResponse;
            });
        })
    );
    return;
  }

  // Network-first strategy for other requests
  event.respondWith(
    fetch(event.request)
      .catch(() => {
        return caches.match(event.request)
          .then(cachedResponse => {
            if (cachedResponse) {
              return cachedResponse;
            }
            if (event.request.mode === 'navigate') {
              return caches.match('/');
            }
            return new Response('Offline content not available');
          });
      })
  );
}); 