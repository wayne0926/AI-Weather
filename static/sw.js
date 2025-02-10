const CACHE_NAME = 'weather-assistant-v1';
const ASSETS_TO_CACHE = [
  '/',
  '/static/index.html',
  '/static/icon.png',
  '/static/style.css',
  '/static/manifest.json',
  'https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js'
];

// Install event - cache static assets
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(ASSETS_TO_CACHE))
      .then(() => self.skipWaiting())
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

// Fetch event - network first, fallback to cache
self.addEventListener('fetch', event => {
  // Skip cross-origin requests
  if (!event.request.url.startsWith(self.location.origin)) {
    return;
  }

  // Handle API requests differently
  if (event.request.url.includes('/get_weather_advice') || 
      event.request.url.includes('/ask_followup') ||
      event.request.url.includes('/get_ip_location')) {
    return; // Let the browser handle API requests normally
  }

  event.respondWith(
    fetch(event.request)
      .catch(() => {
        return caches.match(event.request)
          .then(cachedResponse => {
            if (cachedResponse) {
              return cachedResponse;
            }
            // If the request is not in cache and network is unavailable,
            // return a basic offline page
            if (event.request.mode === 'navigate') {
              return caches.match('/');
            }
            return new Response('Offline content not available');
          });
      })
  );
}); 