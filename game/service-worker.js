// Minimal service worker — exists mainly to satisfy Chrome's installability
// criteria (a registered SW with a fetch handler). No image caching logic
// here on purpose (see game/README.md) — that stays in plain page JS
// (game/js/image.js), simpler to write/debug than intercepted fetches. The
// app shell (HTML/CSS/JS) is cached so a repeat visit opens instantly; data
// (Supabase, IGN images) always goes straight to the network.

const CACHE = 'pv-scroll-shell-v1';
const SHELL = [
    './', './index.html', './css/style.css',
    './js/config.js', './js/store.js', './js/auth.js', './js/image.js',
    './js/campaign.js', './js/swipe.js', './js/menu.js', './js/main.js', './js/help.js',
];

self.addEventListener('install', event => {
    event.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
              .then(() => self.clients.claim())
    );
});

self.addEventListener('fetch', event => {
    const { request } = event;
    if (request.method !== 'GET' || !request.url.startsWith(self.location.origin)) return; // shell only — never touches Supabase/IGN requests
    event.respondWith(
        caches.match(request).then(cached => cached || fetch(request))
    );
});
