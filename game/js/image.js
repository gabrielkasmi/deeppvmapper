// ─── Card image: IGN WMS GetMap, exact crop at a precomputed GSD ──────────
// lat/lng + gsd come straight from campaign_pool (precomputed once in
// scripts/verifications_setup.sql from the polygon's real extent) — no
// geometry math here, just turning that into a GetMap request.

import { IGN_WMS_URL, IGN_WMS_LAYER, CARD_PX } from './config.js';

const WEB_MERCATOR_R = 20037508.342789244; // meters, half the projection's width

function toWebMercator(lat, lng) {
    const x = lng * WEB_MERCATOR_R / 180;
    const y = Math.log(Math.tan((90 + lat) * Math.PI / 360)) / (Math.PI / 180) * WEB_MERCATOR_R / 180;
    return [x, y];
}

/** Builds the GetMap URL for one card: a CARD_PX x CARD_PX crop centered on
 *  (lat, lng), sized so that gsd (meters/pixel) covers the installation
 *  with margin (see the pool-population query for how gsd is chosen). */
export function cardImageUrl(lat, lng, gsd) {
    const [x, y] = toWebMercator(lat, lng);
    const half = (CARD_PX / 2) * gsd;
    const bbox = [x - half, y - half, x + half, y + half].join(',');
    const params = new URLSearchParams({
        SERVICE: 'WMS', VERSION: '1.3.0', REQUEST: 'GetMap',
        LAYERS: IGN_WMS_LAYER, STYLES: '', CRS: 'EPSG:3857',
        BBOX: bbox, WIDTH: String(CARD_PX), HEIGHT: String(CARD_PX),
        FORMAT: 'image/jpeg',
    });
    return `${IGN_WMS_URL}?${params.toString()}`;
}

// Note: no polygon-to-pixels projection here (there used to be one, for a
// contour overlay) — dropped deliberately. Showing the detection's exact
// outline invited people to judge "is this shape drawn correctly" instead
// of the actual question this pool exists to answer, "is there a PV
// installation here at all." The card now just shows a fixed square marker
// at dead center (see index.html #card-marker) — always correct since the
// image is already framed on the centroid, no per-card computation needed.

/** Prefetches an image (browser HTTP cache does the rest) and resolves once
 *  it has actually loaded — so the UI can wait for it rather than flashing
 *  a blank card. Resolves (not rejects) on error too, with ok:false, so one
 *  broken image never blocks the rest of a prefetched batch. */
export function preloadImage(url) {
    return new Promise(resolve => {
        const img = new Image();
        img.onload = () => resolve({ ok: true, url });
        img.onerror = () => resolve({ ok: false, url });
        img.src = url;
    });
}
