// ─── Local admin-boundary lookup: exact contours + admin codes for named-place
// selections ─────────────────────────────────────────────────────────────────
//
// Nominatim (search.js) only gives us a bbox. For région/département/commune
// results we already ship the real boundary locally — the same files used by
// the ?dept=CODE deep link (departements.geojson, regions.geojson, one
// communes-XX.geojson per département, and region_depts.json). This resolves
// a Nominatim pick back to both the exact polygon (drawn on the map, and the
// client-side fallback filter) AND the admin code(s) that let render.js/
// export.js skip the spatial query entirely (store.js's fast insee/dpt path).
// Returns null when nothing local matches (an arrondissement, a hamlet, an
// overseas commune — no file for those — or anything Nominatim names
// differently than our data); the caller falls back to the Nominatim bbox
// rectangle in that case, so nothing breaks.

import { GEO_BASE } from './config.js';

let regionsPromise, departementsPromise, regionDeptsPromise;
const communesPromises = new Map();   // dept code -> Promise<FeatureCollection>

function loadRegions()      { return regionsPromise      ??= fetch(`${GEO_BASE}/regions.geojson`).then(r => r.json()); }
export function loadDepartements() { return departementsPromise ??= fetch(`${GEO_BASE}/departements.geojson`).then(r => r.json()); }
function loadCommunes(dept) {
    if (!communesPromises.has(dept))
        communesPromises.set(dept, fetch(`${GEO_BASE}/communes/communes-${dept}.geojson`).then(r => r.json()));
    return communesPromises.get(dept);
}
function loadRegionDepts()  { return regionDeptsPromise  ??= fetch(`${GEO_BASE}/region_depts.json`).then(r => r.json()); }

/** Accent/case/punctuation-insensitive compare key ("Île-de-France" == "ile de france"). */
function normalize(s) {
    return (s || '').toString().normalize('NFD').replace(/[̀-ͯ]/g, '')
        .toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
}

/** French postcode -> département code. Corse is split by postcode range
 *  (20000-20199 -> 2A, 20200+ -> 2B — the usual practical heuristic); DOM/TOM
 *  postcodes (97x/98x) return null since we don't ship those commune files. */
function deptCodeFromPostcode(postcode) {
    if (!postcode || postcode.length < 2) return null;
    if (postcode.startsWith('97') || postcode.startsWith('98')) return null;
    if (postcode.startsWith('20')) return parseInt(postcode, 10) >= 20200 ? '2B' : '2A';
    return postcode.slice(0, 2);
}

/**
 * item: a raw Nominatim /search result (fetched with addressdetails=1).
 * Returns { feature, admin } — feature is the exact boundary GeoJSON Feature;
 * admin is { inseeCodes } for a commune or { deptCodes } for a département/
 * région (the region's member départements, from region_depts.json) — or null
 * if nothing local matches.
 */
export async function resolveExactFeature(item) {
    const primaryName = normalize((item.display_name || '').split(',')[0]);
    if (!primaryName) return null;

    try {
        const [regions, departements] = await Promise.all([loadRegions(), loadDepartements()]);

        const region = regions.features.find(f => normalize(f.properties.nom) === primaryName);
        if (region) {
            const regionDepts = await loadRegionDepts();
            return { feature: region, admin: { deptCodes: regionDepts[region.properties.code] || [] } };
        }

        const dept = departements.features.find(f => normalize(f.properties.nom) === primaryName);
        if (dept) return { feature: dept, admin: { deptCodes: [dept.properties.code] } };

        const deptCode = deptCodeFromPostcode(item.address?.postcode);
        if (deptCode) {
            const communes = await loadCommunes(deptCode);
            const commune = communes.features.find(f => normalize(f.properties.nom) === primaryName);
            if (commune) return { feature: commune, admin: { inseeCodes: [commune.properties.code] } };
        }
    } catch (e) {
        console.error('Exact boundary lookup failed (falling back to bbox):', e);
    }
    return null;
}

/** Resolve a département's exact polygon + admin code straight from its code
 *  — used by the ?dept=CODE deep link (per-département static Data pages). */
export async function resolveDepartementByCode(code) {
    const departements = await loadDepartements();
    const feature = departements.features.find(f => f.properties.code === code);
    if (!feature) return null;
    return { feature, admin: { deptCodes: [code] } };
}

/** Resolve a région's exact polygon + its member départements' codes straight
 *  from its code — used by the ?region=CODE deep link (per-région static
 *  Data pages). */
export async function resolveRegionByCode(code) {
    const [regions, regionDepts] = await Promise.all([loadRegions(), loadRegionDepts()]);
    const feature = regions.features.find(f => f.properties.code === code);
    if (!feature) return null;
    return { feature, admin: { deptCodes: regionDepts[code] || [] } };
}

/** Resolve a commune's exact polygon + admin code straight from its INSEE
 *  code — used by the ?insee=CODE deep link (per-département static Data
 *  pages link to their largest cities this way). deptCode narrows which
 *  communes-XX.geojson to load; when absent, it's derived from the INSEE
 *  code itself (same départment-prefix heuristic used elsewhere here). */
export async function resolveCommuneByCode(inseeCode, deptCode) {
    const dept = deptCode || deptCodeFromInsee(inseeCode);
    if (!dept) return null;
    try {
        const communes = await loadCommunes(dept);
        const feature = communes.features.find(f => f.properties.code === inseeCode);
        if (!feature) return null;
        return { feature, admin: { inseeCodes: [inseeCode] } };
    } catch (e) {
        console.error('Commune lookup failed:', e);
        return null;
    }
}

/** INSEE code -> département code (same Corse postcode-style split as
 *  deptCodeFromPostcode, but INSEE codes for Corse already use 2A/2B). */
function deptCodeFromInsee(insee) {
    if (!insee || insee.length < 2) return null;
    if (insee.startsWith('97') || insee.startsWith('98')) return insee.slice(0, 3);
    return insee.slice(0, 2);
}

/** "Explore a random location" — picks a random département (the boundary
 *  set is already loaded for every deep link, so this costs nothing extra),
 *  then a random commune inside it (one extra, small, lazily-cached
 *  communes-XX.geojson fetch) for a town-level surprise rather than a whole
 *  département. Falls back to the département itself if that département
 *  happens to ship no commune file. Same { feature, admin } shape as every
 *  other resolver here, ready for setSelectionFromFeature. */
export async function resolveRandomLocation() {
    const departements = await loadDepartements();
    const feats = departements.features;
    const deptFeature = feats[Math.floor(Math.random() * feats.length)];
    const deptCode = deptFeature.properties.code;
    try {
        const communes = await loadCommunes(deptCode);
        const communeFeats = communes.features;
        if (communeFeats && communeFeats.length) {
            const commune = communeFeats[Math.floor(Math.random() * communeFeats.length)];
            return { feature: commune, admin: { inseeCodes: [commune.properties.code] } };
        }
    } catch (e) {
        console.error('Random commune lookup failed, falling back to département:', e);
    }
    return { feature: deptFeature, admin: { deptCodes: [deptCode] } };
}
