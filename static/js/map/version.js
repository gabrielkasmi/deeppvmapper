// ─── Deployed data version ────────────────────────────────────────────────────
//
// Shows which Zenodo release is actually behind the map right now, so anyone
// citing or screenshotting this page can tell exactly which version they saw.
// Backed by a plain local JSON file (static/data/version.json) rather than a
// live Zenodo API call: this way the displayed version always matches what's
// actually deployed, never "whatever Zenodo happens to have today" (those two
// can drift — a new release doesn't necessarily mean this site has been
// redeployed with it yet). Add a new entry to the front of that file's array
// each time you cut a release; older entries stay there as a version history
// even though only the first (current) one is rendered here today.

const VERSION_JSON_URL = '../static/data/version.json';

const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'];

function formatMonthYear(iso) {
    const [y, m] = String(iso).split('-').map(Number);
    if (!y || !m || m < 1 || m > 12) return iso;
    return `${MONTHS[m - 1]} ${y}`;
}

export async function initVersionInfo() {
    const el = document.getElementById('map-version-info');
    if (!el) return;
    try {
        const res = await fetch(VERSION_JSON_URL);
        if (!res.ok) throw new Error(`version.json: HTTP ${res.status}`);
        const releases = await res.json();
        const current = Array.isArray(releases) ? releases[0] : null;
        if (!current?.version) return;

        const dateLabel = current.date ? formatMonthYear(current.date) : '';
        const doiLink = current.doi
            ? ` &mdash; <a href="https://doi.org/${current.doi}" target="_blank" rel="noopener">view on Zenodo &#8599;</a>`
            : '';
        el.innerHTML = `Data version <strong>${current.version}</strong>${dateLabel ? ` (${dateLabel})` : ''}${doiLink}`;
    } catch (e) {
        // Silent by design — a missing/broken version.json shouldn't show an
        // error to visitors, it just means this line stays blank.
        console.error('Version info fetch failed:', e);
    }
}
