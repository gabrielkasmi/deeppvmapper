// ─── Known Issues page: live "Currently open reports" list ───────────────────
//
// Pulled straight from gabrielkasmi/openpvmapper-issues on every page load —
// the public GitHub REST API supports CORS for unauthenticated GET requests,
// so no backend/build step is needed (same live-fetch-on-load pattern as the
// map's own annotation counter, just plain fetch() instead of a Supabase
// RPC). Each visitor's browser makes its own request, so GitHub's 60/hour
// unauthenticated rate limit is per-visitor, not shared across the site.
//
// Issue titles/labels come from a public repo anyone can open an issue on —
// built as DOM nodes (textContent), never innerHTML, so nothing in a
// third-party issue title can inject markup.

const GITHUB_REPO = 'gabrielkasmi/openpvmapper-issues';
const GITHUB_API = `https://api.github.com/repos/${GITHUB_REPO}/issues?state=open&per_page=100`;

function relativeAge(iso) {
    const days = Math.floor((Date.now() - new Date(iso).getTime()) / 86400000);
    if (days <= 0) return 'today';
    if (days === 1) return '1 day ago';
    if (days < 30) return `${days} days ago`;
    const months = Math.floor(days / 30);
    if (months < 12) return `${months} month${months !== 1 ? 's' : ''} ago`;
    const years = Math.floor(days / 365);
    return `${years} year${years !== 1 ? 's' : ''} ago`;
}

function renderEmpty(container, message) {
    container.replaceChildren();
    const p = document.createElement('p');
    p.className = 'report-empty';
    p.textContent = message;
    container.appendChild(p);
}

function renderIssues(container, issues) {
    if (!issues.length) { renderEmpty(container, 'No open reports right now.'); return; }

    container.replaceChildren();
    issues
        .slice()
        .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
        .forEach(issue => {
            const a = document.createElement('a');
            a.className = 'report-item';
            a.href = issue.html_url;
            a.target = '_blank';
            a.rel = 'noopener';

            const title = document.createElement('div');
            title.className = 'report-item-title';
            title.textContent = issue.title;

            const meta = document.createElement('div');
            meta.className = 'report-item-meta';

            (issue.labels || []).forEach(label => {
                const pill = document.createElement('span');
                pill.className = 'report-label';
                pill.textContent = typeof label === 'string' ? label : label.name;
                meta.appendChild(pill);
            });

            const date = document.createElement('span');
            date.className = 'report-item-date';
            date.textContent = `opened ${relativeAge(issue.created_at)}`;
            meta.appendChild(date);

            a.appendChild(title);
            a.appendChild(meta);
            container.appendChild(a);
        });
}

async function loadOpenIssues() {
    const container = document.getElementById('report-list');
    if (!container) return;
    try {
        const res = await fetch(GITHUB_API, { headers: { Accept: 'application/vnd.github+json' } });
        if (!res.ok) throw new Error(`GitHub API responded ${res.status}`);
        const data = await res.json();
        const issues = data.filter(i => !i.pull_request);   // /issues also returns PRs — this repo shouldn't have any, but skip defensively
        renderIssues(container, issues);
    } catch (e) {
        console.error('Failed to load open issues:', e);
        renderEmpty(container, "Couldn't load the list right now — see the tracker directly on GitHub.");
    }
}

document.addEventListener('DOMContentLoaded', loadOpenIssues);
