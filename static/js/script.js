// Smooth scrolling for anchor links
document.addEventListener('DOMContentLoaded', function() {
    // Add smooth scrolling to all links
    const links = document.querySelectorAll('a[href^="#"]:not([href="#"])');

    links.forEach(link => {
        link.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);

            if (targetElement) {
                e.preventDefault();
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add fade-in animation for sections
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observe all sections
    const sections = document.querySelectorAll('section');
    sections.forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(20px)';
        section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(section);
    });

    // GitHub repo badge (top nav): live star/fork counts. Best-effort — the
    // public API is unauthenticated and rate-limited, so this fails silently
    // and just leaves the "–" placeholder if it can't fetch.
    const starsEl = document.getElementById('gh-stars');
    const forksEl = document.getElementById('gh-forks');
    if (starsEl || forksEl) {
        fetch('https://api.github.com/repos/gabrielkasmi/deeppvmapper')
            .then(r => r.ok ? r.json() : Promise.reject())
            .then(data => {
                if (starsEl && typeof data.stargazers_count === 'number') {
                    starsEl.textContent = data.stargazers_count.toLocaleString('en-US');
                }
                if (forksEl && typeof data.forks_count === 'number') {
                    forksEl.textContent = data.forks_count.toLocaleString('en-US');
                }
            })
            .catch(() => {});
    }

    // Community contributions counter (hero stats): sum of the map's
    // crowdsourced annotations table and PV Check's verifications table —
    // two separate crowdsourcing efforts, one combined headline number.
    // Plain REST calls (no supabase-js import on this page), same project
    // credentials as static/js/map/config.js and game/js/config.js.
    // Best-effort — leaves the "—" placeholder on any failure.
    const communityCountEl = document.getElementById('community-count');
    if (communityCountEl) {
        const SUPABASE_URL = 'https://zelhliylrlktnasircwp.supabase.co';
        const SUPABASE_ANON_KEY = 'sb_publishable_rKz4rtTA3hpRxPgN3C3yAg_bbT5iTBi';
        const rpc = fn => fetch(`${SUPABASE_URL}/rest/v1/rpc/${fn}`, {
            method: 'POST',
            headers: {
                apikey: SUPABASE_ANON_KEY,
                Authorization: `Bearer ${SUPABASE_ANON_KEY}`,
                'Content-Type': 'application/json'
            },
            body: '{}'
        }).then(r => r.ok ? r.json() : Promise.reject());

        Promise.all([rpc('annotation_stats'), rpc('verification_stats')])
            .then(([annotations, verifications]) => {
                const total = (annotations?.count ?? 0) + (verifications?.count ?? 0);
                communityCountEl.textContent = total.toLocaleString('en-US');
            })
            .catch(() => {});
    }

    // Citation modal: the quote icon in the top nav opens a small popover
    // with the BibTeX entry (used instead of a full "Citation" page section).
    const citationTrigger = document.getElementById('citation-trigger');
    const citationOverlay = document.getElementById('citation-overlay');
    if (citationTrigger && citationOverlay) {
        const openCitation = () => { citationOverlay.style.display = 'flex'; };
        const closeCitation = () => { citationOverlay.style.display = 'none'; };
        citationTrigger.addEventListener('click', openCitation);
        citationOverlay.addEventListener('click', (e) => {
            if (e.target === citationOverlay) closeCitation();
        });
        const closeBtn = citationOverlay.querySelector('.citation-close');
        if (closeBtn) closeBtn.addEventListener('click', closeCitation);
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeCitation();
        });
    }

    // Open GitHub issues (Contribute → Code issues / Map reports modals):
    // live lists, fetched lazily the first time each modal is opened (not
    // on every pageview), to be gentle with GitHub's unauthenticated rate
    // limit. Best-effort — falls back to a plain "view on GitHub" link if
    // it can't fetch. Issue titles and label names are user-generated
    // content on a public repo, so they're escaped before being inserted
    // as HTML. Two independent lists share this same rendering logic: Code
    // issues (code bugs, gabrielkasmi/deeppvmapper) and Map reports
    // (data-quality reports, gabrielkasmi/openpvmapper-issues — see
    // known-issues.html for the same live-fetch pattern applied there).
    const escapeHtml = (str) => String(str).replace(/[&<>"']/g, (c) => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));

    function renderIssuesInto(container, issues, emptyHtml) {
        if (!issues.length) { container.innerHTML = `<p class="issues-empty">${emptyHtml}</p>`; return; }
        container.innerHTML = issues.map(issue => {
            const labels = (issue.labels || []).map(l => {
                const color = /^[0-9a-fA-F]{6}$/.test(l.color || '') ? l.color : '34495e';
                return `<span class="issue-label" style="background-color:#${color}1a; color:#${color}; border-color:#${color}55;">${escapeHtml(l.name)}</span>`;
            }).join('');
            return `<a href="${issue.html_url}" target="_blank" class="issue-item">
                <span class="issue-title">${escapeHtml(issue.title)}</span>
                ${labels ? `<span class="issue-labels">${labels}</span>` : ''}
            </a>`;
        }).join('');
    }

    function loadIssuesFrom(repo, container, emptyHtml, errorHtml) {
        fetch(`https://api.github.com/repos/${repo}/issues?state=open&per_page=10&sort=created&direction=desc`)
            .then(r => r.ok ? r.json() : Promise.reject())
            .then(data => renderIssuesInto(container, data.filter(i => !i.pull_request).slice(0, 6), emptyHtml))
            .catch(() => { container.innerHTML = `<p class="issues-empty">${errorHtml}</p>`; });
    }

    const issuesList = document.getElementById('gh-issues-list');
    let issuesLoaded = false;
    function loadIssues() {
        if (issuesLoaded || !issuesList) return;
        issuesLoaded = true;
        loadIssuesFrom(
            'gabrielkasmi/deeppvmapper', issuesList,
            'No open issues right now &mdash; check back soon, or <a href="https://github.com/gabrielkasmi/deeppvmapper/issues/new" target="_blank">open one yourself &rarr;</a>',
            'Couldn&rsquo;t load issues right now &mdash; <a href="https://github.com/gabrielkasmi/deeppvmapper/issues" target="_blank">view them directly on GitHub &rarr;</a>'
        );
    }

    const mapperIssuesList = document.getElementById('gh-mapper-issues-list');
    let mapperIssuesLoaded = false;
    function loadMapperIssues() {
        if (mapperIssuesLoaded || !mapperIssuesList) return;
        mapperIssuesLoaded = true;
        loadIssuesFrom(
            'gabrielkasmi/openpvmapper-issues', mapperIssuesList,
            'No open reports right now &mdash; nice. <a href="https://github.com/gabrielkasmi/openpvmapper-issues" target="_blank">Browse the tracker &rarr;</a>',
            'Couldn&rsquo;t load reports right now &mdash; <a href="https://github.com/gabrielkasmi/openpvmapper-issues" target="_blank">view them directly on GitHub &rarr;</a>'
        );
    }

    // MapRoulette challenges — same live-fetch-on-modal-open pattern as the
    // GitHub issues lists above, mirrored per the site owner's request
    // ("fetch challenges the way issues are fetched, so new ones show up
    // automatically"). Project 64195 (DeepPVMapper) has no "list challenges
    // by project id" endpoint that works unauthenticated (confirmed by
    // testing /project/:id/challenges, which returns [] even for a live
    // challenge, and /challenges/listing, which needs auth) — the public,
    // no-auth endpoint that does work is /challenges/extendedFind?ps=<name>,
    // which matches against the project's *name* (its internal
    // "Home_24273900" slug won't match; its displayName "DeepPVMapper"
    // does). Since that's a loose text match, every result is re-checked
    // client-side against parent.id === 64195 before being shown, so a
    // same-named or similarly-named project elsewhere on MapRoulette can
    // never sneak into this list.
    //
    // NB: unlike the GitHub API, MapRoulette's API may or may not send
    // permissive CORS headers for a browser-side fetch from a third-party
    // origin like deeppvmapper.fr — this hasn't been verified from an
    // actual browser. If it turns out to be blocked, fetch() simply rejects
    // and the existing .catch() below falls back to the "view on
    // MapRoulette" link, same as a GitHub API hiccup would — no broken UI
    // either way.
    const mrChallengesList = document.getElementById('mr-challenges-list');
    const mrChallengesBanner = document.getElementById('mr-challenges-banner');
    const MR_PROJECT_ID = 64195;
    let mrChallengesLoaded = false;

    function renderChallengesInto(container, bannerEl, challenges) {
        if (!challenges.length) {
            container.innerHTML = '<p class="issues-empty">No live challenges right now &mdash; check back soon, or <a href="https://maproulette.org/browse/projects/64195" target="_blank">browse the project on MapRoulette &rarr;</a></p>';
            if (bannerEl) bannerEl.style.display = 'none';
            return;
        }
        let totalTasks = 0, doneTasks = 0;
        container.innerHTML = challenges.map(c => {
            const m = c.completionMetrics || {};
            const total = m.total || 0;
            const remaining = m.tasksRemaining != null ? m.tasksRemaining : total;
            const done = Math.max(0, total - remaining);
            totalTasks += total;
            doneTasks += done;
            const pct = total ? Math.round((done / total) * 100) : 0;
            return `<a href="https://maproulette.org/browse/challenges/${c.id}" target="_blank" class="issue-item mr-challenge-item">
                <span class="mr-challenge-top">
                    <span class="issue-title">${escapeHtml(c.name)}</span>
                    <span class="mr-challenge-pct">${pct}%</span>
                </span>
                <span class="mr-challenge-bar"><span class="mr-challenge-bar-fill" style="width:${pct}%"></span></span>
                <span class="mr-challenge-meta">${done.toLocaleString('en-US')} / ${total.toLocaleString('en-US')} tasks reviewed</span>
            </a>`;
        }).join('');
        if (bannerEl) {
            const overallPct = totalTasks ? Math.round((doneTasks / totalTasks) * 100) : 0;
            const label = challenges.length > 1 ? `${challenges.length} live challenges` : '1 live challenge';
            bannerEl.innerHTML = `${overallPct}% complete &mdash; ${doneTasks.toLocaleString('en-US')} of ${totalTasks.toLocaleString('en-US')} tasks reviewed across ${label}.`;
            bannerEl.style.display = '';
        }
    }

    function loadMapperChallenges() {
        if (mrChallengesLoaded || !mrChallengesList) return;
        mrChallengesLoaded = true;
        fetch('https://maproulette.org/api/v2/challenges/extendedFind?ps=DeepPVMapper&limit=50')
            .then(r => r.ok ? r.json() : Promise.reject())
            .then(data => {
                const live = (Array.isArray(data) ? data : [])
                    .filter(c => c.parent && c.parent.id === MR_PROJECT_ID && c.enabled && !c.deleted && !c.isArchived)
                    .sort((a, b) => new Date(b.created) - new Date(a.created));
                renderChallengesInto(mrChallengesList, mrChallengesBanner, live);
            })
            .catch(() => {
                mrChallengesList.innerHTML = '<p class="issues-empty">Couldn&rsquo;t load challenges right now &mdash; <a href="https://maproulette.org/browse/projects/64195" target="_blank">view them directly on MapRoulette &rarr;</a></p>';
                if (mrChallengesBanner) mrChallengesBanner.style.display = 'none';
            });
    }

    // Generic modal triggers: any [data-modal-target] opens the .modal-overlay
    // with that id (used by the Contribute page's Enthusiast/Mapper/Coder
    // cards, instead of duplicating content both as cards and as inline
    // sections). Closes via its own .modal-close, a backdrop click, or
    // Escape. Scoped only to the overlays actually referenced this way, so
    // it can't interfere with the citation modal or outlook.html's own
    // class-toggled related-work modal, which have their own dedicated logic.
    const modalTriggers = document.querySelectorAll('[data-modal-target]');
    if (modalTriggers.length) {
        const targetOverlays = new Set();
        modalTriggers.forEach(trigger => {
            const overlay = document.getElementById(trigger.dataset.modalTarget);
            if (!overlay) return;
            targetOverlays.add(overlay);
            trigger.addEventListener('click', (e) => {
                e.preventDefault();
                overlay.style.display = 'flex';
                if (overlay.id === 'modal-coder') loadIssues();
                if (overlay.id === 'modal-mapreports') loadMapperIssues();
                if (overlay.id === 'modal-maproulette') loadMapperChallenges();
            });
        });
        targetOverlays.forEach(overlay => {
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) overlay.style.display = 'none';
            });
            const closeBtn = overlay.querySelector('.modal-close');
            if (closeBtn) closeBtn.addEventListener('click', () => { overlay.style.display = 'none'; });
        });
        document.addEventListener('keydown', (e) => {
            if (e.key !== 'Escape') return;
            targetOverlays.forEach(overlay => {
                if (overlay.style.display === 'flex') overlay.style.display = 'none';
            });
        });
    }

    // "Report an issue" CTA on département/city (and, potentially, région)
    // pages — a lightweight, no-backend alternative to the interactive
    // map's Supabase-backed report form (static/js/map/report.js): these
    // pages have no database wiring, so a click just folds the visitor's
    // comment and the page URL into a mailto: link instead of an insert.
    const reportBtn = document.getElementById('report-issue-btn');
    if (reportBtn) {
        reportBtn.addEventListener('click', () => {
            const commentEl = document.getElementById('report-comment');
            const comment = (commentEl && commentEl.value || '').trim();
            const targetLabel = reportBtn.dataset.targetLabel || 'DeepPVMapper page';
            const subject = `DeepPVMapper — Issue report: ${targetLabel}`;
            const bodyLines = [
                comment || '(no comment provided)',
                '',
                `Page: ${location.href}`,
            ];
            const mailto = 'mailto:gabriel.kasmi@deeppvmapper.fr'
                + `?subject=${encodeURIComponent(subject)}`
                + `&body=${encodeURIComponent(bodyLines.join('\n'))}`;
            window.location.href = mailto;
        });
    }

    // Back to top: only on the long-form, text-heavy pages that opt in via
    // body.has-back-to-top (About, Pipeline, OpenPVMapper, Use Cases,
    // Registry Audit, Contribute, Software, Publications, In Press, Known
    // Issues, Data Documentation, Resources — not the landing page, the
    // interactive map, the leaderboard, or the generated département/city/
    // région pages, which are stat/table-driven rather than prose-heavy).
    // The button itself is injected here rather than hand-added to every
    // page's HTML, so there's a single place to change its markup/behavior.
    if (document.body.classList.contains('has-back-to-top')) {
        const backToTop = document.createElement('button');
        backToTop.type = 'button';
        backToTop.className = 'back-to-top';
        backToTop.setAttribute('aria-label', 'Back to top');
        backToTop.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 19V5"/><path d="M5 12l7-7 7 7"/></svg>';
        document.body.appendChild(backToTop);

        let ticking = false;
        function updateBackToTop() {
            backToTop.classList.toggle('is-visible', window.scrollY > 600);
            ticking = false;
        }
        window.addEventListener('scroll', () => {
            if (!ticking) {
                window.requestAnimationFrame(updateBackToTop);
                ticking = true;
            }
        }, { passive: true });
        updateBackToTop();

        backToTop.addEventListener('click', () => {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }
});