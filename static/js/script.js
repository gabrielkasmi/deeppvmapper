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

    // Open GitHub issues (Contribute → Coder modal): live list, fetched
    // lazily the first time the Coder modal is opened (not on every
    // pageview), to be gentle with GitHub's unauthenticated rate limit.
    // Best-effort — falls back to a plain "view on GitHub" link if it can't
    // fetch. Issue titles and label names are user-generated content on a
    // public repo, so they're escaped before being inserted as HTML.
    const issuesList = document.getElementById('gh-issues-list');
    let issuesLoaded = false;
    function loadIssues() {
        if (issuesLoaded || !issuesList) return;
        issuesLoaded = true;
        const escapeHtml = (str) => String(str).replace(/[&<>"']/g, (c) => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
        }[c]));
        fetch('https://api.github.com/repos/gabrielkasmi/deeppvmapper/issues?state=open&per_page=10&sort=created&direction=desc')
            .then(r => r.ok ? r.json() : Promise.reject())
            .then(data => {
                const issues = data.filter(i => !i.pull_request).slice(0, 6);
                if (!issues.length) {
                    issuesList.innerHTML = '<p class="issues-empty">No open issues right now &mdash; check back soon, or <a href="https://github.com/gabrielkasmi/deeppvmapper/issues/new" target="_blank">open one yourself &rarr;</a></p>';
                    return;
                }
                issuesList.innerHTML = issues.map(issue => {
                    const labels = (issue.labels || []).map(l => {
                        const color = /^[0-9a-fA-F]{6}$/.test(l.color || '') ? l.color : '34495e';
                        return `<span class="issue-label" style="background-color:#${color}1a; color:#${color}; border-color:#${color}55;">${escapeHtml(l.name)}</span>`;
                    }).join('');
                    return `<a href="${issue.html_url}" target="_blank" class="issue-item">
                        <span class="issue-title">${escapeHtml(issue.title)}</span>
                        ${labels ? `<span class="issue-labels">${labels}</span>` : ''}
                    </a>`;
                }).join('');
            })
            .catch(() => {
                issuesList.innerHTML = '<p class="issues-empty">Couldn&rsquo;t load issues right now &mdash; <a href="https://github.com/gabrielkasmi/deeppvmapper/issues" target="_blank">view them directly on GitHub &rarr;</a></p>';
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
});