#!/usr/bin/env python3
"""
Refill content/known-issues.html's "Currently open reports" list from
gabrielkasmi/openpvmapper-issues (public repo, unauthenticated GitHub API —
no token needed for reading open issues on a public repo).

The rest of the page (intro, per-category prose, contact section) is
hand-authored and untouched — this script only splices what's between the
<!-- OPEN_ISSUES_START --> / <!-- OPEN_ISSUES_END --> markers, the same
splice pattern build_department_pages.py uses for the Leaderboard rankings.

Usage:
  python3 scripts/build_known_issues_page.py

Note: needs outbound internet access (api.github.com) — run it from your
own machine/terminal, not from a network-sandboxed tool. GitHub's
unauthenticated rate limit (60 requests/hour/IP) is far more than this
script needs (one request per ~100 open issues).
"""
import datetime
import json
import os
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE_PATH = os.path.join(ROOT, "content", "known-issues.html")

GITHUB_REPO = "gabrielkasmi/openpvmapper-issues"
GITHUB_API_BASE = "https://api.github.com"

START_MARKER = "<!-- OPEN_ISSUES_START -->"
END_MARKER = "<!-- OPEN_ISSUES_END -->"

REPORT_ITEM = """                <a class="report-item" href="{url}" target="_blank" rel="noopener">
                    <div class="report-item-title">{title}</div>
                    <div class="report-item-meta">
{labels}                        <span class="report-item-date">opened {age}</span>
                    </div>
                </a>"""

LABEL_PILL = '                        <span class="report-label">{name}</span>\n'


def fetch_open_issues():
    """All open issues on GITHUB_REPO, paginated (100/page), skipping pull
    requests (the /issues endpoint returns both — PRs carry a "pull_request"
    key, which this repo shouldn't have any of anyway, but skip defensively)."""
    issues = []
    page = 1
    while True:
        url = (f"{GITHUB_API_BASE}/repos/{GITHUB_REPO}/issues"
               f"?state=open&per_page=100&page={page}")
        req = urllib.request.Request(url, headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "deeppvmapper-build-script",
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                batch = json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            print(f"  ! GitHub API request failed: HTTP {e.code}\n    {body}", file=sys.stderr)
            raise
        if not batch:
            break
        issues.extend(i for i in batch if "pull_request" not in i)
        if len(batch) < 100:
            break
        page += 1
    return issues


def relative_age(iso):
    created = datetime.datetime.fromisoformat(iso.replace("Z", "+00:00"))
    days = (datetime.datetime.now(datetime.timezone.utc) - created).days
    if days <= 0:
        return "today"
    if days == 1:
        return "1 day ago"
    if days < 30:
        return f"{days} days ago"
    months = days // 30
    if months < 12:
        return f"{months} month{'s' if months != 1 else ''} ago"
    years = days // 365
    return f"{years} year{'s' if years != 1 else ''} ago"


def build_report_list(issues):
    if not issues:
        return '                <p class="report-empty">No open reports right now.</p>'
    issues_sorted = sorted(issues, key=lambda i: i["created_at"], reverse=True)
    rows = []
    for issue in issues_sorted:
        labels_html = "".join(
            LABEL_PILL.format(name=l["name"]) for l in issue.get("labels", [])
        )
        rows.append(REPORT_ITEM.format(
            url=issue["html_url"],
            title=issue["title"],
            labels=labels_html,
            age=relative_age(issue["created_at"]),
        ))
    return "\n".join(rows)


def inject_report_list(list_html):
    with open(PAGE_PATH, encoding="utf-8") as f:
        page = f.read()
    if START_MARKER not in page or END_MARKER not in page:
        raise ValueError(
            f"OPEN_ISSUES markers not found in {PAGE_PATH} — has the page "
            "been edited since this script was written?"
        )
    start = page.index(START_MARKER) + len(START_MARKER)
    end = page.index(END_MARKER)
    new_page = page[:start] + "\n" + list_html + "\n                " + page[end:]
    with open(PAGE_PATH, "w", encoding="utf-8") as f:
        f.write(new_page)


def main():
    print(f"Fetching open issues from {GITHUB_REPO}…")
    issues = fetch_open_issues()
    print(f"  {len(issues)} open issue(s) found.")
    list_html = build_report_list(issues)
    inject_report_list(list_html)
    print(f"Updated {PAGE_PATH}")


if __name__ == "__main__":
    main()
