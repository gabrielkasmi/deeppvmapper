#!/usr/bin/env node
// ─── DeepPVMapper Bluesky bot ───────────────────────────────────────────────
//
// Posts to Bluesky when:
//   1. The all-time leaderboard top 5 has changed since the last run.
//   1b. The rolling-7-day leaderboard top 5 has changed, checked ONLY on
//       WEEKLY_LEADERBOARD_DAY (Friday, Europe/Paris) — a real weekly
//       recap of leaderboard(p_window='week')'s current rolling view,
//       rather than a post that could land on any day. Run with
//       --force-weekly to test this check on a non-Friday.
//   2. installations_done (PV Check, season_completion()) crosses a new
//      multiple of 50.
//   3. annotation_stats().count (the map's crowdsourced annotations) crosses
//      a new multiple of 1000.
//
// Meant to run on a schedule (see ../.github/workflows/bluesky-bot.yml —
// once a day), but is idempotent to re-run any time: it never posts the same
// milestone/leaderboard snapshot twice, because "what was last announced" is
// tracked in Supabase (public.bluesky_bot_state — see
// ../bluesky_bot_setup.sql) and only advances after a successful post.
//
// No npm dependencies on purpose (this repo has no package.json/build step
// anywhere else either) — plain fetch() against PostgREST and the AT
// Protocol's raw HTTP API. Requires Node >= 18 for global fetch.
//
// Required environment variables:
//   SUPABASE_URL                 e.g. https://xxxx.supabase.co (not secret —
//                                 same value already public in the site's own
//                                 client-side JS)
//   SUPABASE_SERVICE_ROLE_KEY    Supabase project service_role key — SECRET,
//                                 bypasses RLS, never expose client-side
//   BLUESKY_IDENTIFIER           the bot account's handle, e.g.
//                                 deeppvmapper.bsky.social
//   BLUESKY_APP_PASSWORD         an App Password generated in Bluesky
//                                 Settings → App Passwords — SECRET, never
//                                 the account's real login password
//
// Run with --dry-run to log what WOULD be posted and updated, without
// actually calling Bluesky or writing to Supabase — use this first.
// Run with --force-weekly to also exercise the Friday-only rolling-7-day
// leaderboard check regardless of what day it actually is (testing only —
// combine with --dry-run).

const DRY_RUN = process.argv.includes('--dry-run');
const FORCE_WEEKLY = process.argv.includes('--force-weekly'); // testing only — see main()

const CAMPAIGN_ID = 'season-1'; // see game/js/config.js CAMPAIGN_ID
const INSTALLATIONS_STEP = 50;
const ANNOTATIONS_STEP = 1000;
const VERIFICATIONS_STEP = 1000; // total PV Check contributions (public.verifications) — NOT the same counter as ANNOTATIONS_STEP (public.annotations, the map tool)
const BLUESKY_PDS = 'https://bsky.social';
const WEEKLY_LEADERBOARD_DAY = 'Fri'; // Europe/Paris local day (see isWeeklyLeaderboardDay())

// The workflow itself runs once a day, every day (see
// ../../.github/workflows/bluesky-bot.yml) — the rolling-7-day leaderboard
// check below only actually does anything on this one day of the week, so
// "weekly" reads as a real Friday recap rather than a leaderboard post
// that can land on any day depending on when contributors happen to
// reshuffle. Uses Europe/Paris (Gabriel's timezone), via Node's built-in
// Intl (no extra dependency), rather than the runner's UTC day — the two
// only disagree right at the UTC day boundary, which the 20:00 UTC/22:00
// CEST run time is nowhere near.
function isWeeklyLeaderboardDay(date = new Date()) {
    if (FORCE_WEEKLY) return true;
    const weekday = new Intl.DateTimeFormat('en-US', { timeZone: 'Europe/Paris', weekday: 'short' }).format(date);
    return weekday === WEEKLY_LEADERBOARD_DAY;
}

function requireEnv(name) {
    const v = process.env[name];
    if (!v) {
        console.error(`Missing required environment variable: ${name}`);
        process.exit(1);
    }
    return v;
}

const SUPABASE_URL = requireEnv('SUPABASE_URL');
const SUPABASE_SERVICE_ROLE_KEY = requireEnv('SUPABASE_SERVICE_ROLE_KEY');
const BLUESKY_IDENTIFIER = requireEnv('BLUESKY_IDENTIFIER');
const BLUESKY_APP_PASSWORD = requireEnv('BLUESKY_APP_PASSWORD');

// ─── Supabase (service_role — server-side only, bypasses RLS) ─────────────

async function supabaseRpc(fn, body = {}) {
    const res = await fetch(`${SUPABASE_URL}/rest/v1/rpc/${fn}`, {
        method: 'POST',
        headers: {
            apikey: SUPABASE_SERVICE_ROLE_KEY,
            Authorization: `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
    });
    if (!res.ok) {
        throw new Error(`RPC ${fn} failed: HTTP ${res.status} — ${await res.text()}`);
    }
    return res.json();
}

async function getBotState() {
    const res = await fetch(`${SUPABASE_URL}/rest/v1/bluesky_bot_state?id=eq.1&select=*`, {
        headers: {
            apikey: SUPABASE_SERVICE_ROLE_KEY,
            Authorization: `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
        },
    });
    if (!res.ok) {
        throw new Error(`Fetching bluesky_bot_state failed: HTTP ${res.status} — ${await res.text()}`);
    }
    const rows = await res.json();
    if (!rows.length) {
        throw new Error(
            "bluesky_bot_state has no row with id=1 — run scripts/bluesky_bot_setup.sql in the Supabase SQL Editor first."
        );
    }
    return rows[0];
}

async function patchBotState(patch) {
    if (DRY_RUN) {
        console.log('[dry-run] would PATCH bluesky_bot_state with:', patch);
        return;
    }
    const res = await fetch(`${SUPABASE_URL}/rest/v1/bluesky_bot_state?id=eq.1`, {
        method: 'PATCH',
        headers: {
            apikey: SUPABASE_SERVICE_ROLE_KEY,
            Authorization: `Bearer ${SUPABASE_SERVICE_ROLE_KEY}`,
            'Content-Type': 'application/json',
            Prefer: 'return=minimal',
        },
        body: JSON.stringify({ ...patch, updated_at: new Date().toISOString() }),
    });
    if (!res.ok) {
        throw new Error(`Updating bluesky_bot_state failed: HTTP ${res.status} — ${await res.text()}`);
    }
}

// ─── Bluesky (AT Protocol, raw HTTP — no @atproto/api dependency) ─────────

async function bskyLogin() {
    const res = await fetch(`${BLUESKY_PDS}/xrpc/com.atproto.server.createSession`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ identifier: BLUESKY_IDENTIFIER, password: BLUESKY_APP_PASSWORD }),
    });
    if (!res.ok) {
        throw new Error(`Bluesky login failed: HTTP ${res.status} — ${await res.text()}`);
    }
    const { accessJwt, did } = await res.json();
    return { accessJwt, did };
}

// Bluesky's post-length limit is 300 *graphemes*, not bytes/UTF-16 code
// units — Array.from() splits on Unicode code points, which is close enough
// for the plain-ASCII-plus-emoji text this bot writes (a handful of emoji
// used here are single code points). Truncates defensively; every message
// below is written to comfortably fit anyway.
function truncateForBluesky(text, max = 300) {
    const chars = Array.from(text);
    if (chars.length <= max) return text;
    return chars.slice(0, max - 1).join('') + '…';
}

// The AT Protocol does NOT auto-linkify bare URLs in a post's text — the
// official app's composer only *looks* like it does because it computes
// "facets" (byte-range annotations over the text) client-side before
// posting. Without this, a URL in `text` is just inert text. Facet byte
// offsets are UTF-8 BYTE offsets into the text, not character/grapheme
// offsets, hence the Buffer.byteLength() dance below (matters here because
// our messages contain multi-byte emoji before the URL).
function detectLinkFacets(text) {
    const urlPattern = /https?:\/\/[^\s]+/g;
    const facets = [];
    let match;
    while ((match = urlPattern.exec(text)) !== null) {
        // Bluesky trims common trailing punctuation from auto-detected
        // links; none of our own templates end a URL with punctuation, but
        // strip it defensively in case a future template does.
        const url = match[0].replace(/[.,!?;:)\]]+$/, '');
        const byteStart = Buffer.byteLength(text.slice(0, match.index), 'utf8');
        const byteEnd = byteStart + Buffer.byteLength(url, 'utf8');
        facets.push({
            index: { byteStart, byteEnd },
            features: [{ $type: 'app.bsky.richtext.facet#link', uri: url }],
        });
    }
    return facets;
}

async function bskyPost(session, text) {
    const postText = truncateForBluesky(text);
    const facets = detectLinkFacets(postText);
    const record = {
        $type: 'app.bsky.feed.post',
        text: postText,
        createdAt: new Date().toISOString(),
        ...(facets.length ? { facets } : {}),
    };
    if (DRY_RUN) {
        console.log('[dry-run] would post to Bluesky:\n---\n' + record.text + '\n---');
        if (facets.length) console.log('[dry-run] facets:', JSON.stringify(facets));
        return;
    }
    const res = await fetch(`${BLUESKY_PDS}/xrpc/com.atproto.repo.createRecord`, {
        method: 'POST',
        headers: {
            Authorization: `Bearer ${session.accessJwt}`,
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            repo: session.did,
            collection: 'app.bsky.feed.post',
            record,
        }),
    });
    if (!res.ok) {
        throw new Error(`Bluesky post failed: HTTP ${res.status} — ${await res.text()}`);
    }
    console.log('Posted to Bluesky:\n' + record.text);
}

// ─── Message builders ──────────────────────────────────────────────────────

function medal(rank) {
    return ['🥇', '🥈', '🥉'][rank] ?? `${rank + 1}.`;
}

function formatTop5(top5) {
    return top5
        .map((row, i) => `${medal(i)} ${row.pseudo} — ${row.total.toLocaleString('en-US')}`)
        .join('\n');
}

function top5Changed(a, b) {
    // Order-sensitive on purpose — a reshuffle within the same 5 names
    // (someone overtaking someone else) counts as a change just as much as
    // a new name entering the top 5.
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return true;
    return a.some((row, i) => row.pseudo !== b[i]?.pseudo);
}

// ─── Main ───────────────────────────────────────────────────────────────────

async function main() {
    const state = await getBotState();
    const posts = []; // queued message strings, sent in order with a short gap between each

    // 1. Leaderboard top 5 (all-time) ---------------------------------------
    const top5 = await supabaseRpc('leaderboard', { p_window: 'all', p_limit: 5 });
    if (Array.isArray(top5) && top5.length && top5Changed(state.last_top5, top5)) {
        posts.push({
            text: `🏆 Leaderboard update! Top 5 all-time PV Check contributors:\n\n${formatTop5(top5)}\n\nJoin in: https://deeppvmapper.fr/game/`,
            stateUpdate: { last_top5: top5 },
        });
    }

    // 1b. Leaderboard top 5 (rolling 7 days, Friday only) -------------------
    // leaderboard(p_window='week') is a ROLLING 7-day window (not a
    // calendar week — see verifications_setup.sql): querying it right now
    // always gives "the last 7 days as of this moment". Gating this check
    // to WEEKLY_LEADERBOARD_DAY (see top of file) is what turns that into
    // an actual Friday recap instead of a leaderboard post that could land
    // on any day the workflow happens to run. On every other day this
    // section does nothing at all — no RPC call, no comparison.
    if (isWeeklyLeaderboardDay()) {
        const weekTop5 = await supabaseRpc('leaderboard', { p_window: 'week', p_limit: 5 });
        if (Array.isArray(weekTop5) && weekTop5.length && top5Changed(state.last_week_top5, weekTop5)) {
            posts.push({
                text: `🔥 This week's top 5 PV Check contributors:\n\n${formatTop5(weekTop5)}\n\nJoin in: https://deeppvmapper.fr/game/`,
                stateUpdate: { last_week_top5: weekTop5 },
            });
        }
    }

    // 2. Installations validated (PV Check, every 50) -----------------------
    const completion = await supabaseRpc('season_completion', { p_campaign_id: CAMPAIGN_ID });
    const installationsDone = completion?.installations_done ?? null;
    if (installationsDone != null) {
        const newMilestone = Math.floor(installationsDone / INSTALLATIONS_STEP) * INSTALLATIONS_STEP;
        if (newMilestone > state.last_installations_posted) {
            posts.push({
                text: `✅ ${newMilestone.toLocaleString('en-US')} rooftop PV installations fully validated by the PV Check community! Thank you 🙏\n\nHelp us check more: https://deeppvmapper.fr/game/`,
                stateUpdate: { last_installations_posted: newMilestone },
            });
        }
    }

    // 3. Map annotations (every 1000) ---------------------------------------
    const annotationStats = await supabaseRpc('annotation_stats');
    const annotationsCount = annotationStats?.count ?? null;
    if (annotationsCount != null) {
        const newMilestone = Math.floor(annotationsCount / ANNOTATIONS_STEP) * ANNOTATIONS_STEP;
        if (newMilestone > state.last_annotations_posted) {
            posts.push({
                text: `📍 ${newMilestone.toLocaleString('en-US')} annotations submitted on the DeepPVMapper map! Every correction makes the registry more reliable.\n\nJoin in: https://deeppvmapper.fr/game/`,
                stateUpdate: { last_annotations_posted: newMilestone },
            });
        }
    }

    // 4. Total PV Check contributions (verifications, all campaigns, every
    //    1000) ------------------------------------------------------------
    // A different counter from #3 above: leaderboard_total('all') is a
    // straight count(*) from public.verifications (PV Check votes), not
    // public.annotations (the map annotation tool) — the two features are
    // unrelated and can sit at very different totals. This is the trigger
    // for "N total PV Check contributions", which nothing else here
    // covers: #1 only fires on a top-5 ranking change (not on totals
    // moving), and #2 only tracks fully-validated installations, not raw
    // vote count.
    const totalVerifications = await supabaseRpc('leaderboard_total', { p_window: 'all' });
    if (typeof totalVerifications === 'number') {
        const newMilestone = Math.floor(totalVerifications / VERIFICATIONS_STEP) * VERIFICATIONS_STEP;
        if (newMilestone > state.last_verifications_posted) {
            posts.push({
                text: `🎉 ${newMilestone.toLocaleString('en-US')} PV Check contributions submitted by the community! Thank you 🙏\n\nJoin in: https://deeppvmapper.fr/game/`,
                stateUpdate: { last_verifications_posted: newMilestone },
            });
        }
    }

    if (!posts.length) {
        console.log('Nothing new to post.');
        return;
    }

    const session = DRY_RUN ? null : await bskyLogin();

    for (const post of posts) {
        await bskyPost(session, post.text);
        await patchBotState(post.stateUpdate);
        // Small gap between posts in the same run so multiple simultaneous
        // milestones don't land as one indistinguishable burst.
        if (posts.indexOf(post) < posts.length - 1) await new Promise(r => setTimeout(r, 3000));
    }
}

main().catch(err => {
    console.error(err);
    process.exit(1);
});
