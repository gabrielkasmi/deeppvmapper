# Bluesky progress bot

Posts to Bluesky automatically when:

1. The **all-time leaderboard top 5** changes (new entrant or reordering).
2. **Installations validated** (PV Check, `season_completion().installations_done`) crosses a new multiple of **50**.
3. **Map annotations** (`annotation_stats().count`) crosses a new multiple of **1,000**.

Runs once a day via [`.github/workflows/bluesky-bot.yml`](../../.github/workflows/bluesky-bot.yml) (20:00 UTC — see the comment in that file about DST). Nothing is posted if none of the three conditions changed since the last run: "already announced" state lives in Supabase (`public.bluesky_bot_state`), not in this repo, so re-running the workflow (or triggering it manually) is always safe.

You (Gabriel) can still post manually any time, from your own Bluesky account or the bot's — the bot doesn't need to be the only poster, it just adds these three specific, mechanical updates on top of whatever you post yourself.

## One-time setup

**1. Database.** Run [`scripts/bluesky_bot_setup.sql`](../bluesky_bot_setup.sql) once in the Supabase SQL Editor. It's rerunnable (same convention as the other `*_setup.sql` files here).

**2. Bluesky account.** Create (or pick) the Bluesky account the bot should post as, e.g. `deeppvmapper.bsky.social`. Then, in that account's Settings → **App Passwords**, generate one — **never use the account's real login password here.** Name it something like `github-actions-bot` so you can revoke it individually later without affecting your own login.

**3. GitHub Secrets.** In the repo's Settings → Secrets and variables → Actions, add:

| Secret | Value |
|---|---|
| `SUPABASE_URL` | `https://zelhliylrlktnasircwp.supabase.co` (same one already public in the site's client-side JS — not sensitive, but keeping it as a secret means one less thing to touch if it ever changes) |
| `SUPABASE_SERVICE_ROLE_KEY` | Project Settings → API → `service_role` key. **This bypasses every RLS policy on the project — treat it like a root password, never commit it, never log it.** |
| `BLUESKY_IDENTIFIER` | The bot's handle, e.g. `deeppvmapper.bsky.social` |
| `BLUESKY_APP_PASSWORD` | The App Password from step 2 |

## Testing locally before trusting it

```bash
export SUPABASE_URL=...
export SUPABASE_SERVICE_ROLE_KEY=...
export BLUESKY_IDENTIFIER=...
export BLUESKY_APP_PASSWORD=...

node scripts/bluesky-bot/post-updates.mjs --dry-run
```

`--dry-run` fetches real data and logs exactly what it *would* post, but never calls the Bluesky API and never writes to `bluesky_bot_state` — safe to run as many times as you like while you're checking the wording or thresholds.

You can also trigger the real workflow manually (with its own dry-run checkbox) from the repo's **Actions** tab → "Bluesky progress bot" → **Run workflow**, without waiting for the daily schedule — handy right after a code change.

## Adjusting thresholds or wording

Everything (the 50/1,000 step sizes, the message text, the leaderboard window) is in [`post-updates.mjs`](./post-updates.mjs) — `INSTALLATIONS_STEP`, `ANNOTATIONS_STEP`, and the `posts.push({ text: ... })` calls in `main()`. No other file needs to change.

## If a post looks wrong after the fact

`bluesky_bot_state` only ever moves forward (a milestone/leaderboard snapshot, once posted, is never re-posted for the same value), so if a run posted something wrong, fix the script, then manually update the relevant column in `bluesky_bot_state` in the SQL Editor if you need to force a specific value to be the new "last announced" baseline — e.g.:

```sql
update public.bluesky_bot_state set last_installations_posted = 500 where id = 1;
```
