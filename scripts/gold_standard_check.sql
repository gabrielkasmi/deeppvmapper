-- ─── gold_standard_check — read back the quality-check signal ─────────────
-- Companion to the gold_standard_items table / gold CTE added in
-- verifications_setup.sql. No detection_id is hard-coded here on purpose:
-- this file lives on the public gh-pages branch, and the whole point of a
-- gold-standard item is that the "correct" answer isn't discoverable by
-- reading the site's source. Rows in gold_standard_items are inserted by
-- hand directly in the Supabase SQL editor and are never committed.
--
-- Caveat (see conversation notes): if a gold item's expected_decision was
-- derived purely from PV Check's own unanimous consensus, checking the
-- handful of users who voted on it *before* it became a gold item is
-- circular — they defined the "correct" answer, so they'll trivially
-- match it. This query is informative for:
--   (a) any user who votes on a gold item going forward (new or existing
--       accounts — get_verification_batch() serves it to whoever hasn't
--       voted on it yet, not just brand-new signups), and
--   (b) retroactively, for gold items whose expected_decision came from
--       an independent source (manual inspection, FRPV cross-check)
--       rather than from PV Check's own vote history.

select
    v.user_id,
    v.campaign_id,
    v.detection_id,
    v.decision,
    g.expected_decision,
    (v.decision = g.expected_decision) as passed,
    v.created_at
from public.verifications v
join public.gold_standard_items g
  on g.campaign_id = v.campaign_id
 and g.detection_id = v.detection_id
order by v.created_at desc;

-- Per-user rollup (one row per account that has hit at least one gold item):
-- select user_id,
--        count(*)                                    as gold_items_seen,
--        count(*) filter (where decision = expected_decision) as gold_items_passed
-- from ( <query above> ) t
-- group by user_id;
