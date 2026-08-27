-- ─── One-off: finish the batch_no reshuffle after the SQL Editor version
-- timed out mid-way ───────────────────────────────────────────────────────
-- Run this from a console (psql -f), not the Dashboard's SQL Editor — same
-- statement as the one embedded in scripts/verifications_setup.sql, pulled
-- out standalone so it can be rerun on its own without going through the
-- whole file again. Safe to run as many times as needed: only rows still
-- missing a batch_no (campaign_id = 'season-1') get one.

drop table if exists batch_assignment;
create temporary table batch_assignment as
select id, ceil(row_number() over (partition by dpt order by random())::numeric / 5)::int as batch_no
from public.campaign_pool
where campaign_id = 'season-1'
  and batch_no is null
  and dpt is not null;

create index on batch_assignment (id);

update public.campaign_pool cp set batch_no = ba.batch_no from batch_assignment ba where cp.id = ba.id and cp.id % 20 = 0;
update public.campaign_pool cp set batch_no = ba.batch_no from batch_assignment ba where cp.id = ba.id and cp.id % 20 = 1;
update public.campaign_pool cp set batch_no = ba.batch_no from batch_assignment ba where cp.id = ba.id and cp.id % 20 = 2;
update public.campaign_pool cp set batch_no = ba.batch_no from batch_assignment ba where cp.id = ba.id and cp.id % 20 = 3;
update public.campaign_pool cp set batch_no = ba.batch_no from batch_assignment ba where cp.id = ba.id and cp.id % 20 = 4;
update public.campaign_pool cp set batch_no = ba.batch_no from batch_assignment ba where cp.id = ba.id and cp.id % 20 = 5;
update public.campaign_pool cp set batch_no = ba.batch_no from batch_assignment ba where cp.id = ba.id and cp.id % 20 = 6;
update public.campaign_pool cp set batch_no = ba.batch_no from batch_assignment ba where cp.id = ba.id and cp.id % 20 = 7;
update public.campaign_pool cp set batch_no = ba.batch_no from batch_assignment ba where cp.id = ba.id and cp.id % 20 = 8;
update public.campaign_pool cp set batch_no = ba.batch_no from batch_assignment ba where cp.id = ba.id and cp.id % 20 = 9;
update public.campaign_pool cp set batch_no = ba.batch_no from batch_assignment ba where cp.id = ba.id and cp.id % 20 = 10;
update public.campaign_pool cp set batch_no = ba.batch_no from batch_assignment ba where cp.id = ba.id and cp.id % 20 = 11;
update public.campaign_pool cp set batch_no = ba.batch_no from batch_assignment ba where cp.id = ba.id and cp.id % 20 = 12;
update public.campaign_pool cp set batch_no = ba.batch_no from batch_assignment ba where cp.id = ba.id and cp.id % 20 = 13;
update public.campaign_pool cp set batch_no = ba.batch_no from batch_assignment ba where cp.id = ba.id and cp.id % 20 = 14;
update public.campaign_pool cp set batch_no = ba.batch_no from batch_assignment ba where cp.id = ba.id and cp.id % 20 = 15;
update public.campaign_pool cp set batch_no = ba.batch_no from batch_assignment ba where cp.id = ba.id and cp.id % 20 = 16;
update public.campaign_pool cp set batch_no = ba.batch_no from batch_assignment ba where cp.id = ba.id and cp.id % 20 = 17;
update public.campaign_pool cp set batch_no = ba.batch_no from batch_assignment ba where cp.id = ba.id and cp.id % 20 = 18;
update public.campaign_pool cp set batch_no = ba.batch_no from batch_assignment ba where cp.id = ba.id and cp.id % 20 = 19;

-- Sanity check, printed at the end of the run.
select count(*) filter (where batch_no is null) as sans_batch, count(*) as total
from public.campaign_pool where campaign_id = 'season-1';
