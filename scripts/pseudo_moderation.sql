-- PV Check — pseudo moderation for public.profiles. Replayable: run the whole file in the SQL editor.
--
-- 1. A filter on new and renamed pseudos: profiles_pseudo_ok() refuses a short list of insults and hate terms (FR + EN),
--    after normalising the pseudo (lower case, accents and separators removed, digit swaps 0→o 1→i 3→e 4→a 5→s 7→t,
--    so « S.a.l.0.p.e » is caught). Long words are matched anywhere; short ambiguous ones (con, pd, pute, fdp…) only
--    as a whole word. The constraint is added NOT VALID: existing rows are not checked (and cannot block the script);
--    list them with query 2 below.
--    The page (game/js/auth.js) turns the refusal (Postgres check_violation, 23514) into « Ce nom n'est pas accepté ».
-- 2. Queries to find and fix offending pseudos by hand.
--
-- Same word list as PV Explorer (random-maps/explorer/supabase/pvx_setup.sql, function pvx_pseudo_ok): the two games
-- keep separate functions so that neither script depends on the other. Edit both lists together.

create or replace function public.profiles_pseudo_ok(p text) returns boolean
language sql immutable as $$
    with n as (
        select translate(lower(unaccent_text), '013457', 'oieast') as w,
               regexp_replace(translate(lower(unaccent_text), '013457', 'oieast'), '[^a-z]', '', 'g') as glued
        from (select translate(p, 'ÀÂÄÁÃÇÉÈÊËÍÎÏÌÑÓÔÖÒÕÚÛÜÙÝàâäáãçéèêëíîïìñóôöòõúûüùýÿŒœÆæ',
                                  'AAAAACEEEEIIIINOOOOOUUUUYaaaaaceeeeiiiinooooouuuuyyOoAa') as unaccent_text) u
    )
    select not (
        n.glued ~ '(connard|connasse|salope|salaud|encule|enculer|batard|bougnoul|negro|negre|youpin|tapette|tarlouze|nazi|hitler|fuhrer|nigger|nigga|faggot|cunt|whore|bitch|asshole|motherfuck|fuck|shit|merdeux|pouffiasse|niquetamere|suceur|couille|pornhub|porno|kkk|siegheil|heilhitler|jihad|terroris)'
        or n.w ~ '(^|[^a-z])(con|cons|conne|connes|pd|pede|pute|putes|fdp|tg|ntm|nique|bite|chatte|viol|pedale|retard|ass|sex|cul|merde)([^a-z]|$)'
    ) from n;
$$;

alter table public.profiles drop constraint if exists profiles_pseudo_ok;
alter table public.profiles add constraint profiles_pseudo_ok check (public.profiles_pseudo_ok(pseudo)) not valid;

-- ─── moderation (SQL editor, as owner) ─────────────────────────────────────
-- existing pseudos the filter would refuse:
-- select id, pseudo, created_at from public.profiles where not public.profiles_pseudo_ok(pseudo) order by created_at;
--
-- rename one (the player keeps their votes and their place in the leaderboard):
-- update public.profiles set pseudo = 'Joueur ' || left(id::text, 4) where id = '…';
--
-- once nothing is listed any more, check every row from now on:
-- alter table public.profiles validate constraint profiles_pseudo_ok;
