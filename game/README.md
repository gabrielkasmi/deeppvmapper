# PV Check — vérification communautaire par swipe

Jeu/outil de crowdsourcing pour vérifier les détections DeepPVMapper à faible
confiance : l'utilisateur swipe une image aérienne recadrée sur une
installation et confirme/infirme que c'est bien un système PV. Positionnement :
"Check. Validate. Improve the map." Sert à transformer les ~installations
mono-source (jamais recoupées avec OSM/FRPV/correction manuelle) en
détections vérifiées par 10 votes indépendants, sans passer par la carte
principale.

Statut : v1 codée et en cours de test en conditions réelles avec Gabriel
(rebrand "PV Check" fait, écran d'accueil à une seule étape, header/footer
de la landing raccords avec le reste du site, marqueur croix allégé, thème
sombre avec la banner en fond sur les deux écrans, compteurs
confirm/reject/unsure côté session + côté menu, menu réorganisé en deux
onglets Leaderboard/Progress — les deux consultables avant même de choisir
un pseudo, depuis la landing — onglet Progress avec une carte de France par
département (colorée selon la part des votes de la communauté sur
l'ensemble de la saison, pas le % de complétion — voir
season_progress_by_department() dans scripts/verifications_setup.sql).
Scheduler revu en "batches" : le pool de season-1 (~655k installations) est
maintenant traversé par lots de ~65k (ntile(10) sur campaign_pool.batch_no),
chaque lot poussé jusqu'à ses 10 votes/installation avant que le suivant ne
s'ouvre (get_verification_batch() ne sert que le lot le moins avancé). Les
% affichés (national + par département) sont scopés au lot actif, donc le
dénominateur réel est ~65k et pas ~655k — ça bouge visiblement au lieu de
rester proche de 0% pendant des semaines, sans rien changer à la vraie
cible de redondance (toujours 10 votes indépendants par installation). Le
SQL doit être ré-exécuté dans Supabase après chaque changement de
schéma/RPC — le script est maintenant rejouable sans erreur ("relation
already exists" etc. ; les fonctions à `returns table(...)` se font
`drop function if exists` avant d'être recréées, sinon Postgres refuse un
changement de colonnes) — voir "Comment tester" en bas de ce document.
L'appel WMS IGN (game/js/image.js) n'a toujours pas pu être validé depuis
une session sandboxée ; à confirmer dans le navigateur si ce n'est pas déjà
fait.

## Pourquoi un module à part

Interaction fondamentalement différente de `content/map.html` (plein écran,
mobile-first, swipe/scroll, PWA installable) — pas de Leaflet, pas de
leaflet-geoman, pas de Chart.js. Réutilise le même projet Supabase et le
même pattern RLS insert-only que `annotations`/`issue_reports`, mais avec
un nouveau jeu de tables dédié (voir Données).

## Architecture

```
                       ┌────────────────────────────────────────┐
   GitHub Pages        │  Supabase (même projet que la carte)    │
   deeppvmapper.fr/game│  • campaigns, campaign_pool — file de   │
   (site statique,     │    travail avec claim + bail            │
   dossier game/,   ───┤  • verifications — insert-only, RLS,    │
   même repo/CNAME     │    modération manuelle (annotations-like)│
   que le reste)       │  • profiles — pseudo, uuid = auth.uid() │
                       │  • auth.users — anon / email OTP / (Google│
                       │    plus tard), linkIdentity pour claim  │
                       └────────────────────────────────────────┘
                                        │
                       Géoplateforme IGN — WMS GetMap
                       (crop 400×400px, GSD variable, PAS de WMTS)
```

Sous-chemin, pas sous-domaine : GitHub Pages n'autorise qu'un seul domaine
personnalisé par dépôt (un seul `CNAME`) — un sous-domaine impliquerait un
second dépôt/déploiement. Aucun besoin PWA (scope du service worker,
installabilité) ne justifie un sous-domaine ici. Seul coût accepté si une
migration future vers un hébergement différent s'avère nécessaire : les
icônes déjà installées sur les écrans d'accueil pointeront vers une URL
morte — acceptable si ça arrive après une vraie traction communautaire (cf.
philosophie générale ci-dessous).

## Sélection de la cible : le pool de la saison 1

Installations dont `detections.sources` est mono-valeur et vaut `'0'`
(DPVM seul) ou `'1'` (FRPV seul) — donc jamais recoupées avec OSM/correction
manuelle/recall, qui sont déjà considérées fiables. Taille réelle du pool à
confirmer par requête (`count(*) from detections where sources !~ ','`
et valeur dans `('0','1')`) avant de dimensionner quoi que ce soit — ne pas
supposer que c'est le chiffre 600k évoqué au départ (celui-ci compte tout,
pas seulement le mono-source).

Cible : 10 vérifications indépendantes par installation.

## Format des images

400×400px, 20cm/px de GSD (soit ~80m×80m de couverture terrain), polygone
centré. Si le polygone ne rentre pas dans ce footprint, le GSD est élargi
(toujours 400×400px en sortie, mais un terrain plus grand) — calculé une
fois, au moment où l'item est sélectionné pour la file, pas recalculé côté
client à chaque affichage.

Source : Géoplateforme IGN, service **WMS** (`GetMap`, bbox + taille pixel
arbitraires) sur `ORTHOIMAGERY.ORTHOPHOTOS` — pas le WMTS déjà utilisé pour
la couche satellite de la carte (celui-ci ne sert que des tuiles à des
zooms fixes, ce qui obligerait à stitcher/cropper côté client). **Non
testé** — à valider en premier (voir Plan). Le volume attendu (lots de
10-15 images, jusqu'à ~100 joueurs simultanés dans le pire des cas) est
jugé raisonnable pour ce service sans validation de charge dédiée.

Chargement : par paquets de 10-15 en arrière-plan pendant que l'utilisateur
swipe le paquet précédent (prefetch). Fait en JS de page normal (Cache API
appelée directement, ou simples blobs en mémoire) — pas via interception
de fetch dans le service worker, plus simple à écrire et déboguer.

## Interaction

Rendu en pile de cartes façon flux continu ("swipe"), mais la décision reste
un geste explicite par carte — un scroll pur ne porte pas de jugement, ce
qui casserait le signal de vérification. Le marqueur au centre de la carte
est une simple croix fine (pas le contour de la détection — voir plus bas
"pourquoi pas de contour"), volontairement discrète pour ne pas cacher
l'installation elle-même.

Convention (identique mobile/desktop) :
- **→ / swipe droite** — confirme (c'est un PV)
- **← / swipe gauche** — infirme
- **↓ / bouton dédié** — passe sans juger (image illisible, etc. — distinct
  du tag "ambigu", qui enregistre un jugement incertain plutôt qu'une
  absence de jugement)
- **⌫ / backspace** — annule le dernier geste
- Desktop : clic sur zones ✕/✓ ou clavier ; mobile : swipe tactile

Chaque swipe déclenche un court délai optimiste (quelques centaines de ms,
le temps de l'animation) avant l'insert réel — façon "annuler l'envoi" de
Gmail. Le bouton "retour" pendant cette fenêtre annule un envoi qui n'a
jamais été committé ; passé ce délai, c'est écrit et immuable, comme le
reste du schéma (insert-only, jamais de update/delete pour l'utilisateur).
Pas de batching de plusieurs votes côté client : ça avait été écarté après
revue — un lot tamponné trop longtemps risque de se perdre entièrement si
l'utilisateur ferme l'app avant l'envoi, ce que l'insert quasi-immédiat
évite.

## Identité et auth

Trois entrées possibles à l'écran de démarrage, pas à poids égal :

- **Anonyme + pseudo** (gros bouton, le cas dominant) — `signInAnonymously()`
  Supabase, aucune donnée personnelle, juste un pseudo unique choisi par
  l'utilisateur (insert direct sur `profiles`, contrainte unique attrapée
  plutôt qu'un check préalable séparé).
- **Email/pseudo** (lien discret en dessous, pour qui veut la persistance
  cross-device tout de suite) — OTP à 6 chiffres tapé dans l'app plutôt
  qu'un lien magique cliqué (un lien magique risque de rouvrir un
  navigateur externe plutôt que la PWA installée et de casser le fil).
- **Google** (bouton visible, grisé "bientôt disponible") — pas de
  verification Google requise pour un login basé sur les scopes basiques
  (email/profil/openid) : ni plafond de 100 comptes, ni revue manuelle,
  publication immédiate possible. Reporté par choix de séquencement, pas
  par contrainte technique.

**Claim** : `linkIdentity()` convertit un compte anonyme en compte
permanent (email, puis Google plus tard) sans changer l'UUID — tout
l'historique (`verifications`, `profiles`) reste rattaché automatiquement,
aucune migration de données. Bouton "claim my account" dans le menu, câblé
sur email dès le lancement.

Limite acceptée : quelqu'un qui reste anonyme sur plusieurs appareils se
retrouve avec autant de comptes cloisonnés, sans fusion automatique
possible après coup (`linkIdentity` ne fusionne pas deux comptes déjà
permanents). Acceptable vu que le cas dominant est un seul joueur sur un
seul téléphone.

## Distribution du travail — revu après audit, simplifié

Le premier design (file avec claim + bail Postgres, `SKIP LOCKED`, passes
discrètes) a été écarté après relecture avant l'implémentation : c'est le
pattern standard des files de tâches, où une unité de travail doit être
traitée exactement une fois par un seul worker — mais notre tâche est
l'inverse, elle veut explicitement 10 votes indépendants par item. Le
mécanisme protégeait contre un scénario (deux utilisateurs sur le même
item en même temps) qui n'est pas un problème ici, juste deux des dix
votes attendus. Retour au standard du domaine (Zooniverse, LabelStudio,
MapRoulette et consorts pour la classification/le labeling communautaire) :
échantillonnage aléatoire pondéré vers les items les moins couverts, pas
de claim, pas de bail, pas de passes.

**Schéma** : une ligne par installation ciblée dans `campaign_pool`
(`detection_id`, `campaign_id`, `votes_received` int, défaut 0) — pas de
`belongs_to_user`/`claimed_at`, pas de duplication physique. Le compteur
est maintenu par un trigger `AFTER INSERT ON verifications`, pas recalculé
à la volée par agrégation.

**Sélection d'un lot** : `WHERE campaign_id = $c AND votes_received < 10
AND detection_id NOT IN (mes détections déjà votées) ORDER BY
votes_received ASC, random() LIMIT $n`. La couverture en largeur d'abord
ressort naturellement du tri (les moins votés sortent en premier) — cible
statistique, pas garantie dure, ce qui est suffisant puisque la redondance
est l'objectif et non un risque à éviter.

**Filet de sécurité** : contrainte unique `(user_id, detection_id,
campaign_id)` sur `verifications` — empêche qu'un même utilisateur vote
deux fois le même item, seul cas qui compte vraiment.

**Fin de saison** : quand `campaign_pool` est entièrement à
`votes_received = 10`, la requête de sélection revient vide — l'écran
affiche "bravo, tu as tout couvert, reviens plus tard." Option
"préviens-moi" au même endroit, via le canal email déjà prévu pour le
claim, pas via de vraies notifications push (infra Web Push jugée
disproportionnée pour la v1).

## Menu

- Mes stats (compteur perso, via une RPC `security definer` sur
  `auth.uid()` — pas de policy `select` ouverte sur `verifications`)
- Leaderboard (semaine glissante / mois glissant / all-time — RPC
  paramétrée, filtrée sur `verifications.created_at`)
- % de complétion de la saison en cours
- Bouton "claim my account"

## Modération

Manuelle, comme `annotations`/`issue_reports` aujourd'hui — mais sur une
**vue agrégée par installation** (`verifications_summary` : compte
confirm/reject/ambigu par `detection_id`), pas ligne par ligne comme pour
les annotations géométriques. À construire dès la phase 0, pas quand le
volume deviendra pénible. Pas de promotion automatique par consensus —
choix assumé, cohérent avec "je préfère gérer ça manuellement."

## Anti-abus

Aucun garde-fou technique contre le swipe aléatoire pour l'instant — on
présume la bonne volonté des joueurs, et la modération manuelle avant tout
merge dans la base officielle est le vrai filet de sécurité (rien
n'entache le jeu de données tant que ça n'a pas été relu). À réévaluer
seulement si un usage public/compétitif fait apparaître un vrai problème
de qualité de signal (ex. via l'écart-type des votes sur les items
attendus non-ambigus).

## Saison 2+

Pas à résoudre maintenant, juste à ne pas fermer la porte : le schéma ne
doit pas coder en dur la sémantique "confirmer/infirmer un PV" —
`campaign_id` + un vocabulaire de `decision` non figé permettent de
réutiliser toute l'infra (auth, profils, leaderboard, écran de swipe) pour
une tâche différente (ex. validation de forme de polygone pour une tranche
de puissance donnée) sans réécriture.

## Plan de construction

0. **Fondations Supabase** — tables `verifications`, `profiles`,
   `campaigns`, `campaign_pool` (+ trigger de comptage) ; RPCs (lot de
   vérification, mes stats, leaderboard, % complétion) ; vue
   `verifications_summary` pour la modération. **Fait** — voir
   `scripts/verifications_setup.sql`.
1. **Spike image WMS** — valider le `GetMap` IGN (crop exact, GSD
   variable, pas de souci à ce volume) avant d'investir dans l'UI. Seule
   inconnue technique réelle du projet.
2. **Auth minimal** — anon + pseudo, email/OTP en option discrète, bouton
   Google grisé.
3. **Écran de swipe** — shell, pile de cartes, gestes (tactile/souris/
   clavier), overlay du contour, formulaire ambigu/commentaire, batching
   local des votes.
4. **Menu** — stats/leaderboard/% complétion, bouton claim.
5. **PWA** — manifest + service worker minimal (juste de quoi passer les
   critères d'installabilité, pas de logique de cache d'images dedans).
6. **Modération** — sur la vue agrégée, même flux Dashboard que
   l'existant.
7. **Lancement soft** — friends & family d'abord (comme pour les
   annotations), avant le vrai push public/conférence.

## Comment tester

1. **Exécuter `scripts/verifications_setup.sql`** dans le SQL Editor
   Supabase (même projet que le reste — `supabase_setup.sql` et
   `supabase_detections_setup.sql` doivent déjà avoir tourné, `detections`
   doit déjà être peuplée). Ça crée les tables/RPCs et peuple
   `campaign_pool` à partir des détections mono-source actuelles. Vérifier
   à la fin combien de lignes sont entrées (`select count(*) from
   campaign_pool`) — si c'est 0, le filtre `sources` n'a rien matché,
   regarder les valeurs réelles de la colonne avant d'aller plus loin.
2. **Activer l'auth anonyme** dans Supabase : Dashboard → Authentication →
   Providers → Anonymous Sign-Ins → activer (désactivé par défaut).
   Activer aussi Email (OTP) si pas déjà fait pour un autre flux.
3. **Servir le site en local**, comme pour la carte :
   `python3 -m http.server` à la racine du repo, puis
   `http://localhost:8000/game/`.
4. **Ouvrir dans un vrai navigateur** (pas de preview intégrée ici) : saisir
   un pseudo (ou tirer un nom aléatoire au dé), le bouton "Play" se dégrise
   → si une carte s'affiche avec une vraie image aérienne et une petite
   croix au centre, le point le plus incertain du plan (l'appel WMS)
   fonctionne. Si l'image est cassée, regarder l'onglet réseau du
   navigateur pour voir ce que `data.geopf.fr` renvoie exactement.
5. **Menu (☰)** — vérifier que "My verifications" (avec le détail
   confirm/reject/unsure), le leaderboard et le % de progression de la
   saison remontent après quelques swipes.

Rien de tout ça n'a pu être exécuté depuis cette session — la sandbox n'a
pas de route réseau vers Supabase ni vers l'IGN, et aucun navigateur n'était
connecté. La suite se fait ensemble : tu ouvres, tu me dis ce qui casse
(erreur console, image blanche, écran qui reste bloqué...), je corrige.

## Ce qui n'est pas encore fait

Écran "ambigu" fonctionnel mais pas peaufiné visuellement ; bouton
"préviens-moi" en fin de saison encore un stub (toast "bientôt
disponible") plutôt que branché sur un vrai envoi ; claim par email
fonctionnel mais via de simples `prompt()` navigateur, pas un vrai
formulaire ; bouton Google visuellement présent et désactivé, non câblé.
Rien de tout ça ne bloque le test de la boucle principale (auth → swipe →
vote → stats).

## Philosophie générale

Prolonger un pattern backend qui marche déjà (RLS insert-only, RPC
`security definer`, modération offline) plutôt qu'introduire une nouvelle
infra. Dimensionné pour l'échelle réelle attendue (dizaines à ~100 joueurs
simultanés), pas pour une hypothétique mise à l'échelle — pas de scheduler
précalculé superflu, pas de promotion automatique par consensus, pas de
compte obligatoire. Si une vraie communauté émerge, la migration vers
quelque chose de plus permanent (sous-domaine, hébergement dédié, vraie
notification push, vérification Google) se justifiera et son coût
(quelques comptes anonymes orphelins, quelques icônes d'écran d'accueil
mortes) sera alors acceptable — pas avant.
