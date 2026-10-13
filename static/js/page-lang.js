/*
 * page-lang.js - French / English for the generated pages (content/cities, content/data,
 * content/regions).
 *
 * scripts/build_department_pages.py writes these pages in English; scripts/frenchify_pages.js then
 * rewrites each one IN FRENCH, in place, so French is what search engines index (same URLs, no twin
 * files). The English originals stay in the page as JSON (<script id="dpm-en">), and this script,
 * loaded at the end of every page, does the rest in the browser:
 *   - language = ?lang=fr|en, else the last choice remembered in localStorage ("dpm_lang", also set
 *     by every Jekyll page), else the language the page was shipped in (French);
 *   - on a French page, EN puts the original English back (restoreEnglish): no English URL exists;
 *   - on a page still in English (raw generator output), FR translates it (translateDocument);
 *   - the EN | FR switcher is a toggle that reloads the page in the chosen language.
 * The translation is a dictionary of fixed strings plus one pattern per sentence template of the
 * generator; a string it does not recognise stays in English and never breaks the page.
 *
 * KEEP IN SYNC with the sentence templates in scripts/build_department_pages.py
 * (build_city_intro / build_dept_intro / build_region_intro, labels, charts, <meta> descriptions).
 * scripts/check_page_lang.js reports what is left untranslated and checks the EN round trip.
 */
(function (root) {
  'use strict';

  var KEY = 'dpm_lang';
  var NB = ' ';

  /* ------------------------------------------------------------------ places */
  function setOf(s) { var o = {}; s.split('|').forEach(function (k) { o[k] = 1; }); return o; }
  // Départements by the article French puts before them ("dans le Nord", "dans l'Ain",
  // "dans les Landes", "dans la Loire"); the rest take "en" ("en Dordogne").
  var D_M = setOf("Calvados|Cantal|Cher|Doubs|Finistère|Gard|Gers|Jura|Loir-et-Cher|Loiret|Lot|Lot-et-Garonne|Maine-et-Loire|Morbihan|Nord|Pas-de-Calais|Puy-de-Dôme|Bas-Rhin|Haut-Rhin|Rhône|Tarn|Tarn-et-Garonne|Var|Vaucluse|Territoire de Belfort|Val-de-Marne|Val-d'Oise");
  var D_F = setOf("Creuse|Drôme|Loire|Manche|Marne|Meuse|Nièvre|Sarthe|Somme|Vienne");
  var D_P = setOf("Alpes-de-Haute-Provence|Hautes-Alpes|Alpes-Maritimes|Ardennes|Bouches-du-Rhône|Côtes-d'Armor|Landes|Pyrénées-Atlantiques|Hautes-Pyrénées|Pyrénées-Orientales|Yvelines|Deux-Sèvres|Vosges|Hauts-de-Seine");
  var D_V = setOf("Isère|Ain|Aisne|Allier|Aube|Aude|Aveyron|Eure|Hérault|Indre|Oise|Orne|Yonne|Essonne");
  var DEPTS = setOf("Ardèche|Ariège|Charente|Charente-Maritime|Corrèze|Corse-du-Sud|Haute-Corse|Côte-d'Or|Dordogne|Eure-et-Loir|Haute-Garonne|Gironde|Ille-et-Vilaine|Indre-et-Loire|Haute-Loire|Loire-Atlantique|Lozère|Haute-Marne|Mayenne|Meurthe-et-Moselle|Moselle|Haute-Saône|Saône-et-Loire|Savoie|Haute-Savoie|Paris|Seine-Maritime|Seine-et-Marne|Seine-Saint-Denis|Vendée|Haute-Vienne");
  [D_M, D_F, D_P, D_V].forEach(function (g) { for (var k in g) DEPTS[k] = 1; });

  function vowel(s) { return /^[AEIOUYÉÈÊÎÔÂHaeiouyéèêîôâh]/.test(s); }
  function dIn(n) {
    if (n === 'Paris') return 'à Paris';
    if (D_M[n]) return 'dans le ' + n;
    if (D_F[n]) return 'dans la ' + n;
    if (D_P[n]) return 'dans les ' + n;
    if (D_V[n]) return "dans l'" + n;
    return 'en ' + n;
  }
  function dOf(n) {
    if (D_M[n]) return 'du ' + n;
    if (D_F[n]) return 'de la ' + n;
    if (D_P[n]) return 'des ' + n;
    if (D_V[n]) return "de l'" + n;
    return (vowel(n) ? "d'" : 'de ') + n;
  }
  var R_IN = {
    'Auvergne-Rhône-Alpes': 'en Auvergne-Rhône-Alpes', 'Bourgogne-Franche-Comté': 'en Bourgogne-Franche-Comté',
    'Bretagne': 'en Bretagne', 'Centre-Val de Loire': 'en Centre-Val de Loire', 'Corse': 'en Corse',
    'Grand Est': 'dans le Grand Est', 'Hauts-de-France': 'dans les Hauts-de-France',
    'Île-de-France': 'en Île-de-France', 'Normandie': 'en Normandie', 'Nouvelle-Aquitaine': 'en Nouvelle-Aquitaine',
    'Occitanie': 'en Occitanie', 'Pays de la Loire': 'dans les Pays de la Loire',
    "Provence-Alpes-Côte d'Azur": "en Provence-Alpes-Côte d'Azur"
  };
  var R_OF = {
    'Auvergne-Rhône-Alpes': "d'Auvergne-Rhône-Alpes", 'Bourgogne-Franche-Comté': 'de Bourgogne-Franche-Comté',
    'Bretagne': 'de Bretagne', 'Centre-Val de Loire': 'du Centre-Val de Loire', 'Corse': 'de Corse',
    'Grand Est': 'du Grand Est', 'Hauts-de-France': 'des Hauts-de-France',
    'Île-de-France': "d'Île-de-France", 'Normandie': 'de Normandie', 'Nouvelle-Aquitaine': 'de Nouvelle-Aquitaine',
    'Occitanie': "d'Occitanie", 'Pays de la Loire': 'des Pays de la Loire',
    "Provence-Alpes-Côte d'Azur": "de Provence-Alpes-Côte d'Azur"
  };
  function rIn(n) { return R_IN[n] || ('en ' + n); }
  function rOf(n) { return R_OF[n] || ('de ' + n); }
  function inCity(n) {            // "à Fontenay", "au Havre", "aux Sables-d'Olonne"
    if (/^Le /.test(n)) return 'au ' + n.slice(3);
    if (/^Les /.test(n)) return 'aux ' + n.slice(4);
    return 'à ' + n;
  }

  var ZONE = {
    'in the Paris region, at the heart of northern France': 'dans la région parisienne, au cœur du nord de la France',
    'in the center of France': 'au centre de la France',
    'in eastern-central France': 'dans le centre-est de la France',
    'in northwestern France, along the Channel coast': 'dans le nord-ouest de la France, le long de la Manche',
    'in the far north of France': "à l'extrême nord de la France",
    'in northeastern France, along the German and Swiss borders': 'dans le nord-est de la France, le long des frontières allemande et suisse',
    'on the Atlantic coast, in western France': "sur la côte atlantique, dans l'ouest de la France",
    'on the northwestern peninsula of France': 'sur la péninsule du nord-ouest de la France',
    'in southwestern France, along the Atlantic coast': 'dans le sud-ouest de la France, le long de la côte atlantique',
    'in southern France, along the Mediterranean and the Pyrenees': 'dans le sud de la France, entre Méditerranée et Pyrénées',
    'in east-central France, along the Alps': 'dans le centre-est de la France, le long des Alpes',
    'in southeastern France, along the Mediterranean coast': 'dans le sud-est de la France, le long de la côte méditerranéenne',
    'the Mediterranean island region of France': "sur l'île de Beauté, en Méditerranée"
  };
  var LANDMARK = {
    'the Promenade des Anglais in Nice': 'la Promenade des Anglais à Nice',
    'the medieval citadel of Carcassonne': 'la cité médiévale de Carcassonne',
    'the Millau Viaduct': 'le viaduc de Millau',
    "the Calanques and Marseille's Vieux-Port": 'les calanques et le Vieux-Port de Marseille',
    'the D-Day landing beaches': 'les plages du Débarquement',
    'the Burgundy vineyards': 'le vignoble bourguignon',
    'Chartres Cathedral': 'la cathédrale de Chartres',
    'the Pointe du Raz': 'la pointe du Raz',
    'the Pont du Gard': 'le pont du Gard',
    'the Bordeaux vineyards and the Dune du Pilat': 'le vignoble bordelais et la dune du Pilat',
    'the Place de la Comédie in Montpellier': 'la place de la Comédie à Montpellier',
    'the walled city of Saint-Malo': 'la cité fortifiée de Saint-Malo',
    'the Château de Chenonceau': 'le château de Chenonceau',
    'the Vercors and Chartreuse massifs': 'les massifs du Vercors et de la Chartreuse',
    'the Château de Chambord': 'le château de Chambord',
    'Mont-Saint-Michel': 'le Mont-Saint-Michel',
    'the Verdun battlefields': 'les champs de bataille de Verdun',
    'the Carnac megaliths': 'les mégalithes de Carnac',
    'the Puy de Dôme volcano': 'le volcan du puy de Dôme',
    'the Basque coast and Béarn Pyrenees': 'la côte basque et les Pyrénées béarnaises',
    'the Cathar castles of the Pyrénées-Orientales': 'les châteaux cathares des Pyrénées-Orientales',
    'Strasbourg Cathedral': 'la cathédrale de Strasbourg',
    'the Basilica of Notre-Dame de Fourvière in Lyon': 'la basilique Notre-Dame de Fourvière à Lyon',
    'Mont Blanc': 'le mont Blanc',
    'the Eiffel Tower and Notre-Dame Cathedral': 'la tour Eiffel et la cathédrale Notre-Dame',
    "Rouen's Gothic cathedral and the Etretat cliffs": "la cathédrale gothique de Rouen et les falaises d'Étretat",
    'Saint-Tropez': 'Saint-Tropez',
    'the Palais des Papes in Avignon': 'le palais des Papes à Avignon',
    'the Futuroscope park': 'le parc du Futuroscope',
    'the Corsican coastline near Ajaccio': "le littoral corse près d'Ajaccio",
    'the Cap Corse coastline': 'le littoral du Cap Corse'
  };

  /* ------------------------------------------------------------------ numbers etc. */
  function N(s) {                     // "4,361" | "5 741" | "45.0"  ->  "4 361" | "5 741" | "45,0"
    return String(s).replace(/,(?=\d{3}(\D|$))/g, NB).replace(/ (?=\d{3}(\D|$))/g, NB).replace(/(\d)\.(\d)/g, '$1,$2');
  }
  function U(u) { return u.replace(/Wp$/, 'Wc'); }
  function ORD(s, g) {                // "17th" -> "17e" ; 1 -> "1er"/"1re"
    var n = parseInt(s, 10);
    if (n === 1) return g === 'f' ? '1re' : '1er';
    return n + 'e';
  }
  function RANK(s, g) {               // "12th out of 13" -> "12e sur 13"
    var m = /^(\d+)(?:st|nd|rd|th)(?: out of (\d+))?$/.exec(s);
    return m ? ORD(m[1], g) + (m[2] ? ' sur ' + m[2] : '') : s;
  }
  function RANG(s) {                  // "12th out of 13" -> "12e rang sur 13"
    var m = /^(\d+)(?:st|nd|rd|th)(?: out of (\d+))?$/.exec(s);
    return m ? ORD(m[1], 'm') + ' rang' + (m[2] ? ' sur ' + m[2] : '') : s;
  }
  function SPAN(y) {
    var m = /^(\d{4})[–-](\d{4})$/.exec(y);
    if (m) return m[1] === m[2] ? "l'année " + m[1] : 'la période ' + m[1] + '–' + m[2];
    return y === 'the observation period' ? "la période d'observation" : y;
  }
  function PL(n, one, many) { return (parseInt(String(n).replace(/\D/g, ''), 10) === 1 ? one : many); }
  function LIST(s) { return s.replace(/ and (?=[^,]*$)/, ' et '); }

  /* ------------------------------------------------------------------ fixed strings */
  var EXACT = {
    // nav / footer (same wording as _data/i18n.yml)
    'DeepPVMapper — Home': 'DeepPVMapper — Accueil',
    'About': 'À propos', 'Data': 'Données', 'Software': 'Logiciels', 'Contribute': 'Contribuer',
    'Use Cases': "Cas d'usage", 'Docs ▾': 'Documentation ▾', 'Publications ▾': 'Publications ▾',
    'Detection architecture & deployment': 'Architecture de détection et déploiement',
    'Multi-source database methodology': 'Méthodologie de la base multi-sources',
    'Papers & Preprints': 'Articles et prépublications',
    'Peer-reviewed papers, preprints & posters': 'Articles revus par les pairs, prépublications et posters',
    'Press Coverage': 'Presse',
    'Popular-science & media coverage': 'Vulgarisation et couverture médiatique',
    'Language': 'Langue',
    'Maintained by Gabriel Kasmi. This work is licensed under': 'Maintenu par Gabriel Kasmi. Ce travail est publié sous licence',
    // page body
    'PV systems detected': 'Systèmes PV détectés',
    'Estimated installed capacity': 'Puissance installée estimée',
    'Source composition': 'Répartition des sources',
    'Installed capacity distribution': 'Répartition de la puissance installée',
    'Installed capacity distribution (log scale)': 'Répartition de la puissance installée (échelle logarithmique)',
    'Panel orientation': 'Orientation des panneaux',
    'Panel orientation (azimuth) distribution': "Répartition de l'orientation des panneaux (azimut)",
    'Source-combination shares': 'Parts des combinaisons de sources',
    'log scale': 'échelle log.',
    'DPVM only': 'DPVM seul', 'FRPV only': 'FRPV seul', 'OSM only': 'OSM seul',
    'click here to see it on the map': 'cliquez ici pour les voir sur la carte',
    'Click here to see it on the map': 'Cliquez ici pour les voir sur la carte',
    'See something odd here?': "Quelque chose d'anormal ?",
    'Report an issue': 'Signaler un problème',
    'What looks wrong? A count that seems off, a system that looks misclassified...':
      "Qu'est-ce qui semble incorrect ? Un nombre qui paraît faux, un système mal classé…",
    '← Back to all départements': '← Retour à tous les départements',
    '← Back to all régions': '← Retour à toutes les régions',
    'Cities': 'Villes', 'Installed capacity by département': 'Puissance installée par département',
    'Installed capacity share by département': 'Part de la puissance installée par département',
    "Don’t see your commune above? It may not have enough confirmed detections yet —":
      "Vous ne voyez pas votre commune ? Elle n’a peut-être pas encore assez de détections confirmées —",
    'let us know which one you had in mind': 'dites-nous laquelle vous aviez en tête',
    // the institutional card is already French in the English pages
    // compass letter in SVG labels (W = West -> O = Ouest)
    '__svg:W': 'O'
  };
  var COMPASS = { N: 'N', NNE: 'NNE', NE: 'NE', ENE: 'ENE', E: 'E', ESE: 'ESE', SE: 'SE', SSE: 'SSE', S: 'S',
                  SSW: 'SSO', SW: 'SO', WSW: 'OSO', W: 'O', WNW: 'ONO', NW: 'NO', NNW: 'NNO' };

  /* ------------------------------------------------------------------ sentence patterns */
  var NUM = '([\\d,.  ]+?)';
  var ORDI = '(\\d+(?:st|nd|rd|th))';
  var RK = '(\\d+(?:st|nd|rd|th)(?: out of \\d+)?)';
  var PHR = '(counted [\\d,. ]+ inhabitants|has an undocumented population)';
  function pop(p) {                  // "counted 4,361 inhabitants" -> "comptait 4 361 habitants"
    var m = /^counted ([\d,. ]+) inhabitants$/.exec(p);
    return m ? 'comptait ' + N(m[1]) + ' habitants' : null;
  }
  function pref(p) { return p ? ', dont la préfecture est ' + p : ''; }

  // Each rule: [RegExp on one sentence (with its final punctuation), function(match, ctx) -> French]
  var RULES = [
    /* ---------------- commune ---------------- */
    [/^(.+?) is a commune of (.+?), in ([^,]+)\.$/, function (m) {
      return m[1] + ' est une commune ' + dOf(m[2]) + ', ' + rIn(m[3]) + '.'; }],
    [/^As of the latest census it (counted [\d,. ]+ inhabitants|has an undocumented population)\.$/, function (m) {
      var p = pop(m[1]); return 'Lors du dernier recensement, elle ' + (p ? p : "n'a pas de population documentée") + '.'; }],
    [new RegExp('^Part of (.+?) \\((.+?)\\), (.+?) ' + PHR + ' as of the latest census\\.$'), function (m) {
      var p = pop(m[4]); return m[3] + ', commune ' + dOf(m[1]) + ' (' + m[2] + '), ' +
        (p ? p + ' lors du dernier recensement.' : "n'a pas de population documentée au dernier recensement."); }],
    [new RegExp('^(.+?), in (.+?) \\((.+?)\\), ' + PHR + ' as of the latest census\\.$'), function (m) {
      var p = pop(m[4]); return m[1] + ', commune ' + dOf(m[2]) + ' (' + m[3] + '), ' +
        (p ? p + ' lors du dernier recensement.' : "n'a pas de population documentée au dernier recensement."); }],
    [new RegExp('^OpenPVMapper has identified ' + NUM + ' rooftop PV installations in (.+?), totalling ' + NUM +
      ' (kWp|MWp) — the ' + ORDI + ' commune in (.+?) by installed capacity, out of ' + NUM + ' communes with detections\\.$'),
      function (m) {
        return 'OpenPVMapper a identifié ' + N(m[1]) + ' installation' + PL(m[1], '', 's') + ' PV en toiture ' + inCity(m[2]) +
          ', pour un total de ' + N(m[3]) + ' ' + U(m[4]) + ' — ' + ORD(m[5], 'f') + ' commune ' + dOf(m[6]) +
          ' en puissance installée, sur ' + N(m[7]) + ' communes avec détections.'; }],
    [new RegExp('^' + NUM + ' rooftop PV installations totalling ' + NUM + ' (kWp|MWp) have been identified in (.+?) by OpenPVMapper, ranking it ' +
      ORDI + ' in (.+?) out of ' + NUM + ' communes with detections\\.$'),
      function (m) {
        return N(m[1]) + ' installation' + PL(m[1], '', 's') + ' PV en toiture, pour un total de ' + N(m[2]) + ' ' + U(m[3]) +
          ', ' + PL(m[1], 'a été identifiée', 'ont été identifiées') + ' ' + inCity(m[4]) +
          ' par OpenPVMapper, ce qui place la commune au ' + ORD(m[5], 'm') + ' rang ' + dIn(m[6]) +
          ', sur ' + N(m[7]) + ' communes avec détections.'; }],
    [new RegExp("^OpenPVMapper's latest release puts (.+?) at " + NUM + ' (kWp|MWp) of estimated rooftop PV capacity over ' + NUM +
      ' installations — ' + ORDI + ' in (.+?) out of ' + NUM + ' communes with detections\\.$'),
      function (m) {
        return "La dernière version d'OpenPVMapper estime à " + N(m[2]) + ' ' + U(m[3]) + ' la capacité PV en toiture ' +
          inCity(m[1]) + ', répartie sur ' + N(m[4]) + ' installation' + PL(m[4], '', 's') + ' — ' + ORD(m[5], 'f') + ' ' +
          dIn(m[6]) + ', sur ' + N(m[7]) + ' communes avec détections.'; }],

    /* ---------------- département ---------------- */
    [/^(.+?) is a département of ([^,]+?)(?:, with (.+) as its préfecture)?\.$/, function (m) {
      return m[1] + ' est un département ' + rOf(m[2]) + pref(m[3]) + '.'; }],
    [/^As of 2021 it (counted [\d,. ]+ inhabitants|has an undocumented population)\.$/, function (m) {
      var p = pop(m[1]); return 'En 2021, il ' + (p ? p : "n'a pas de population documentée") + '.'; }],
    [/^It is known for (.+)\.$/, function (m) { return 'Il est connu pour ' + (LANDMARK[m[1]] || m[1]) + '.'; }],
    [new RegExp('^Part of ([^,]+), ([^,]+?)(?:, with (.+?) as its préfecture)? ' + PHR + ' as of 2021\\.$'), function (m) {
      var p = pop(m[4]);
      return 'Département ' + rOf(m[1]) + ', ' + m[2] + (m[3] ? ' (préfecture : ' + m[3] + ')' : '') + ' ' +
        (p ? p + ' en 2021.' : "n'a pas de population documentée en 2021."); }],
    [new RegExp('^([^,]+), in ([^,]+?)(?:, with (.+?) as its préfecture)?, ' + PHR + ' as of 2021\\.$'), function (m) {
      var p = pop(m[4]);
      return m[1] + ', ' + rIn(m[2]) + (m[3] ? ', a pour préfecture ' + m[3] + ' et ' : ', ') +
        (p ? p + ' en 2021.' : "n'a pas de population documentée en 2021."); }],
    [new RegExp('^The latest OpenPVMapper release has identified ' + NUM + ' MWp of rooftop PV across ' + NUM +
      ' installations in (.+?) — the ' + ORDI + ' département in France and ' + RK + ' within (.+?) by installed capacity\\.$'),
      function (m) {
        return "La dernière version d'OpenPVMapper a identifié " + N(m[1]) + ' MWc de PV en toiture répartis sur ' + N(m[2]) +
          ' installations ' + dIn(m[3]) + ' — ' + ORD(m[4], 'm') + ' département de France et ' + RANK(m[5], 'm') + ' ' + rIn(m[6]) +
          ' en puissance installée.'; }],
    [new RegExp('^' + NUM + ' rooftop PV installations totalling ' + NUM + ' MWp have been identified in (.+?) by OpenPVMapper, ranking it ' +
      ORDI + ' nationally and ' + RK + ' in (.+?)\\.$'),
      function (m) {
        return N(m[1]) + ' installations PV en toiture, pour un total de ' + N(m[2]) + ' MWc, ont été identifiées ' + dIn(m[3]) +
          ' par OpenPVMapper, ce qui place le département au ' + ORD(m[4], 'm') + ' rang national et au ' + RANG(m[5]) + ' ' + rIn(m[6]) + '.'; }],
    [new RegExp("^OpenPVMapper's latest release puts (.+?) at " + NUM + ' MWp of estimated rooftop PV capacity over ' + NUM +
      ' installations — ' + ORDI + ' in France, ' + RK + ' in (.+?)\\.$'),
      function (m) {
        return "La dernière version d'OpenPVMapper estime à " + N(m[2]) + ' MWc la capacité PV en toiture ' + dOf(m[1]) +
          ', répartie sur ' + N(m[3]) + ' installations — ' + ORD(m[4], 'm') + ' en France, ' + RANK(m[5], 'm') + ' ' + rIn(m[6]) + '.'; }],
    [/^(\d+(?:\.\d+)?)% of installations in (.+?) are confirmed by more than two independent sources\.$/, function (m) {
      return N(m[1]) + ' % des installations ' + dOf(m[2]) + ' sont confirmées par plus de deux sources indépendantes.'; }],

    /* ---------------- région ---------------- */
    [/^(.+?) is located (.+)\.$/, function (m) { return m[1] + ' se situe ' + (ZONE[m[2]] || m[2]) + '.'; }],
    [new RegExp('^As of 2021 it counted ' + NUM + ' inhabitants \\(the ' + ORDI + ' most populous French région\\), across ' + NUM +
      ' km² of land \\(the ' + ORDI + ' largest by area\\)(\\.|, with a GDP of €' + NUM + ' million in 2024 \\(' + ORDI + ' nationally\\)\\.)$'),
      function (m) {
        return 'En 2021, la région comptait ' + N(m[1]) + ' habitants (' + ORD(m[2], 'f') + ' région la plus peuplée de France), pour ' +
          N(m[3]) + ' km² (' + ORD(m[4], 'f') + ' plus vaste)' +
          (m[5] === '.' ? '.' : ', et un PIB de ' + N(m[6]) + ' millions d’euros en 2024 (' + ORD(m[7], 'm') + ' rang national).'); }],
    [new RegExp('^Its GDP stood at €' + NUM + ' million in 2024 \\(provisional\\), ranking it ' + ORDI + ' nationally\\.$'), function (m) {
      return 'Son PIB s’élevait à ' + N(m[1]) + ' millions d’euros en 2024 (estimation provisoire), au ' + ORD(m[2], 'm') + ' rang national.'; }],
    [/^Its largest cities are (.+)\.$/, function (m) { return 'Ses plus grandes villes sont ' + LIST(m[1]) + '.'; }],
    [new RegExp("^Situated (.+), ([^,]+?) spans " + NUM + " km² — the " + ORDI + " largest of France's 13 métropole régions\\.$"), function (m) {
      return 'Située ' + (ZONE[m[1]] || m[1]) + ', la région ' + m[2] + ' s’étend sur ' + N(m[3]) + ' km² — la ' +
        ORD(m[4], 'f') + ' plus vaste des 13 régions de France métropolitaine.'; }],
    [new RegExp('^It was home to ' + NUM + ' people as of 2021 \\(' + ORDI + ' most populous\\), and generated an estimated €' + NUM +
      ' million in GDP in 2024 \\(ranked ' + ORDI + '\\)\\.$'), function (m) {
      return 'Elle comptait ' + N(m[1]) + ' habitants en 2021 (' + ORD(m[2], 'f') + ' plus peuplée) et a généré un PIB estimé à ' +
        N(m[3]) + ' millions d’euros en 2024 (' + ORD(m[4], 'm') + ' rang).'; }],
    [/^(.+) are among its largest cities\.$/, function (m) { return LIST(m[1]) + ' figurent parmi ses plus grandes villes.'; }],
    [new RegExp('^With ' + NUM + ' inhabitants as of 2021 \\(' + ORDI + ' nationally\\) spread across ' + NUM + ' km² \\(' + ORDI +
      ' by area\\), (.+?) lies (.+)\\.$'), function (m) {
      return 'Avec ' + N(m[1]) + ' habitants en 2021 (' + ORD(m[2], 'm') + ' rang national) répartis sur ' + N(m[3]) + ' km² (' +
        ORD(m[4], 'f') + ' par la superficie), la région ' + m[5] + ' se situe ' + (ZONE[m[6]] || m[6]) + '.'; }],
    [new RegExp('^Its economy was estimated at €' + NUM + ' million in GDP in 2024 \\(' + ORDI + ' nationally\\)\\.$'), function (m) {
      return 'Son économie était estimée à ' + N(m[1]) + ' millions d’euros de PIB en 2024 (' + ORD(m[2], 'f') + ' rang national).'; }],
    [/^(.+) rank among its largest cities\.$/, function (m) { return LIST(m[1]) + ' comptent parmi ses plus grandes villes.'; }],
    [new RegExp('^The latest OpenPVMapper release has identified ' + NUM + ' MWp of rooftop PV across ' + NUM +
      ' installations in (.+?), making it the ' + ORDI + ' région in France by installed capacity\\.$'), function (m) {
      return "La dernière version d'OpenPVMapper a identifié " + N(m[1]) + ' MWc de PV en toiture répartis sur ' + N(m[2]) +
        ' installations ' + rIn(m[3]) + ', ce qui place la région au ' + ORD(m[4], 'm') + ' rang français en puissance installée.'; }],
    [/^The most equipped départements? (were|was) (.+)\.$/, function (m) {
      return m[1] === 'were' ? 'Les départements les plus équipés étaient ' + LIST(m[2]) + '.' : 'Le département le plus équipé était ' + m[2] + '.'; }],
    [new RegExp('^' + NUM + ' rooftop PV installations totalling ' + NUM + ' MWp have been identified across (.+?) by OpenPVMapper — the ' +
      ORDI + ' highest installed capacity among French régions\\.$'), function (m) {
      return N(m[1]) + ' installations PV en toiture, pour un total de ' + N(m[2]) + ' MWc, ont été identifiées ' + rIn(m[3]) +
        ' par OpenPVMapper — ' + ORD(m[4], 'm') + ' rang des régions françaises en puissance installée.'; }],
    [/^(.+?) leads? the région's départements by capacity\.$/, function (m) {
      return LIST(m[1]) + (/ and |, /.test(m[1]) ? ' arrivent' : ' arrive') + ' en tête des départements de la région pour la puissance installée.'; }],
    [new RegExp("^OpenPVMapper's latest release puts (.+?) at " + NUM + ' MWp of estimated rooftop PV capacity over ' + NUM +
      ' installations — the ' + ORDI + ' région in France on that measure\\.$'), function (m) {
      return "La dernière version d'OpenPVMapper estime à " + N(m[2]) + ' MWc la capacité PV en toiture de la région ' + m[1] +
        ', répartie sur ' + N(m[3]) + ' installations — ' + ORD(m[4], 'f') + ' région de France sur ce critère.'; }],
    [/^(.+?) accounts? for the largest shares? within the région\.$/, function (m) {
      var pl = / and |, /.test(m[1]);
      return LIST(m[1]) + (pl ? ' concentrent les plus grandes parts' : ' concentre la plus grande part') + ' de la puissance de la région.'; }],

    /* ---------------- shared ---------------- */
    [/^(Detections|These detections|The detections) span (.+?)([.;])$/, function (m) {
      var subj = m[1] === 'These detections' ? 'Ces détections' : 'Les détections';
      return subj + ' couvrent ' + SPAN(m[2]) + (m[3] === ';' ? ' ;' : '.'); }],
    [/^To explore every detection in (.+),$/, function (m, c) {
      var where = c.kind === 'cities' ? inCity(m[1]) : c.kind === 'regions' ? rIn(m[1]) : dIn(m[1]);
      return 'Pour explorer toutes les détections ' + where + ','; }],
    [/^Report it — you'll help improve data quality for (.+) and across France\.$/, function (m) {
      return 'Signalez-le : vous contribuerez à améliorer la qualité des données pour ' + m[1] + ' et pour toute la France.'; }]
  ];

  // Whole-string patterns (labels, tooltips) — applied to the full text, not per sentence.
  var SHORT = [
    [/^by installed capacity, out of ([\d,. ]+) (communes|départements|régions)(?: in (.+))?$/, function (m) {
      return 'au classement par puissance installée, sur ' + N(m[1]) + ' ' + m[2] + (m[3] ? ' ' + dIn(m[3]) : ''); }],
    [/^← Back to (.+)$/, function (m) {
      var n = m[1];
      if (DEPTS[n]) return '← Retour au département\u00a0: ' + n;
      return '← Retour à la région ' + n; }],
    [/^(.+) · DeepPVMapper Data$/, function (m) { return m[1] + ' · Données DeepPVMapper'; }],
    [/^Darker = larger share of (.+?)’s installed capacity\.$/, function (m) {
      return 'Plus la teinte est foncée, plus la part de la région ' + m[1] + ' dans la puissance installée est élevée.'; }],
    [/^(.+): ([\d,.]+) MWp · #(\d+) by installed capacity in the région$/, function (m) {
      return m[1] + ' : ' + N(m[2]) + ' MWc · n°' + NB + m[3] + ' de la région en puissance installée'; }],
    [/^(\d[\d,. ]*) systems$/, function (m) { return N(m[1]) + ' système' + PL(m[1], '', 's'); }],
    [/^(\d+(?:\.\d+)?)% of capacity that year$/, function (m) { return N(m[1]) + ' % de la puissance cette année'; }],
    // orientation tooltip: "NNE: 13 installations (11%)"
    [/^([NESW]{1,3}): (\d+) installations? \((\d+)%\)$/, function (m) {
      return (COMPASS[m[1]] || m[1]) + ' : ' + m[2] + ' installation' + PL(m[2], '', 's') + ' (' + m[3] + ' %)'; }],
    // power-class tooltip: "< 9 kWp: 106 installations" / "9–36 kWp: 13 installations"
    [/^([<>] )?([\d.]+(?:–[\d.]+)?) kWp: (\d+) installations?$/, function (m) {
      return (m[1] || '') + m[2] + ' kWc : ' + m[3] + ' installation' + PL(m[3], '', 's'); }],
    // source-combination legend / tooltip: "DPVM + FRPV · 41%" , "OSM only: 22%"
    [/^((?:DPVM|FRPV|OSM)(?: only| \+ [A-Z+ ]+)?)( · |: )(\d+)%$/, function (m) {
      var lab = EXACT[m[1]] || m[1];
      return lab + (m[2] === ': ' ? ' : ' : ' · ') + m[3] + ' %'; }],
    // <meta> descriptions (head)
    [/^Rooftop PV systems detected by DeepPVMapper in (.+?) \((.+?)\): ([\d,. ]+) systems, ([\d,. ]+) (kWp|MWp) estimated installed capacity\. Explore on the map or download the data\.$/, function (m) {
      return 'Systèmes photovoltaïques en toiture détectés par DeepPVMapper ' + inCity(m[1]) + ' (' + m[2] + ') : ' +
        N(m[3]) + ' systèmes, ' + N(m[4]) + ' ' + U(m[5]) + ' de puissance installée estimée. Explorez-les sur la carte ou téléchargez les données.'; }],
    [/^([\d,. ]+) rooftop PV systems detected (?:in|across) (.+?), ([\d,. ]+) (kWp|MWp) estimated installed capacity\.( Open data, free to download\.)?$/, function (m, c) {
      var where = c.kind === 'cities' ? inCity(m[2]) : c.kind === 'regions' ? rIn(m[2]) : dIn(m[2]);
      return N(m[1]) + ' systèmes PV en toiture détectés ' + where + ', ' + N(m[3]) + ' ' + U(m[4]) + ' de puissance installée estimée.' +
        (m[5] ? ' Données ouvertes, téléchargement gratuit.' : ''); }],
    [/^Open-data registry of rooftop PV systems in (.+?) \((.+?)\): ([\d,. ]+) systems, ([\d,. ]+) MWp estimated capacity\. Free GeoJSON\/CSV download for PCAET & territorial energy planning, or explore on the map\.$/, function (m) {
      return 'Registre open data des systèmes PV en toiture ' + dIn(m[1]) + ' (' + m[2] + ') : ' + N(m[3]) + ' systèmes, ' + N(m[4]) +
        ' MWc de puissance estimée. Export GeoJSON/CSV gratuit pour les PCAET et la planification énergétique territoriale, ou exploration sur la carte.'; }],
    [/^Open-data registry of rooftop PV systems across (.+?): ([\d,. ]+) systems, ([\d,. ]+) MWp over (\d+) départements\. Free GeoJSON\/CSV download for PCAET, SRADDET & territorial energy planning\.$/, function (m) {
      return 'Registre open data des systèmes PV en toiture ' + rIn(m[1]) + ' : ' + N(m[2]) + ' systèmes, ' + N(m[3]) + ' MWc répartis sur ' + m[4] +
        ' département' + PL(m[4], '', 's') + '. Export GeoJSON/CSV gratuit pour les PCAET, le SRADDET et la planification énergétique territoriale.'; }],
    // stat numbers
    [/^([\d,.  ]+) (kWp|MWp|GWp)$/, function (m) { return N(m[1]) + ' ' + U(m[2]); }],
    [/^#(\d+)$/, function (m) { return 'n°' + NB + m[1]; }],
    [/^\d{1,3}(?:[, ]\d{3})+$/, function (m) { return N(m[0]); }]
  ];

  /* ------------------------------------------------------------------ translation core */
  function trSentence(s, ctx) {
    for (var i = 0; i < RULES.length; i++) {
      var m = RULES[i][0].exec(s);
      if (m) { var out = RULES[i][1](m, ctx); if (out) return out; }
    }
    return null;
  }

  function trString(raw, ctx) {
    var lead = /^\s*/.exec(raw)[0], trail = /\s*$/.exec(raw)[0];
    var core = raw.slice(lead.length, raw.length - trail.length);
    if (!core) return null;
    if (ctx.svg && Object.prototype.hasOwnProperty.call(EXACT, '__svg:' + core)) return lead + EXACT['__svg:' + core] + trail;
    if (Object.prototype.hasOwnProperty.call(EXACT, core)) return lead + EXACT[core] + trail;
    var i, m;
    for (i = 0; i < SHORT.length; i++) {
      m = SHORT[i][0].exec(core);
      if (m) return lead + SHORT[i][1](m, ctx) + trail;
    }
    // sentence by sentence (a paragraph may hold several; some end right before a link)
    var parts = core.split(/([.;])(\s+)(?=[A-Z0-9])/);
    var out = '', changed = false, k;
    for (k = 0; k < parts.length; k += 3) {
      var sent = parts[k] + (parts[k + 1] || '');
      var t = trSentence(sent, ctx);
      if (t !== null) changed = true;
      out += (t !== null ? t : sent) + (parts[k + 2] || '');
    }
    return changed ? lead + out + trail : null;
  }

  function kindOf(path) {
    var m = /(?:^|\/)(cities|data|regions)\/[^\/]*$/.exec(path);
    return m ? m[1] : 'cities';
  }

  /* ------------------------------------------------------------------ DOM */
  var PAGES = 'about|data|software|contribute|outlook|pipeline|openpvmapper|data-documentation|legal|local-statistics';
  var ATTRS = ['data-tip', 'aria-label', 'title', 'placeholder', 'alt'];
  var METAS = ['meta[name="description"]', 'meta[property="og:title"]', 'meta[property="og:description"]'];

  function textNodes(doc) {          // same filter + order at build time and at run time
    var walker = doc.createTreeWalker(doc.body, 4 /* SHOW_TEXT */, null), out = [], n;
    while ((n = walker.nextNode())) {
      var p = n.parentNode && n.parentNode.nodeName;
      if (p === 'SCRIPT' || p === 'STYLE' || p === 'NOSCRIPT') continue;
      out.push(n);
    }
    return out;
  }

  /* English -> French rewrite of one document. When `log` is given, the English original of every
   * change is recorded in it (text nodes by index, attributes by element index) so the page can be
   * switched back to English later (see restoreEnglish). */
  function translateDocument(doc, kind, log) {
    var nodes = textNodes(doc), i;
    nodes.forEach(function (t, idx) {
      var par = t.parentNode;
      var ctx = { kind: kind, svg: !!(par && par.namespaceURI === 'http://www.w3.org/2000/svg') };
      var out = trString(t.nodeValue, ctx);
      if (out !== null && out !== t.nodeValue) {
        if (log) log.t.push([idx, t.nodeValue]);
        t.nodeValue = out;
      }
    });
    var ctxA = { kind: kind, svg: false };
    var els = doc.body.querySelectorAll('*');
    function setAttr(el, idx, name, val) {
      if (log) log.a.push([idx, name, el.getAttribute(name)]);
      el.setAttribute(name, val);
    }
    for (i = 0; i < els.length; i++) {
      var el = els[i];
      ATTRS.forEach(function (a) {
        if (!el.hasAttribute(a)) return;
        var out = trString(el.getAttribute(a), ctxA);
        if (out !== null) setAttr(el, i, a, out);
      });
      // links to pages that have a French twin go to the twin
      if (el.nodeName === 'A' && el.hasAttribute('href') && !(el.parentNode && el.parentNode.className === 'topnav-lang')) {
        var h = el.getAttribute('href'), m;
        var re = new RegExp('^((?:\\.\\./)+)(' + PAGES + ')\\.html((?:[?#].*)?)$');
        if ((m = re.exec(h))) setAttr(el, i, 'href', m[1] + '../fr/content/' + m[2] + '.html' + m[3]);
        else if ((m = /^((?:\.\.\/)+)index\.html((?:[?#].*)?)$/.exec(h))) setAttr(el, i, 'href', m[1] + 'fr/index.html' + m[2]);
      }
    }
    METAS.forEach(function (sel) {
      var e = doc.querySelector(sel);
      if (!e) return;
      var out = trString(e.getAttribute('content'), ctxA);
      if (out !== null) { if (log) log.m.push([sel, 'content', e.getAttribute('content')]); e.setAttribute('content', out.replace(/\u00a0/g, ' ')); }
    });
    var dt = trString(doc.title, ctxA);
    if (dt !== null) { if (log) log.d = doc.title; doc.title = dt; }
  }

  /* Build step (scripts/frenchify_pages.js): turn an English generated page into the French page,
   * embedding the English originals as JSON for the EN view. Returns nothing; edits `doc`. */
  function frenchify(doc, kind) {
    var log = { t: [], a: [], m: [], d: null };
    translateDocument(doc, kind, log);
    doc.documentElement.setAttribute('lang', 'fr');
    var head = doc.head;
    if (head && !head.querySelector('meta[property="og:locale"]')) {
      var m = doc.createElement('meta'); m.setAttribute('property', 'og:locale'); m.setAttribute('content', 'fr_FR');
      var ref = head.querySelector('meta[property="og:type"]');
      head.insertBefore(m, ref ? ref.nextSibling : null);
    }
    var box = doc.querySelector('.topnav-lang');
    if (box) {
      var en = box.querySelector('[hreflang="en"]'), fr = box.querySelector('[hreflang="fr"]');
      if (en && fr) {
        en.setAttribute('href', '#lang-en'); en.classList.remove('is-active'); en.removeAttribute('aria-current');
        fr.setAttribute('href', '#lang-fr'); fr.classList.add('is-active'); fr.setAttribute('aria-current', 'true'); fr.removeAttribute('title');
      }
    }
    var s = doc.createElement('script');
    s.setAttribute('type', 'application/json'); s.setAttribute('id', 'dpm-en');
    s.textContent = JSON.stringify(log).replace(/</g, '\\u003c');
    var anchor = doc.body.querySelector('script[src$="page-lang.js"]');
    anchor ? anchor.parentNode.insertBefore(s, anchor) : doc.body.appendChild(s);
    return log;
  }

  /* Run time, on a French page: put the English originals back. */
  function restoreEnglish(doc) {
    var el = doc.getElementById('dpm-en');
    if (!el) return false;
    var log;
    try { log = JSON.parse(el.textContent); } catch (e) { return false; }
    var nodes = textNodes(doc), els = doc.body.querySelectorAll('*');
    (log.t || []).forEach(function (p) { if (nodes[p[0]]) nodes[p[0]].nodeValue = p[1]; });
    (log.a || []).forEach(function (p) { if (els[p[0]]) els[p[0]].setAttribute(p[1], p[2]); });
    (log.m || []).forEach(function (p) { var e = doc.querySelector(p[0]); if (e) e.setAttribute(p[1], p[2]); });
    if (log.d) doc.title = log.d;
    doc.documentElement.setAttribute('lang', 'en');
    return true;
  }

  function wireSwitcher(doc, lang, staticMode) {
    var box = doc.querySelector('.topnav-lang');
    if (!box) return;
    var en = box.querySelector('[hreflang="en"]'), fr = box.querySelector('[hreflang="fr"]');
    if (!en || !fr) return;
    fr.removeAttribute('title');
    en.setAttribute('href', '#lang-en'); fr.setAttribute('href', '#lang-fr');
    en.classList.toggle('is-active', lang === 'en'); fr.classList.toggle('is-active', lang === 'fr');
    en.removeAttribute('aria-current'); fr.removeAttribute('aria-current');
    (lang === 'en' ? en : fr).setAttribute('aria-current', 'true');
    function pick(target) {
      return function (e) {
        if (e.metaKey || e.ctrlKey || e.shiftKey || e.button === 1) return;
        e.preventDefault();
        if (target === lang) return;
        try { doc.defaultView.localStorage.setItem(KEY, target); } catch (x) {}
        var u = doc.defaultView.location;
        u.href = u.pathname + '?lang=' + target + u.hash;
      };
    }
    en.addEventListener('click', pick('en'));
    fr.addEventListener('click', pick('fr'));
  }

  function resolveLang(win, def) {
    var param = null;
    try { param = new win.URLSearchParams(win.location.search).get('lang'); } catch (e) {}
    var stored = null, ok = true;
    try { stored = win.localStorage.getItem(KEY); } catch (e) { ok = false; }
    var lang = (param === 'fr' || param === 'en') ? param : (stored === 'fr' || stored === 'en') ? stored : def;
    if (param === 'fr' || param === 'en') {
      try { win.localStorage.setItem(KEY, param); } catch (e) { ok = false; }
      try {  // drop ?lang= from the address bar
        var p = new win.URLSearchParams(win.location.search); p.delete('lang');
        var q = p.toString();
        win.history.replaceState(null, '', win.location.pathname + (q ? '?' + q : '') + win.location.hash);
      } catch (e) {}
    }
    return { lang: lang, storageOK: ok };
  }

  function carryLang(doc, lang) {    // without localStorage, keep the language in the internal links
    var as = doc.querySelectorAll('a[href]');
    for (var i = 0; i < as.length; i++) {
      var h = as[i].getAttribute('href');
      if (/^(?:\.\.\/|\.\/)*(?:content\/)?(cities|data|regions)\/[^?#]+\.html$/.test(h)) as[i].setAttribute('href', h + '?lang=' + lang);
    }
  }

  function run(win) {
    var doc = win.document;
    // Generated pages are shipped in French (the indexed version); a page still in English comes
    // straight from the generator, before scripts/frenchify_pages.js has run.
    var shipped = (doc.documentElement.getAttribute('lang') || 'en') === 'fr' ? 'fr' : 'en';
    var st = resolveLang(win, shipped);
    var kind = kindOf(win.location.pathname);
    if (shipped === 'fr' && st.lang === 'en') restoreEnglish(doc);
    else if (shipped === 'en' && st.lang === 'fr') {
      doc.documentElement.setAttribute('lang', 'fr');
      translateDocument(doc, kind);
    }
    wireSwitcher(doc, st.lang);
    if (!st.storageOK && st.lang !== shipped) carryLang(doc, st.lang);
    if (st.lang === 'fr') {
      // The hexbin tooltip text is written by the page's own script on mousemove; translate after it.
      var tip = doc.getElementById('chart-tooltip');
      if (tip) {
        var cells = doc.querySelectorAll('.chart-hover-target');
        for (var i = 0; i < cells.length; i++) {
          cells[i].addEventListener('mousemove', function () {
            var out = trString(tip.textContent, { kind: kind, svg: false });
            if (out !== null) tip.textContent = out;
          });
        }
      }
    }
  }

  root.DPVMLang = { trString: trString, kindOf: kindOf, translateDocument: translateDocument, frenchify: frenchify,
                    restoreEnglish: restoreEnglish, run: run };
  if (root.document && !root.__DPVM_LANG_NO_AUTORUN) run(root);
})(typeof window !== 'undefined' ? window : this);
