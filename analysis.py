"""
analysis.py — Synthèses de progression.

VERSION 2026-08-25-C  (relance : trois séries sur un axe)

Répond à deux questions, et à elles seules : où en suis-je, et où dois-je
concentrer mes efforts.

LE COÛT DE RELANCE

C'est la mesure que personne ne fait, et probablement la plus utile ici.
Pousser en montée et en descente ne sert à rien si le prix se paie sur le
plat qui suit. On mesure donc, après chaque montée, l'allure sur trois
fenêtres de distance, comparée à l'allure sur le plat « libre » de la même
sortie — celui qui ne suit ni montée ni descente.

    0-200 m      la relance      es-tu capable de repartir
    200-600 m    l'effort        tiens-tu l'allure retrouvée
    600-1500 m   la récupération combien de temps la bosse te coûte

Le profil de décroissance porte l'information. Un écart de -10 % à 200 m
qui s'annule à 600 m signale une bonne récupération. Un écart qui tient
encore à 1 500 m signifie que chaque bosse se paie bien au-delà de la
relance.

Deux précautions sans lesquelles la mesure ne veut rien dire.

1. On ne compte QUE les portions plates dans chaque fenêtre. Sur un
   parcours de montagne, la fenêtre des 600-1500 m tombe souvent dans la
   descente suivante : la comparer à du plat n'aurait aucun sens.
2. Sous une distance plate minimale dans la fenêtre, on renvoie « non
   mesurable » plutôt qu'un chiffre bancal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.metrics import wmean

VERSION = "2026-08-25-C"

FLAT = 0.03          # |pente| en deçà de laquelle on considère du plat
CLIMB = 0.05         # pente au-delà de laquelle on est en montée
MIN_CLIMB_M = 200.0  # une bosse plus courte n'est pas une montée
MIN_FLAT_M = 120.0   # plat minimal dans une fenêtre pour qu'elle compte
MAX_GAP_M = 1000.0   # au-delà, le plat n'est plus la relance de cette bosse
WINDOWS = [(0, 200, "relance"), (200, 600, "effort"), (600, 1500, "recuperation")]


def _segments(mask: np.ndarray, cum: np.ndarray, min_len: float):
    """Plages contiguës où mask est vrai, d'au moins min_len mètres."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if cum[i - 1] - cum[start] >= min_len:
                out.append((start, i - 1))
            start = None
    if start is not None and cum[-1] - cum[start] >= min_len:
        out.append((start, len(mask) - 1))
    return out


def transition_cost(d: pd.DataFrame, kind: str = "up") -> dict:
    """
    Coût de relance après les montées (kind="up") ou les descentes ("down").

    LA RÉFÉRENCE EST LE POINT DÉLICAT, et la première version s'y est cassé
    les dents. Comparer au plat « libre » — celui qui ne suit aucune
    transition — suppose qu'il en existe. Sur un parcours vallonné où une
    bosse survient tous les 1 300 m, la fenêtre de contamination de 1 500 m
    couvre tout le tracé : aucune référence, donc aucune mesure. Ce n'est
    pas un cas limite, c'est le terrain courant.

    La référence est donc cherchée le plus loin possible d'une transition,
    puis rapprochée tant qu'il n'y a pas assez de plat : 1 500 m, puis
    800 m, puis 400 m. Le seuil retenu est renvoyé dans `ref_distance_m`,
    parce qu'il change l'interprétation — une référence à 400 m est déjà
    partiellement contaminée par la relance qu'on cherche à mesurer, donc
    elle SOUS-estime l'effet.

    On renvoie aussi le profil relatif, chaque fenêtre rapportée à la
    dernière mesurable. Celui-là ne dépend d'aucune référence externe et
    reste lisible même en montagne : il dit si tu récupères, pas de combien.
    """
    slope = d["slope"].to_numpy()
    cum = d["cum_dist"].to_numpy()
    dt = d["dt"].to_numpy()
    gap = d["gap"].to_numpy()
    flat = np.abs(slope) < FLAT

    empty = {"n_transitions": 0, "plat_libre_kmh": np.nan,
             "ref_distance_m": np.nan,
             **{w[2]: np.nan for w in WINDOWS},
             **{f"{w[2]}_n": 0 for w in WINDOWS},
             **{f"{w[2]}_rel": np.nan for w in WINDOWS}}

    zones = _segments(slope > CLIMB if kind == "up" else slope < -CLIMB,
                      cum, MIN_CLIMB_M)
    if not zones:
        return empty

    # ORIGINE DES FENÊTRES : le début du plat qui suit, pas la fin de la
    # montée. Après un col vient presque toujours une descente, jamais du
    # plat : mesurée depuis la fin de la montée, la fenêtre 0-200 m tombait
    # systématiquement dans la descente et n'était jamais exploitable.
    # La relance, c'est le moment où tu retrouves du plat — quel que soit
    # le temps qu'il a fallu pour y arriver.
    # Et il faut une portion plate SOUTENUE, pas un point isolé. Entre une
    # montée à +11 % et une descente à -13 %, la pente passe par zéro : ce
    # point unique était pris pour le début du plat, la fenêtre s'ouvrait
    # au sommet du col et ne contenait que trois points.
    flat_zones = _segments(flat, cum, MIN_FLAT_M)
    ends = []
    for _, e in zones:
        suivant = [cum[a] for a, _ in flat_zones
                   if cum[a] > cum[e] and cum[a] <= cum[e] + MAX_GAP_M]
        if suivant:
            ends.append(float(min(suivant)))
    if not ends:
        return {**empty, "n_transitions": len(zones)}

    # Allure ET coût cardiaque par fenêtre.
    #
    # Pourquoi les deux. Relancer plus vite en poussant plus fort n'est pas
    # une progression : c'est une dette qui se paie en fin de course longue.
    # Le seul indicateur qui tranche est le coût cardiaque — battements par
    # W/kg produit. Vitesse en hausse ET coût en baisse, c'est un vrai gain
    # d'efficacité. Vitesse en hausse et coût en hausse, tu pousses
    # simplement plus fort.
    hr = d["hr"].to_numpy(dtype=float) if "hr" in d.columns else np.full(len(d), np.nan)
    cost = d["hr_cost"].to_numpy(dtype=float) if "hr_cost" in d.columns else np.full(len(d), np.nan)

    per_window, hr_window, cost_window = {}, {}, {}
    for lo, hi, label in WINDOWS:
        vals, hrs, costs, weights = [], [], [], []
        for e in ends:
            w = (cum > e + lo) & (cum <= e + hi) & flat
            if w.sum() == 0:
                continue
            if (cum[w].max() - cum[w].min()) < MIN_FLAT_M or dt[w].sum() < 20:
                continue
            vals.append(wmean(gap[w], dt[w]))
            hrs.append(wmean(hr[w], dt[w]))
            costs.append(wmean(cost[w], dt[w]))
            weights.append(dt[w].sum())
        per_window[label] = (wmean(vals, weights) if vals else np.nan, len(vals))
        hr_window[label] = wmean(hrs, weights) if hrs else np.nan
        cost_window[label] = wmean(costs, weights) if costs else np.nan

    # RÉFÉRENCE : LE PLAT « LIBRE » ÉTAIT BIAISÉ VERS LE BAS.
    #
    # Définir la référence comme le plat éloigné de toute transition
    # sélectionne, sur un parcours vallonné, presque exclusivement le début
    # et la fin de la sortie — c'est-à-dire l'échauffement et le retour au
    # calme, les deux moments les plus lents. La référence était donc trop
    # basse et TOUTES les fenêtres ressortaient positives : +10 % après une
    # montée, ce qui n'a aucun sens physiologique.
    #
    # On prend maintenant l'allure de plat de l'ensemble de la sortie, en
    # écartant les cinq premières minutes. Les fenêtres post-transition en
    # font partie, ce qui dilue légèrement l'effet mesuré — mais un biais
    # de dilution connu vaut mieux qu'un biais de sélection qui inverse le
    # signe du résultat.
    #
    # Ordre de grandeur de cette dilution, mesuré en simulation : un coût
    # réel de -12 % ressort à -5 % quand les fenêtres représentent 40 % du
    # plat total. L'amplitude est donc sous-estimée, mais le sens et le
    # classement entre fenêtres restent justes — ce qui suffit pour suivre
    # une évolution dans le temps, qui est l'usage visé.
    t_cum = np.cumsum(dt)
    apres_echauffement = t_cum > 300
    base_flat = flat & apres_echauffement
    if dt[base_flat].sum() < 120:
        base_flat = flat
    ok = dt[base_flat].sum() >= 60
    ref = wmean(gap[base_flat], dt[base_flat]) if ok else np.nan
    ref_hr = wmean(hr[base_flat], dt[base_flat]) if ok else np.nan
    ref_cost = wmean(cost[base_flat], dt[base_flat]) if ok else np.nan
    ref_dist = float(dt[base_flat].sum() / 60)

    out = {"n_transitions": len(ends),
           "plat_libre_kmh": ref * 3.6 if ref and not np.isnan(ref) else np.nan,
           "ref_fc": ref_hr, "ref_cout": ref_cost, "ref_distance_m": ref_dist}

    # Profil relatif : chaque fenêtre rapportée à la dernière mesurable.
    mesurables = [l for _, _, l in WINDOWS if not np.isnan(per_window[l][0])]
    base = per_window[mesurables[-1]][0] if mesurables else np.nan

    for _, _, label in WINDOWS:
        v, n = per_window[label]
        out[label] = (v / ref) if (ref and not np.isnan(ref) and not np.isnan(v)) else np.nan
        out[f"{label}_n"] = n
        out[f"{label}_rel"] = (v / base) if (base and not np.isnan(base)
                                             and not np.isnan(v)) else np.nan
        out[f"{label}_fc"] = hr_window[label] - ref_hr if not np.isnan(ref_hr) else np.nan
        c = cost_window[label]
        out[f"{label}_cout"] = (c / ref_cost) if (ref_cost and not np.isnan(ref_cost)
                                                  and not np.isnan(c)) else np.nan
    return out


def summarize_transitions(d: pd.DataFrame) -> dict:
    """Aplatit les deux analyses pour stockage en base."""
    row = {}
    for kind, prefix in [("up", "apres_montee"), ("down", "apres_descente")]:
        r = transition_cost(d, kind)
        for lo, hi, label in WINDOWS:
            row[f"{prefix}_{label}"] = r.get(label, np.nan)
            row[f"{prefix}_{label}_rel"] = r.get(f"{label}_rel", np.nan)
            row[f"{prefix}_{label}_fc"] = r.get(f"{label}_fc", np.nan)
            row[f"{prefix}_{label}_cout"] = r.get(f"{label}_cout", np.nan)
        row[f"{prefix}_n"] = r.get("n_transitions", 0)
        row[f"{prefix}_ref_kmh"] = r.get("plat_libre_kmh", np.nan)
        row[f"{prefix}_ref_m"] = r.get("ref_distance_m", np.nan)
    return row


# ── Comparaison de périodes ──────────────────────────────────────────────────

def split_recent(hist: pd.DataFrame, months: int | None,
                 n_recent: int = 5, sport: str = "trail") -> tuple:
    """
    Sépare les n dernières sorties trail (référence mobile, environ deux
    semaines) du reste de la période choisie.

    Une référence mobile plutôt qu'un mois fixe : ta forme se lit sur ce que
    tu viens de faire, pas sur une case du calendrier.
    """
    d = hist.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce", utc=True).dt.tz_localize(None)
    d = d[d["sport"] == sport].dropna(subset=["date"]).sort_values("date")
    if months:
        d = d[d["date"] >= d["date"].max() - pd.DateOffset(months=months)]
    # n_recent=0 sert à compter les sorties disponibles : tout doit alors
    # basculer en référence. Attention, d.iloc[:-0] renvoie une table VIDE
    # en Python, pas la table entière — le piège classique de la négation
    # de zéro sur les tranches.
    if n_recent <= 0:
        return d.iloc[0:0], d
    recent = d.tail(n_recent)
    passe = d.iloc[:-n_recent] if len(d) > n_recent else d.iloc[0:0]
    return recent, passe


def compare_bands(bins: pd.DataFrame, recent_ids, past_ids,
                  metric: str = "vitesse_kmh", uphill: bool = True,
                  min_dist_km: float = 0.4,
                  min_total_recent_km: float = 2.0,
                  min_total_ref_km: float = 6.0) -> pd.DataFrame:
    """
    Compare une métrique par bande de pente entre sorties récentes et
    référence.

    POURQUOI LE VERDICT BINAIRE A ÉTÉ ABANDONNÉ

    La première version testait l'écart contre un seuil à 95 % et
    répondait « significatif » ou « dans le bruit ». Sur quatre ou cinq
    sorties récentes, presque rien ne franchit ce seuil : l'app répétait
    « je ne peux rien en déduire ». Ce n'est pas une analyse, c'est un
    refus d'analyser — et c'est faux, parce qu'un écart non significatif à
    95 % n'est pas pour autant une absence d'information.

    Deux mesures indépendantes sont donc produites.

    1. L'AMPLEUR, rapportée au seuil : `force` = |écart| / seuil à 95 %.
       Une valeur continue, qu'on gradue plutôt que de la trancher.

    2. LA CONSTANCE, par test des signes : combien de tes sorties récentes
       se situent au-dessus de la médiane de référence. Quatre sorties sur
       quatre du même côté, c'est une chance sur seize d'arriver par
       hasard — informatif là où la comparaison de moyennes ne conclut pas.
       Cette mesure est puissante précisément quand l'échantillon est
       petit, ce qui est exactement le cas d'usage.

    Le niveau retenu combine les deux : une progression modeste mais
    constante sur toutes les sorties vaut mieux qu'un grand écart porté
    par une seule.
    """
    if bins.empty or "pente_centre" not in bins.columns:
        return pd.DataFrame()

    b = bins.copy()
    b["activity_id"] = b["activity_id"].astype(str)

    # RICHESSE MESURÉE EN DISTANCE, PAS EN TEMPS NI EN NOMBRE DE SORTIES.
    # Deux minutes passées dans une bande, c'est 200 m en montée raide et
    # 600 m sur du roulant : le même seuil temporel accepte donc des
    # échantillons trois fois moins riches là où l'allure est basse — soit
    # exactement les bandes de forte pente, celles qui produisaient des
    # écarts aberrants. La distance normalise cela.
    if "distance_km" not in b.columns:
        b["distance_km"] = b["vitesse_kmh"] * b["temps_min"] / 60

    sel = b["pente_centre"] > 0.03 if uphill else b["pente_centre"] < -0.03
    b = b[sel & b[metric].notna() & (b["distance_km"] >= min_dist_km)]
    if b.empty:
        return pd.DataFrame()

    rset, pset = set(map(str, recent_ids)), set(map(str, past_ids))

    # CORRECTION POUR COMPARAISONS MULTIPLES. On teste trois à cinq bandes
    # et on met en avant la plus forte : c'est trois à cinq chances de
    # tomber juste par hasard. Sans correction, la simulation donne 23 à
    # 36 % de verdicts « net » sur des données SANS aucune progression.
    # Le seuil est donc relevé par la correction de Šidák.
    n_bandes = max(1, b["bande"].nunique())
    z = Z_SIDAK.get(min(n_bandes, 9), 2.77)

    rows = []
    for band, grp in b.groupby("bande", observed=True):
        r = grp[grp["activity_id"].isin(rset)]
        p = grp[grp["activity_id"].isin(pset)]
        km_r, km_p = float(r["distance_km"].sum()), float(p["distance_km"].sum())
        if (len(r) < 2 or len(p) < 4
                or km_r < min_total_recent_km or km_p < min_total_ref_km):
            continue

        # Pondération par la distance : une sortie qui a couvert 3 km dans
        # la bande doit peser trois fois plus qu'une qui en a couvert 1.
        vr = float(np.average(r[metric], weights=r["distance_km"]))
        vp = float(np.average(p[metric], weights=p["distance_km"]))
        med = float(p[metric].median())
        sd = float(p[metric].std(ddof=1))
        seuil = z * sd / np.sqrt(len(r)) if sd > 0 else np.inf
        force = abs(vr - vp) / seuil if np.isfinite(seuil) and seuil > 0 else 0.0

        # Test des signes : constance du sens de l'écart. Inutilisable
        # sous 5 sorties — avec 4, même un résultat parfait de 4 sur 4 ne
        # descend qu'à p = 0,125, ce qui ne prouve rien et déclenchait des
        # verdicts « probable » sur du bruit pur.
        au_dessus = int((r[metric] > med).sum())
        n = len(r)
        p_sign = (2 * min(_binom_tail(au_dessus, n), _binom_tail(n - au_dessus, n))
                  if n >= 5 else np.nan)
        signes_nets = (not np.isnan(p_sign)) and (p_sign * n_bandes <= 0.15)

        # SEUILS CALIBRÉS PAR SIMULATION, pas choisis à vue. Sur 400
        # tirages de données SANS aucune progression, en prenant comme
        # l'interface le fait la plus forte des trois bandes :
        #
        #   seuil 1,0 -> 14 % de faux « net »   (trop permissif)
        #   seuil 1,3 ->  5 % de faux « net »   retenu
        #   seuil 1,6 ->  2 %, mais ne détecte plus qu'un gain de 10 %
        #                 une fois sur deux à quatre sorties
        #
        # 1,30 détecte un gain réel de 10 % dans 77 % des cas à quatre
        # sorties et 98 % à dix. C'est le compromis retenu.
        #
        # « probable » à 0,90 laisse passer 22 % de faux positifs : c'est
        # assumé et dit dans le libellé, parce qu'un signal faible reste
        # une information tant qu'on ne le présente pas comme un fait.
        if force >= 1.30:
            niveau = "net"
        elif force >= 0.90 or signes_nets:
            niveau = "probable"
        elif force >= 0.60:
            niveau = "faible"
        else:
            niveau = "nul"

        # Un échantillon maigre ne peut pas produire un verdict fort,
        # même si l'écart calculé est grand.
        if km_r < min_total_recent_km * 2 and niveau == "net":
            niveau = "probable"

        rows.append({
            "bande": band, "pente": float(grp["pente_centre"].iloc[0]),
            "km_recent": km_r, "km_ref": km_p,
            "recent": vr, "reference": vp, "mediane_ref": med,
            "ecart": vr - vp,
            "ecart_pct": (vr / vp - 1) * 100 if vp else np.nan,
            "n_recent": n, "n_ref": len(p),
            "au_dessus": au_dessus, "p_sign": p_sign,
            "force": force, "niveau": niveau,
            "signal": niveau in ("net", "probable"),
        })
    return pd.DataFrame(rows).sort_values("pente")


# Quantiles normaux corrigés de Šidák pour k comparaisons, alpha = 5 %.
Z_SIDAK = {1: 1.96, 2: 2.24, 3: 2.39, 4: 2.50, 5: 2.58,
           6: 2.64, 7: 2.69, 8: 2.73, 9: 2.77}


def _binom_tail(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k) pour une binomiale, sans dépendance externe."""
    from math import comb
    return sum(comb(n, i) * p ** n for i in range(k, n + 1))


def read_bands(table: pd.DataFrame, label: str, unit: str = "km/h",
               sens_positif: bool = True, noms: dict | None = None) -> str:
    """
    Une ligne par bande, toutes au même format.

    La version précédente détaillait la bande la plus forte et résumait les
    autres à un pourcentage nu : l'œil ne pouvait pas les comparer, et rien
    ne justifiait ce traitement inégal. Le verdict textuel a disparu
    également — une bande qui s'affiche a déjà passé les filtres de
    distance, donc dire « trop faible pour conclure » revenait à contredire
    sa propre présence à l'écran. La solidité reste lisible dans l'opacité
    des barres.
    """
    noms = noms or {}
    if table.empty:
        return f"Pas assez de sorties comparables en {label}."

    t = table.copy()
    t["bon"] = (t["ecart"] > 0) == sens_positif
    t = t.sort_values("pente", ascending=False)

    lignes = []
    for _, r in t.iterrows():
        nom = noms.get(r["bande"], r["bande"])
        couleur = GOOD_HEX if r["bon"] else BAD_HEX
        lignes.append(
            f"<b>{nom}</b> · {r['recent']:.2f} contre {r['reference']:.2f} "
            f"{unit} · <b style='color:{couleur}'>{r['ecart_pct']:+.0f} %</b> "
            f"<span style='color:#5C665F'>· {r['km_recent']:.1f} / "
            f"{r['km_ref']:.0f} km</span>"
        )
    return "<br>".join(lignes)


GOOD_HEX, BAD_HEX = "#3E6146", "#B23A2B"


def band_summary(bins: pd.DataFrame, activity_ids, uphill: bool | None = None,
                 min_dist_km: float = 0.4) -> pd.DataFrame:
    """
    Photographie de l'historique : moyennes et volumes par bande, sans
    aucune comparaison.

    C'est la question « qu'est-ce que je vaux, et où ai-je couru », par
    opposition à « est-ce que je progresse ». Les deux méritent une page
    distincte : mélanger un état et une tendance sur le même écran est ce
    qui rendait la lecture confuse.
    """
    if bins.empty:
        return pd.DataFrame()
    b = bins.copy()
    b["activity_id"] = b["activity_id"].astype(str)
    if "distance_km" not in b.columns:
        b["distance_km"] = b["vitesse_kmh"] * b["temps_min"] / 60
    b = b[b["activity_id"].isin(set(map(str, activity_ids)))]
    b = b[b["distance_km"] >= min_dist_km]
    if uphill is True:
        b = b[b["pente_centre"] > 0.03]
    elif uphill is False:
        b = b[b["pente_centre"] < -0.03]
    if b.empty:
        return pd.DataFrame()

    def agg(g):
        w = g["distance_km"]
        out = {"pente": float(g["pente_centre"].iloc[0]),
               "sorties": int(g["activity_id"].nunique()),
               "distance_km": float(w.sum()),
               "temps_h": float(g["temps_min"].sum() / 60),
               "part_distance": np.nan}
        for c, k in [("vitesse_kmh", "vitesse_kmh"), ("fc", "fc"),
                     ("cadence", "cadence"), ("cout_fc", "cout_fc"),
                     ("part_marche", "part_marche")]:
            if c in g.columns and g[c].notna().any():
                m = g[c].notna()
                out[k] = float(np.average(g[c][m], weights=w[m]))
            else:
                out[k] = np.nan
        return pd.Series(out)

    t = b.groupby("bande", observed=True).apply(agg, include_groups=False).reset_index()
    total = t["distance_km"].sum()
    t["part_distance"] = t["distance_km"] / total * 100 if total else np.nan
    return t.sort_values("pente", ascending=False)


HORIZONS = (3, 5, 10, 20)


def multi_horizon(bins: pd.DataFrame, hist: pd.DataFrame, months: int | None,
                  metric: str = "vitesse_kmh", uphill: bool = True,
                  horizons=HORIZONS) -> pd.DataFrame:
    """
    Le même écart mesuré sur 3, 5, 10 et 20 dernières sorties.

    L'intérêt n'est pas d'avoir quatre chiffres au lieu d'un : c'est que la
    CONSTANCE entre horizons est elle-même une mesure de confiance, et une
    mesure plus honnête que n'importe quel test statistique sur un
    échantillon étroit.

    Un gain visible sur 3 sorties mais absent sur 20, c'est de la forme du
    moment — ou du bruit. Visible sur les quatre horizons, c'est une
    progression installée. Visible sur 20 mais pas sur 3, c'est un progrès
    ancien en train de s'éroder. Ces trois situations appellent des
    décisions d'entraînement différentes, et un chiffre unique les
    confondait.
    """
    lignes = {}
    for n in horizons:
        recent, past = split_recent(hist, months, n_recent=n)
        if len(recent) < 2 or len(past) < 4:
            continue
        t = compare_bands(bins, recent["activity_id"], past["activity_id"],
                          metric, uphill)
        if t.empty:
            continue
        for _, r in t.iterrows():
            lignes.setdefault(r["bande"], {"bande": r["bande"],
                                           "pente": r["pente"]})
            lignes[r["bande"]][f"n{n}"] = r["ecart_pct"]
            lignes[r["bande"]][f"niv{n}"] = r["niveau"]

    if not lignes:
        return pd.DataFrame()
    t = pd.DataFrame(lignes.values())

    cols = [f"n{n}" for n in horizons if f"n{n}" in t.columns]
    if cols:
        vals = t[cols]
        # Constance : part des horizons qui vont dans le même sens que la
        # médiane. 1,00 = les quatre s'accordent.
        med = vals.median(axis=1)
        t["constance"] = (np.sign(vals).eq(np.sign(med), axis=0)
                          .sum(axis=1) / vals.notna().sum(axis=1))
        t["ecart_median"] = med
        t["n_horizons"] = vals.notna().sum(axis=1)
    return t.sort_values("pente", ascending=False)


def read_multi(t: pd.DataFrame, label: str, noms: dict | None = None,
               sens_positif: bool = True) -> str:
    """Lecture d'un tableau multi-horizons."""
    noms = noms or {}
    if t.empty or "constance" not in t.columns:
        return f"Pas assez de sorties pour comparer plusieurs horizons en {label}."

    solides = t[(t["constance"] >= 0.99) & (t["n_horizons"] >= 3)
                & (t["ecart_median"].abs() >= 2)]
    if solides.empty:
        pire = t.loc[t["ecart_median"].abs().idxmax()]
        return (f"Aucune évolution installée en {label}. Le plus gros "
                f"mouvement ({noms.get(pire['bande'], pire['bande'])}, "
                f"{pire['ecart_median']:+.0f} % médian) change de sens selon "
                "l'horizon — c'est de la variabilité, pas une tendance.")

    lignes = []
    for _, r in solides.iterrows():
        bon = (r["ecart_median"] > 0) == sens_positif
        lignes.append(
            f"<b>{noms.get(r['bande'], r['bande'])}</b> · "
            f"<b style='color:{GOOD_HEX if bon else BAD_HEX}'>"
            f"{r['ecart_median']:+.0f} %</b> sur les "
            f"{int(r['n_horizons'])} horizons "
            f"<span style='color:#5C665F'>· cohérent</span>")
    autres = len(t) - len(solides)
    txt = "<br>".join(lignes)
    if autres:
        txt += (f"<br><span style='color:#5C665F'>{autres} autre(s) bande(s) "
                "sans tendance cohérente entre horizons.</span>")
    return txt


def descent_quadrant(bins: pd.DataFrame, recent_ids, past_ids) -> pd.DataFrame:
    """
    Descente en deux dimensions : vitesse ET économie.

    Un score unique de descente serait trompeur. Descendre vite en payant
    cher n'est pas la même chose que descendre vite en étant relâché, et
    les deux appellent un travail différent. On croise donc l'écart de
    vitesse et l'écart de coût cardiaque, ce qui produit quatre situations
    lisibles d'un coup d'œil.
    """
    v = compare_bands(bins, recent_ids, past_ids, "vitesse_kmh", uphill=False)
    c = compare_bands(bins, recent_ids, past_ids, "cout_fc", uphill=False)
    if v.empty or c.empty:
        return pd.DataFrame()

    t = v[["bande", "pente", "recent", "reference", "ecart_pct", "niveau",
           "km_recent", "km_ref"]].merge(
        c[["bande", "ecart_pct", "niveau"]], on="bande",
        suffixes=("_vitesse", "_cout"))

    def cadran(r):
        vite = r["ecart_pct_vitesse"] > 1.5
        lent = r["ecart_pct_vitesse"] < -1.5
        eco = r["ecart_pct_cout"] < -1.5
        cher = r["ecart_pct_cout"] > 1.5
        if vite and eco:
            return "rapide et économique"
        if vite and cher:
            return "rapide mais coûteuse"
        if lent and eco:
            return "lente mais économique"
        if lent and cher:
            return "lente et coûteuse"
        return "stable"
    t["cadran"] = t.apply(cadran, axis=1)
    return t.sort_values("pente")


def read_quadrant(t: pd.DataFrame, noms: dict | None = None) -> str:
    noms = noms or {}
    if t.empty:
        return "Pas assez de descente comparable."
    lignes = []
    for _, r in t.iterrows():
        nom = noms.get(r["bande"], r["bande"])
        coul = {"rapide et économique": GOOD_HEX,
                "lente et coûteuse": BAD_HEX,
                "rapide mais coûteuse": "#9A6636",
                "lente mais économique": "#2E6B8C"}.get(r["cadran"], "#5C665F")
        lignes.append(
            f"<b>{nom}</b> · vitesse <b>{r['ecart_pct_vitesse']:+.0f} %</b>, "
            f"coût <b>{r['ecart_pct_cout']:+.0f} %</b> → "
            f"<b style='color:{coul}'>{r['cadran']}</b>")
    return "<br>".join(lignes)


def walk_run_threshold(bins: pd.DataFrame, activity_ids,
                       min_minutes: float = 20.0) -> pd.DataFrame:
    """
    Seuil de bascule marche / course, MUTUALISÉ sur l'historique.

    Pourquoi ça ne marchait pas sortie par sortie : dans une bande donnée,
    tu as rarement marché ET couru le même jour assez longtemps pour que la
    comparaison tienne. Le calcul renvoyait « non mesurable » presque
    toujours. En cumulant sur des centaines de sorties, chaque bande
    finit par contenir les deux régimes.

    On compare, dans chaque bande de montée, la vitesse et le coût
    cardiaque des portions marchées et des portions courues. La bascule est
    la première bande où marcher est au moins aussi rapide que courir pour
    un coût égal ou inférieur.
    """
    if bins.empty or "temps_marche_min" not in bins.columns:
        return pd.DataFrame()

    b = bins.copy()
    b["activity_id"] = b["activity_id"].astype(str)
    b = b[b["activity_id"].isin(set(map(str, activity_ids)))]
    b = b[b["pente_centre"] > 0.03]
    if b.empty:
        return pd.DataFrame()

    rows = []
    for band, g in b.groupby("bande", observed=True):
        tm = g["temps_marche_min"].fillna(0)
        tc = g["temps_course_min"].fillna(0)
        if tm.sum() < min_minutes or tc.sum() < min_minutes:
            continue
        mm = g["v_marche_kmh"].notna() & (tm > 0)
        mc = g["v_course_kmh"].notna() & (tc > 0)
        if mm.sum() < 3 or mc.sum() < 3:
            continue
        vm = float(np.average(g["v_marche_kmh"][mm], weights=tm[mm]))
        vc = float(np.average(g["v_course_kmh"][mc], weights=tc[mc]))
        cm = (float(np.average(g["cout_marche"][mm & g["cout_marche"].notna()],
                               weights=tm[mm & g["cout_marche"].notna()]))
              if (mm & g["cout_marche"].notna()).sum() >= 3 else np.nan)
        cc = (float(np.average(g["cout_course"][mc & g["cout_course"].notna()],
                               weights=tc[mc & g["cout_course"].notna()]))
              if (mc & g["cout_course"].notna()).sum() >= 3 else np.nan)
        rows.append({
            "bande": band, "pente": float(g["pente_centre"].iloc[0]),
            "v_marche": vm, "v_course": vc,
            "cout_marche": cm, "cout_course": cc,
            "min_marche": float(tm.sum()), "min_course": float(tc.sum()),
            "gain_vitesse_pct": (vc / vm - 1) * 100 if vm else np.nan,
            "gain_cout_pct": (cc / cm - 1) * 100 if cm and not np.isnan(cm)
                             and not np.isnan(cc) else np.nan,
            "marche_gagnante": bool(vm >= vc * 0.97 and
                                    (np.isnan(cm) or np.isnan(cc) or cm <= cc)),
        })
    return pd.DataFrame(rows).sort_values("pente")


def read_walk_run(t: pd.DataFrame, noms: dict | None = None) -> str:
    noms = noms or {}
    if t.empty:
        return ("Seuil marche/course non mesurable : il faut au moins 20 min "
                "de marche ET 20 min de course dans une même bande, "
                "cumulées sur l'historique.")
    gagnantes = t[t["marche_gagnante"]]
    lignes = []
    for _, r in t.iterrows():
        nom = noms.get(r["bande"], r["bande"])
        verdict = ("<b style='color:#3E6146'>marcher</b>" if r["marche_gagnante"]
                   else "courir")
        cout = (f", coût {r['gain_cout_pct']:+.0f} %"
                if not np.isnan(r["gain_cout_pct"]) else "")
        lignes.append(
            f"<b>{nom}</b> · marche {r['v_marche']:.2f} contre course "
            f"{r['v_course']:.2f} km/h{cout} → {verdict}")
    txt = "<br>".join(lignes)
    if len(gagnantes):
        premiere = noms.get(gagnantes.iloc[0]["bande"], gagnantes.iloc[0]["bande"])
        txt += (f"<br><b>Bascule : {premiere}.</b> Au-delà, marcher ne te "
                "coûte pas de temps et t'économise.")
    else:
        txt += ("<br><span style='color:#5C665F'>Aucune bascule : courir reste "
                "plus rapide sur toutes les bandes mesurables.</span>")
    return txt


def read_transitions(recent: pd.DataFrame, past: pd.DataFrame,
                     prefix: str, label: str) -> tuple[pd.DataFrame, str]:
    """
    Profil de relance : vitesse ET coût cardiaque, par fenêtre.

    La version précédente n'affichait que l'écart de vitesse. C'était
    trompeur : relancer 12 % plus vite en montant de 8 battements n'est
    pas un progrès, c'est une dépense anticipée qui se paie en fin de
    course longue. Les deux mesures sont donc présentées côte à côte, sans
    être fondues en un score unique — un score cacherait précisément
    l'arbitrage que tu dois voir.
    """
    rows = []
    for _, _, w in WINDOWS:
        col, cfc, ccout = f"{prefix}_{w}", f"{prefix}_{w}_fc", f"{prefix}_{w}_cout"
        if col not in recent.columns:
            continue
        r, p = recent[col].dropna(), past[col].dropna()
        if len(r) < 2 or len(p) < 4:
            continue
        ligne = {
            "fenetre": {"relance": "0-200 m", "effort": "200-600 m",
                        "recuperation": "600-1500 m"}[w],
            "vitesse_pct": (r.mean() - 1) * 100,
            "vitesse_ref_pct": (p.mean() - 1) * 100,
            "n_recent": len(r), "n_ref": len(p),
        }
        for src, dest in [(cfc, "fc_delta"), (ccout, "cout_pct")]:
            if src in recent.columns and recent[src].notna().sum() >= 2:
                v = recent[src].dropna().mean()
                ligne[dest] = (v - 1) * 100 if dest == "cout_pct" else v
            else:
                ligne[dest] = np.nan

        # FC exprimée en POURCENTAGE, pour que les trois séries partagent un
        # seul axe. Superposer un axe en bpm et un axe en pourcentage rendait
        # le graphe illisible : Plotly n'aligne pas les groupes de barres
        # entre deux axes, elles se chevauchaient.
        #
        # La conversion ne nécessite aucune donnée supplémentaire. Par
        # construction, coût = FC / puissance et puissance ∝ allure, donc :
        #     FC_fenêtre / FC_référence = ratio_coût × ratio_allure
        rc = 1 + ligne["cout_pct"] / 100 if not np.isnan(ligne["cout_pct"]) else np.nan
        rv = 1 + ligne["vitesse_pct"] / 100
        ligne["fc_pct"] = (rc * rv - 1) * 100 if not np.isnan(rc) else np.nan
        rows.append(ligne)

    t = pd.DataFrame(rows)
    if t.empty:
        return t, (f"Relance {label} non mesurable : pas assez de plat "
                   "après les transitions sur ces parcours.")

    r0 = t.iloc[0]
    v, c = r0["vitesse_pct"], r0.get("cout_pct", np.nan)
    # La FC en bpm n'est disponible que sur les imports récents. À défaut,
    # on utilise le pourcentage, qui se déduit du coût et de l'allure.
    fbpm = r0.get("fc_delta", np.nan)
    fpct = r0.get("fc_pct", np.nan)
    f = fbpm if not np.isnan(fbpm) else fpct
    unite_fc = "bpm" if not np.isnan(fbpm) else "%"
    seuil = 3.0 if not np.isnan(fbpm) else 2.0
    vit = f"<b>{v:+.0f} %</b> de vitesse"

    # LE SIGNAL « JE POUSSE PLUS FORT » EST PORTÉ PAR LA FC ABSOLUE, PAS
    # PAR LE COÛT.
    #
    # Vérifié en simulation : relancer 12 % plus vite avec 6 battements de
    # plus fait BAISSER le coût cardiaque de 3 %, puisque la puissance
    # produite croît plus vite que la fréquence. Le coût dit « tu es plus
    # efficace », ce qui est exact — mais tu cours quand même cette portion
    # à une intensité supérieure, et c'est cela qui se paie sur une course
    # longue. Le verdict s'appuie donc d'abord sur le delta de FC.
    seuil_fc = seuil
    if np.isnan(f):
        msg = f"À 200 m {label} : {vit}. Fréquence cardiaque non mesurable."
    elif v > 2 and f > seuil_fc:
        msg = (f"À 200 m {label} : {vit}, mais <b>{f:+.0f} {unite_fc}</b> de FC. Tu "
               "relances en montant d'intensité — acceptable sur une sortie "
               "courte, coûteux sur un format long où cette dépense se "
               "rattrape en fin de course.")
    elif v > 2 and f <= seuil_fc:
        msg = (f"À 200 m {label} : {vit} pour seulement <b>{f:+.0f} {unite_fc}</b> de FC. "
               "Vrai gain d'efficacité : tu vas plus vite sans monter "
               "d'intensité.")
    elif abs(v) <= 2 and f < -seuil_fc:
        msg = (f"À 200 m {label} : allure inchangée pour <b>{f:+.0f} {unite_fc}</b> de FC. "
               "Gain d'économie pur — même relance, moins cher.")
    elif v < -2 and f > seuil_fc:
        msg = (f"À 200 m {label} : {vit} et <b>{f:+.0f} {unite_fc}</b> de FC. Tu subis la "
               "relance des deux côtés : plus lent ET plus cher. C'est là "
               "qu'il y a du temps à récupérer.")
    elif v < -2:
        msg = (f"À 200 m {label} : {vit} à intensité stable. La bosse te "
               "coûte de l'allure, pas de l'intensité.")
    else:
        msg = f"À 200 m {label} : {vit}, FC {f:+.0f} {unite_fc}. Rien de marquant."
    if not np.isnan(c):
        msg += (f" <span style='color:#5C665F'>Coût cardiaque "
                f"{c:+.0f} %.</span>")

    dernier = t.iloc[-1]["vitesse_pct"]
    if len(t) > 1:
        msg += (f"<br><span style='color:#5C665F'>À 1,5 km : "
                f"{dernier:+.0f} % de vitesse — "
                + ("tu as retrouvé ton allure." if dernier > -3
                   else "la bosse se paie encore.") + "</span>")
    return t, msg


def read_drift(recent: pd.DataFrame, past: pd.DataFrame) -> str:
    r = recent["drift"].dropna() if "drift" in recent.columns else pd.Series(dtype=float)
    p = past["drift"].dropna() if "drift" in past.columns else pd.Series(dtype=float)
    if len(r) < 2 or len(p) < 4:
        return "Découplage cardiaque non mesurable sur cet échantillon."
    mr, mp = float(r.mean()), float(p.mean())
    seuil = 1.96 * float(p.std(ddof=1)) / np.sqrt(len(r))
    pct = (mr - 1) * 100
    etat = ("bonne tenue" if mr < 1.05 else
            "usure modérée" if mr < 1.12 else "usure marquée")
    tete = (f"<b>{pct:+.1f} %</b> · {etat}<br>"
            "<span style='color:#5C665F'>À effort mécanique égal et à "
            "terrain égal, ton cœur bat "
            f"{abs(pct):.0f} % {'plus vite' if pct > 0 else 'moins vite'} "
            "en seconde moitié de sortie qu'en première. "
            "Sous 5 %, la fatigue ne se voit pas encore ; au-delà de 12 %, "
            "elle pèse sur une course longue.</span>")
    if abs(mr - mp) <= seuil:
        return tete + f"<br>Stable par rapport à ta référence ({(mp - 1) * 100:+.1f} %)."
    sens = "amélioration" if mr < mp else "dégradation"
    return tete + (f"<br>En {sens} : ta référence était à "
                   f"{(mp - 1) * 100:+.1f} %.")


# TRIMP produit par heure d'endurance fondamentale, à 65 % de réserve.
# 0,65 x 0,64 x exp(1,92 x 0,65) x 60 minutes.
TRIMP_PAR_HEURE_EF = 87.0


def en_heures_ef(trimp: float) -> str:
    """
    Traduit un TRIMP en heures d'endurance fondamentale équivalentes.

    Le TRIMP est une unité sans référence physique : 535 ne dit rien à
    personne. Rapporté à ce que produit une heure en endurance
    fondamentale, il devient une durée — quelque chose qu'on ressent.
    """
    if trimp is None or np.isnan(trimp):
        return "—"
    h = trimp / TRIMP_PAR_HEURE_EF
    return f"{int(h)}h{int(round((h % 1) * 60)):02d}"


def read_load(load: pd.DataFrame) -> str:
    if load.empty or "ratio_ac" not in load.columns:
        return "Charge non calculable : il faut la fréquence cardiaque."
    ratio = load["ratio_ac"].dropna()
    if ratio.empty:
        return "Pas assez de semaines pour un ratio aigu/chronique."
    r = float(ratio.iloc[-1])
    moy4 = (float(load["chronique"].iloc[-1]) if "chronique" in load.columns
            else float(load["total"].tail(5).head(4).mean()))
    cette = float(load["total"].iloc[-1])
    detail = (f"<span style='color:#5C665F'>{cette:.0f} cette semaine contre "
              f"{moy4:.0f} en moyenne sur les 4 précédentes — soit "
              f"l'équivalent de {en_heures_ef(cette)} en endurance "
              f"fondamentale contre {en_heures_ef(moy4)}. La semaine en "
              "cours est incomplète tant qu'elle n'est pas finie, le ratio "
              "monte donc mécaniquement en fin de semaine.</span>"
              if not np.isnan(moy4) else "")
    if r > 1.5:
        return (f"<b>×{r:.2f}</b> · montée de charge rapide<br>{detail}<br>"
                "<span style='color:#5C665F'>Au-delà de ×1,5, l'augmentation "
                "brutale est le facteur de risque de blessure le mieux "
                "documenté — plus que le volume absolu.</span>")
    if r < 0.8:
        return (f"<b>×{r:.2f}</b> · charge en baisse<br>{detail}<br>"
                "<span style='color:#5C665F'>Normal en semaine d'assimilation "
                "ou d'affûtage.</span>")
    return (f"<b>×{r:.2f}</b> · progression régulière<br>{detail}<br>"
            "<span style='color:#5C665F'>La zone ×0,8 à ×1,3 correspond à une "
            "montée en charge que le corps absorbe.</span>")
