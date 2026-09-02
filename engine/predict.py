"""
predict.py — Prédiction de temps et plan de course.

Choix de méthode, et pourquoi il est volontairement conservateur.

Avec 20 à 100 sorties, un modèle de machine learning à 8 features
surapprend garanti : tu obtiendras un R² de 0,95 en apprentissage et des
prédictions absurdes sur une course jamais vue. On utilise donc un modèle
paramétrique à DEUX paramètres, calibré sur ton historique :

    T = a · Deq^b        (loi d'endurance, Riegel 1981)

où Deq est la distance équivalente plat obtenue en intégrant le coût
énergétique de Minetti sur le profil réel, et non par la règle de pouce
"distance + D+/100" qui sous-estime lourdement les fortes pentes et ignore
totalement le coût de la descente raide.

b vaut typiquement 1,06 sur route et 1,10 à 1,18 en trail long. Un b estimé
hors de [1,00 ; 1,30] signale un historique non représentatif, pas un
athlète exceptionnel : le modèle le signale au lieu de le masquer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import physio
from .metrics import prepare
from .physio import CR_FLAT

MIN_ACTIVITIES = 6
B_PLAUSIBLE = (1.00, 1.30)


def km_effort(distance_km: float, d_plus: float) -> float:
    """
    Kilomètre-effort : distance + D+/100. L'unité usuelle du trail.

    Pourquoi elle a été retenue contre l'intégration de Minetti, après
    comparaison sur les données réelles. Sept formules ont été mises en
    concurrence par validation croisée un-contre-tous sur 25 sorties
    longues. Erreur type : 14,0 % pour Minetti seul, 13,0 % pour le
    km-effort, 12,2 % pour Minetti augmenté d'un terme excentrique en
    D-/100. L'écart entre la meilleure et la pire n'est pas significatif —
    test apparié t = 1,23 — donc rien ne justifie une formule plus
    compliquée que celle que tout le monde utilise.

    Note importante : le km-effort ne crédite pas la descente, alors que
    descendre à -10 % coûte moins cher que courir à plat. Il surestime donc
    les parcours très descendants. En sens inverse, il ignore le coût
    musculaire excentrique, qui est bien réel sur 3 000 m de D-. Les deux
    biais se compensent en partie, ce qui explique qu'il tienne aussi bien.

    Le chiffre à retenir de toute cette comparaison n'est pas le classement
    des formules, c'est le NIVEAU d'erreur : 12 à 14 %, soit plus ou moins
    une heure sur une prédiction à neuf heures.
    """
    return float(distance_km) + float(d_plus or 0) / 100


def flat_equivalent_distance(d: pd.DataFrame) -> float:
    """
    Distance équivalente plat, en km, par intégration du coût de Minetti.

    Deq = Σ dist_i · Cr(pente_i) / Cr(0)

    Sur un 70 km / 2000 D+, la règle "distance + D+/100" donne 90 km.
    Cette intégration donne un résultat différent selon que le D+ est
    concentré sur deux cols ou étalé sur vingt bosses — ce qui est
    précisément la nuance que l'ITRA reconnaît ne pas capturer.
    """
    dist = d["dist"].to_numpy()
    cr = physio.cost_running(d["slope"].to_numpy())
    return float(np.sum(dist * cr / physio.CR_FLAT) / 1000)


def build_history(activities: list[dict]) -> pd.DataFrame:
    """
    activities : liste de dicts {date, deq_km, duration_h, hrr_mean, sport}.
    Retourne le DataFrame d'entraînement du modèle.
    """
    df = pd.DataFrame(activities)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def fit_endurance_model(hist: pd.DataFrame, months: int = 24,
                        target_deq: float | None = None,
                        min_deq: float | None = None,
                        floor_ratio: float = 6.0,
                        envelope_quantile: float = 0.35) -> dict:
    """
    Calibre T = a · Deq^b sur les sorties TRAIL, dans la plage de distance
    pertinente pour la cible.

    DEUX ERREURS DE CONCEPTION CORRIGÉES ICI, l'une après l'autre.

    Erreur 1 — mélanger route et trail avec une pente commune.
    L'idée était d'estimer l'exposant sur toutes les courses (beaucoup de
    points) et le niveau sur le trail seul (peu de points), en neutralisant
    l'écart de niveau par un effet fixe de terrain. Les données l'ont
    réfutée : exposant 0,815 sur route contre 0,994 sur trail. Ce ne sont
    pas deux niveaux autour d'une pente commune, ce sont deux régimes
    distincts. Les footings route sont homogènes et faciles ; les trails
    mettent en situation de course. Une pente commune ne décrit ni l'un ni
    l'autre. On n'ajuste donc plus que sur le trail.

    Erreur 2 — ajuster sur toute la plage de distances.
    L'exposant estimé sur le trail monte régulièrement avec le seuil bas :

        deq >  3 km   n=67   b = 0,994
        deq > 10 km   n=52   b = 1,058
        deq > 15 km   n=27   b = 1,111
        deq > 20 km   n=21   b = 1,153
        deq > 25 km   n=14   b = 1,213

    Une dérive aussi nette signifie qu'une loi de puissance unique ne
    convient pas sur toute l'étendue. En bas de plage, les sorties sont des
    entraînements à allure choisie et non des efforts limites : elles ne
    relèvent pas de la loi de Riegel, qui décrit des performances
    maximales. En haut de plage, ce sont des courses.

    On ajuste donc dans une fenêtre basse fixée par la cible : par défaut au
    huitième environ de la distance visée. Prédire un 90 km équivalent se
    fait sur les sorties de plus de 15 km, pas sur les footings de 6 km.
    Le compromis est explicite — plus le seuil est haut, plus l'exposant
    est pertinent mais moins il y a de points. Le nombre retenu est
    toujours rapporté pour que tu juges.
    """
    if hist.empty:
        return {"ok": False, "reason": "Historique vide."}

    d = hist.copy()
    # Unité de référence : le km-effort. Calculé ici si absent de la base.
    if "ke_km" not in d.columns or d["ke_km"].isna().all():
        if {"distance_km", "d_plus"}.issubset(d.columns):
            d["ke_km"] = d["distance_km"] + d["d_plus"].fillna(0) / 100
        else:
            d["ke_km"] = d.get("deq_km")
    d = d.dropna(subset=["ke_km", "duration_h"])
    d["date"] = pd.to_datetime(d["date"], errors="coerce", utc=True).dt.tz_localize(None)
    d = d[(d["ke_km"] > 0) & (d["duration_h"] > 0)]

    cutoff = d["date"].max() - pd.DateOffset(months=months)
    d = d[d["date"] >= cutoff]

    # Seules les séances à effort continu. Le fractionné, l'indéterminé et
    # les sorties trop courtes ont un temps total qui ne décrit aucun
    # effort soutenu — et ce sont eux qui écrasaient l'exposant à 0,84.
    # EXCLUSION RESTREINTE AU FRACTIONNÉ ET AUX SORTIES TROP COURTES.
    #
    # La version précédente ne gardait que « continu », « variable »,
    # « sortie longue » et « course », écartant donc « indéterminée ».
    # Mesuré sur la base réelle : au-delà du seuil de distance, les six
    # séances « indéterminée » sont de vraies sorties longues où la
    # détection n'a pas pu trancher — fréquence cardiaque manquante ou trop
    # peu de blocs courus. C'est la détection qui a échoué, pas la séance.
    #
    # Les exclure coûtait 6 points sur 32 et dégradait l'erreur de
    # prédiction de 12 à 13 %. Le seuil de distance suffit à écarter le
    # fractionné, qui dépasse rarement 19 km-effort ; on ne conserve donc
    # l'exclusion explicite que pour les cas où le temps total ne décrit
    # manifestement aucun effort soutenu.
    if "session_type" in d.columns:
        d = d[~d["session_type"].isin(["fractionné", "trop courte"])]

    # La DÉCLARATION fait foi. `terrain`, déduit du dénivelé par kilomètre,
    # n'est plus qu'un contrôle de cohérence : c'est toi qui sais ce que tu
    # as couru, pas un seuil de 15 m/km.
    all_runs = d.copy()
    if "discipline" in d.columns and (d["discipline"] == "trail").sum() >= MIN_ACTIVITIES:
        d = d[d["discipline"] == "trail"]
        base_label = "trail déclaré"
    elif "terrain" in d.columns and (d["terrain"] == "trail").sum() >= MIN_ACTIVITIES:
        d = d[d["terrain"] == "trail"]
        base_label = "trail déduit du D+"
    else:
        base_label = "toutes courses"

    if min_deq is None:
        min_deq = max(10.0, target_deq / floor_ratio) if target_deq else 12.0
    kept = d[d["ke_km"] >= min_deq]

    # Si le seuil vide l'échantillon, on le rabaisse jusqu'à retrouver de
    # quoi ajuster, en signalant que la plage n'est plus celle voulue.
    relaxed = False
    while len(kept) < MIN_ACTIVITIES + 4 and min_deq > 6:
        min_deq *= 0.75
        kept = d[d["ke_km"] >= min_deq]
        relaxed = True

    if len(kept) < MIN_ACTIVITIES:
        return {"ok": False, "reason":
                f"{len(kept)} sortie(s) trail au-dessus de {min_deq:.0f} "
                f"km-effort sur {months} mois. Il en faut au moins "
                f"{MIN_ACTIVITIES}."}

    x = np.log(kept["ke_km"].to_numpy())
    y = np.log(kept["duration_h"].to_numpy())
    if float(x.max() - x.min()) < 0.5:
        return {"ok": False, "reason":
                "Plage de distances trop étroite pour estimer un exposant."}

    b, log_a = np.polyfit(x, y, 1)
    resid = y - (b * x + log_a)
    log_a_env = log_a + float(np.quantile(resid, envelope_quantile))

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    # Erreur de prédiction réelle, par validation croisée un-contre-tous.
    # C'est elle qui doit piloter l'intervalle affiché, pas le résidu
    # d'ajustement — un modèle prédit toujours bien les points qu'il a vus.
    if len(x) > 8:
        errs = [y[i] - np.polyval(np.polyfit(np.delete(x, i), np.delete(y, i), 1), x[i])
                for i in range(len(x))]
        cv_error = float(np.sqrt(np.mean(np.square(errs))))
    else:
        cv_error = float("nan")

    # Robustesse : sensibilité au retrait d'un point.
    if len(x) > 8:
        jack = [float(np.polyfit(np.delete(x, i), np.delete(y, i), 1)[0])
                for i in range(len(x))]
        b_span = (round(min(jack), 3), round(max(jack), 3))
    else:
        b_span = None

    # Écart route / trail, à titre informatif : il quantifie ce que le sol
    # technique coûte au-delà de ce que Minetti modélise.
    penalty = None
    col_ref = "discipline" if "discipline" in all_runs.columns else "terrain"
    if col_ref in all_runs.columns:
        road = all_runs[(all_runs[col_ref] != "trail")
                        & (all_runs["ke_km"] >= min_deq)]
        if len(road) >= 6:
            r_road = np.log(road["duration_h"].to_numpy()) - b * np.log(road["ke_km"].to_numpy())
            penalty = float(np.exp(log_a_env - np.quantile(r_road, envelope_quantile)))

    return {
        "ok": True,
        "a": float(np.exp(log_a_env)),
        "b": float(b),
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n": int(len(kept)),
        "min_deq": float(min_deq),
        "relaxed": relaxed,
        "base": base_label,
        "b_span": b_span,
        "terrain_penalty": penalty,
        "resid_sd": float(np.std(resid)),
        "cv_error": cv_error,
        "deq_range": (float(kept["ke_km"].min()), float(kept["ke_km"].max())),
        "b_plausible": B_PLAUSIBLE[0] <= b <= B_PLAUSIBLE[1],
    }


def predict_time(deq_km: float, model: dict, pacing: float = 1.0) -> dict:
    """
    Temps estimé en heures, avec intervalle.

    pacing : 1,0 = allure d'enveloppe (course). 1,10 = gestion prudente.
    L'intervalle vient de la dispersion résiduelle du modèle, pas d'une
    invention : il reflète l'hétérogénéité réelle de ton historique.
    """
    if not model.get("ok"):
        return {"ok": False, "reason": model.get("reason")}

    t = model["a"] * deq_km ** model["b"] * pacing

    # L'écart-type des résidus d'AJUSTEMENT sous-estime l'incertitude réelle :
    # le modèle a vu ces points. La validation croisée un-contre-tous donne
    # 12 à 14 % sur cet historique. On retient un plancher de 12 %, sinon
    # l'intervalle affiché serait faussement rassurant.
    sd = max(float(model.get("cv_error", 0.0)), float(model["resid_sd"]), 0.12)
    lo, hi = t * np.exp(-sd), t * np.exp(sd)

    dmin, dmax = model["deq_range"]
    extrapolation = deq_km > dmax * 1.25 or deq_km < dmin * 0.75

    return {
        "ok": True,
        "hours": float(t),
        "low": float(lo),
        "high": float(hi),
        "extrapolation": bool(extrapolation),
        "note": (
            f"Extrapolation hors historique ({dmin:.0f}-{dmax:.0f} km eq.) : "
            "l'incertitude réelle dépasse l'intervalle affiché."
            if extrapolation else ""
        ),
    }


def segment_course(d: pd.DataFrame, min_len_m: float = 400) -> pd.DataFrame:
    """
    Découpe automatique d'un parcours en segments homogènes.

    Critère : changement de classe de terrain (montée / plat / descente)
    maintenu sur au moins min_len_m. Le seuil de longueur évite de produire
    quarante micro-segments sur un profil ondulant, qui seraient illisibles
    et statistiquement vides.
    """
    slope = d["slope"].to_numpy()
    kind = np.where(slope > 0.05, "montée", np.where(slope < -0.05, "descente", "plat"))
    cum = d["cum_dist"].to_numpy()

    segments, start, current = [], 0, kind[0]
    for i in range(1, len(kind)):
        if kind[i] != current:
            if cum[i] - cum[start] >= min_len_m:
                segments.append((start, i, current))
                start, current = i, kind[i]
            # sinon on absorbe le micro-changement dans le segment courant
    segments.append((start, len(kind) - 1, current))

    rows = []
    for a, b, k in segments:
        sub = d.iloc[a:b + 1]
        dist_m = float(sub["dist"].sum())
        if dist_m < min_len_m / 2:
            continue
        pente = float(np.average(sub["slope"], weights=np.maximum(sub["dist"], 1e-9)))
        # On réétiquette sur la pente moyenne du segment : le premier point
        # d'un segment est calculé sur une fenêtre incomplète et n'est pas
        # représentatif.
        label = "montée" if pente > 0.05 else "descente" if pente < -0.05 else "plat"
        rows.append({
            "km_debut": cum[a] / 1000,
            "km_fin": cum[b] / 1000,
            "distance_km": dist_m / 1000,
            "d_plus": float(sub["d_plus"].sum()),
            "d_minus": float(sub["d_minus"].sum()),
            "pente_moy": pente,
            "pente_max": float(sub["slope"].max()),
            "pente_min": float(sub["slope"].min()),
            "type": label,
            "deq_km": flat_equivalent_distance(sub),
        })
    return pd.DataFrame(rows)


def match_history(segment: dict, hist_bins: pd.DataFrame,
                  tolerance: float = 0.03) -> pd.DataFrame:
    """
    Retrouve dans l'historique les bandes de pente comparables à ce segment.

    hist_bins : concaténation des tables by_slope_bin() de chaque sortie,
    enrichie des colonnes date / distance_km / d_plus / duration_h pour que
    tu puisses relier chaque référence à une sortie précise et la confronter
    à ton ressenti — ce que tu demandais explicitement.
    """
    if hist_bins.empty:
        return hist_bins
    lo, hi = segment["pente_moy"] - tolerance, segment["pente_moy"] + tolerance
    m = hist_bins["pente_centre"].between(lo, hi)
    return hist_bins[m].sort_values("date", ascending=False)


def band_profile(bins: pd.DataFrame, activity_ids,
                min_dist_km: float = 0.2) -> tuple:
    """
    Ta vitesse par pente, telle que mesurée. Renvoie (pentes, vitesses).

    Sert à répartir le temps le long d'un parcours : c'est la seule source
    qui sache que tu montes à 6 km/h à 12 % et descends à 13 km/h à −9 %.
    """
    if bins.empty or "pente_centre" not in bins.columns:
        return np.array([]), np.array([])
    b = bins.copy()
    b["activity_id"] = b["activity_id"].astype(str)
    b = b[b["activity_id"].isin(set(map(str, activity_ids)))]
    if "distance_km" not in b.columns:
        b["distance_km"] = b["vitesse_kmh"] * b["temps_min"] / 60
    b = b[b["vitesse_kmh"].notna() & (b["distance_km"] > min_dist_km)]
    if b.empty:
        return np.array([]), np.array([])
    profil = {}
    for band, g in b.groupby("bande", observed=True):
        if len(g) < 3 or g["distance_km"].sum() < 2:
            continue
        profil[float(g["pente_centre"].iloc[0])] = float(
            np.average(g["vitesse_kmh"], weights=g["distance_km"]))
    if len(profil) < 3:
        return np.array([]), np.array([])
    pentes = np.array(sorted(profil))
    return pentes, np.array([profil[p] for p in pentes])


def expected_pace(d: pd.DataFrame, pentes, vitesses,
                  base_kmh: float | None = None) -> np.ndarray:
    """
    Temps attendu point par point, en secondes, avant calage sur le total.

    On travaille au POINT et non au segment : la pente moyenne d'un
    tronçon de dix kilomètres ne dit rien de sa difficulté réelle, alors
    que la distribution de ses pentes la porte entièrement. Un tronçon
    plat-descendant et un tronçon montant-descendant peuvent partager la
    même pente moyenne et demander des temps très différents.
    """
    slope = d["slope"].to_numpy()
    dist = d["dist"].to_numpy()
    if len(pentes) >= 3:
        v = np.interp(slope, pentes, vitesses)
    else:
        ref = base_kmh or 10.0
        v = ref * CR_FLAT / physio.cost_running(slope)
    v = np.clip(v, 1.0, 30.0)
    return dist / 1000 / v * 3600


def split_at_marks(d: pd.DataFrame, marques: list[dict], total_h: float,
                   bins: pd.DataFrame, activity_ids,
                   drift: float | None = None,
                   base_kmh: float | None = None,
                   depart: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Temps de passage aux repères que TU places sur le parcours.

    marques : [{"nom": "Ravito 1", "km": 12.5}, ...]

    Trois principes, les mêmes que pour le total.

    1. LA FORME VIENT DE TON PROFIL par bande de pente, appliqué point par
       point puis cumulé entre repères.
    2. LE TOTAL VIENT DU MODÈLE D'ENDURANCE. Les temps par tronçon sont
       renormalisés pour que leur somme soit exactement le temps prédit —
       ton profil est mesuré à intensité d'entraînement, il ne sait rien de
       ce qui se passe à la huitième heure.
    3. LA DÉRIVE RÉPARTIT, elle n'ajoute pas. L'exposant d'endurance porte
       déjà la fatigue ; la dérive cardiaque sert seulement à ralentir la
       fin par rapport au début, à total constant.
    """
    if d.empty or not marques:
        return pd.DataFrame()

    cum = d["cum_dist"].to_numpy() / 1000
    total_km = float(cum[-1])

    # Bornes : départ, repères triés et dédoublonnés, arrivée.
    kms = sorted({round(float(m["km"]), 3) for m in marques
                  if 0 < float(m["km"]) < total_km})
    noms = {round(float(m["km"]), 3): str(m.get("nom") or "Repère")
            for m in marques}
    bornes = [0.0] + kms + [total_km]
    labels = ["Départ"] + [noms[k] for k in kms] + ["Arrivée"]

    pentes, vitesses = band_profile(bins, activity_ids)
    sec = expected_pace(d, pentes, vitesses, base_kmh)

    # Modulation : plus lent à la fin, à somme constante.
    amp = float(np.clip((drift or 1.0) - 1.0, 0.0, 0.30))
    frac = cum / max(total_km, 1e-9)
    sec = sec * (1.0 + amp * (frac - 0.5))

    facteur = total_h * 3600 / max(sec.sum(), 1e-9)
    sec = sec * facteur

    dplus = d["d_plus"].to_numpy()
    dmoins = d["d_minus"].to_numpy()
    dist = d["dist"].to_numpy() / 1000

    lignes, cumule = [], 0.0
    for i in range(len(bornes) - 1):
        a, b = bornes[i], bornes[i + 1]
        m = (cum > a) & (cum <= b) if i else (cum <= b)
        if m.sum() == 0:
            continue
        t_h = float(sec[m].sum() / 3600)
        cumule += t_h
        km_seg = float(dist[m].sum())
        dp = float(dplus[m].sum())
        lignes.append({
            "de": labels[i], "a": labels[i + 1],
            "km_debut": a, "km_fin": b,
            "distance_km": km_seg,
            "d_plus": dp, "d_minus": float(dmoins[m].sum()),
            "ke_km": km_effort(km_seg, dp),
            "temps_h": t_h,
            "cumule_h": cumule,
            "vitesse_kmh": km_seg / t_h if t_h > 0 else np.nan,
            "heure": (depart + pd.Timedelta(hours=cumule)).strftime("%H:%M")
                     if depart is not None else None,
        })
    t = pd.DataFrame(lignes)
    if not t.empty:
        t["source"] = "profil" if len(pentes) >= 3 else "modèle"
    return t


def race_plan(course: pd.DataFrame, bins: pd.DataFrame, activity_ids,
              total_h: float, drift: float | None = None,
              base_kmh: float | None = None) -> pd.DataFrame:
    """
    Temps par segment, à partir de TON profil par bande de pente.

    ARCHITECTURE, ET POURQUOI ELLE EST HYBRIDE

    Trois façons de procéder, dont une seule tient.

    1. Modèle global seul. Donne un temps total crédible mais aucune
       répartition : incapable de dire à quelle heure tu passeras au col.
    2. Profil par bande seul. Donne une répartition fine, mais un total
       faux — ton historique par bande est mesuré à intensité
       d'entraînement, sur des sorties de deux heures, et ne contient
       aucune information sur ce qui se passe à la huitième.
    3. Les deux : les bandes donnent la FORME, le modèle d'endurance donne
       le TOTAL. C'est ce qui est implémenté.

    LA FATIGUE N'EST PAS AJOUTÉE DEUX FOIS

    Point qui mérite d'être explicite. L'exposant d'endurance b EST un
    terme de fatigue : b = 1,133 signifie que doubler la distance
    équivalente coûte 2^1,133 = 2,19 fois le temps, soit 9,5 % de
    ralentissement par doublement. Ajouter par-dessus un facteur de fatigue
    calibré sur la dérive cardiaque compterait deux fois le même
    phénomène et surestimerait le temps.

    La dérive ne sert donc PAS à allonger le total. Elle sert uniquement à
    RÉPARTIR le ralentissement le long du parcours : à total constant, un
    coureur qui dérive peu part et finit à allure proche, un coureur qui
    dérive beaucoup part vite et finit lentement. C'est la seule chose que
    la dérive puisse légitimement dire ici.
    """
    if course.empty:
        return pd.DataFrame()

    b = bins.copy()
    if not b.empty:
        b["activity_id"] = b["activity_id"].astype(str)
        b = b[b["activity_id"].isin(set(map(str, activity_ids)))]
        if "distance_km" not in b.columns:
            b["distance_km"] = b["vitesse_kmh"] * b["temps_min"] / 60

    # ── 1. Vitesse de référence par bande, depuis l'historique ────────────
    profil = {}
    if not b.empty:
        for band, g in b.groupby("bande", observed=True):
            m = g["vitesse_kmh"].notna() & (g["distance_km"] > 0.2)
            if m.sum() < 3 or g["distance_km"][m].sum() < 3:
                continue
            profil[float(g["pente_centre"].iloc[0])] = float(
                np.average(g["vitesse_kmh"][m], weights=g["distance_km"][m]))

    pentes = np.array(sorted(profil)) if profil else np.array([])
    vitesses = np.array([profil[p] for p in pentes]) if profil else np.array([])

    def v_attendue(pente: float) -> tuple[float, str]:
        """Vitesse brute pour une pente, par interpolation du profil."""
        if len(pentes) >= 3:
            return float(np.interp(pente, pentes, vitesses)), "profil"
        # Repli : coût de Minetti appliqué à une vitesse de plat
        ref = base_kmh or 10.0
        return float(ref * CR_FLAT / physio.cost_running(pente)), "modèle"

    rows = []
    for _, seg in course.iterrows():
        v, src = v_attendue(float(seg["pente_moy"]))
        rows.append({**seg.to_dict(), "v_brute_kmh": v, "source": src})
    plan = pd.DataFrame(rows)

    # ── 2. Calage sur le total du modèle d'endurance ──────────────────────
    t_brut = float((plan["distance_km"] / plan["v_brute_kmh"].clip(lower=0.5)).sum())
    if t_brut <= 0:
        return plan
    facteur = total_h / t_brut
    plan["facteur_global"] = facteur

    # ── 3. Répartition du ralentissement selon la dérive ──────────────────
    #
    # À total constant, on module l'allure du début vers la fin. Le profil
    # est linéaire en fraction de parcours, d'amplitude proportionnelle à
    # la dérive observée, et normalisé pour que le temps total soit
    # exactement celui du modèle.
    amp = 0.0
    if drift is not None and not np.isnan(drift):
        # Une dérive de +18 % sur une sortie donne une amplitude de 18 %
        # entre début et fin, bornée à 30 % pour ne pas extrapoler
        # aveuglément d'une sortie de 2 h vers une course de 9 h.
        amp = float(np.clip(drift - 1.0, 0.0, 0.30))

    frac = (plan["km_fin"] / plan["km_fin"].max()).to_numpy()
    modul = 1.0 + amp * (frac - 0.5)          # < 1 au départ, > 1 à l'arrivée
    plan["modulation"] = modul

    v_finale = plan["v_brute_kmh"] * facteur / modul
    plan["vitesse_prevue_kmh"] = v_finale
    plan["temps_h"] = plan["distance_km"] / v_finale.clip(lower=0.5)

    # Renormalisation exacte : la modulation déplace du temps, elle n'en
    # crée pas.
    plan["temps_h"] *= total_h / float(plan["temps_h"].sum())
    plan["temps_cumule_h"] = plan["temps_h"].cumsum()
    plan["vitesse_prevue_kmh"] = plan["distance_km"] / plan["temps_h"]

    plan["mode"] = np.where(plan["pente_moy"] > 0.15, "marche", "course")
    plan["allure_min_km"] = 60 / plan["vitesse_prevue_kmh"].clip(lower=0.5)
    return plan


def build_plan(course: pd.DataFrame, hist_bins: pd.DataFrame,
               model: dict, fallback_gap_kmh: float,
               walk_threshold: float = 0.15) -> pd.DataFrame:
    """
    Plan de course segment par segment.

    Pour chaque segment : allure attendue dérivée de tes bandes de pente
    historiques (et non d'une moyenne globale), décision marche/course,
    temps estimé. Cumul en fin de table.
    """
    rows = []
    for _, seg in course.iterrows():
        matches = match_history(seg.to_dict(), hist_bins)
        if len(matches) >= 3:
            v_kmh = float(np.average(matches["vitesse_kmh"],
                                     weights=matches["temps_min"]))
            source = f"{len(matches)} référence(s)"
        else:
            cr = physio.cost_running(seg["pente_moy"])
            v_kmh = fallback_gap_kmh * physio.CR_FLAT / cr
            source = "modèle (pas de référence)"

        mode = "marche" if seg["pente_moy"] > walk_threshold else "course"
        t_h = seg["distance_km"] / max(v_kmh, 0.5)
        rows.append({
            **seg.to_dict(),
            "vitesse_prevue_kmh": v_kmh,
            "mode": mode,
            "temps_h": t_h,
            "source": source,
        })
    plan = pd.DataFrame(rows)
    plan["temps_cumule_h"] = plan["temps_h"].cumsum()
    return plan


def fmt_hours(h: float) -> str:
    """3.7 -> '3h42'."""
    if h is None or np.isnan(h):
        return "—"
    total = int(round(h * 60))
    return f"{total // 60}h{total % 60:02d}"
