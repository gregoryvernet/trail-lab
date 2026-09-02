"""
metrics.py — Du DataFrame brut aux indicateurs.

Trois corrections structurantes par rapport à la version précédente :

1. LISSAGE. La pente point-à-point sur des pas de 2-3 m avec une altitude
   barométrique résolue au mètre est du bruit pur : on obtient couramment
   des pentes de ±40 % sur du plat. Tous les indicateurs construits dessus
   (technicité, buckets >10/20/30 %, cadence vs pente, coût vs pente) étaient
   donc non interprétables. On lisse l'altitude puis on calcule la pente sur
   une fenêtre glissante de 30 m de distance parcourue.

2. PONDÉRATION PAR LE TEMPS. np.mean() sur un vecteur de points suppose un
   échantillonnage régulier. Polar en mode "smart recording" ne l'est pas.
   Toutes les moyennes sont ici pondérées par dt.

3. COMPARAISON À TERRAIN ÉGAL. Comparer la 1re et la 2e moitié d'une sortie
   est faux si le profil n'est pas symétrique. La dérive est calculée par
   classe de pente, puis agrégée.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import physio

EARTH_R = 6_371_000.0

# Bandes de pente, découpage nommé plutôt que numérique. Les seuils
# suivent les paliers où le geste change réellement : au-delà de 7 % la
# foulée se raccourcit, au-delà de 12 % la marche devient compétitive,
# au-delà de 17 % elle s'impose presque toujours.
SLOPE_BINS = [-0.45, -0.17, -0.12, -0.07, -0.03,
              0.03, 0.07, 0.12, 0.17, 0.45]
SLOPE_LABELS = [
    "Descente raide", "Descente difficile", "Descente dure",
    "Descente moyenne", "Plat",
    "Montée moyenne", "Montée dure", "Montée difficile", "Montée raide",
]

# Bornes affichées, pour que le nom ne soit jamais ambigu.
SLOPE_RANGES = {
    "Descente raide": "au-delà de −17 %", "Descente difficile": "−12 à −17 %",
    "Descente dure": "−7 à −12 %", "Descente moyenne": "−3 à −7 %",
    "Plat": "−3 à +3 %",
    "Montée moyenne": "3 à 7 %", "Montée dure": "7 à 12 %",
    "Montée difficile": "12 à 17 %", "Montée raide": "au-delà de 17 %",
}


def haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(x)
    return s.rolling(window, center=True, min_periods=1).median().to_numpy()


# Plafonds anti-décrochage GPS, par sport. Au-delà, c'est une
# téléportation — en dessous, on ampute des données valides.
MAX_SPEED = {"trail": 8.0, "route": 8.0, "rando": 6.0,
             "velo": 25.0, "vtt": 25.0, "velo_route": 30.0, "gravel": 28.0}


def max_speed_for(sport: str | None) -> float:
    return MAX_SPEED.get(str(sport or "").lower(), 8.0)


def guess_max_speed(df: pd.DataFrame) -> float:
    """
    Plafond déduit de la trace elle-même, quand le sport n'est pas connu.

    Le plafond de 8 m/s conçu pour la course écartait 1 454 segments sur
    3 789 d'une sortie vélo, ramenant 26,7 km à 13,1 km — sans le moindre
    avertissement. Or l'onglet d'analyse d'une sortie ne connaît pas le
    sport avant d'avoir lu le fichier.
    
    On regarde donc la vitesse médiane brute : au-delà de 4 m/s soutenus,
    aucun coureur n'est concerné, et le plafond vélo s'applique.
    """
    lat, lon = df["lat"].to_numpy(dtype=float), df["lon"].to_numpy(dtype=float)
    if np.isnan(lat).all() or len(df) < 30:
        return 8.0
    dt = df["t"].diff().dt.total_seconds().to_numpy()[1:]
    d = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
    ok = (dt > 0) & (dt < 60) & ~np.isnan(d)
    if ok.sum() < 20:
        return 8.0
    v = d[ok] / dt[ok]
    v = v[v < 40]                        # écarte les décrochages francs
    if len(v) < 20:
        return 8.0
    med = float(np.median(v))
    if med > 4.0:                        # 14,4 km/h de médiane
        return 30.0
    # Un coureur peut atteindre 7 m/s en descente : on garde de la marge.
    return 8.0


def prepare(df: pd.DataFrame, slope_window_m: float = 30.0,
            max_speed_ms: float | None = None,
            route_mode: bool = False) -> pd.DataFrame:
    """
    Enrichit le DataFrame brut : distance, dt, vitesse, altitude lissée,
    pente fiable, GAP, allure marche/course.

    max_speed_ms : garde-fou anti-décrochage GPS (8 m/s = 28,8 km/h,
    au-dessus c'est une téléportation en trail). À relever pour le vélo.

    route_mode : la trace décrit un PARCOURS, pas une sortie enregistrée.

    C'est une distinction qui n'a rien de cosmétique. Une trace de course
    fournie par l'organisateur est une géométrie : points espacés de
    plusieurs dizaines de mètres, horodatage absent ou fictif, parfois un
    seul instant répété. Les garde-fous conçus pour une sortie réelle —
    écarter les segments de plus de 120 s ou dépassant 8 m/s — y suppriment
    alors l'essentiel du tracé et amputent la distance. Symptôme observé :
    57 km affichés pour un parcours de 70 km.

    En mode parcours on ne mesure donc que la géométrie : distance, pente,
    dénivelé. Vitesse, GAP, marche et cardio n'ont aucun sens ici et ne
    sont pas exploités.
    """
    d = df.copy().reset_index(drop=True)
    n = len(d)
    if n < 10:
        raise ValueError("Trace trop courte ou illisible (moins de 10 points).")
    if max_speed_ms is None:
        max_speed_ms = guess_max_speed(d)

    # ── dt et distance ────────────────────────────────────────────────────
    # .dt.total_seconds() est indépendant de la résolution du dtype : pandas 2
    # stocke en nanosecondes, pandas 3 en microsecondes. Diviser un int64 par
    # 1e9 marche sur l'un et donne un facteur 1000 d'erreur sur l'autre.
    dt = np.array(d["t"].diff().dt.total_seconds().fillna(0.0), dtype=float)

    lat, lon = d["lat"].to_numpy(), d["lon"].to_numpy()
    has_gps = ~np.isnan(lat) & ~np.isnan(lon)
    dist = np.zeros(n)
    if has_gps.sum() > 2:
        dist[1:] = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
        dist = np.nan_to_num(dist, nan=0.0)

    # ── Filtrage des points aberrants ─────────────────────────────────────
    if route_mode:
        # Aucun filtre temporel : seuls comptent les sauts géométriques
        # manifestement impossibles (téléportation de plus de 2 km).
        dist[dist > 2000] = 0.0
        dt = np.ones(n)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_speed = np.where(dt > 0, dist / dt, 0.0)
        bad = (dt <= 0) | (dt > 120) | (raw_speed > max_speed_ms)
        dist[bad] = 0.0
        dt[bad] = 0.0

    d["dt"] = dt
    d["dist"] = dist
    d["cum_dist"] = np.cumsum(dist)
    d["cum_time"] = np.cumsum(dt)

    # ── Altitude lissée ───────────────────────────────────────────────────
    ele = d["ele"].to_numpy(dtype=float)
    if np.isnan(ele).all():
        ele = np.zeros(n)
    else:
        ele = pd.Series(ele).interpolate(limit_direction="both").to_numpy()
    # Médiane glissante (retire les spikes) puis moyenne (retire le grain)
    # Fenêtres de lissage exprimées en points : il faut les adapter à
    # l'espacement réel, sinon une trace clairsemée (un point tous les 40 m)
    # se voit lisser sur 600 m et perd son relief.
    spacing = float(np.median(dist[dist > 0])) if (dist > 0).any() else 3.0
    w_med = int(np.clip(round(27 / max(spacing, 0.5)), 3, 15))
    w_avg = int(np.clip(round(45 / max(spacing, 0.5)), 3, 25))
    ele_s = _rolling_median(ele, w_med)
    ele_s = pd.Series(ele_s).rolling(w_avg, center=True, min_periods=1).mean().to_numpy()
    d["ele_smooth"] = ele_s

    # ── Pente sur fenêtre de distance ─────────────────────────────────────
    # Pour chaque point, on cherche l'indice j tel que cum_dist[i]-cum_dist[j]
    # >= slope_window_m, et on prend la pente entre j et i.
    cum = d["cum_dist"].to_numpy()
    j_idx = np.searchsorted(cum, cum - slope_window_m, side="left")
    j_idx = np.clip(j_idx, 0, n - 1)
    dd = cum - cum[j_idx]
    de = ele_s - ele_s[j_idx]
    slope = np.where(dd >= 5.0, de / np.maximum(dd, 1e-6), 0.0)
    d["slope"] = np.clip(slope, -0.45, 0.45)

    # ── Vitesse (lissée légèrement, sinon le GAP est bruité) ──────────────
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, dist / dt, np.nan)
    speed = pd.Series(speed).interpolate(limit_direction="both").fillna(0.0)
    d["speed"] = speed.rolling(5, center=True, min_periods=1).mean().to_numpy()

    # ── D+ / D- sur altitude lissée uniquement ────────────────────────────
    de_step = np.diff(ele_s, prepend=ele_s[0])
    d["d_plus"] = np.where(de_step > 0, de_step, 0.0)
    d["d_minus"] = np.where(de_step < 0, -de_step, 0.0)

    # ── Marche / course, GAP, puissance métabolique ───────────────────────
    d["walking"] = physio.classify_gait(
        d["cad"].to_numpy(), d["speed"].to_numpy(), d["slope"].to_numpy()
    )
    d["gap"] = physio.gap(d["speed"].to_numpy(), d["slope"].to_numpy())
    d["p_met"] = physio.metabolic_power(
        d["speed"].to_numpy(), d["slope"].to_numpy(), d["walking"].to_numpy()
    )

    # ── Coût cardiaque : bpm par W/kg. Bas = économique. ──────────────────
    hr = d["hr"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d["hr_cost"] = np.where(d["p_met"] > 1.0, hr / d["p_met"], np.nan)

    d["slope_bin"] = pd.cut(d["slope"], bins=SLOPE_BINS, labels=SLOPE_LABELS)
    d.attrs["max_speed"] = float(max_speed_ms)
    return d


def wmean(values, weights) -> float:
    """Moyenne pondérée robuste aux NaN. Retourne NaN si aucun poids valide."""
    v, w = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    m = ~np.isnan(v) & ~np.isnan(w) & (w > 0)
    return float(np.sum(v[m] * w[m]) / np.sum(w[m])) if m.sum() and np.sum(w[m]) > 0 else float("nan")


# ── Indicateurs de sortie ────────────────────────────────────────────────────

def summarize(d: pd.DataFrame, hr_rest: float = 50, hr_max: float = 190) -> dict:
    """KPI d'une sortie. Toutes les moyennes sont pondérées par le temps."""
    dt = d["dt"].to_numpy()
    slope = d["slope"].to_numpy()
    up = slope > 0.05
    down = slope < -0.05
    flat = ~up & ~down

    total_t = float(dt.sum())
    if total_t <= 0:
        raise ValueError(
            "Aucun intervalle de temps valide. Trace sans horodatage, "
            "ou entièrement filtrée comme aberrante."
        )
    out = {
        "distance_km": float(d["dist"].sum() / 1000),
        "d_plus": float(d["d_plus"].sum()),
        "d_minus": float(d["d_minus"].sum()),
        "duration_h": total_t / 3600,
        "gap_kmh": wmean(d["gap"], dt) * 3.6,
        "share_up": float(dt[up].sum() / total_t),
        "share_down": float(dt[down].sum() / total_t),
        "share_flat": float(dt[flat].sum() / total_t),
        "walk_share": float(dt[d["walking"].to_numpy()].sum() / total_t),
    }

    # VAM sur les seules portions réellement montantes
    t_up = dt[up].sum()
    out["vam"] = float(d["d_plus"].to_numpy()[up].sum() / t_up * 3600) if t_up > 60 else np.nan

    # Descente : vitesse ET économie. Une descente rapide mais coûteuse
    # (freinage, crispation) ne vaut pas une descente rapide et relâchée.
    out["desc_kmh"] = wmean(d["speed"][down], dt[down]) * 3.6
    out["desc_hr_cost"] = wmean(d["hr_cost"][down], dt[down])
    out["up_hr_cost"] = wmean(d["hr_cost"][up], dt[up])
    out["hr_cost"] = wmean(d["hr_cost"], dt)

    # Efficacité descente : km/h par unité de coût cardiaque.
    out["desc_efficiency"] = (
        out["desc_kmh"] / out["desc_hr_cost"]
        if out["desc_hr_cost"] and not np.isnan(out["desc_hr_cost"]) else np.nan
    )

    # Intensité
    hr = d["hr"].to_numpy()
    if not np.isnan(hr).all():
        out["hr_mean"] = wmean(hr, dt)
        hrr = physio.hr_reserve(hr, hr_rest, hr_max)
        out["hrr_mean"] = wmean(hrr, dt)
        out["time_above_85"] = float(dt[np.nan_to_num(hrr) > 0.85].sum() / total_t)
    else:
        out.update(hr_mean=np.nan, hrr_mean=np.nan, time_above_85=np.nan)

    out["cad_up"] = wmean(d["cad"][up], dt[up])
    out["drift"] = decoupling(d)
    return out


def decoupling(d: pd.DataFrame, min_seconds: float = 300) -> float:
    """
    Découplage cardiaque (Pa:HR) corrigé du terrain.

    Pour chaque bande de pente présente dans les deux moitiés temporelles,
    on compare le coût cardiaque bpm/(W/kg). On agrège ensuite en pondérant
    par le temps passé dans la bande. Cela neutralise le biais d'une sortie
    qui monte d'abord et descend ensuite — biais qui rendait l'indicateur
    précédent ininterprétable.

    > 1 = dégradation (même effort mécanique, FC plus haute en 2e moitié).
    """
    dt = d["dt"].to_numpy()
    cum = np.cumsum(dt)
    if cum[-1] < 2 * min_seconds:
        return float("nan")
    half = cum[-1] / 2
    first, second = cum <= half, cum > half

    num = den = 0.0
    for label in SLOPE_LABELS:
        m = (d["slope_bin"] == label).to_numpy()
        m1, m2 = m & first, m & second
        if dt[m1].sum() < min_seconds / 3 or dt[m2].sum() < min_seconds / 3:
            continue
        c1 = wmean(d["hr_cost"][m1], dt[m1])
        c2 = wmean(d["hr_cost"][m2], dt[m2])
        if np.isnan(c1) or np.isnan(c2) or c1 <= 0:
            continue
        w = dt[m].sum()
        num += (c2 / c1) * w
        den += w
    return float(num / den) if den > 0 else float("nan")


def by_slope_bin(d: pd.DataFrame) -> pd.DataFrame:
    """Table pente × (vitesse, GAP, FC, cadence, coût, part de marche, temps)."""
    rows = []
    for label in SLOPE_LABELS:
        m = (d["slope_bin"] == label).to_numpy()
        w = d["dt"].to_numpy()[m]
        if w.sum() < 30:
            continue
        # Séparation marche / course DANS la bande. Sans ces colonnes, le
        # seuil de bascule ne peut être mesuré que sortie par sortie — et
        # il l'est rarement, faute d'avoir fait les deux dans une même
        # bande le même jour. Mutualisées sur l'historique, elles le
        # rendent calculable.
        walk = d["walking"].to_numpy()[m]
        wk, rn = w[walk], w[~walk]
        sub = d[m]
        rows.append({
            "bande": label,
            "temps_min": w.sum() / 60,
            "distance_km": float(d["dist"].to_numpy()[m].sum() / 1000),
            "vitesse_kmh": wmean(d["speed"][m], w) * 3.6,
            "gap_kmh": wmean(d["gap"][m], w) * 3.6,
            "fc": wmean(d["hr"][m], w),
            "cadence": wmean(d["cad"][m], w),
            "cout_fc": wmean(d["hr_cost"][m], w),
            "part_marche": float(w[walk].sum() / w.sum()),
            "temps_marche_min": float(wk.sum() / 60),
            "temps_course_min": float(rn.sum() / 60),
            "v_marche_kmh": wmean(sub["speed"][walk], wk) * 3.6 if wk.sum() > 20 else np.nan,
            "v_course_kmh": wmean(sub["speed"][~walk], rn) * 3.6 if rn.sum() > 20 else np.nan,
            "cout_marche": wmean(sub["hr_cost"][walk], wk) if wk.sum() > 20 else np.nan,
            "cout_course": wmean(sub["hr_cost"][~walk], rn) if rn.sum() > 20 else np.nan,
        })
    return pd.DataFrame(rows)


def walk_run_threshold(d: pd.DataFrame, min_seconds: float = 60) -> dict | None:
    """
    Seuil marche/course MESURÉ, pas postulé.

    Méthode : dans chaque bande de pente montante, on sépare les points
    marchés des points courus (cadence), puis on compare la vitesse
    obtenue à coût cardiaque comparable. La bande où la marche devient au
    moins aussi rapide que la course pour un coût égal ou inférieur est
    le point de bascule.

    Retourne None si les données ne permettent pas de trancher — cas
    fréquent et normal si tu cours tout ou marches tout.
    """
    dt = d["dt"].to_numpy()
    walking = d["walking"].to_numpy()
    rows = []
    for label in SLOPE_LABELS:
        if not label.startswith("M"):
            continue
        m = (d["slope_bin"] == label).to_numpy()
        mw, mr = m & walking, m & ~walking
        if dt[mw].sum() < min_seconds or dt[mr].sum() < min_seconds:
            continue
        v_w = wmean(d["speed"][mw], dt[mw]) * 3.6
        v_r = wmean(d["speed"][mr], dt[mr]) * 3.6
        c_w = wmean(d["hr_cost"][mw], dt[mw])
        c_r = wmean(d["hr_cost"][mr], dt[mr])
        rows.append({
            "bande": label, "v_marche": v_w, "v_course": v_r,
            "cout_marche": c_w, "cout_course": c_r,
            "marche_gagnante": bool(v_w >= v_r * 0.95 and (np.isnan(c_w) or np.isnan(c_r) or c_w <= c_r)),
        })
    if not rows:
        return None
    table = pd.DataFrame(rows)
    winners = table[table["marche_gagnante"]]
    return {
        "table": table,
        "bascule": winners["bande"].iloc[0] if len(winners) else None,
        "n_bandes_comparables": len(table),
    }


def session_profile(d: pd.DataFrame) -> dict:
    """
    SUGGÈRE un type de séance. Ne le décide pas.

    Pourquoi une suggestion et pas un verdict — le chemin parcouru vaut
    d'être documenté, parce qu'il explique pourquoi le champ reste manuel.

    Premier essai : coefficient de variation du GAP. Échec. Une sortie
    vallonnée continue donne 0,30 et une séance d'intervalles 0,33 : les
    distributions se chevauchent, tout seuil se trompe une fois sur trois.

    Deuxième essai : bimodalité du GAP, plus discriminante que la variance.
    Échec plus instructif. Le GAP normalise la pente — c'est sa raison
    d'être. Sur des côtes répétées, il efface donc exactement ce qu'on
    cherche à détecter : 8 × 2' en côte ressortent avec un CV de 0,13,
    indistinguables d'un footing plat. Et une puissance métabolique ne
    sauverait rien : elle est proportionnelle au GAP par construction.

    Ce qui alterne réellement dans une séance de qualité, c'est l'EFFORT,
    et le seul capteur qui le voie indépendamment du terrain est le
    cardiofréquencemètre. D'où le critère retenu : bimodalité de la FC par
    blocs de 60 s. Il reste imparfait — la FC traîne de 20 à 30 s derrière
    l'effort, ce qui lisse les intervalles courts.

    Conclusion assumée : aucune heuristique fiable à 100 %. La valeur
    retournée pré-remplit un champ que tu confirmes au dépôt. Une seconde
    de ton temps vaut mieux qu'un classement faux qui écarte silencieusement
    une sortie longue clé du calibrage.
    """
    dt = d["dt"].to_numpy()
    out = {"cv_gap": float("nan"), "bimodalite": float("nan"),
           "gap_trend": float("nan"), "session_type": "indéterminée",
           "type_fiabilite": "faible"}
    if dt.sum() < 900:
        out["session_type"] = "trop courte"
        return out

    # Agrégation par blocs de 60 s en calcul vectorisé.
    #
    # La version précédente utilisait groupby.apply avec une lambda
    # renvoyant une Series : 63 ms par activité, soit les deux tiers du
    # temps total d'import. Pandas y crée un objet Python par bloc — cent
    # quatre-vingts objets pour une sortie de trois heures, deux cent
    # cinquante fois pour un rattrapage complet.
    #
    # np.bincount fait les mêmes sommes pondérées en une passe C.
    block = (np.cumsum(dt) // 60).astype(int)
    keep = dt > 0
    b_ok = block[keep]
    if len(b_ok) == 0:
        return out
    b_ok = b_ok - b_ok.min()
    nb = int(b_ok.max()) + 1
    w = dt[keep]

    def par_bloc(valeurs):
        v = np.asarray(valeurs, dtype=float)[keep]
        m = ~np.isnan(v)
        num = np.bincount(b_ok[m], weights=(v[m] * w[m]), minlength=nb)
        den = np.bincount(b_ok[m], weights=w[m], minlength=nb)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, num / den, np.nan)

    per_block = pd.DataFrame({
        "gap": par_bloc(d["gap"].to_numpy()),
        "hr": par_bloc(d["hr"].to_numpy()),
        "walk": par_bloc(d["walking"].to_numpy().astype(float)),
    })

    # Les blocs majoritairement marchés sont exclus : une sortie longue en
    # rando-course alterne 3 km/h en montée et 11 km/h sur le plat, ce qui
    # gonfle toute mesure de variabilité. Or ce sont les séances clés du
    # plan — le faux positif coûterait cher.
    run = per_block[(per_block["walk"] < 0.30) & (per_block["gap"] > 0.3)]
    if len(run) < 10:
        return out

    gap = run["gap"].to_numpy()
    out["cv_gap"] = round(float(gap.std() / gap.mean()), 3)
    out["gap_trend"] = round(float(np.corrcoef(np.arange(len(gap)), gap)[0, 1]), 3)

    hr = run["hr"].dropna().to_numpy()
    if len(hr) < 10:
        out["session_type"] = "indéterminée"
        return out

    sep = _bimodality(hr)
    swing = float((np.quantile(hr, 0.9) - np.quantile(hr, 0.1)))
    out["bimodalite"] = round(sep, 3)

    # Deux régimes cardiaques nets ET un écart d'au moins 12 bpm entre le
    # haut et le bas. Le second critère écarte les distributions
    # techniquement bimodales mais physiologiquement plates.
    if sep > 0.85 and swing >= 12 and abs(out["gap_trend"]) < 0.45:
        out["session_type"] = "fractionné"
        out["type_fiabilite"] = "moyenne"
    elif out["cv_gap"] > 0.18:
        out["session_type"] = "variable"
    else:
        out["session_type"] = "continu"
    return out


def _bimodality(x: np.ndarray) -> float:
    """
    Part de la variance expliquée par la meilleure partition en deux groupes
    (k-moyennes exact en dimension 1, par balayage des coupures).

    Environ 0,64 sur une distribution unimodale · au-delà de 0,85, deux
    régimes réellement séparés.
    """
    v = np.sort(np.asarray(x, dtype=float))
    n = len(v)
    total = float(((v - v.mean()) ** 2).sum())
    if n < 6 or total <= 0:
        return float("nan")

    cum = np.cumsum(v)
    best = total
    for k in range(2, n - 1):
        m1 = cum[k - 1] / k
        m2 = (cum[-1] - cum[k - 1]) / (n - k)
        within = float(((v[:k] - m1) ** 2).sum() + ((v[k:] - m2) ** 2).sum())
        best = min(best, within)
    return 1 - best / total


def analyze(df_raw: pd.DataFrame, hr_rest: float = 50, hr_max: float = 190,
            max_speed_ms: float | None = None) -> dict:
    """Pipeline complet pour une sortie."""
    d = prepare(df_raw, max_speed_ms=max_speed_ms)
    return {
        "points": d,
        "summary": {**summarize(d, hr_rest, hr_max), **session_profile(d)},
        "slope_table": by_slope_bin(d),
        "walk_run": walk_run_threshold(d),
    }
