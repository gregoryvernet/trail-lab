"""
sync.py — Rattrapage et synchronisation Strava.

Le problème concret : environ 150 activités à importer. Chaque activité
coûte une requête de flux, plus deux pages de liste. Le quota "read" par
défaut est de 100 requêtes par quart d'heure et 1 000 par jour. Le
rattrapage complet tient donc largement dans une journée, mais PAS en une
seule passe.

D'où trois exigences :

1. REPRENABLE. On enregistre après chaque activité, pas à la fin. Si le
   quota tombe ou si Streamlit met le conteneur en veille au milieu, on
   reprend où on s'est arrêté au lieu de tout rejouer.
2. INCRÉMENTAL. Les identifiants déjà en base ne sont jamais retéléchargés.
   La synchronisation quotidienne ne coûte alors que 2 ou 3 requêtes.
3. PILOTÉ PAR LES EN-TÊTES. Strava renvoie X-RateLimit-Limit et
   X-RateLimit-Usage à chaque réponse. On s'arrête proprement à 85 % du
   quota au lieu d'attendre le 429 — un 429 compte quand même dans le
   quota journalier, donc foncer dans le mur coûte des requêtes perdues.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from engine import analysis, bike, efforts, ingest, metrics, predict
from engine.store import ACTIVITIES, SLOPE_BINS

BIN_CENTERS = {
    "Descente raide": -0.28, "Descente difficile": -0.145,
    "Descente dure": -0.095, "Descente moyenne": -0.05, "Plat": 0.0,
    "Montée moyenne": 0.05, "Montée dure": 0.095,
    "Montée difficile": 0.145, "Montée raide": 0.28,
}

# Un trail se distingue d'une sortie route par le dénivelé rapporté à la
# distance. 15 m/km est un seuil bas et volontairement inclusif : mieux
# vaut classer une sortie vallonnée en trail que l'inverse, car le niveau
# du modèle se cale sur le sous-ensemble trail.
TRAIL_DPLUS_PER_KM = 15.0


def classify_terrain(distance_km: float, d_plus: float) -> str:
    if not distance_km or distance_km <= 0:
        return "route"
    return "trail" if (d_plus or 0) / distance_km >= TRAIL_DPLUS_PER_KM else "route"


def sync(client, store, hr_rest: float = 50, hr_max: float = 190,
         ftp: float | None = None, limit: int = 400,
         progress=None) -> dict:
    """
    client   : integrations.strava.Strava
    store    : engine.store.Store
    progress : callable(done, total, label) pour la barre Streamlit

    Retourne un rapport : importées, ignorées, échecs, raison d'arrêt.
    """
    known = store.known_ids(ACTIVITIES)
    report = {"imported": 0, "skipped": 0, "failed": [], "duplicates": 0,
              "stopped": None}

    todo = []
    for a in client.activities(max_pages=10):
        aid = str(a["id"])
        if aid in known:
            report["skipped"] += 1
            continue
        sport = _sport(a)
        if sport not in ("trail", "velo", "rando"):
            report["skipped"] += 1
            continue
        todo.append(a)
        if len(todo) >= limit:
            break

    total = len(todo)
    for i, a in enumerate(todo, 1):
        aid = str(a["id"])
        if progress:
            progress(i, total, a.get("name", aid))

        if _quota_exhausted(client):
            report["stopped"] = (
                "Quota Strava à 85 %. Import interrompu proprement. "
                "Relance dans un quart d'heure : les activités déjà "
                "importées ne seront pas retéléchargées."
            )
            break

        try:
            row, bins = _process(client, a, hr_rest, hr_max, ftp)
            # La même sortie peut déjà être en base via un import de fichier
            # local : identifiants différents, effort identique.
            if store.find_overlap(row["date"], row.get("duration_h", 0)):
                report["duplicates"] += 1
                continue
        except Exception as e:
            report["failed"].append((aid, a.get("name"), str(e)))
            continue

        # Écriture immédiate : la reprise après interruption en dépend.
        store.upsert(ACTIVITIES, pd.DataFrame([row]), key="activity_id")
        if not bins.empty:
            store.upsert(SLOPE_BINS, bins, key=["activity_id", "bande"])
        report["imported"] += 1

    return report


def _sport(a: dict) -> str:
    from integrations.strava import SPORT_MAP
    return SPORT_MAP.get(a.get("sport_type") or a.get("type"), "autre")


def _quota_exhausted(client, threshold: float = 0.85) -> bool:
    """Lit les en-têtes de quota mémorisés par le dernier appel."""
    usage = getattr(client, "last_rate", None)
    if not usage:
        return False
    short_u, day_u = usage["usage"]
    short_l, day_l = usage["limit"]
    return (short_u >= short_l * threshold) or (day_u >= day_l * threshold)


def _process(client, a: dict, hr_rest, hr_max, ftp):
    from integrations.strava import normalize_activity

    meta = normalize_activity(a)
    streams = client.streams(a["id"])
    raw = ingest.from_strava_streams(streams, pd.Timestamp(a["start_date"]))

    if len(raw) < 30:
        raise ValueError("Flux trop court ou absent (activité indoor ?).")

    d = metrics.prepare(raw, max_speed_ms=metrics.max_speed_for(meta["sport"]))

    row = {
        "activity_id": meta["activity_id"],
        "source": "strava",
        "sport": meta["sport"],
        "date": meta["date"],
        "name": meta["name"],
    }

    if meta["sport"] == "velo":
        row.update(bike.summarize_ride(d, ftp, meta["device_watts"], hr_rest, hr_max))
        row["terrain"] = "velo"
        bins = pd.DataFrame()
    else:
        s = metrics.summarize(d, hr_rest, hr_max)
        row.update(s)
        row["deq_km"] = predict.flat_equivalent_distance(d)
        row["terrain"] = classify_terrain(s["distance_km"], s["d_plus"])
        row["trimp"] = bike.trimp(d["hr"].to_numpy(), d["dt"].to_numpy(),
                                  hr_rest, hr_max)
        row.update(analysis.summarize_transitions(d))
        row.update(efforts.summarize(d))
        bins = metrics.by_slope_bin(d)
        if not bins.empty:
            bins["activity_id"] = meta["activity_id"]
            bins["date"] = meta["date"]
            bins["pente_centre"] = bins["bande"].map(BIN_CENTERS)

    return row, bins


def weekly_load(hist: pd.DataFrame, weeks: int = 16) -> pd.DataFrame:
    """
    Charge hebdomadaire tous sports confondus, en TRIMP.

    C'est le seul agrégat qui permette d'additionner trail et vélo. La
    distance ne le permet pas (30 km à vélo ≠ 30 km à pied), le D+ non
    plus, le TSS non plus tant que la puissance manque à pied.
    """
    if hist.empty or "trimp" not in hist.columns:
        return pd.DataFrame()
    d = hist.dropna(subset=["trimp"]).copy()
    d["date"] = pd.to_datetime(d["date"])
    d["semaine"] = d["date"].dt.to_period("W").dt.start_time
    agg = (d.groupby(["semaine", "sport"])["trimp"].sum()
             .unstack(fill_value=0).sort_index().tail(weeks))
    agg["total"] = agg.sum(axis=1)

    # RATIO AIGU/CHRONIQUE : la charge de la semaine rapportée à la moyenne
    # des QUATRE SEMAINES PRÉCÉDENTES, celle-ci exclue.
    #
    # La première version incluait la semaine courante dans sa propre
    # référence. Outre l'incohérence conceptuelle — comparer une valeur à
    # une moyenne qui la contient —, cela produisait des chiffres
    # invérifiables : 535 contre 429 affichait un ratio de 1,67 au lieu de
    # 1,25, parce que le dénominateur n'était pas celui annoncé.
    chronique = agg["total"].shift(1).rolling(4, min_periods=2).mean()
    agg["chronique"] = chronique
    agg["ratio_ac"] = agg["total"] / chronique
    return agg
