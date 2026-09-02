#!/usr/bin/env python
"""
disciplines.py — Reclassement manuel des disciplines, par fichier CSV.

    python disciplines.py export            toutes les activités
    python disciplines.py export --devinees seulement celles déduites du D+/km
    python disciplines.py import disciplines.csv

Le fichier exporté contient une colonne `discipline` que tu modifies dans
Excel, plus les éléments de décision — distance, D+, D+/km, titre, date.
Les autres colonnes sont là pour t'aider à trancher, elles ne sont pas
relues.

Valeurs acceptées : trail, route, vtt, velo_route, gravel, rando.

À la réinjection, `discipline_source` passe à « manuel » sur les lignes
modifiées. C'est ce qui empêchera un futur repair.py d'écraser ton travail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from engine.archive import DISCIPLINE_LABELS, SPORT_FROM_DISCIPLINE
from engine.store import get_store, ACTIVITIES

VALIDES = set(SPORT_FROM_DISCIPLINE)
FICHIER = "disciplines.csv"

COLS = ["activity_id", "discipline", "date", "name", "distance_km", "d_plus",
        "dplus_par_km", "duration_h", "ke_km", "discipline_source",
        "strava_type", "session_type"]


def secrets() -> dict:
    p = Path(".streamlit/secrets.toml")
    if not p.exists():
        return {}
    try:
        import tomllib
        with open(p, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def exporter(store, devinees_seulement: bool) -> int:
    d = store.read(ACTIVITIES)
    if d.empty:
        print("Base vide.")
        return 1

    if "dplus_par_km" not in d.columns or d["dplus_par_km"].isna().all():
        with np.errstate(divide="ignore", invalid="ignore"):
            d["dplus_par_km"] = np.where(d.get("distance_km", 0) > 0.5,
                                         d.get("d_plus") / d.get("distance_km"),
                                         np.nan)
    if devinees_seulement and "discipline_source" in d.columns:
        d = d[d["discipline_source"] == "déduit du D+/km"]

    cols = [c for c in COLS if c in d.columns]
    out = d[cols].copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce",
                                     format="mixed").dt.strftime("%Y-%m-%d")
    # Tri par dénivelé décroissant : les cas douteux arrivent en premier.
    if "dplus_par_km" in out.columns:
        out = out.sort_values("dplus_par_km", ascending=False)
    for c in ("distance_km", "d_plus", "dplus_par_km", "duration_h", "ke_km"):
        if c in out.columns:
            out[c] = out[c].round(1)

    # point-virgule et virgule décimale : Excel français ouvre sans réglage
    out.to_csv(FICHIER, index=False, sep=";", decimal=",",
               encoding="utf-8-sig")
    print(f"{len(out)} ligne(s) écrite(s) dans {FICHIER}")
    print(f"\nRépartition actuelle")
    for k, n in out["discipline"].value_counts().items():
        print(f"  {DISCIPLINE_LABELS.get(k, k):<16s} {n:>4d}")
    print(f"\nModifie la colonne `discipline`, valeurs acceptées :")
    print("  " + ", ".join(sorted(VALIDES)))
    print(f"\nPuis : python disciplines.py import {FICHIER}")
    return 0


def importer(store, chemin: Path) -> int:
    if not chemin.exists():
        print(f"Fichier introuvable : {chemin}")
        return 1
    # Excel peut réenregistrer en virgule ou en point-virgule : on teste.
    for sep in (";", ","):
        d = pd.read_csv(chemin, sep=sep, dtype={"activity_id": str})
        if {"activity_id", "discipline"}.issubset(d.columns):
            break
    else:
        print("Colonnes `activity_id` et `discipline` introuvables.")
        return 1

    d["discipline"] = d["discipline"].astype(str).str.strip().str.lower()
    invalides = sorted(set(d["discipline"]) - VALIDES)
    if invalides:
        print(f"Valeurs non reconnues : {invalides}")
        print(f"Acceptées : {', '.join(sorted(VALIDES))}")
        return 1

    actuel = store.read(ACTIVITIES)
    if actuel.empty:
        print("Base vide.")
        return 1
    avant = dict(zip(actuel["activity_id"].astype(str),
                     actuel.get("discipline", pd.Series(dtype=str))))

    change = d[d.apply(
        lambda r: avant.get(r["activity_id"]) != r["discipline"], axis=1)]
    if change.empty:
        print("Aucune modification détectée.")
        return 0

    maj = change[["activity_id", "discipline"]].copy()
    maj["sport"] = maj["discipline"].map(SPORT_FROM_DISCIPLINE)
    maj["discipline_source"] = "manuel"
    store.upsert(ACTIVITIES, maj, key="activity_id")

    print(f"{len(maj)} activité(s) reclassée(s)")
    print("\nDétail des changements")
    for _, r in change.head(40).iterrows():
        ancien = avant.get(r["activity_id"], "?")
        nom = str(r.get("name", ""))[:32]
        print(f"  {nom:<34s} {ancien:>12s} -> {r['discipline']}")
    if len(change) > 40:
        print(f"  … et {len(change) - 40} autre(s)")

    apres = store.read(ACTIVITIES)
    print("\nRépartition après reclassement")
    for k, n in apres["discipline"].value_counts().items():
        print(f"  {DISCIPLINE_LABELS.get(k, k):<16s} {n:>4d}")
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("export", "import"):
        print(__doc__)
        return 1
    store = get_store(secrets())
    print(f"Stockage : {store.backend}\n")
    if sys.argv[1] == "export":
        return exporter(store, "--devinees" in sys.argv)
    chemin = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(FICHIER)
    return importer(store, chemin)


if __name__ == "__main__":
    sys.exit(main())
