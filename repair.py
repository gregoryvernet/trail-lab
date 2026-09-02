#!/usr/bin/env python
"""
repair.py — Renseigne la discipline sur les activités déjà en base.

    python repair.py chemin/vers/archive.zip

La discipline vient du fichier `activities.csv` de l'archive Strava, donc
de ce que TU as déclaré, et non d'une déduction à partir du dénivelé.
Aucune trace n'est retraitée : réimporter trois cents sorties pour remplir
une colonne n'aurait aucun sens.

Le script affiche ensuite les incohérences entre déclaration et profil.
Il ne les corrige pas — c'est à toi de trancher, et l'enjeu est réel :
le modèle de prédiction se calibre sur les seules sorties trail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from engine import archive
from engine.archive import DISCIPLINE_LABELS
from engine.store import get_store, ACTIVITIES


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


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    src = Path(sys.argv[1])
    if not src.exists():
        print(f"Archive introuvable : {src}")
        return 1

    store = get_store(secrets())
    print(f"Stockage : {store.backend}\n")

    rep = archive.repair_disciplines(src, store)
    if not rep.get("ok"):
        print(rep["reason"])
        return 1

    print(f"{rep['mises_a_jour']} activité(s) mise(s) à jour")
    if rep.get("preserves"):
        print(f"{rep['preserves']} activité(s) reclassée(s) à la main : "
              "préservée(s)")
    if rep["hors_archive"]:
        print(f"{rep['hors_archive']} activité(s) absente(s) de l'archive "
              "(import de fichier local) — discipline inchangée")
    print("\nRépartition par discipline")
    for d, n in sorted(rep["repartition"].items(), key=lambda x: -x[1]):
        print(f"  {DISCIPLINE_LABELS.get(d, d):<16s} {n:>4d}")
    if rep.get("sources"):
        print("\nD'où vient la discipline")
        for src, n in sorted(rep["sources"].items(), key=lambda x: -x[1]):
            print(f"  {src:<16s} {n:>4d}")
        print("  « déduit du D+/km » = sortie renommée à la main, "
              "Strava n'a pas gardé le type fin")

    hist = store.read(ACTIVITIES)
    inc = archive.coherence_disciplines(hist)
    print(f"\n{len(inc)} incohérence(s) entre déclaration et profil")
    if not inc.empty:
        aff = inc.head(20).copy()
        if "date" in aff.columns:
            aff["date"] = pd.to_datetime(aff["date"], errors="coerce",
                                         format="mixed").dt.strftime("%d/%m/%y")
        cols = [c for c in ["date", "name", "discipline", "distance_km",
                            "d_plus", "dplus_par_km", "signal"] if c in aff.columns]
        print(aff[cols].round(1).to_string(index=False, max_colwidth=28))
        print("\nPour corriger une sortie, change son type dans Strava puis "
              "réexporte l'archive, ou modifie la colonne `discipline` "
              "directement dans Supabase.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
