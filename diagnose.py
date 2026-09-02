#!/usr/bin/env python
"""
diagnose.py — Pourquoi l'exposant d'endurance est-il hors plage ?

    python diagnose.py

Un exposant sous 1,00 signifie que tu courrais relativement PLUS VITE sur
long que sur court. C'est physiologiquement impossible sur des efforts
soutenus. La cause est donc dans les données, pas dans la physiologie : des
sorties dont le temps total ne décrit pas un effort continu.

Ce script ne corrige rien. Il montre où est le problème, en testant
plusieurs filtres et en affichant les points qui pèsent le plus.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from engine.store import get_store, ACTIVITIES
from engine import predict


def fit_raw(d: pd.DataFrame) -> tuple[float, float, int]:
    """Ajustement brut log-log, sans effet de terrain, pour comparaison."""
    if len(d) < 5:
        return float("nan"), float("nan"), len(d)
    x, y = np.log(d["deq_km"].to_numpy()), np.log(d["duration_h"].to_numpy())
    b, a = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (b * x + a)) ** 2) / np.sum((y - y.mean()) ** 2)
    return b, r2, len(d)


def main() -> int:
    store = get_store({})
    hist = store.read(ACTIVITIES)
    if hist.empty:
        print("Base vide.")
        return 1

    hist["date"] = pd.to_datetime(hist["date"], errors="coerce", utc=True).dt.tz_localize(None)
    runs = hist[hist["sport"].isin(["trail", "rando"])].dropna(
        subset=["deq_km", "duration_h"]).copy()
    runs = runs[(runs["deq_km"] > 0) & (runs["duration_h"] > 0)]
    print(f"{len(runs)} sortie(s) à pied avec distance et durée\n")

    # ── 1. Répartition des durées ─────────────────────────────────────────
    print("1. RÉPARTITION PAR DISTANCE ÉQUIVALENTE")
    bins = [0, 5, 8, 12, 18, 25, 40, 1000]
    labels = ["<5", "5-8", "8-12", "12-18", "18-25", "25-40", ">40"]
    runs["tranche"] = pd.cut(runs["deq_km"], bins=bins, labels=labels)
    tab = runs.groupby("tranche", observed=True).agg(
        n=("deq_km", "size"),
        deq_med=("deq_km", "median"),
        h_med=("duration_h", "median"),
        h_min=("duration_h", "min"),
    )
    tab["kmh_med"] = tab["deq_med"] / tab["h_med"]
    tab["kmh_max"] = tab["deq_med"] / tab["h_min"]
    print(tab.round(2).to_string())
    print("\n   La vitesse médiane DOIT décroître quand la distance monte.")
    print("   Si elle remonte, une tranche contient des sorties mal mesurées.\n")

    # ── 2. Effet des filtres ──────────────────────────────────────────────
    print("2. EFFET DE CHAQUE FILTRE SUR L'EXPOSANT")
    tests = [("aucun filtre", runs)]

    if "session_type" in runs.columns:
        tests.append(("sans fractionné", runs[runs["session_type"] != "fractionné"]))
        tests.append(("continu + variable seulement",
                      runs[runs["session_type"].isin(["continu", "variable"])]))
    tests.append(("Deq > 6 km", runs[runs["deq_km"] > 6]))
    tests.append(("Deq > 8 km", runs[runs["deq_km"] > 8]))
    if "session_type" in runs.columns:
        tests.append(("continu/variable ET Deq > 6",
                      runs[runs["session_type"].isin(["continu", "variable"])
                           & (runs["deq_km"] > 6)]))
    if "hrr_mean" in runs.columns and runs["hrr_mean"].notna().any():
        tests.append(("intensité >= 0,65",
                      runs[runs["hrr_mean"].fillna(1) >= 0.65]))
    cut12 = runs["date"].max() - pd.DateOffset(months=12)
    tests.append(("12 derniers mois", runs[runs["date"] >= cut12]))

    print(f"   {'filtre':<32s} {'n':>4s} {'exposant':>9s} {'R²':>6s}")
    for name, sub in tests:
        b, r2, n = fit_raw(sub)
        flag = "" if 1.0 <= b <= 1.30 else "   <-- hors plage"
        print(f"   {name:<32s} {n:>4d} {b:>9.3f} {r2:>6.3f}{flag}")

    # ── 3. Les points qui tirent la pente vers le bas ─────────────────────
    print("\n3. SORTIES LES PLUS LENTES POUR LEUR DISTANCE")
    print("   (résidu positif = beaucoup plus lent que le modèle ne prédit)")
    x, y = np.log(runs["deq_km"].to_numpy()), np.log(runs["duration_h"].to_numpy())
    b, a = np.polyfit(x, y, 1)
    runs["residu"] = y - (b * x + a)
    cols = ["date", "name", "deq_km", "duration_h", "session_type", "residu"]
    cols = [c for c in cols if c in runs.columns]

    courtes = runs[runs["deq_km"] < 10].nlargest(12, "residu")[cols]
    print("\n   Sur DISTANCES COURTES (< 10 km eq.) — ce sont elles qui")
    print("   aplatissent la pente si leur temps inclut de l'échauffement :")
    print(courtes.to_string(index=False, max_colwidth=32))

    print("\n4. SORTIES LES PLUS RAPIDES POUR LEUR DISTANCE")
    print("   (résidu négatif = courses, ou durée sous-estimée)")
    print(runs.nsmallest(10, "residu")[cols].to_string(index=False, max_colwidth=32))

    # ── 5. Le modèle actuel ───────────────────────────────────────────────
    print("\n5. MODÈLE ACTUEL (filtres en place)")
    m = predict.fit_endurance_model(runs)
    if m["ok"]:
        for k in ["b", "r2", "n", "n_trail", "n_road", "terrain_penalty",
                  "resid_sd", "level_source"]:
            v = m.get(k)
            print(f"   {k:<18s} {v if not isinstance(v, float) else round(v, 4)}")
    else:
        print(f"   non calibré : {m['reason']}")

    # ── 6. Types de séance ────────────────────────────────────────────────
    if "session_type" in runs.columns:
        print("\n6. TYPES DE SÉANCE ET DISTANCE MÉDIANE")
        t = runs.groupby("session_type", observed=True).agg(
            n=("deq_km", "size"), deq_med=("deq_km", "median"),
            h_med=("duration_h", "median"))
        t["kmh"] = (t["deq_med"] / t["h_med"]).round(2)
        print(t.round(2).to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
