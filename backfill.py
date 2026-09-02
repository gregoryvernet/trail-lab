#!/usr/bin/env python
"""
backfill.py — Rattrapage initial depuis l'archive Strava.

À LANCER UNE SEULE FOIS, EN LOCAL, PAS DANS L'APPLICATION.

Deux raisons de ne pas passer par l'interface web :

1. Une archive de 150 activités pèse couramment 100 à 300 Mo. Streamlit
   Cloud plafonne les téléversements bien en dessous, et le conteneur se
   met en veille pendant les traitements longs.
2. Le rattrapage est une opération unique. La faire tourner sur ton PC,
   qui écrit directement dans Supabase, évite d'ajouter au déploiement un
   chemin de code qui ne servira qu'une fois.

USAGE

    # Depuis un dossier de TCX déjà sur ton disque — le plus rapide
    python backfill.py "D:/Greg/Code_Indice_UTMB/1. Fichiers GPX/TCX Polar"

    # Depuis l'archive Strava
    python backfill.py export_12345678.zip --hr-rest 48 --hr-max 188

    python backfill.py <source> --dry-run     # inventaire, aucune écriture
    python backfill.py <source> --limit 20    # essai sur 20 sorties

Les identifiants Supabase sont lus depuis .streamlit/secrets.toml. Sans
eux, l'écriture se fait en local dans ./data — utile pour tester, mais
ces fichiers ne survivront pas au déploiement.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def load_secrets() -> dict:
    path = Path(".streamlit/secrets.toml")
    if not path.exists():
        return {}
    try:
        import tomllib
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        print(f"  Lecture des secrets impossible : {e}")
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Rattrapage initial")
    ap.add_argument("source", help="ZIP d'export Strava, OU dossier de fichiers TCX/FIT/GPX")
    ap.add_argument("--hr-rest", type=float, default=50)
    ap.add_argument("--hr-max", type=float, default=190)
    ap.add_argument("--ftp", type=float, default=None)
    ap.add_argument("--poids", type=float, default=None,
                    help="Poids en kg, pour les W/kg à vélo.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sports", default="trail,rando,velo")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--months", type=int, default=None,
                    help="Ne garder que les N derniers mois.")
    ap.add_argument("--fill-elevation", action="store_true",
                    help="Reconstitue l'altitude manquante (TCX/FIT Polar) "
                         "depuis un modèle de terrain public.")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        print(f"Introuvable : {src}")
        return 1
    is_folder = src.is_dir()

    from engine import archive
    from engine.store import get_store, ACTIVITIES

    secrets = load_secrets()
    store = get_store(secrets)
    print(f"Stockage : {store.backend}")
    if store.backend == "local":
        print("  Attention : stockage local. Configure Supabase avant de "
              "déployer, sinon ce travail sera perdu.")

    if args.dry_run:
        if is_folder:
            files = [f for f in src.rglob("*")
                     if f.suffix.lower() in (".tcx", ".fit", ".gpx", ".gz")]
            print(f"\n{len(files)} fichier(s) dans {src}")
            ext = pd.Series([f.suffix.lower() for f in files]).value_counts()
            print(ext.to_string())
        else:
            import zipfile
            idx = archive.read_index(zipfile.ZipFile(src))
            if args.months:
                cut = pd.Timestamp.now() - pd.DateOffset(months=args.months)
                before = len(idx)
                idx = idx[pd.to_datetime(idx["date"], errors="coerce") >= cut]
                print(f"\nFiltre {args.months} mois : {len(idx)} retenue(s) "
                      f"sur {before}")
            print(f"\n{len(idx)} activité(s) dans l'archive")
            print(idx["sport"].value_counts().to_string())
            print(f"\nPériode : {idx['date'].min():%Y-%m-%d} → {idx['date'].max():%Y-%m-%d}")
            print("\nFormats réellement présents (c'est ce que ta montre a "
                  "poussé vers Strava, pas un GPX réexporté) :")
            print(archive.inventory(src).to_string(index=False))
        print("\nAucune écriture (--dry-run).")
        return 0

    t0 = time.time()
    last = [0]

    def progress(i, total, label):
        if i - last[0] >= 5 or i == total:
            last[0] = i
            pct = i / max(total, 1) * 100
            eta = (time.time() - t0) / i * (total - i) if i else 0
            print(f"  [{pct:5.1f}%] {i}/{total}  reste ~{eta / 60:.0f} min  {label}")

    if args.fill_elevation:
        print("Reconstitution d'altitude activée (Open Topo Data, EU-DEM 25 m).")
        print("  Environ 1 requête/seconde, 100 points par requête.")
        print("  Le cache est figé sur disque : rien n'est jamais redemandé.\n")

    if is_folder:
        rep = archive.import_folder(src, store, hr_rest=args.hr_rest,
                                    hr_max=args.hr_max, ftp=args.ftp, poids_kg=args.poids,
                                    progress=progress,
                                    fill_elevation=args.fill_elevation)
    else:
        rep = archive.import_archive(
            src, store, hr_rest=args.hr_rest, hr_max=args.hr_max, ftp=args.ftp,
            poids_kg=args.poids, sports=tuple(args.sports.split(",")),
            limit=args.limit, progress=progress,
            fill_elevation=args.fill_elevation, months=args.months,
        )

    print(f"\nTerminé en {(time.time() - t0) / 60:.1f} min")
    print(f"  Importées      : {rep['imported']}")
    print(f"  Déjà en base   : {rep['skipped']}")
    print(f"  Sans GPS       : {rep['no_gps']}  (tapis, home-trainer, saisie manuelle)")
    print(f"  Doublons       : {rep.get('duplicates', 0)}  (déjà en base via une autre source)")
    print(f"  Sans altitude  : {rep.get('no_elevation', 0)}  (rejetées — relance avec --fill-elevation)")
    if rep.get("enriched"):
        print(f"\n  Altitude reconstituée sur {rep['enriched']} sortie(s)")
        q, c = rep.get("dem_queried", 0), rep.get("dem_cached", 0)
        if q + c:
            print(f"    {q} point(s) demandés · {c} déjà en cache "
                  f"({c / (q + c):.0%} de réutilisation)")
        print(f"    cache : {rep.get('dem_points', 0)} points figés sur disque")
    if rep.get("repli_cache"):
        print(f"    dont {rep['repli_cache']} sortie(s) enrichie(s) depuis le "
              "cache seul (service indisponible, profil un peu plus grossier)")
    if rep.get("sans_gps_alt"):
        print(f"  Sans GPS (alt.) : {rep['sans_gps_alt']}  (indoor — altitude "
              "non reconstituable, ce n'est pas un échec)")
    if rep.get("quota_reached"):
        print("\n  QUOTA ATTEINT — import interrompu proprement.")
        print("  Relance la même commande demain : rien ne sera redemandé.")
    if rep.get("reseau_ko"):
        print("\n  SERVICE D'ALTITUDE INJOIGNABLE — import interrompu.")
        print("  " + rep["reseau_ko"])
        print("\n  Diagnostic : python -c \"from engine.elevation import ping;"
              " print(ping())\"")
    print(f"  Échecs         : {len(rep['failed'])}")
    for aid, name, err in rep["failed"][:10]:
        print(f"      {aid}  {str(name)[:35]:35s}  {err[:60]}")
    if len(rep["failed"]) > 10:
        print(f"      … et {len(rep['failed']) - 10} autre(s)")

    hist = store.read(ACTIVITIES)
    if hist.empty:
        return 0

    print("\nRécapitulatif")
    print(hist["sport"].value_counts().to_string())
    runs = hist[hist["sport"].isin(["trail", "rando"])]
    if "session_type" in runs.columns and runs["session_type"].notna().any():
        print("\nTypes de séance détectés")
        print(runs["session_type"].value_counts().to_string())

    if not runs.empty and "deq_km" in runs.columns:
        from engine import predict
        runs = runs.copy()
        runs["date"] = pd.to_datetime(runs["date"])
        print("\nModèle d'endurance")
        # Deux calibrages : générique, puis ciblé sur une course longue.
        # L'exposant dépend de la plage de distances retenue, donc afficher
        # un seul chiffre sans dire pour quelle cible n'aurait aucun sens.
        for cible, libelle in [(None, "générique"), (114.0, "cible 80 km / 3 400 D+")]:
            m = predict.fit_endurance_model(runs, target_deq=cible)
            if not m["ok"]:
                print(f"  {libelle:<24s} non calibré : {m['reason']}")
                continue
            drapeau = "" if m["b_plausible"] else "  HORS PLAGE"
            print(f"  {libelle:<24s} b={m['b']:.3f}{drapeau}  R²={m['r2']:.3f}  "
                  f"n={m['n']} (>{m['min_deq']:.0f} km-effort)")
            if not np.isnan(m.get("cv_error", float("nan"))):
                print(f"  {'':24s} erreur de prédiction : {m['cv_error'] * 100:.0f} %")
            if m.get("b_span"):
                print(f"  {'':24s} sensibilité : {m['b_span'][0]:.3f} à {m['b_span'][1]:.3f}")
            if cible:
                p = predict.predict_time(cible, m, 1.05)
                if p["ok"]:
                    print(f"  {'':24s} temps estimé : {predict.fmt_hours(p['hours'])} "
                          f"[{predict.fmt_hours(p['low'])} – {predict.fmt_hours(p['high'])}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
