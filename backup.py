#!/usr/bin/env python
"""
backup.py — Sauvegarde des tables et maintien de la base en éveil.

    python backup.py                    sauvegarde dans sauvegardes/
    python backup.py --dossier ailleurs
    python backup.py --ping             interroge la base sans rien écrire

DEUX RÔLES, ET LE SECOND EST LE PLUS IMPORTANT.

Supabase suspend les projets gratuits après une semaine sans aucune
requête. Ce script, exécuté chaque semaine par une action GitHub, maintient
donc la base en vie — la sauvegarde n'en est que l'effet secondaire utile.

Les identifiants viennent de `.streamlit/secrets.toml` en local, ou des
variables d'environnement SUPABASE_URL et SUPABASE_KEY en exécution
automatisée. Aucune valeur n'est jamais affichée : les journaux d'une
action GitHub sont publics sur un dépôt public.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from engine.store import (get_store, ACTIVITIES, SLOPE_BINS, JOURNAL,
                          COURSES, TOKENS)

# TOKENS est délibérément exclu : ce sont des jetons d'accès Strava, qui
# n'ont pas à se retrouver dans un fichier de sauvegarde.
TABLES = [ACTIVITIES, SLOPE_BINS, JOURNAL, COURSES]


def secrets() -> dict:
    """Fichier local d'abord, variables d'environnement ensuite."""
    p = Path(".streamlit/secrets.toml")
    if p.exists():
        try:
            import tomllib
            with open(p, "rb") as f:
                s = tomllib.load(f)
            if s.get("SUPABASE_URL"):
                return s
        except Exception:
            pass
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
    return {"SUPABASE_URL": url, "SUPABASE_KEY": key} if url and key else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dossier", default="sauvegardes")
    ap.add_argument("--ping", action="store_true",
                    help="Interroge la base sans écrire de fichier.")
    args = ap.parse_args()

    sec = secrets()
    if not sec.get("SUPABASE_URL"):
        print("Identifiants Supabase absents : ni secrets.toml, ni variables "
              "d'environnement.")
        return 1

    store = get_store(sec)
    if store.backend != "supabase":
        print("Backend local — rien à maintenir en éveil.")
        return 1

    horodatage = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"Sauvegarde Trail Lab · {horodatage}\n")

    dossier = Path(args.dossier)
    if not args.ping:
        dossier.mkdir(parents=True, exist_ok=True)

    jour = date.today().isoformat()
    total, echecs = 0, []
    for table in TABLES:
        try:
            df = store.read(table)
        except Exception as e:
            # Une table absente ne doit pas faire échouer l'ensemble : le
            # but premier est d'avoir interrogé la base.
            echecs.append(f"{table} : {type(e).__name__}")
            print(f"  {table:<14s} ÉCHEC — {type(e).__name__}")
            continue
        total += len(df)
        if args.ping:
            print(f"  {table:<14s} {len(df):>6} lignes")
            continue
        chemin = dossier / f"{table}_{jour}.csv"
        df.to_csv(chemin, index=False, encoding="utf-8-sig")
        taille = chemin.stat().st_size / 1024
        print(f"  {table:<14s} {len(df):>6} lignes  {taille:>7.1f} Ko  "
              f"{chemin}")

    print(f"\n{total} lignes au total")
    if not args.ping:
        _elaguer(dossier)

    if echecs:
        print("\nTables en échec : " + ", ".join(echecs))
        # On sort en succès malgré tout : la base a été interrogée, donc
        # l'objectif de maintien en éveil est atteint. Une sortie en erreur
        # déclencherait une alerte pour un problème mineur.
    return 0


def _elaguer(dossier: Path, garder: int = 8) -> None:
    """
    Ne conserve que les huit sauvegardes les plus récentes de chaque table.

    Sans élagage, une exécution hebdomadaire remplit le dépôt de fichiers
    quasi identiques. Huit correspond à deux mois d'historique, ce qui
    suffit pour revenir en arrière après une erreur.
    """
    for table in TABLES:
        fichiers = sorted(dossier.glob(f"{table}_*.csv"))
        for vieux in fichiers[:-garder]:
            vieux.unlink()
            print(f"  élagué : {vieux.name}")


if __name__ == "__main__":
    sys.exit(main())
