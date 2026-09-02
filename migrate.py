#!/usr/bin/env python
"""
migrate.py — Récupère les saisies de journal faites avant le changement de clé.

    python migrate.py            aperçu, aucune écriture
    python migrate.py --ecrire   applique la migration

L'ancien format de clé était « AAAA-MM-JJ_sport ». Il a été remplacé par
l'identifiant du plan (« s1-d1 ») parce que deux séances vélo tombaient le
même jour et se écrasaient mutuellement. Les saisies antérieures sont
toujours en base, mais sous des clés que le formulaire ne retrouve plus.

Ce script les transfère, sans écraser ce qui a été saisi depuis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from engine import plan as plan_mod
from engine.store import get_store, JOURNAL

PLAN_CSV = Path("data_plan_templiers_2026.csv")


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
    if not PLAN_CSV.exists():
        print(f"{PLAN_CSV} introuvable — dépose d'abord le plan dans l'app.")
        return 1
    plan = pd.read_csv(PLAN_CSV, parse_dates=["date"])
    store = get_store(secrets())
    print(f"Stockage : {store.backend}\n")

    journal = store.read(JOURNAL)
    r = plan_mod.migrate_journal_keys(plan, journal)
    if not r.get("ok"):
        print(r["reason"])
        return 1
    if r["migrees"] == 0:
        print(r.get("reason", "Rien à migrer."))
        return 0

    d = r["a_ecrire"]
    print(f"{r['migrees']} ligne(s) à transférer\n")
    cols = [c for c in ["planned_key", "fait", "temps_min", "commentaire"]
            if c in d.columns]
    print(d[cols].to_string(index=False, max_colwidth=44))

    if r["conflits"]:
        print(f"\n{len(r['conflits'])} clé(s) ambiguë(s), non migrée(s) — "
              "deux séances du même sport le même jour :")
        print("  " + ", ".join(sorted(set(r["conflits"]))))
    if r["perdues"]:
        print(f"\n{len(r['perdues'])} clé(s) sans séance correspondante :")
        print("  " + ", ".join(sorted(set(r["perdues"]))))

    if "--ecrire" not in sys.argv:
        print("\nAperçu seulement. Relance avec --ecrire pour appliquer.")
        return 0

    store.upsert(JOURNAL, d, key="planned_key")
    print(f"\n{len(d)} ligne(s) écrite(s).")
    print("\nLes anciennes lignes subsistent en base sans gêner : elles ne "
          "correspondent à aucune séance du plan. Pour les supprimer, dans "
          "l'éditeur SQL Supabase :")
    print("  delete from journal where planned_key ~ '^\\d{4}-\\d{2}-\\d{2}_';")
    return 0


if __name__ == "__main__":
    sys.exit(main())
