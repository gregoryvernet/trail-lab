#!/usr/bin/env python
"""
check.py — Autodiagnostic. À lancer AVANT tout le reste.

    python check.py                      # vérifie l'installation
    python check.py mon_fichier.tcx      # vérifie aussi la lecture d'un fichier

Objectif : distinguer « le moteur est cassé » de « mon fichier pose
problème ». Sans ce script, la première erreur rencontrée peut venir de
six endroits différents et tu perds une soirée à chercher.

Chaque test affiche OK ou ÉCHEC avec la cause. Le script s'arrête au
premier blocage réel plutôt que d'enchaîner des erreurs en cascade.
"""

from __future__ import annotations

import sys
from pathlib import Path

OK, KO, WARN = "  OK   ", " ÉCHEC ", " ATTENTION "
_fails = 0


def check(label: str, fn, critical: bool = True):
    global _fails
    try:
        detail = fn()
        print(f"[{OK}] {label}" + (f" — {detail}" if detail else ""))
        return True
    except Exception as e:
        _fails += 1 if critical else 0
        tag = KO if critical else WARN
        print(f"[{tag}] {label}")
        print(f"          {type(e).__name__}: {e}")
        return False


def main() -> int:
    print("=" * 62)
    print("  TRAIL LAB — autodiagnostic")
    print("=" * 62)

    # ── 1. Environnement ──────────────────────────────────────────────────
    print("\n1. Environnement")

    def py_version():
        v = sys.version_info
        if v < (3, 10):
            raise RuntimeError(f"Python {v.major}.{v.minor} trop ancien, il faut 3.10 ou plus")
        return f"Python {v.major}.{v.minor}.{v.micro}"
    if not check("Version de Python", py_version):
        return 1

    def deps():
        missing = []
        for mod, pkg in [("numpy", "numpy"), ("pandas", "pandas"),
                         ("gpxpy", "gpxpy"), ("fitparse", "fitparse"),
                         ("plotly", "plotly"), ("streamlit", "streamlit")]:
            try:
                __import__(mod)
            except ImportError:
                missing.append(pkg)
        if missing:
            raise RuntimeError(
                f"paquets manquants : {', '.join(missing)}. "
                "Lance : pip install -r requirements.txt")
        import numpy, pandas
        return f"numpy {numpy.__version__}, pandas {pandas.__version__}"
    if not check("Dépendances", deps):
        return 1

    def optional():
        try:
            import supabase
            return "supabase présent"
        except ImportError:
            raise RuntimeError("supabase absent — normal tant que tu restes en local")
    check("Dépendance optionnelle", optional, critical=False)

    # ── 2. Modules du projet ──────────────────────────────────────────────
    print("\n2. Modules du projet")
    sys.path.insert(0, str(Path(__file__).parent))

    def modules():
        for m in ["engine.physio", "engine.ingest", "engine.metrics",
                  "engine.predict", "engine.store", "engine.bike",
                  "engine.plan", "engine.archive", "engine.sync", "engine.analysis",
                  "engine.efforts",
                  "engine.elevation",
                  "integrations.strava"]:
            __import__(m)
        return "13 modules"
    if not check("Import des modules", modules):
        print("\n  → Lance ce script depuis le dossier du projet "
              "(celui qui contient app.py).")
        return 1

    # ── 3. Moteur de calcul ───────────────────────────────────────────────
    print("\n3. Moteur de calcul")
    import numpy as np
    import pandas as pd
    from engine import metrics, physio, predict

    def minetti():
        cr0 = float(physio.cost_running(0.0))
        if abs(cr0 - 3.6) > 0.01:
            raise RuntimeError(f"coût sur le plat = {cr0}, attendu 3,60")
        # La descente doit être MOINS chère que le plat jusqu'à environ -20 %
        if physio.cost_running(-0.10) >= cr0:
            raise RuntimeError("la descente ressort plus chère que le plat")
        gap_down = float(physio.gap(np.array([10 / 3.6]), np.array([-0.10]))[0]) * 3.6
        gap_up = float(physio.gap(np.array([6 / 3.6]), np.array([0.15]))[0]) * 3.6
        return f"10 km/h à -10 % = {gap_down:.1f} km/h eq · 6 km/h à +15 % = {gap_up:.1f}"
    check("Coûts énergétiques (Minetti)", minetti)

    def engine_run():
        rng = np.random.default_rng(0)
        n = 3600
        cum = np.arange(n) * 2.8
        ele = 500 + 150 * np.sin(cum / 1000) + 0.02 * cum + rng.normal(0, 0.6, n)
        df = pd.DataFrame({
            "t": pd.date_range("2026-01-01T08:00:00Z", periods=n, freq="2s"),
            "lat": 45 + cum / 111320, "lon": np.full(n, 6.0), "ele": ele,
            "hr": 150 + rng.normal(0, 4, n),
            "cad": np.full(n, 168.0) + rng.normal(0, 4, n),
            "power": np.nan,
        })
        res = metrics.analyze(df, 48, 188)
        s = res["summary"]
        if not (9.5 < s["distance_km"] < 10.5):
            raise RuntimeError(f"distance calculée {s['distance_km']:.2f} km, attendu ~10")
        if not (1.9 < s["duration_h"] < 2.1):
            raise RuntimeError(f"durée {s['duration_h']:.2f} h, attendu ~2")
        deq = predict.flat_equivalent_distance(res["points"])
        return (f"{s['distance_km']:.1f} km · {s['d_plus']:.0f} D+ · "
                f"GAP {s['gap_kmh']:.1f} km/h · {deq:.1f} km eq. plat")
    check("Analyse d'une trace synthétique", engine_run)

    def model():
        rng = np.random.default_rng(1)
        deq = rng.uniform(8, 45, 40)
        hist = pd.DataFrame({
            "date": pd.Timestamp("2025-09-01") + pd.to_timedelta(rng.integers(0, 340, 40), "D"),
            "deq_km": deq, "duration_h": 0.08 * deq ** 1.09 * rng.lognormal(0, .07, 40),
            "hrr_mean": rng.uniform(.65, .85, 40),
            "terrain": rng.choice(["trail", "route"], 40),
            "session_type": "continu",
        })
        m = predict.fit_endurance_model(hist)
        if not m["ok"]:
            raise RuntimeError(m["reason"])
        if not m["b_plausible"]:
            raise RuntimeError(f"exposant {m['b']:.3f} hors plage")
        return f"exposant {m['b']:.3f} (vrai 1,090) · R² {m['r2']:.3f}"
    check("Modèle d'endurance", model)

    def curve_test():
        from engine import efforts
        rng = np.random.default_rng(4)
        h = pd.DataFrame([{"activity_id": str(i), "sport": "trail",
                           "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i * 4),
                           "v30": 13.2 * rng.lognormal(0, .05),
                           "v60": 12.3 * rng.lognormal(0, .05),
                           "v120": 11.4 * rng.lognormal(0, .05)} for i in range(40)])
        c = efforts.build_curve(h)
        if not c["ok"]:
            raise RuntimeError(c["reason"])
        if not c["coherente"]:
            raise RuntimeError("courbe non décroissante")
        bs = efforts.base_speed(c, 9.0)
        if not bs["beta_plausible"]:
            raise RuntimeError(f"beta = {bs['beta']:.3f} hors plage")
        return (f"perte {bs['perte_par_doublement_pct']:.1f} %/doublement · "
                f"exposant implicite {bs['b_implicite']:.3f}")
    check("Courbe vitesse-durée", curve_test)

    # ── 4. Stockage ───────────────────────────────────────────────────────
    print("\n4. Stockage")
    from engine.store import Store, ACTIVITIES, get_store

    def secrets_locaux() -> dict:
        p = Path(".streamlit/secrets.toml")
        if not p.exists():
            return {}
        try:
            import tomllib
            with open(p, "rb") as f:
                return tomllib.load(f)
        except Exception:
            return {}

    def backend_reel():
        """
        Teste le backend RÉELLEMENT utilisé par l'application.

        La version précédente testait Store("local") en dur : elle
        affichait « OK » même quand la connexion Supabase était cassée,
        ce qui est exactement le cas où l'on a besoin d'un diagnostic.
        """
        sec = secrets_locaux()
        if not (sec.get("SUPABASE_URL") and sec.get("SUPABASE_KEY")):
            raise RuntimeError("Supabase non configuré — stockage local, "
                               "les données seront perdues au déploiement")
        st = get_store(sec)
        if st.backend != "supabase":
            raise RuntimeError("secrets présents mais backend local")
        n = len(st.read(ACTIVITIES))
        return f"supabase joignable · {n} activité(s) en base"
    check("Backend de production", backend_reel, critical=False)

    def storage():
        import shutil
        st = Store("local")
        # Date volontairement absurde : le test doit être ISOLÉ de tes
        # données. Avec une date plausible, la détection tombait sur une
        # vraie sortie chevauchant le créneau et le test échouait alors
        # que le mécanisme fonctionnait.
        st.upsert(ACTIVITIES, pd.DataFrame([{
            "activity_id": "_selftest", "date": "2099-06-15T03:17:00",
            "sport": "trail", "duration_h": 2.0, "name": "test"}]))
        dup = st.find_overlap("2099-06-15T03:18:00", 1.98)
        df = st.read(ACTIVITIES)
        # Nettoyage : on ne laisse pas de ligne de test dans l'historique
        df = df[df["activity_id"] != "_selftest"]
        (Path("data") / f"{ACTIVITIES}.csv").unlink(missing_ok=True)
        if not df.empty:
            df.to_csv(Path("data") / f"{ACTIVITIES}.csv", index=False, encoding="utf-8-sig")
        else:
            shutil.rmtree("data", ignore_errors=True)
        if dup != "_selftest":
            raise RuntimeError("la détection de doublon n'a pas fonctionné")
        return "écriture, lecture et anti-doublon"
    check("Base locale (CSV)", storage)

    # ── 5. Plan d'entraînement ────────────────────────────────────────────
    print("\n5. Plan d'entraînement")
    from engine import plan as plan_mod

    def plan_csv():
        p = Path("data_plan_templiers_2026.csv")
        if not p.exists():
            raise RuntimeError("data_plan_templiers_2026.csv absent du dossier")
        df = pd.read_csv(p, parse_dates=["date"])
        keys = int(df["seance_cle"].astype(str).str.lower().eq("true").sum())
        return (f"{len(df)} séances · {df['semaine'].nunique()} semaines · "
                f"{keys} séances clés · départ {df['date'].min():%d/%m/%Y}")
    check("Plan Templiers", plan_csv, critical=False)

    # ── 6. Fichier réel ───────────────────────────────────────────────────
    if len(sys.argv) > 1:
        print("\n6. Ton fichier")
        from engine import ingest

        target = Path(sys.argv[1])

        def read_file():
            if not target.exists():
                raise RuntimeError(f"introuvable : {target}")
            raw = ingest.load(str(target), target.name)
            if len(raw) < 30:
                raise RuntimeError(f"{len(raw)} points seulement — fichier vide ou indoor")
            cols = []
            for c, label in [("hr", "FC"), ("cad", "cadence"),
                             ("ele", "altitude"), ("power", "puissance")]:
                pct = raw[c].notna().mean()
                if pct > 0.5:
                    cols.append(label)
            manque = [l for c, l in [("hr", "FC"), ("cad", "cadence")]
                      if raw[c].notna().mean() <= 0.5]
            msg = f"{len(raw)} points · {', '.join(cols) or 'aucun capteur'}"
            if manque:
                msg += f" · MANQUE : {', '.join(manque)}"
            return msg

        if check("Lecture", read_file):
            def analyse_file():
                raw = ingest.load(str(target), target.name)
                res = metrics.analyze(raw, 48, 188)
                s = res["summary"]
                return (f"{s['distance_km']:.1f} km · {s['d_plus']:.0f} D+ · "
                        f"{predict.fmt_hours(s['duration_h'])} · "
                        f"GAP {s['gap_kmh']:.1f} km/h · "
                        f"marche {s['walk_share']:.0%} · "
                        f"type suggéré : {s.get('session_type', '?')}")
            check("Analyse", analyse_file)
    else:
        print("\n6. Ton fichier — non testé")
        print("       Relance avec : python check.py chemin/vers/une_sortie.tcx")

    # ── Bilan ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    if _fails == 0:
        print("  Tout est bon. Tu peux lancer :  streamlit run app.py")
    else:
        print(f"  {_fails} blocage(s). Corrige-les avant de continuer.")
    print("=" * 62)
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
