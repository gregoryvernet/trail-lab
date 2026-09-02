"""
archive.py — Import de l'archive Strava (export en masse).

POURQUOI CE MODULE EXISTE

La documentation officielle de Strava l'écrit deux fois sur la même page :
« A Strava subscription is a prerequisite for creating an app » et, à
l'étape de création, « Note: A Strava subscription is required ».

Sans abonnement, tu ne peux pas créer d'application, donc pas obtenir de
Client ID, donc pas utiliser l'API du tout. Ce n'est pas un quota qu'on
contourne : c'est la porte d'entrée qui est fermée.

LA VOIE DE CONTOURNEMENT

L'export en masse de tes données est gratuit et complet :
Strava → Paramètres → Mon compte → « Télécharger ou supprimer votre
compte » → Demander votre archive. Tu reçois par mail, sous quelques
heures, un ZIP contenant :

    activities.csv          index de toutes tes activités
    activities/*.fit.gz     les fichiers ORIGINAUX, compressés
    activities/*.gpx.gz
    activities/*.tcx.gz

Pour un rattrapage de 150 activités, c'est objectivement SUPÉRIEUR à
l'API : fichiers en résolution native (l'API rééchantillonne les flux),
aucun quota, aucune expiration de jeton, une seule opération. L'API
n'aurait été meilleure que pour la synchronisation quotidienne.

Pour le flux courant sans abonnement, trois options par ordre de
préférence : réexporter l'archive tous les trimestres ; exporter le TCX
depuis Polar Flow après chaque sortie ; ou prendre l'abonnement et
rebrancher `sync.py`, qui reste fonctionnel.
"""

from __future__ import annotations

import gzip
import io
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from engine import analysis, bike, efforts, elevation, ingest, metrics, predict
from engine.store import ACTIVITIES, SLOPE_BINS
from engine.sync import BIN_CENTERS, classify_terrain

# Les en-têtes de activities.csv changent selon la langue du compte et la
# version de l'export. On mappe par motif plutôt que par égalité stricte.
CSV_PATTERNS = {
    "activity_id": r"activity\s*id|id de l'?activit",
    "date": r"activity\s*date|date de l'?activit",
    "name": r"activity\s*name|nom de l'?activit",
    "type": r"activity\s*type|type d'?activit",
    "filename": r"^filename$|nom du fichier",
    "workout_type": r"workout\s*type|type de s[ée]ance|type d'?entra",
}

# DISCIPLINE : ce que TU as déclaré à Strava, sans réinterprétation.
#
# La version précédente déduisait « trail » ou « route » du dénivelé par
# kilomètre, en ignorant la déclaration. C'était présomptueux : tu sais ce
# que tu as couru, un seuil de 15 m/km ne le sait pas. La déclaration fait
# donc foi, et les incohérences sont signalées plutôt que corrigées en
# silence.
DISCIPLINE_FROM_CSV = {
    "trail run": "trail",
    "run": "route", "course à pied": "route", "virtual run": "route",
    "mountain bike ride": "vtt", "vtt": "vtt",
    "ride": "velo_route", "vélo": "velo_route", "cycling": "velo_route",
    "virtual ride": "velo_route", "e-bike ride": "velo_route",
    "ebikeride": "velo_route",
    "gravel ride": "gravel",
    "hike": "rando", "randonnée": "rando", "walk": "rando", "marche": "rando",
}

# Regroupement grossier, qui pilote les onglets de l'application.
SPORT_FROM_DISCIPLINE = {
    "trail": "trail", "route": "trail", "rando": "rando",
    "vtt": "velo", "velo_route": "velo", "gravel": "velo",
}

DISCIPLINE_LABELS = {
    "trail": "Trail", "route": "Course à pied", "vtt": "VTT",
    "velo_route": "Vélo route", "gravel": "Gravel", "rando": "Randonnée",
    "autre": "Autre",
}

SPORT_FROM_CSV = {k: SPORT_FROM_DISCIPLINE.get(v, "autre")
                  for k, v in DISCIPLINE_FROM_CSV.items()}

# TITRES AUTO-GÉNÉRÉS PAR STRAVA : la seule trace du type fin.
#
# L'archive Strava ne contient QUE le type grossier — Run, Ride, Virtual
# Ride, Hike. Le TrailRun n'y figure pas, alors que c'est lui qui distingue
# trail et route. Vérifié sur 556 sorties : aucune colonne ne le porte.
#
# Mais Strava compose ses titres automatiques à partir du type déclaré :
# « Lunch Trail Run » signifie que la sortie a été déclarée en Trail Run.
# Quand le titre n'a pas été renommé à la main, il porte donc l'information
# que le fichier CSV a perdu.
TITRE_VERS_DISCIPLINE = [
    (r"\btrail\s*run\b", "trail"),
    (r"\bmountain\s*bike\b|\bvtt\b", "vtt"),
    (r"\bgravel\b", "gravel"),
    (r"\bhike\b|\brando", "rando"),
    (r"\bvirtual\s*ride\b|\bhome\s*trainer\b|\bzwift\b", "velo_route"),
    (r"\bride\b|\bv[ée]lo\b", "velo_route"),
    (r"\brun\b|\bcourse\b|\bfooting\b", "route"),
]

# Seuil de repli, quand ni le CSV ni le titre ne tranchent.
DPLUS_PAR_KM_TRAIL = 20.0


def discipline_from_title(titre: str) -> str | None:
    """Type fin déduit d'un titre auto-généré. None si le titre ne dit rien."""
    if not titre:
        return None
    t = str(titre).lower()
    for motif, disc in TITRE_VERS_DISCIPLINE:
        if re.search(motif, t):
            return disc
    return None


def resolve_discipline(type_csv: str, titre: str, distance_km: float | None,
                       d_plus: float | None) -> tuple[str, str]:
    """
    Discipline par cascade de trois sources, de la plus fiable à la moins.

    Retourne (discipline, source) — la source importe autant que le
    résultat, puisqu'elle dit ce qu'on peut croire.

    1. LE TITRE, s'il porte le type. Auto-généré, il vient directement de
       ta déclaration : « Lunch Trail Run » est une déclaration, pas une
       interprétation.
    2. LE TYPE DU CSV, pour tout ce qui n'est pas de la course — Ride,
       Virtual Ride, Hike ne souffrent pas d'ambiguïté.
    3. LE DÉNIVELÉ PAR KILOMÈTRE, en dernier recours, pour les sorties que
       tu as renommées à la main — « Technique +++ », « KV », « 🥵🥵🥵 ».
       Ces cas sont signalés comme déduits, pas déclarés.
    """
    par_titre = discipline_from_title(titre)
    brut = str(type_csv or "").strip().lower()
    par_csv = DISCIPLINE_FROM_CSV.get(brut)

    # Le titre ne prime que s'il est cohérent avec la famille du CSV :
    # un titre « Trail Run » sur une activité déclarée Ride serait une
    # incohérence, pas une précision.
    if par_titre and par_csv:
        if SPORT_FROM_DISCIPLINE.get(par_titre) == SPORT_FROM_DISCIPLINE.get(par_csv):
            return par_titre, "titre"
    elif par_titre and not par_csv:
        return par_titre, "titre"

    if par_csv and par_csv != "route":
        return par_csv, "type Strava"

    # Reste la course à pied indistincte : on tranche au dénivelé.
    if distance_km and distance_km > 0.5 and d_plus is not None:
        dpkm = float(d_plus) / float(distance_km)
        if dpkm >= DPLUS_PAR_KM_TRAIL:
            return "trail", "déduit du D+/km"
        return "route", "déduit du D+/km"

    return par_csv or "autre", "type Strava"


def _discipline_repli(sport: str) -> str:
    """Discipline par défaut quand l'archive ne déclare rien."""
    return {"trail": "trail", "velo": "velo_route", "rando": "rando"}.get(
        sport, "autre")


def coherence_disciplines(hist: pd.DataFrame,
                          seuil_trail: float = 10.0,
                          seuil_route: float = 30.0) -> pd.DataFrame:
    """
    Sorties dont la déclaration semble contredire le profil.

    Ne corrige rien : signale. Deux cas.

      - déclarée « Run » mais au-delà de 30 m de D+ par kilomètre : c'était
        probablement du trail, et elle est aujourd'hui exclue du calibrage ;
      - déclarée « Trail Run » mais sous 10 m/km : elle tire le niveau du
        modèle trail sans en relever.

    L'enjeu n'est pas cosmétique. Le modèle de prédiction se calibre sur les
    seules sorties trail : chaque étiquette erronée déplace le niveau.
    """
    if hist.empty or "discipline" not in hist.columns:
        return pd.DataFrame()
    d = hist.copy()
    if "dplus_par_km" not in d.columns or d["dplus_par_km"].isna().all():
        with np.errstate(divide="ignore", invalid="ignore"):
            d["dplus_par_km"] = np.where(
                d.get("distance_km", pd.Series(0, index=d.index)) > 0.5,
                d.get("d_plus", np.nan) / d.get("distance_km", np.nan), np.nan)
    d = d[d["dplus_par_km"].notna()]
    if d.empty:
        return pd.DataFrame()

    # Une sortie déjà résolue PAR LE DÉNIVELÉ ne peut pas être en
    # contradiction avec le dénivelé : la signaler serait absurde. On ne
    # contrôle donc que ce qui a été déclaré — par le titre ou par le type.
    if "discipline_source" in d.columns:
        d = d[d["discipline_source"].isin(["titre", "type Strava"])
              | d["discipline_source"].isna()]
    if d.empty:
        return pd.DataFrame()

    a = d[(d["discipline"] == "route") & (d["dplus_par_km"] >= seuil_route)].copy()
    a["signal"] = "déclarée route, profil de trail"
    b = d[(d["discipline"] == "trail") & (d["dplus_par_km"] <= seuil_trail)].copy()
    b["signal"] = "déclarée trail, profil plat"
    # Filtrer les colonnes sur le RÉSULTAT, pas sur la table d'entrée :
    # `signal` n'existe que sur a et b, elle était donc écartée.
    out = pd.concat([a, b])
    cols = [c for c in ["activity_id", "date", "name", "discipline",
                        "discipline_source", "distance_km", "d_plus",
                        "dplus_par_km", "ke_km", "signal"]
            if c in out.columns]
    return out[cols].sort_values("dplus_par_km", ascending=False)


def repair_disciplines(zip_source, store) -> dict:
    """
    Renseigne `discipline` sur les activités DÉJÀ en base, sans retraiter
    les traces.

    La discipline vient du fichier activities.csv de l'archive, pas de
    l'analyse du parcours : aucun recalcul n'est nécessaire. Réimporter
    trois cents sorties pour remplir une colonne serait absurde.
    """
    zf = zipfile.ZipFile(zip_source)
    index = read_index(zf)
    index["activity_id"] = index["activity_id"].astype(str)
    index = index.drop_duplicates(subset=["activity_id"]).set_index("activity_id")

    hist = store.read(ACTIVITIES)
    if hist.empty:
        return {"ok": False, "reason": "Base vide."}

    # Les reclassements MANUELS sont préservés. Sans cela, relancer ce
    # script après un travail de tri à la main l'effacerait entièrement —
    # et sans le signaler.
    manuels = set()
    if "discipline_source" in hist.columns:
        manuels = set(hist.loc[hist["discipline_source"] == "manuel",
                               "activity_id"].astype(str))

    maj, hors, preserves = [], 0, 0
    for aid in hist["activity_id"].astype(str):
        if aid in manuels:
            preserves += 1
            continue
        if aid not in index.index:
            hors += 1
            continue
        r = index.loc[aid]
        maj.append({"activity_id": aid,
                    "discipline": r["discipline"],
                    "discipline_source": r.get("discipline_source"),
                    "strava_type": r["strava_type"],
                    "sport": r["sport"]})
    if maj:
        store.upsert(ACTIVITIES, pd.DataFrame(maj), key="activity_id")
    return {"ok": True, "mises_a_jour": len(maj), "hors_archive": hors,
            "preserves": preserves,
            "repartition": (pd.DataFrame(maj)["discipline"].value_counts().to_dict()
                            if maj else {}),
            "sources": (pd.DataFrame(maj)["discipline_source"].value_counts().to_dict()
                        if maj else {})}


def has_elevation(raw: pd.DataFrame, min_range_m: float = 5.0) -> bool:
    """
    La trace porte-t-elle une altitude réellement mesurée ?

    Deux cas à écarter, et le second est le piège : une colonne absente,
    mais aussi une colonne présente et constante. Les montres Polar sans
    altimètre barométrique écrivent le champ et le laissent vide ou figé —
    on ne peut donc pas se contenter de tester la présence de la colonne.
    """
    ele = raw["ele"].dropna()
    if len(ele) < 30:
        return False
    return float(ele.max() - ele.min()) >= min_range_m


def _map_columns(df: pd.DataFrame) -> dict:
    found = {}
    for key, pattern in CSV_PATTERNS.items():
        for col in df.columns:
            if re.search(pattern, str(col).strip().lower()):
                found[key] = col
                break
    return found


def read_index(zf: zipfile.ZipFile) -> pd.DataFrame:
    """Lit activities.csv et normalise ses colonnes."""
    name = next((n for n in zf.namelist()
                 if os.path.basename(n).lower() == "activities.csv"), None)
    if not name:
        raise ValueError(
            "activities.csv introuvable dans l'archive. Vérifie que tu as "
            "bien téléchargé l'export complet et non un fichier isolé."
        )
    with zf.open(name) as f:
        df = pd.read_csv(f, low_memory=False)

    cols = _map_columns(df)
    missing = {"activity_id", "filename"} - set(cols)
    if missing:
        raise ValueError(
            f"Colonnes introuvables dans activities.csv : {missing}. "
            f"Colonnes présentes : {list(df.columns)[:12]}"
        )

    out = pd.DataFrame({
        "activity_id": df[cols["activity_id"]].astype(str),
        "filename": df[cols["filename"]].astype(str),
    })
    out["date"] = pd.to_datetime(df[cols["date"]], errors="coerce",
                                 format="mixed") if "date" in cols else pd.NaT
    out["name"] = df[cols["name"]] if "name" in cols else ""
    raw_type = (df[cols["type"]].astype(str).str.strip().str.lower()
                if "type" in cols else pd.Series("run", index=df.index))
    out["strava_type"] = raw_type
    # Distance et D+ du CSV, pour la cascade de résolution.
    def num(motif):
        for c in df.columns:
            if re.search(motif, str(c).lower()):
                return pd.to_numeric(df[c], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    dist_km = num(r"^distance$|distance.*km|distance\s*\(")
    if dist_km.notna().any() and float(dist_km.median()) > 1000:
        dist_km = dist_km / 1000          # certains exports sont en mètres
    dpl = num(r"elevation gain|d[ée]nivel|positif")

    res = [resolve_discipline(t, n, d, e) for t, n, d, e
           in zip(raw_type, out["name"], dist_km, dpl)]
    out["discipline"] = [r[0] for r in res]
    out["discipline_source"] = [r[1] for r in res]
    out["sport"] = out["discipline"].map(SPORT_FROM_DISCIPLINE).fillna("autre")
    out["workout_type"] = (df[cols["workout_type"]].astype(str).str.lower()
                           if "workout_type" in cols else "")
    return out[out["filename"].notna() & (out["filename"] != "nan")]


def _open_track(zf: zipfile.ZipFile, filename: str):
    """Décompresse un fichier de trace et renvoie (objet fichier, extension)."""
    candidates = [n for n in zf.namelist() if n.endswith(filename)
                  or os.path.basename(n) == os.path.basename(filename)]
    if not candidates:
        raise FileNotFoundError(filename)
    inner = candidates[0]
    data = zf.read(inner)
    if inner.endswith(".gz"):
        data = gzip.decompress(data)
        inner = inner[:-3]
    return io.BytesIO(data), os.path.splitext(inner)[-1].lower()


def import_archive(zip_source, store, hr_rest: float = 50, hr_max: float = 190,
                   ftp: float | None = None, poids_kg: float | None = None,
                   sports=("trail", "rando", "velo"),
                   limit: int | None = None, progress=None,
                   fill_elevation: bool = False,
                   months: int | None = None) -> dict:
    """
    zip_source : chemin ou objet fichier du ZIP d'export Strava.
    Écrit en base au fil de l'eau : l'import est reprenable.

    fill_elevation : reconstitue l'altitude manquante depuis un modèle de
    terrain public. Indispensable pour les TCX/FIT Polar, qui n'en ont pas.

    months : ne garde que les N derniers mois. Sur une archive couvrant
    douze ans, importer l'intégralité coûterait du quota de modèle de
    terrain pour des sorties sans valeur pour le modèle actuel.
    """
    zf = zipfile.ZipFile(zip_source)
    index = read_index(zf)
    index = index[index["sport"].isin(sports)]

    if months:
        cutoff = pd.Timestamp.now().tz_localize(None) - pd.DateOffset(months=months)
        dates = pd.to_datetime(index["date"], errors="coerce")
        if getattr(dates.dtype, "tz", None) is not None:
            dates = dates.dt.tz_localize(None)
        index = index[dates >= cutoff]

    known = store.known_ids(ACTIVITIES)
    todo = index[~index["activity_id"].isin(known)]
    if limit:
        todo = todo.head(limit)

    dem = elevation.ElevationCache() if fill_elevation else None
    report = {"imported": 0, "skipped": int(len(index) - len(todo)),
              "failed": [], "no_gps": 0, "duplicates": 0, "no_elevation": 0,
              "enriched": 0, "dem_queried": 0, "dem_cached": 0,
              "quota_reached": False, "reseau_ko": None, "repli_cache": 0,
              "sans_gps_alt": 0}
    total = len(todo)

    for i, (_, meta) in enumerate(todo.iterrows(), 1):
        if progress:
            progress(i, total, str(meta["name"])[:40])
        try:
            fh, ext = _open_track(zf, meta["filename"])
            raw = ingest.load(fh, f"x{ext}")
            if len(raw) < 30:
                report["no_gps"] += 1
                continue
            # Altitude absente (TCX/FIT Polar sans altimètre) : on la
            # reconstitue depuis le modèle de terrain plutôt que de perdre
            # la sortie. La rejeter perdrait 430 trails ; la traiter comme
            # du plat serait pire, elle entrerait à 0 D+ et serait classée
            # « route », faussant l'effet de terrain du modèle.
            if not has_elevation(raw):
                if dem is None:
                    report["no_elevation"] += 1
                    continue
                raw, st = elevation.enrich(raw, dem)
                report["enriched"] += 1
                report["dem_queried"] += st["queried"]
                report["dem_cached"] += st["cached"]
                report["repli_cache"] += int(st.get("repli_cache", False))
            row, bins = _process(raw, meta, hr_rest, hr_max, ftp, poids_kg)
            dup = store.find_overlap(row["date"], row.get("duration_h", 0))
            if dup:
                report["duplicates"] = report.get("duplicates", 0) + 1
                continue
        except elevation.Quota:
            # Quota du modèle de terrain épuisé : on s'arrête proprement en
            # conservant tout ce qui a été obtenu. Relancer demain reprendra
            # où on en est, sans redemander une seule altitude.
            report["quota_reached"] = True
            break
        except elevation.Reseau as e:
            # Service injoignable : on s'arrête AU PREMIER constat. Laisser
            # l'erreur tomber dans le except générique marquait chaque
            # activité en échec — 138 d'affilée pour une cause unique.
            report["reseau_ko"] = str(e)
            break
        except ValueError as e:
            # Trace sans coordonnées : ce n'est pas un échec technique mais
            # une catégorie à part, sans quoi la liste des échecs devient
            # illisible et masque les vrais problèmes.
            if "coordonnées" in str(e) or "GPS" in str(e):
                report["sans_gps_alt"] += 1
                continue
            report["failed"].append((meta["activity_id"], meta["name"], str(e)))
            continue
        except FileNotFoundError:
            report["no_gps"] += 1
            continue
        except Exception as e:
            report["failed"].append((meta["activity_id"], meta["name"], str(e)))
            continue

        store.upsert(ACTIVITIES, pd.DataFrame([row]), key="activity_id")
        if not bins.empty:
            store.upsert(SLOPE_BINS, bins, key=["activity_id", "bande"])
        report["imported"] += 1

        # Sauvegarde régulière du cache : une coupure ne doit pas coûter
        # les requêtes déjà payées.
        if dem is not None and dem.new >= 500:
            dem.save()

    if dem is not None:
        dem.save()
        report["dem_points"] = len(dem)
    return report


def build_row(raw: pd.DataFrame, meta: pd.Series, hr_rest=50, hr_max=190,
              ftp=None, poids_kg=None):
    """
    Point d'entrée unique pour transformer une trace en ligne de base.

    Partagé par l'import d'archive, l'import de dossier ET le dépôt manuel
    depuis l'application. Une seule implémentation : sinon les trois chemins
    divergent au premier changement de métrique et l'historique devient
    incohérent selon la façon dont chaque sortie est entrée.
    """
    return _process(raw, meta, hr_rest, hr_max, ftp, poids_kg)


def _process(raw: pd.DataFrame, meta: pd.Series, hr_rest, hr_max, ftp,
             poids_kg=None):
    sport = meta["sport"]
    # Plafond anti-décrochage calé sur la DISCIPLINE, plus fine que le
    # sport : un vélo de route et un VTT ne décrochent pas aux mêmes
    # vitesses. Un plafond trop bas ampute la distance sans avertir — il
    # ramenait 26,7 km à 13,1 km sur une sortie vélo.
    discipline = meta.get("discipline") or _discipline_repli(sport)
    d = metrics.prepare(raw, max_speed_ms=metrics.max_speed_for(discipline))

    row = {
        "activity_id": str(meta["activity_id"]),
        "source": "archive",
        "sport": sport,
        "discipline": meta.get("discipline") or _discipline_repli(sport),
        "strava_type": meta.get("strava_type"),
        "date": meta["date"],
        "name": meta["name"],
    }

    if sport == "velo":
        # Une archive contient les fichiers originaux : si le champ power
        # est présent, il vient du capteur, pas d'une estimation Strava.
        has_power = raw["power"].notna().any()
        row.update(bike.summarize_ride(d, ftp, has_power, hr_rest, hr_max))
        row.update(bike.summarize_power(d, poids_kg, has_power))
        row["poids_kg"] = poids_kg
        row["terrain"] = "velo"
        bins = pd.DataFrame()
    else:
        s = metrics.summarize(d, hr_rest, hr_max)
        row.update(s)
        row["deq_km"] = predict.flat_equivalent_distance(d)
        row["ke_km"] = predict.km_effort(s["distance_km"], s["d_plus"])
        row.update(analysis.summarize_transitions(d))
        row.update(efforts.summarize(d))
        # `terrain` reste calculé, mais comme CONTRÔLE de la déclaration,
        # plus comme vérité. Il sert à signaler les incohérences.
        row["terrain"] = classify_terrain(s["distance_km"], s["d_plus"])
        row["dplus_par_km"] = (s["d_plus"] / s["distance_km"]
                               if s["distance_km"] > 0.5 else np.nan)
        row["trimp"] = bike.trimp(d["hr"].to_numpy(), d["dt"].to_numpy(),
                                  hr_rest, hr_max)
        prof = metrics.session_profile(d)
        row.update(prof)
        row["session_type"] = reconcile_type(prof["session_type"],
                                             meta.get("workout_type", ""))
        bins = metrics.by_slope_bin(d)
        if not bins.empty:
            bins["activity_id"] = str(meta["activity_id"])
            bins["date"] = meta["date"]
            bins["pente_centre"] = bins["bande"].map(BIN_CENTERS)

    return row, bins


def reconcile_type(detected: str, strava_label: str) -> str:
    """
    L'étiquette Strava prime quand elle existe : c'est toi qui l'as posée.
    La détection automatique sert de filet pour les activités non étiquetées,
    qui sont la majorité.
    """
    lab = str(strava_label).strip().lower()
    if "race" in lab or "course" in lab or lab == "1":
        return "course"
    if "workout" in lab or "séance" in lab or "fractionn" in lab or lab == "3":
        return "fractionné"
    if "long" in lab or lab == "2":
        return "sortie longue"
    return detected


# ── Import d'un dossier local ────────────────────────────────────────────────

def import_folder(folder, store, hr_rest: float = 50, hr_max: float = 190,
                  ftp: float | None = None, poids_kg: float | None = None,
                  progress=None,
                  fill_elevation: bool = False) -> dict:
    """
    Importe un dossier de fichiers TCX / FIT / GPX déjà présents sur le disque.

    Chemin le plus court quand tu as déjà exporté tes séances depuis Polar
    Flow : ni archive à demander, ni abonnement, ni quota. L'identifiant
    d'activité est dérivé du nom de fichier, donc un même fichier réimporté
    ne crée pas de doublon.

    Le sport est déduit du contenu, pas du nom : présence de puissance et
    vitesse moyenne élevée → vélo. C'est imparfait sur une sortie vélo lente
    sans capteur ; le paramètre `sport` de la ligne reste corrigeable en base.
    """
    folder = Path(folder)
    files = sorted(f for f in folder.rglob("*")
                   if f.suffix.lower() in (".tcx", ".fit", ".gpx", ".gz"))
    known = store.known_ids(ACTIVITIES)
    dem = elevation.ElevationCache() if fill_elevation else None
    report = {"imported": 0, "skipped": 0, "failed": [], "no_gps": 0,
              "duplicates": 0, "no_elevation": 0, "enriched": 0,
              "dem_queried": 0, "dem_cached": 0, "quota_reached": False}

    todo = [f for f in files if _file_id(f) not in known]
    report["skipped"] = len(files) - len(todo)

    for i, path in enumerate(todo, 1):
        if progress:
            progress(i, len(todo), path.name[:40])
        try:
            if path.suffix.lower() == ".gz":
                fh = io.BytesIO(gzip.decompress(path.read_bytes()))
                ext = Path(path.stem).suffix.lower()
            else:
                fh, ext = io.BytesIO(path.read_bytes()), path.suffix.lower()
            raw = ingest.load(fh, f"x{ext}")
            if len(raw) < 30:
                report["no_gps"] += 1
                continue
            if not has_elevation(raw):
                if dem is None:
                    report["no_elevation"] += 1
                    continue
                raw, st = elevation.enrich(raw, dem)
                report["enriched"] += 1
                report["dem_queried"] += st["queried"]
                report["dem_cached"] += st["cached"]

            meta = pd.Series({
                "activity_id": _file_id(path),
                "filename": path.name,
                "date": raw["t"].iloc[0],
                "name": path.stem,
                "sport": _guess_sport(raw),
                "workout_type": "",
            })
            row, bins = _process(raw, meta, hr_rest, hr_max, ftp, poids_kg)
            if store.find_overlap(row["date"], row.get("duration_h", 0)):
                report["duplicates"] += 1
                continue
        except elevation.Quota:
            report["quota_reached"] = True
            break
        except Exception as e:
            report["failed"].append((path.name, path.stem, str(e)))
            continue

        store.upsert(ACTIVITIES, pd.DataFrame([row]), key="activity_id")
        if not bins.empty:
            store.upsert(SLOPE_BINS, bins, key=["activity_id", "bande"])
        report["imported"] += 1
        if dem is not None and dem.new >= 500:
            dem.save()

    if dem is not None:
        dem.save()
        report["dem_points"] = len(dem)
    return report


def _file_id(path: Path) -> str:
    return f"file_{re.sub(r'[^A-Za-z0-9]+', '_', path.stem)[:60]}"


def _guess_sport(raw: pd.DataFrame) -> str:
    """
    Détection du sport par faisceau d'indices, avec la cadence en juge de paix.

    LA VERSION PRÉCÉDENTE SE TROMPAIT SOUVENT. Elle testait la présence de
    puissance, puis une vitesse médiane supérieure à 5,5 m/s. Deux défauts :
    une sortie VTT technique descend sous 5,5 m/s de médiane et sortait en
    trail ; un home-trainer sans GPS n'avait pas de vitesse du tout.

    La cadence est le discriminant le plus net qui existe ici, parce que les
    deux gestes n'ont aucun recouvrement : une foulée de course tourne entre
    150 et 185 pas par minute, un coup de pédale entre 60 et 100 tours. Une
    médiane sous 120 tranche donc sans ambiguïté, et cela fonctionne aussi
    en intérieur, sans GPS.

    Trois indices secondaires servent quand la cadence manque :
      - la vitesse de pointe : un vélo dépasse presque toujours 10 m/s en
        descente, un coureur jamais ;
      - la roue libre : des passages à vitesse élevée sans cadence, ce qui
        n'a pas d'équivalent en course ;
      - la puissance mesurée, qui n'existe qu'à vélo dans cette base.
    """
    score, motifs = 0.0, []

    cad = raw["cad"].to_numpy(dtype=float)
    cad = cad[~np.isnan(cad) & (cad > 0)]
    if len(cad) > 50:
        med = float(np.median(cad))
        if med < 120:
            score += 3.0
            motifs.append(f"cadence {med:.0f}")
        elif med > 140:
            score -= 3.0
            motifs.append(f"cadence {med:.0f}")

    if raw["power"].notna().sum() > len(raw) * 0.3:
        score += 1.5
        motifs.append("puissance")

    lat, lon = raw["lat"].to_numpy(), raw["lon"].to_numpy()
    if not np.isnan(lat).all():
        dt = raw["t"].diff().dt.total_seconds().to_numpy()[1:]
        dist = metrics.haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
        ok = (dt > 0) & (dt < 30) & ~np.isnan(dist)
        if ok.sum() > 30:
            v = dist[ok] / dt[ok]
            v = v[v < 30]                      # anti-décrochage GPS
            if len(v) > 30:
                pointe = float(np.quantile(v, 0.97))
                if pointe > 10.0:
                    score += 2.0
                    motifs.append(f"pointe {pointe * 3.6:.0f} km/h")
                elif pointe < 6.0:
                    score -= 1.5
                    motifs.append(f"pointe {pointe * 3.6:.0f} km/h")
                if float(np.median(v)) > 5.5:
                    score += 1.0

                # Roue libre : vite sans pédaler.
                if len(cad) > 50 and len(raw) > 100:
                    c = raw["cad"].to_numpy(dtype=float)[1:][ok]
                    rapide = v > np.quantile(v, 0.6)
                    libre = rapide & ((c == 0) | np.isnan(c))
                    part = float(libre.sum() / max(rapide.sum(), 1))
                    if part > 0.15:
                        score += 1.5
                        motifs.append(f"roue libre {part:.0%}")

    return "velo" if score >= 2.0 else "trail"


def guess_sport_detail(raw: pd.DataFrame) -> dict:
    """Même détection, en exposant le raisonnement — utile au débogage."""
    sport = _guess_sport(raw)
    cad = raw["cad"].dropna()
    return {"sport": sport,
            "cadence_mediane": float(cad.median()) if len(cad) > 50 else None,
            "a_puissance": bool(raw["power"].notna().sum() > len(raw) * 0.3)}


def inventory(zip_source) -> pd.DataFrame:
    """
    Inventaire d'une archive AVANT import : formats de fichiers réellement
    présents. Utile car le format dépend de ce que ta montre a poussé vers
    Strava, pas d'un choix de ta part.
    """
    zf = zipfile.ZipFile(zip_source)
    idx = read_index(zf)
    idx["format"] = (idx["filename"].str.replace(r"\.gz$", "", regex=True)
                     .str.extract(r"\.(\w+)$")[0].str.lower())
    return (idx.groupby(["sport", "format"]).size()
            .rename("n").reset_index().sort_values("n", ascending=False))


def missing_from_store(zip_source, store, months: int | None = None,
                       sports=("trail", "rando", "velo")) -> pd.DataFrame:
    """
    Activités présentes dans l'archive mais absentes de la base.

    Répond à la question qui se pose après tout import partiel : lesquelles
    manquent, et de quand datent-elles. Le rapport d'import n'en liste que
    les dix premières, ce qui ne permet pas de décider s'il vaut la peine
    d'insister.
    """
    zf = zipfile.ZipFile(zip_source)
    index = read_index(zf)
    index = index[index["sport"].isin(sports)]
    if months:
        cutoff = pd.Timestamp.now().tz_localize(None) - pd.DateOffset(months=months)
        dates = pd.to_datetime(index["date"], errors="coerce")
        if getattr(dates.dtype, "tz", None) is not None:
            dates = dates.dt.tz_localize(None)
        index = index[dates >= cutoff]

    connus = store.known_ids(ACTIVITIES)
    manquantes = index[~index["activity_id"].isin(connus)].copy()
    manquantes["format"] = (manquantes["filename"]
                            .str.replace(r"\.gz$", "", regex=True)
                            .str.extract(r"\.(\w+)$")[0].str.lower())
    cols = [c for c in ["activity_id", "date", "name", "sport", "discipline",
                        "discipline_source", "format"] if c in manquantes.columns]
    return manquantes.sort_values("date", ascending=False)[cols]
