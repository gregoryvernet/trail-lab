"""
store.py — Persistance.

Point de vigilance majeur pour le déploiement : le système de fichiers de
Streamlit Community Cloud est ÉPHÉMÈRE. Un SQLite ou un CSV écrit sur le
disque disparaît à chaque redémarrage du conteneur (mise en veille après
inactivité, redéploiement, maintenance). Une base locale n'est donc utilisable
qu'en développement.

Deux backends :
  - "local"    : CSV dans ./data — développement uniquement
  - "supabase" : Postgres hébergé, offre gratuite suffisante ici

Le schéma est délibérément plat : une ligne = une activité résumée. Les
séries temporelles complètes (des dizaines de milliers de points par sortie)
ne sont pas stockées ; elles sont recalculées à la demande depuis Strava.
Stocker les points bruts ferait exploser la base pour un gain nul.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import datetime

DATA_DIR = Path("data")
ACTIVITIES = "activities"
SLOPE_BINS = "slope_bins"
TOKENS = "tokens"
JOURNAL = "journal"
COURSES = "courses"


def _json_safe(df: pd.DataFrame) -> list[dict]:
    """
    Convertit un DataFrame en liste de dicts sérialisables en JSON.

    DEUX PIÈGES, tous deux rencontrés en production.

    1. `df.where(pd.notna(df), None)` NE MARCHE PAS sur une colonne
       numérique. Pandas y conserve le type float64, et None y est
       reconverti en NaN. Postgres refuse alors l'écriture avec
       « Out of range float values are not JSON compliant: nan » — message
       qui ne désigne pas la cause. Il faut passer par le type objet.

    2. Les types numpy ne sont pas sérialisables. np.float64, np.int64 et
       np.bool_ ressemblent à des nombres Python mais le module json les
       refuse. On les ramène aux types natifs.
    """
    d = df.astype(object).where(pd.notna(df), None)
    out = []
    for rec in d.to_dict("records"):
        propre = {}
        for k, v in rec.items():
            if v is None:
                propre[k] = None
            elif isinstance(v, (np.floating, float)):
                propre[k] = None if not np.isfinite(v) else float(v)
            elif isinstance(v, (np.integer,)):
                propre[k] = int(v)
            elif isinstance(v, (np.bool_, bool)):
                propre[k] = bool(v)
            elif isinstance(v, (pd.Timestamp, datetime)):
                propre[k] = pd.Timestamp(v).isoformat()
            elif isinstance(v, np.ndarray):
                propre[k] = v.tolist()
            else:
                propre[k] = v
        out.append(propre)
    return out


def clean_supabase_url(url: str) -> str:
    """
    Ramène l'URL à la RACINE du projet.

    Supabase affiche selon les écrans soit `https://xxx.supabase.co`, soit
    la même chose suivie de `/rest/v1/`. La bibliothèque cliente ajoute
    elle-même ce chemin : lui donner la forme complète produit
    `//rest/v1/rest/v1/...` et une erreur PGRST125 « Invalid path specified
    in request URL » — dont le message ne dit rien de la cause réelle.

    On accepte donc les deux formes, ainsi que la barre finale et les
    espaces d'un copier-coller.
    """
    u = str(url).strip().rstrip("/")
    for suffixe in ("/rest/v1", "/rest", "/auth/v1", "/storage/v1"):
        if u.endswith(suffixe):
            u = u[: -len(suffixe)]
    return u.rstrip("/")


def _parse_dates(col) -> pd.Series:
    """
    Lecture de dates en formats HÉTÉROGÈNES.

    Bug réel, trouvé au test. Un import d'archive écrit
    « 2026-08-22 15:11:27 » (séparateur espace, via Timestamp), une synchro
    Strava écrit « 2026-08-22T15:11:27Z » (ISO avec T et Z). Les deux
    cohabitent dans le même fichier.

    Or pandas déduit UN SEUL format depuis les premières lignes, puis
    transforme en NaT tout ce qui n'y correspond pas — silencieusement,
    puisque errors="coerce". Résultat observé : sur 258 lignes, celle au
    format minoritaire devenait NaT et échappait à la détection de
    doublons. En production, cela aurait laissé passer des doublons entre
    sorties importées par des voies différentes, donc un volume
    hebdomadaire faux et un modèle nourri deux fois par les mêmes points.

    format="mixed" force l'analyse ligne par ligne. C'est plus lent, sans
    conséquence sur des tables de quelques centaines de lignes.
    """
    d = pd.to_datetime(col, errors="coerce", utc=True, format="mixed")
    return d.dt.tz_localize(None)


class Store:
    def __init__(self, backend: str = "local", secrets: dict | None = None):
        self.backend = backend
        self.secrets = secrets or {}
        if backend == "local":
            DATA_DIR.mkdir(exist_ok=True)
            self.client = None
        elif backend == "supabase":
            from supabase import create_client
            self.client = create_client(
                clean_supabase_url(self.secrets["SUPABASE_URL"]),
                str(self.secrets["SUPABASE_KEY"]).strip(),
            )
        else:
            raise ValueError(f"Backend inconnu : {backend}")

    # ── Lecture ───────────────────────────────────────────────────────────
    def read(self, table: str) -> pd.DataFrame:
        if self.backend == "local":
            # CSV et non parquet : le backend local sert au développement et
            # au débogage, où pouvoir ouvrir le fichier à la main vaut mieux
            # qu'un format binaire. Cela évite aussi une dépendance à pyarrow
            # dont l'absence ne se manifesterait qu'au premier import.
            path = DATA_DIR / f"{table}.csv"
            if not path.exists():
                return pd.DataFrame()
            return pd.read_csv(path, low_memory=False)
        return self._read_paged(table)

    # PostgREST plafonne toute réponse à 1 000 lignes par défaut, SANS
    # erreur ni avertissement. Sur slope_bins — sept à neuf bandes par
    # activité, soit environ 2 000 lignes pour 305 sorties — la moitié des
    # données devenait invisible et toutes les analyses par bande étaient
    # faussées à l'insu de l'utilisateur. Le pire cas de figure : un
    # résultat plausible mais faux.
    PAGE = 1000

    def _read_paged(self, table: str) -> pd.DataFrame:
        morceaux, debut = [], 0
        while True:
            res = (self.client.table(table).select("*")
                   .range(debut, debut + self.PAGE - 1).execute())
            lot = res.data or []
            if not lot:
                break
            morceaux.append(pd.DataFrame(lot))
            if len(lot) < self.PAGE:
                break
            debut += self.PAGE
            if debut > 200_000:              # garde-fou anti-boucle
                break
        return (pd.concat(morceaux, ignore_index=True) if morceaux
                else pd.DataFrame())

    # ── Écriture (upsert sur une clé simple ou composite) ─────────────────
    def upsert(self, table: str, df: pd.DataFrame,
               key: str | list[str] = "activity_id") -> None:
        """
        key accepte une liste, et ce n'est pas un raffinement.

        La table des bandes de pente contient six à neuf lignes par
        activité, une par bande. Dédoublonner sur `activity_id` seul y
        écrasait donc toutes les bandes sauf une à chaque écriture : la
        table paraissait remplie, mais 85 % des données avaient disparu
        silencieusement. Symptôme observé côté interface — une seule bande
        de montée affichée, aucune en descente, sur 232 sorties.
        """
        if df.empty:
            return
        # Normalisation des dates à l'ÉCRITURE : on ne crée plus
        # d'hétérogénéité de format dans le fichier.
        df = df.copy()
        for c in ("date", "importe_le", "maj", "realise_date"):
            if c in df.columns:
                v = pd.to_datetime(df[c], errors="coerce", utc=True, format="mixed")
                df[c] = v.dt.tz_localize(None).dt.strftime("%Y-%m-%dT%H:%M:%S")
        keys = [key] if isinstance(key, str) else list(key)

        if self.backend == "local":
            path = DATA_DIR / f"{table}.csv"
            existing = self.read(table)
            present = [k for k in keys if k in df.columns]

            # MISE À JOUR PARTIELLE : ne pas écraser les colonnes absentes.
            #
            # Un simple concat suivi de drop_duplicates(keep="last") remplace
            # la ligne entière par la nouvelle. Une mise à jour ne portant que
            # sur deux colonnes — reclasser une discipline, par exemple —
            # effaçait donc silencieusement le nom, la distance et les
            # soixante autres. PostgREST ne souffre pas de ce défaut, il ne
            # met à jour que les colonnes fournies ; il faut reproduire ce
            # comportement en local.
            if not existing.empty and present:
                for k in present:
                    existing[k] = existing[k].astype(str)
                    df[k] = df[k].astype(str)
                idx = existing.set_index(present)
                neuf = df.set_index(present)
                communs = idx.index.intersection(neuf.index)
                if len(communs):
                    for col in neuf.columns:
                        if col not in idx.columns:
                            idx[col] = pd.NA
                        idx.loc[communs, col] = neuf.loc[communs, col]
                ajouts = neuf.loc[neuf.index.difference(communs)]
                merged = pd.concat([idx, ajouts]).reset_index()
            else:
                merged = pd.concat([existing, df], ignore_index=True)
                if present:
                    for k in present:
                        merged[k] = merged[k].astype(str)
                    merged = merged.drop_duplicates(subset=present, keep="last")
            merged.to_csv(path, index=False, encoding="utf-8-sig")
        else:
            self.client.table(table).upsert(
                _json_safe(df), on_conflict=",".join(keys)).execute()

    def known_ids(self, table: str = ACTIVITIES, key: str = "activity_id") -> set:
        df = self.read(table)
        return set(df[key].astype(str)) if key in df.columns else set()

    def find_overlap(self, start, duration_h: float,
                     start_tol_min: float = 3.0,
                     dur_tol_min: float = 5.0) -> str | None:
        """
        Cherche une activité déjà en base qui correspond au même effort réel.

        Nécessaire dès qu'on mélange les sources. Une même sortie importée
        depuis un TCX local porte l'identifiant `file_2026_08_29_run`, et
        depuis Strava l'identifiant `15234567890` : la déduplication par
        clé primaire ne voit rien et tu te retrouves avec deux lignes, donc
        un volume hebdomadaire doublé et un modèle d'endurance nourri deux
        fois par le même point.

        On rapproche donc sur l'empreinte physique de la séance : heure de
        départ à quelques minutes près et durée à quelques minutes près.
        Les tolérances ne sont pas cosmétiques — Strava renvoie un temps de
        déplacement (moving_time) tandis qu'un fichier brut donne un temps
        écoulé, et l'écart atteint couramment deux à trois minutes.
        """
        df = self.read(ACTIVITIES)
        if df.empty or "date" not in df.columns or "duration_h" not in df.columns:
            return None

        start = pd.Timestamp(start)
        if start.tzinfo is not None:
            start = start.tz_convert("UTC").tz_localize(None)

        dates = _parse_dates(df["date"])

        ecart = (dates - start).abs()
        near_start = ecart <= pd.Timedelta(minutes=start_tol_min)
        near_dur = (df["duration_h"].astype(float) - duration_h).abs() * 60 <= dur_tol_min
        hit = df[near_start & near_dur]
        if hit.empty:
            return None
        # La correspondance la PLUS PROCHE, pas la première rencontrée : sur
        # un historique fourni, plusieurs sorties peuvent tomber dans la
        # fenêtre de tolérance et l'ordre du fichier n'a aucun sens.
        return str(hit.loc[ecart[hit.index].idxmin(), "activity_id"])


def get_store(secrets: dict) -> Store:
    """Choisit automatiquement le backend selon les secrets disponibles."""
    if secrets.get("SUPABASE_URL") and secrets.get("SUPABASE_KEY"):
        return Store("supabase", secrets)
    return Store("local")


SUPABASE_SCHEMA = """
-- À exécuter dans l'éditeur SQL de Supabase. Réexécutable sans risque :
-- les tables existantes ne sont pas recréées, les colonnes manquantes sont
-- ajoutées. Postgres refuse toute colonne inconnue à l'écriture, d'où
-- l'exhaustivité de cette liste — un oubli fait échouer l'import entier.

create table if not exists activities (
  activity_id   text primary key
);

alter table activities
  add column if not exists source        text,
  add column if not exists sport         text,
  add column if not exists strava_type   text,
  add column if not exists terrain       text,
  add column if not exists discipline    text,
  add column if not exists discipline_source text,
  add column if not exists dplus_par_km  double precision,
  add column if not exists date          timestamptz,
  add column if not exists name          text,
  add column if not exists notes         text,
  add column if not exists planned_key   text,
  add column if not exists session_type  text,
  add column if not exists type_fiabilite text,
  -- volumes
  add column if not exists distance_km   double precision,
  add column if not exists d_plus        double precision,
  add column if not exists d_minus       double precision,
  add column if not exists duration_h    double precision,
  add column if not exists deq_km        double precision,
  add column if not exists ke_km         double precision,
  -- allure et physiologie
  add column if not exists gap_kmh       double precision,
  add column if not exists vam           double precision,
  add column if not exists desc_kmh      double precision,
  add column if not exists desc_hr_cost  double precision,
  add column if not exists desc_efficiency double precision,
  add column if not exists up_hr_cost    double precision,
  add column if not exists hr_cost       double precision,
  add column if not exists drift         double precision,
  add column if not exists hr_mean       double precision,
  add column if not exists hrr_mean      double precision,
  add column if not exists time_above_85 double precision,
  add column if not exists cad_up        double precision,
  add column if not exists walk_share    double precision,
  add column if not exists share_up      double precision,
  add column if not exists share_down    double precision,
  add column if not exists share_flat    double precision,
  add column if not exists trimp         double precision,
  add column if not exists cv_gap        double precision,
  add column if not exists bimodalite    double precision,
  add column if not exists gap_trend     double precision,
  -- meilleurs efforts soutenus
  add column if not exists v30           double precision,
  add column if not exists v60           double precision,
  add column if not exists v120          double precision,
  -- vélo
  add column if not exists poids_kg      double precision,
  add column if not exists has_power     boolean,
  add column if not exists device_watts  boolean,
  add column if not exists power_source  text,
  add column if not exists speed_kmh     double precision,
  add column if not exists power_mean    double precision,
  add column if not exists np            double precision,
  add column if not exists variability_index double precision,
  add column if not exists work_kj       double precision,
  add column if not exists intensity_factor double precision,
  add column if not exists tss           double precision,
  add column if not exists w_moyen       double precision,
  add column if not exists w_max_5s      double precision,
  add column if not exists w15           double precision,
  add column if not exists w30           double precision,
  add column if not exists w60           double precision,
  add column if not exists wkg_moyen     double precision,
  add column if not exists wkg_max_5s    double precision,
  add column if not exists wkg15         double precision,
  add column if not exists wkg30         double precision,
  add column if not exists wkg60         double precision;

-- Coût de relance : trois fenêtres x trois mesures, pour montée et descente.
alter table activities
  add column if not exists apres_montee_n            double precision,
  add column if not exists apres_montee_ref_kmh      double precision,
  add column if not exists apres_montee_ref_m        double precision,
  add column if not exists apres_montee_relance      double precision,
  add column if not exists apres_montee_relance_rel  double precision,
  add column if not exists apres_montee_relance_fc   double precision,
  add column if not exists apres_montee_relance_cout double precision,
  add column if not exists apres_montee_effort       double precision,
  add column if not exists apres_montee_effort_rel   double precision,
  add column if not exists apres_montee_effort_fc    double precision,
  add column if not exists apres_montee_effort_cout  double precision,
  add column if not exists apres_montee_recuperation      double precision,
  add column if not exists apres_montee_recuperation_rel  double precision,
  add column if not exists apres_montee_recuperation_fc   double precision,
  add column if not exists apres_montee_recuperation_cout double precision,
  add column if not exists apres_descente_n            double precision,
  add column if not exists apres_descente_ref_kmh      double precision,
  add column if not exists apres_descente_ref_m        double precision,
  add column if not exists apres_descente_relance      double precision,
  add column if not exists apres_descente_relance_rel  double precision,
  add column if not exists apres_descente_relance_fc   double precision,
  add column if not exists apres_descente_relance_cout double precision,
  add column if not exists apres_descente_effort       double precision,
  add column if not exists apres_descente_effort_rel   double precision,
  add column if not exists apres_descente_effort_fc    double precision,
  add column if not exists apres_descente_effort_cout  double precision,
  add column if not exists apres_descente_recuperation      double precision,
  add column if not exists apres_descente_recuperation_rel  double precision,
  add column if not exists apres_descente_recuperation_fc   double precision,
  add column if not exists apres_descente_recuperation_cout double precision;

create index if not exists activities_date_idx on activities (date);
create index if not exists activities_sport_idx on activities (sport);

create table if not exists slope_bins (
  activity_id   text references activities(activity_id) on delete cascade,
  bande         text,
  primary key (activity_id, bande)
);

alter table slope_bins
  add column if not exists date          timestamptz,
  add column if not exists pente_centre  double precision,
  add column if not exists temps_min     double precision,
  add column if not exists distance_km   double precision,
  add column if not exists vitesse_kmh   double precision,
  add column if not exists gap_kmh       double precision,
  add column if not exists fc            double precision,
  add column if not exists cadence       double precision,
  add column if not exists cout_fc       double precision,
  add column if not exists part_marche   double precision,
  add column if not exists temps_marche_min double precision,
  add column if not exists temps_course_min double precision,
  add column if not exists v_marche_kmh  double precision,
  add column if not exists v_course_kmh  double precision,
  add column if not exists cout_marche   double precision,
  add column if not exists cout_course   double precision;

create index if not exists slope_bins_pente_idx on slope_bins (pente_centre);

create table if not exists journal (
  planned_key   text primary key
);
alter table journal
  add column if not exists fait          boolean,
  add column if not exists temps_min     double precision,
  add column if not exists rpe           double precision,
  add column if not exists genou_j1      double precision,
  add column if not exists genou_j2      double precision,
  add column if not exists commentaire   text,
  add column if not exists maj           timestamptz;

create table if not exists courses (
  course_id     text primary key
);
alter table courses
  add column if not exists nom           text,
  add column if not exists distance_km   double precision,
  add column if not exists d_plus        double precision,
  add column if not exists d_minus       double precision,
  add column if not exists ke_km         double precision,
  add column if not exists deq_km        double precision,
  add column if not exists profil        text,
  add column if not exists reperes       text,
  add column if not exists importe_le    timestamptz;

create table if not exists tokens (
  provider      text primary key
);
alter table tokens
  add column if not exists access_token  text,
  add column if not exists refresh_token text,
  add column if not exists expires_at    bigint;

-- SÉCURITÉ AU NIVEAU DES LIGNES.
--
-- Supabase active RLS par défaut. Sans règle définie, toute lecture ET
-- toute écriture par la clé anon sont refusées — la lecture renvoie alors
-- zéro ligne SANS erreur, ce qui est le pire cas : un résultat plausible
-- et faux. Le Table Editor du tableau de bord utilise le rôle postgres,
-- qui contourne RLS : les données y semblent présentes alors que
-- l'application ne voit rien.
--
-- Ce projet ne contient qu'un historique d'entraînement personnel, dont
-- l'accès est protégé par la confidentialité de la clé anon (exclue du
-- dépôt par .gitignore, stockée dans les secrets côté serveur).
--
-- Ces lignes sont rejouées à chaque exécution du schéma : c'est
-- volontaire, RLS s'étant réactivée une fois en cours de route.
alter table activities  disable row level security;
alter table slope_bins  disable row level security;
alter table journal     disable row level security;
alter table courses     disable row level security;
alter table tokens      disable row level security;
"""
