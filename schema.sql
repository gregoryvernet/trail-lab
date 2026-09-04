
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

-- POINTS GPS BRUTS, COMPRESSES.
--
-- Ce choix inverse une décision initiale, et la raison a changé. À
-- l'origine, stocker les points ne servait à rien : tout ce qui s'en déduit
-- est déjà dans activities et slope_bins. Mais pour une sauvegarde
-- indépendante — pouvoir tout reconstruire sans l'archive Strava, et
-- recalculer de nouveaux indicateurs plus tard — ils sont indispensables.
--
-- Coût mesuré : 47 Ko par sortie d'1h30 après arrondi à 6 décimales et
-- compression gzip, soit 14 Mo pour 305 sorties. Trois pour cent du quota
-- gratuit, et l'analyse recalculée depuis la trace stockée s'écarte de
-- 0,01 % de l'originale.
create table if not exists traces (
  activity_id   text primary key references activities(activity_id) on delete cascade
);
alter table traces
  add column if not exists date          timestamptz,
  add column if not exists points        integer,
  add column if not exists octets        integer,
  add column if not exists format        text,
  add column if not exists donnees       text;

create table if not exists tokens (
  provider      text primary key
);
alter table traces disable row level security;

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
