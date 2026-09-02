"""
app.py — Interface.

Conçue mobile d'abord : pas de layout="wide", pas de st.columns(4) qui
s'écrase sur iPhone, navigation par onglets. Les graphiques Plotly sont
configurés sans barre d'outils, qui est inutilisable au doigt.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from engine import analysis, archive, bike, efforts, ingest, metrics, plan as plan_mod, predict, physio, sync as sync_mod
from engine.archive import DISCIPLINE_LABELS
from engine.store import get_store, ACTIVITIES, SLOPE_BINS, TOKENS, JOURNAL, COURSES

st.set_page_config(page_title="Trail Lab", page_icon="⛰️", initial_sidebar_state="collapsed")

APP_VERSION = "2026-08-26-I"
VERSION = "2026-08-26-I"

PLOTLY_CFG = {"displayModeBar": False, "scrollZoom": False}

# Charte reprise du plan d'entraînement : papier, encre, bistre.
PAPER, CARD = "#EFEDE4", "#FAF9F4"
INK, SOFT = "#1F2C27", "#5C665F"
ACCENT = "#9A6636"      # bistre
GOOD = "#3E6146"        # forêt
GOOD_HEX, BAD_HEX = "#3E6146", "#B23A2B"
BAD = "#B23A2B"         # drapeau
COOL = "#2E6B8C"        # eau
MUTED = "#9A9382"
STONE, HAIR = "#D7D2C3", "#C9C3B2"

# Plotly n'accepte pas les piles CSS complètes : il faut une liste de noms
# de familles réelles, sinon il retombe silencieusement sur sa police par
# défaut — ce qui faisait perdre la charte sur les graphes.
PASTEL_FORCE = 0.42

DISPLAY = ('Bahnschrift SemiCondensed, Bahnschrift, "DIN Alternate", '
           '"Avenir Next Condensed", "Roboto Condensed", "Arial Narrow", '
           'Tahoma, sans-serif')
BODY = ('"Palatino Linotype", Palatino, "Book Antiqua", Charter, '
        'Georgia, serif')
MONO = 'Consolas, "SF Mono", ui-monospace, monospace'

# Bandes de pente : libellés explicites à l'affichage. On ne renomme pas
# ceux stockés en base — cela obligerait à tout réimporter pour un simple
# gain de lisibilité.
from engine.metrics import SLOPE_LABELS, SLOPE_RANGES
BAND_LABELS = {b: b for b in SLOPE_LABELS}

CSS = """
<style>
  html, body, [class*="css"] { font-family: %(body)s; }
  .stApp { background: %(paper)s; }
  h1, h2, h3, h4 {
      font-family: %(display)s !important;
      text-transform: uppercase; letter-spacing: .06em;
      color: %(ink)s !important; font-weight: 700 !important;
  }
  h1 { letter-spacing: -.005em; font-size: 2.6rem !important; }
  h3 { font-size: 1rem !important; letter-spacing: .18em;
       color: %(accent)s !important; margin-top: 2.2rem !important; }
  .stTabs [data-baseweb="tab"] {
      font-family: %(display)s; text-transform: uppercase;
      letter-spacing: .14em; font-size: 12px;
  }
  [data-testid="stMetricValue"] {
      font-family: %(display)s; font-weight: 700; color: %(ink)s;
  }
  [data-testid="stMetricLabel"] {
      font-family: %(display)s; text-transform: uppercase;
      letter-spacing: .16em; font-size: 11px; color: %(accent)s;
  }
  /* La flèche et la couleur du delta portent un jugement — hausse bonne,
     baisse mauvaise — qui n'a pas de sens sur un taux de réalisation. */
  [data-testid="stMetricDelta"] svg { display: none; }
  [data-testid="stMetricDelta"] {
      font-family: %(mono)s; font-size: 12.5px; color: %(soft)s !important;
      background: none; padding: 0;
  }
  [data-testid="stMetricDelta"] div { color: %(soft)s !important; }
  [data-testid="stMetric"] {
      background: %(card)s; border: none;
      border-top: 2px solid %(accent)s;
      padding: 12px 14px 14px;
  }
  .stDataFrame, .stDataEditor { font-family: %(mono)s; font-size: 13px; }
  .lecture {
      background: %(card)s; border-left: 2px solid %(accent)s;
      padding: 14px 18px; margin: 2px 0 26px; font-size: 16px;
      line-height: 1.65; color: #2A3630;
  }
  .lecture b { color: %(ink)s; }
  .note { font-family: %(mono)s; font-size: 11.5px; color: %(soft)s;
          margin: -10px 0 20px; line-height: 1.5; }
  hr { border-color: %(hair)s; }
  .wk-head { display:flex; align-items:baseline; gap:14px; flex-wrap:wrap;
             margin: 8px 0 2px; }
  .wk-num { font-family:%(display)s; font-size:34px; font-weight:700;
            color:%(accent)s; line-height:1; }
  .wk-dates { font-family:%(display)s; text-transform:uppercase;
              letter-spacing:.14em; font-size:15px; color:%(ink)s; }
  .wk-obj { font-style:italic; color:%(soft)s; font-size:15px; }
  .wk-stats { font-family:%(mono)s; font-size:12.5px; color:%(soft)s;
              border-top:1px solid %(hair)s; border-bottom:1px solid %(hair)s;
              padding:7px 0; margin:6px 0 14px; }
  .wk-stats b { color:%(ink)s; }
  .seance { border-left:3px solid %(hair)s; padding-left:12px;
            margin: 2px 0 4px; }
  .s-date { font-family:%(display)s; text-transform:uppercase;
            letter-spacing:.12em; font-size:13px; color:%(soft)s; }
  .s-titre { font-family:%(display)s; text-transform:uppercase;
             letter-spacing:.09em; font-size:14px; color:%(ink)s;
             font-weight:700; }
  .s-cle { font-family:%(display)s; font-size:9.5px; letter-spacing:.12em;
           text-transform:uppercase; background:%(accent)s; color:%(paper)s;
           padding:1px 5px; vertical-align:middle; }
  .s-dur { font-family:%(mono)s; font-size:13px; color:%(ink)s; }
  .s-txt { font-size:14px; color:%(soft)s; line-height:1.45; }
  .lg-lab { font-family:%(display)s; text-transform:uppercase;
            letter-spacing:.13em; font-size:9.5px; color:%(soft)s; }
  .lg-out { font-family:%(mono)s; font-size:15px; }
  [data-testid="stForm"] { border:none; padding:0; }
</style>
""" % {"body": BODY, "display": DISPLAY, "mono": MONO, "paper": PAPER,
       "card": CARD, "ink": INK, "accent": ACCENT, "hair": HAIR,
       "soft": SOFT, "paper": PAPER}


def lecture(txt: str):
    """Encart de lecture : la phrase qui explique le graphe au-dessus."""
    st.markdown(f'<div class="lecture">{txt}</div>', unsafe_allow_html=True)


def note(txt: str):
    st.markdown(f'<div class="note">{txt}</div>', unsafe_allow_html=True)

# Centres des bandes de pente, pour le matching historique
BIN_CENTERS = {
    "D -25/-45%": -0.35, "D -15/-25%": -0.20, "D -8/-15%": -0.115,
    "D -3/-8%": -0.055, "Plat": 0.0, "M 3/8%": 0.055,
    "M 8/15%": 0.115, "M 15/25%": 0.20, "M 25/45%": 0.35,
}


def gate() -> bool:
    """Mot de passe simple. Le dépôt est public sur Streamlit Cloud ;
    tes données de santé ne doivent pas l'être."""
    pwd = st.secrets.get("APP_PASSWORD")
    if not pwd:
        return True
    if st.session_state.get("auth"):
        return True
    entered = st.text_input("Mot de passe", type="password")
    if entered and entered == pwd:
        st.session_state["auth"] = True
        st.rerun()
    elif entered:
        st.error("Mot de passe incorrect.")
    return False


def line(x, y, name, color, y2=False):
    return go.Scatter(x=x, y=y, name=name, mode="lines",
                      line=dict(color=color, width=2),
                      yaxis="y2" if y2 else "y")


def show(fig, height=300):
    fig.update_layout(
        height=height, margin=dict(l=8, r=18, t=74, b=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        # Tailles calées sur le corps de texte (16 px) : à 11-12 px les
        # graphes paraissaient d'un autre document, et l'écart devient
        # franchement gênant sur un écran de téléphone.
        font=dict(family=BODY, size=14, color=INK),
        title=dict(font=dict(family=DISPLAY, size=16, color=ACCENT),
                   x=0, xanchor="left", y=0.97, yanchor="top"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(family=DISPLAY, size=14, color=SOFT)),
        hoverlabel=dict(bgcolor=CARD, bordercolor=HAIR,
                        font=dict(family=MONO, size=14, color=INK)),
    )
    pale = _rgba(HAIR, .55)
    fig.update_xaxes(gridcolor=pale, zeroline=False, showline=False,
                     tickfont=dict(family=MONO, size=13, color=SOFT))
    fig.update_yaxes(gridcolor=pale, zeroline=False, showline=False,
                     tickfont=dict(family=MONO, size=13, color=SOFT))
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)


# ── Onglet 1 : analyse d'une sortie ──────────────────────────────────────────

DISCIPLINES_SORTIE = ["trail", "route", "rando", "vtt", "velo_route", "gravel"]
FAMILLE = {"trail": "pied", "route": "pied", "rando": "pied",
           "vtt": "velo", "velo_route": "velo", "gravel": "velo"}


def tab_sortie(hr_rest, hr_max, store_for_save):
    st.caption("Dépose un GPX, TCX ou FIT. Le FIT est le format le plus riche.")
    up = st.file_uploader("Fichier", type=["gpx", "tcx", "fit"],
                          label_visibility="collapsed")
    if not up:
        return

    try:
        raw = ingest.load(up, up.name)
    except Exception as e:
        st.error(f"Lecture impossible : {e}")
        return

    # ── Choix de la discipline, obligatoire ───────────────────────────────
    #
    # Il conditionne à la fois le plafond anti-décrochage GPS et les
    # indicateurs affichés. Le laisser deviner conduisait à afficher une
    # VAM et un découplage cardiaque sur une sortie vélo, où ils ne
    # mesurent rien — et surtout à amputer la distance de moitié quand le
    # plafond de la course s'appliquait à 26 km/h de moyenne.
    suggere = archive._guess_sport(raw)
    defaut = "vtt" if suggere == "velo" else "trail"
    disc = st.radio(
        "Discipline", DISCIPLINES_SORTIE,
        index=DISCIPLINES_SORTIE.index(defaut), horizontal=True,
        format_func=lambda d: DISCIPLINE_LABELS.get(d, d), key="sortie_disc")
    note(f"Détection automatique : {DISCIPLINE_LABELS.get(defaut, defaut)}. "
         f"Plafond anti-décrochage GPS appliqué : "
         f"{metrics.max_speed_for(disc) * 3.6:.0f} km/h — un plafond trop "
         "bas ampute la distance.")

    try:
        d = metrics.prepare(raw, max_speed_ms=metrics.max_speed_for(disc))
    except Exception as e:
        st.error(f"Analyse impossible : {e}")
        return

    if FAMILLE[disc] == "velo":
        _sortie_velo(store_for_save, raw, d, disc, up.name, hr_rest, hr_max)
    else:
        _sortie_pied(store_for_save, raw, d, disc, up.name, hr_rest, hr_max)


def _sortie_velo(store, raw, d, disc, filename, hr_rest, hr_max):
    """Indicateurs vélo : distance, D+, puissance. Ni GAP ni VAM, qui
    supposent qu'on court."""
    poids = st.session_state.get("poids_kg")
    a_power = raw["power"].notna().sum() > len(raw) * 0.3
    s = bike.summarize_ride(d, st.session_state.get("ftp"), a_power,
                            hr_rest, hr_max)
    bp = bike.best_power(d, poids) if a_power else {}

    c1, c2 = st.columns(2)
    c1.metric("Distance", f"{s['distance_km']:.1f} km")
    c2.metric("Durée", predict.fmt_hours(s["duration_h"]))
    c1.metric("D+", f"{s['d_plus']:.0f} m")
    c2.metric("Vitesse moyenne", f"{s['speed_kmh']:.1f} km/h")
    c1.metric("TRIMP", f"{s['trimp']:.0f}" if not np.isnan(s["trimp"]) else "—",
              plan_mod.fmt_minutes(s["trimp"] / 87 * 60)
              if not np.isnan(s["trimp"]) else None, delta_color="off")
    c2.metric("D−", f"{d['d_minus'].sum():.0f} m")

    if not a_power:
        st.info(f"Aucune puissance de capteur ({s['power_source']}). "
                "Le bloc puissance apparaîtra dès que le 4iiii sera apparié. "
                "Strava fournit parfois une puissance estimée : elle n'est "
                "pas exploitée, son erreur dépasse 25 % en peloton.")
    else:
        st.subheader("Puissance")
        p1, p2 = st.columns(2)
        p1.metric("Moyenne", f"{s['power_mean']:.0f} W",
                  f"{s['power_mean'] / poids:.2f} W/kg" if poids else None,
                  delta_color="off")
        p2.metric("Normalisée", f"{s['np']:.0f} W",
                  f"variabilité {s['variability_index']:.2f}",
                  delta_color="off")
        p1.metric("Pointe 5 s", f"{bp.get('w_max_5s', float('nan')):.0f} W",
                  f"{bp['wkg_max_5s']:.2f} W/kg" if poids and
                  not np.isnan(bp.get("wkg_max_5s", float("nan"))) else None,
                  delta_color="off")
        p2.metric("Travail", f"{s['work_kj']:.0f} kJ")
        if not np.isnan(s.get("tss", float("nan"))):
            p1.metric("TSS", f"{s['tss']:.0f}")
            p2.metric("Intensité", f"{s['intensity_factor']:.2f}")

        blocs = [(m, bp.get(f"w{m}")) for m in bike.DUREES_W
                 if not np.isnan(bp.get(f"w{m}", float("nan")))]
        if blocs:
            fig = go.Figure(go.Scatter(
                x=[m for m, _ in blocs], y=[v for _, v in blocs],
                mode="lines+markers+text",
                text=[f"{v:.0f} W" for _, v in blocs], textposition="top center",
                textfont=dict(family=MONO, size=14, color=INK),
                line=dict(color=_pastel(ACCENT, .18), width=3.5, shape="spline"),
                marker=dict(size=13, color=_pastel(ACCENT, .28))))
            fig.update_layout(title="MEILLEURS EFFORTS DE LA SORTIE",
                              yaxis=dict(title="watts"),
                              xaxis=dict(title="minutes", type="log",
                                         tickvals=[m for m, _ in blocs],
                                         ticktext=[str(m) for m, _ in blocs]))
            show(fig, 300)
        st.caption(bike.SINGLE_SIDED_WARNING)

    km = d["cum_dist"] / 1000
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=km, y=d["ele_smooth"], name="Altitude",
                             fill="tozeroy", line=dict(color=MUTED, width=1),
                             fillcolor=_rgba(MUTED, .16)))
    fig.update_layout(title="PROFIL", yaxis=dict(title="m"))
    show(fig, 240)

    if raw["power"].notna().any() or d["hr"].notna().any():
        fig = go.Figure()
        if a_power:
            fig.add_trace(go.Scatter(
                x=km, y=d["power"].rolling(30, min_periods=5).mean(),
                name="Puissance (30 s)", mode="lines",
                line=dict(color=_pastel(ACCENT, .2), width=2)))
        if d["hr"].notna().any():
            fig.add_trace(go.Scatter(x=km, y=d["hr"], name="FC", mode="lines",
                                     line=dict(color=_pastel(BAD, .3), width=1.5),
                                     yaxis="y2"))
        fig.update_layout(title="PUISSANCE ET FRÉQUENCE CARDIAQUE",
                          yaxis=dict(title="W"),
                          yaxis2=dict(title="bpm", overlaying="y",
                                      side="right", showgrid=False))
        show(fig, 280)

    st.divider()
    _save_block(store, raw, {"summary": {**s, "session_type": "continu"},
                             "points": d}, filename, disc)


def _sortie_pied(store, raw, d, disc, filename, hr_rest, hr_max):
    """Indicateurs course : GAP, VAM, découplage, bandes de pente."""
    res = {"points": d,
           "summary": {**metrics.summarize(d, hr_rest, hr_max),
                       **metrics.session_profile(d)},
           "slope_table": metrics.by_slope_bin(d),
           "walk_run": metrics.walk_run_threshold(d)}
    s = res["summary"]
    ke = predict.km_effort(s["distance_km"], s["d_plus"])

    a, b = st.columns(2)
    a.metric("Distance", f"{s['distance_km']:.1f} km", f"{ke:.1f} km-effort",
             delta_color="off")
    b.metric("Durée", predict.fmt_hours(s["duration_h"]))
    a.metric("D+", f"{s['d_plus']:.0f} m")
    b.metric("D−", f"{s['d_minus']:.0f} m")
    a.metric("GAP moyen", f"{s['gap_kmh']:.1f} km/h")
    b.metric("VAM", f"{s['vam']:.0f} m/h" if not np.isnan(s["vam"]) else "—")
    a.metric("Descente", f"{s['desc_kmh']:.1f} km/h")
    b.metric("Découplage",
             f"{(s['drift'] - 1) * 100:+.1f} %"
             if not np.isnan(s["drift"]) else "—",
             help="Coût cardiaque de la 2e moitié contre la 1re, à terrain "
                  "égal. Sous 5 %, la fatigue ne se voit pas.")

    st.divider()
    km = d["cum_dist"] / 1000
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=km, y=d["ele_smooth"], name="Altitude",
                             fill="tozeroy", line=dict(color=MUTED, width=1),
                             fillcolor=_rgba(MUTED, .16)))
    fig.update_layout(title="PROFIL", yaxis=dict(title="m"))
    show(fig, 240)

    fig = go.Figure()
    if d["hr"].notna().any():
        fig.add_trace(go.Scatter(x=km, y=d["hr"], name="FC", mode="lines",
                                 line=dict(color=_pastel(BAD, .3), width=2)))
    fig.add_trace(go.Scatter(x=km, y=d["gap"] * 3.6, name="GAP km/h",
                             mode="lines", yaxis="y2",
                             line=dict(color=_pastel(COOL, .3), width=2)))
    fig.update_layout(title="FC ET VITESSE ÉQUIVALENTE PLAT",
                      yaxis=dict(title="bpm"),
                      yaxis2=dict(title="km/h", overlaying="y", side="right",
                                  showgrid=False))
    show(fig, 300)

    st.subheader("Par bande de pente")
    tbl = res["slope_table"]
    if tbl.empty:
        st.info("Pas assez de temps par bande pour une lecture fiable.")
    else:
        st.dataframe(
            tbl.style.format({
                "temps_min": "{:.0f}", "distance_km": "{:.1f}",
                "vitesse_kmh": "{:.1f}", "gap_kmh": "{:.1f}", "fc": "{:.0f}",
                "cadence": "{:.0f}", "cout_fc": "{:.1f}",
                "part_marche": "{:.0%}"}),
            use_container_width=True, hide_index=True)

    wr = res["walk_run"]
    if wr and wr.get("bascule"):
        lecture(f"Bascule marche/course mesurée sur cette sortie : "
                f"<b>{wr['bascule']}</b>.")

    st.divider()
    _compare_block(store, res)
    st.divider()
    _save_block(store, raw, res, filename, disc)


def _compare_block(store, res):
    """Cette sortie face au backlog, sur la période choisie."""
    hist = store.read(ACTIVITIES)
    bins = store.read(SLOPE_BINS)
    if hist.empty or bins.empty:
        return

    st.subheader("Face à ton historique")
    label, months = period_picker("periode_sortie")

    d = hist.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce", utc=True).dt.tz_localize(None)
    d = d[d["sport"] == "trail"].dropna(subset=["date"])
    if months:
        d = d[d["date"] >= d["date"].max() - pd.DateOffset(months=months)]
    if len(d) < 5:
        st.info(f"Seulement {len(d)} sortie(s) sur « {label} ». Élargis la période.")
        return

    ref = bins[bins["activity_id"].astype(str).isin(set(d["activity_id"].astype(str)))]
    cur = res["slope_table"]
    if ref.empty or cur.empty:
        st.info("Pas de bande de pente comparable.")
        return

    rows = []
    for _, r in cur.iterrows():
        g = ref[(ref["bande"] == r["bande"]) & ref["vitesse_kmh"].notna()]
        if len(g) < 4:
            continue
        med = float(g["vitesse_kmh"].median())
        rang = float((g["vitesse_kmh"] < r["vitesse_kmh"]).mean() * 100)
        rows.append({"bande": r["bande"], "cette sortie": r["vitesse_kmh"],
                     "médiane historique": med,
                     "écart %": (r["vitesse_kmh"] / med - 1) * 100,
                     "percentile": rang, "n_ref": len(g)})
    t = pd.DataFrame(rows)
    if t.empty:
        st.info("Aucune bande avec assez de références.")
        return

    fig = go.Figure()
    fig.add_trace(go.Bar(x=t["bande"], y=t["médiane historique"],
                         name=f"Médiane {label}", marker_color=MUTED, opacity=0.6))
    fig.add_trace(go.Bar(x=t["bande"], y=t["cette sortie"], name="Cette sortie",
                         marker_color=ACCENT))
    fig.update_layout(title="Vitesse par bande de pente", barmode="group",
                      yaxis_title="km/h")
    show(fig, 260)

    forts = t[t["percentile"] >= 70]
    faibles = t[t["percentile"] <= 30]
    if not forts.empty:
        st.write("Au-dessus de ton niveau habituel : "
                 + ", ".join(f"{r['bande']} ({r['percentile']:.0f}ᵉ pct)"
                             for _, r in forts.iterrows()) + ".")
    if not faibles.empty:
        st.write("En retrait : "
                 + ", ".join(f"{r['bande']} ({r['percentile']:.0f}ᵉ pct)"
                             for _, r in faibles.iterrows()) + ".")
    if forts.empty and faibles.empty:
        st.write("Sortie conforme à ton niveau habituel sur toutes les bandes.")

    tr = analysis.transition_cost(res["points"], "up")
    if not np.isnan(tr.get("relance", np.nan)):
        st.caption(
            f"Relance après montée : {(tr['relance'] - 1) * 100:+.0f} % sur "
            f"200 m, {(tr.get('effort', np.nan) - 1) * 100:+.0f} % sur 200-600 m "
            f"({tr['n_transitions']} montée(s) analysée(s))."
        )

    with st.expander("Détail chiffré"):
        st.dataframe(t.round(2), hide_index=True, use_container_width=True)


def _save_block(store, raw, res, filename, disc="trail"):
    """Enregistrement dans l'historique, avec garde-fou anti-doublon."""
    s = res["summary"]
    start = pd.Timestamp(raw["t"].iloc[0])
    dup = store.find_overlap(start, s["duration_h"])
    if dup:
        st.info(f"Déjà dans l'historique (`{dup}`). Rien à enregistrer.")
        return

    sport = archive.SPORT_FROM_DISCIPLINE.get(disc, "trail")
    st.caption(f"Sera enregistré comme **{DISCIPLINE_LABELS.get(disc, disc)}**.")
    nom = st.text_input("Nom", value=Path(filename).stem, key="save_name")

    # Type de séance : suggéré, jamais imposé. Le détecteur rate les côtes
    # courtes (inertie cardiaque de 20-30 s), et un « fractionné » à tort
    # écarte silencieusement la séance du calibrage du modèle.
    types = ["continu", "sortie longue", "variable", "fractionné", "course"]
    if sport == "velo":
        # Le type de séance repose sur la bimodalité de la FC en course :
        # sans signification ici, on ne demande donc rien.
        seance = "continu"
        st.session_state.setdefault("save_type", "continu")
    else:
        suggestion = s.get("session_type", "continu")
        idx = types.index(suggestion) if suggestion in types else 0
        seance = st.selectbox("Type de séance", types, index=idx, key="save_type",
                          help="Pré-rempli par détection automatique. Les "
                               "séances marquées « fractionné » sont exclues "
                               "du calibrage du modèle de prédiction : leur "
                               "temps total inclut échauffement et "
                               "récupérations. Vérifie avant d'enregistrer.")
        if s.get("type_fiabilite") == "faible":
            st.caption("Détection peu fiable sur cette sortie — vérifie le champ.")
    if not st.button("Enregistrer dans l'historique", type="primary"):
        return

    meta = pd.Series({
        "activity_id": archive._file_id(Path(filename)),
        "filename": filename,
        "date": start,
        "name": nom,
        "sport": sport,
        "discipline": disc,
        "discipline_source": "manuel",
        "workout_type": "",
    })
    try:
        row, bins = archive.build_row(raw, meta, hr_rest=st.session_state.get("hr_rest", 50),
                                      hr_max=st.session_state.get("hr_max", 190),
                                      ftp=st.session_state.get("ftp"),
                                      poids_kg=st.session_state.get("poids_kg"))
        row["session_type"] = seance          # le choix manuel prime toujours
        store.upsert(ACTIVITIES, pd.DataFrame([row]), key="activity_id")
        if not bins.empty:
            store.upsert(SLOPE_BINS, bins, key=["activity_id", "bande"])
        st.success(f"Enregistré comme « {seance} ».")
    except Exception as e:
        st.error(f"Enregistrement impossible : {e}")


# Fenêtres de comparaison proposées dans les onglets Historique, Tendance
# et Vélo. « Tout » vaut None : aucun filtre de date.
PERIODES = {"Tout": None, "12 mois": 12, "6 mois": 6, "3 mois": 3}


def discipline_picker(hist: pd.DataFrame, key: str,
                      defaut: str = "trail") -> str | None:
    """Filtre par discipline déclarée. None = toutes."""
    if "discipline" not in hist.columns or hist["discipline"].isna().all():
        return None
    dispo = [d for d in ["trail", "route", "vtt", "velo_route", "gravel", "rando"]
             if (hist["discipline"] == d).sum() >= 3]
    if len(dispo) < 2:
        return dispo[0] if dispo else None
    idx = dispo.index(defaut) if defaut in dispo else 0
    return st.radio("Discipline", dispo, index=idx, horizontal=True, key=key,
                    format_func=lambda d: DISCIPLINE_LABELS.get(d, d),
                    label_visibility="collapsed")


def period_picker(key: str) -> tuple[str, int | None]:
    label = st.radio("Période de référence", list(PERIODES), index=1,
                     horizontal=True, key=key, label_visibility="collapsed")
    return label, PERIODES[label]


def recent_picker(key: str, dispo: int) -> int:
    """
    Combien de sorties récentes comparer à l'historique.

    Le curseur n'est pas un confort : il change la nature de la question.
    3 sorties répondent à « comment vais-je en ce moment », au prix d'une
    forte sensibilité au bruit. 20 sorties répondent à « ai-je progressé
    ce trimestre », avec un signal stable mais lent à bouger. Aucun des
    deux réglages n'est meilleur, ils ne disent pas la même chose.
    """
    haut = max(3, min(30, dispo - 4))
    n = st.slider("Nombre de sorties récentes à comparer", 3, haut,
                  min(5, haut), key=key,
                  help="Peu de sorties : lecture réactive mais bruitée. "
                       "Beaucoup : lecture stable mais lente. "
                       "Les sorties restantes servent de référence.")
    return int(n)


# Assez d'écart entre les niveaux pour se lire, assez de retenue pour que
# le graphe reste aéré. Le plancher évite la barre invisible.
NIVEAU_ALPHA = {"net": 0.88, "probable": 0.62, "faible": 0.40, "nul": 0.24}


def _rgba(hexa: str, alpha: float) -> str:
    h = hexa.lstrip("#")
    r, g, bl = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{bl},{alpha:.2f})"


def _pastel(hexa: str, force: float = 0.42) -> str:
    """
    Désature une teinte en la mélangeant vers le papier.

    L'opacité seule ne suffisait pas : un rouge drapeau à 82 % d'opacité
    sur fond papier reste un rouge franc, et il jurait avec le bistre et
    les gris de la charte. Mélanger vers le fond réduit la saturation en
    conservant la teinte — c'est ce que fait un lavis.
    """
    h = hexa.lstrip("#")
    pap = PAPER.lstrip("#")
    out = []
    for i in (0, 2, 4):
        c = int(h[i:i + 2], 16)
        p = int(pap[i:i + 2], 16)
        out.append(int(round(c + (p - c) * force)))
    return "#%02X%02X%02X" % tuple(out)


def bands_chart(table: pd.DataFrame, titre: str, unite: str = "km/h",
                sens_positif: bool = True):
    """
    Barres divergentes de l'écart, une par bande, opacité graduée selon la
    solidité du signal.

    L'opacité remplace le noir-ou-couleur binaire de la version précédente :
    un écart probable mais non certain n'est ni un fait ni un néant, et
    l'afficher en gris uni revenait à le nier.
    """
    if table.empty:
        return
    t = table.sort_values("pente")
    couleurs, textes = [], []
    for _, r in t.iterrows():
        bon = (r["ecart"] > 0) == sens_positif
        couleurs.append(_rgba(_pastel(GOOD if bon else BAD, .22),
                              NIVEAU_ALPHA.get(r["niveau"], 0.25)))
        textes.append(f"{r['ecart_pct']:+.1f} %")

    libelles = [BAND_LABELS.get(x, x) for x in t["bande"]]
    fig = go.Figure(go.Bar(
        x=t["ecart_pct"], y=libelles, orientation="h",
        marker=dict(color=couleurs, line=dict(width=0)),
        text=textes, textposition="outside", cliponaxis=False,
        textfont=dict(family=MONO, size=14, color=INK),
        customdata=np.stack([t["recent"], t["reference"], t["n_recent"],
                             t["n_ref"], t["km_recent"], t["km_ref"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>Récent : %{customdata[0]:.2f} " + unite
                       + " sur %{customdata[4]:.1f} km (%{customdata[2]} sorties)"
                       + "<br>Référence : %{customdata[1]:.2f} " + unite
                       + " sur %{customdata[5]:.0f} km (%{customdata[3]} sorties)"
                         "<extra></extra>")))
    fig.add_vline(x=0, line=dict(color=_rgba(SOFT, .8), width=1))
    fig.update_layout(title=titre.upper(), showlegend=False, bargap=0.5)
    span = max(8.0, float(t["ecart_pct"].abs().max()) * 1.55)
    # Aucune grille : les valeurs sont écrites au bout des barres, une
    # grille ne fait que du bruit visuel.
    fig.update_xaxes(range=[-span, span], title=None, showgrid=False,
                     showticklabels=False, zeroline=False)
    fig.update_yaxes(title=None, showgrid=False, ticklen=10,
                     tickcolor="rgba(0,0,0,0)",
                     tickfont=dict(family=DISPLAY, size=15, color=INK))
    show(fig, max(210, 58 * len(t) + 96))
    note("Barre dense = signal solide. À droite = progression.")


def transitions_chart(table: pd.DataFrame, titre: str):
    """
    Trois séries, UN SEUL AXE, toutes en pourcentage.

    La version précédente mettait la FC sur un axe secondaire en bpm.
    Plotly n'aligne pas les groupes de barres entre deux axes : les séries
    se chevauchaient et le graphe était inexploitable. La FC est donc
    convertie en pourcentage — elle se déduit du coût et de l'allure sans
    donnée supplémentaire.
    """
    if table.empty:
        return
    # Teintes lavées : le rouge drapeau et le bleu eau à pleine saturation
    # juraient avec le bistre et les gris du reste de la page.
    series = [("vitesse_pct", "Vitesse", _pastel(ACCENT, 0.30)),
              ("fc_pct", "Fréquence cardiaque", _pastel(BAD, 0.46)),
              ("cout_pct", "Coût cardiaque", _pastel(COOL, 0.46))]
    fig = go.Figure()
    for col, nom, coul in series:
        if col not in table.columns or table[col].isna().all():
            continue
        fig.add_trace(go.Bar(
            x=table["fenetre"], y=table[col], name=nom,
            marker=dict(color=coul, line=dict(width=0)),
            text=[("" if np.isnan(v) else f"{v:+.0f} %") for v in table[col]],
            textposition="outside", cliponaxis=False,
            textfont=dict(family=MONO, size=14, color=INK),
            hovertemplate=nom + " %{y:+.1f} %<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=_rgba(SOFT, .8), width=1))

    vals = pd.concat([table[c] for c, _, _ in series if c in table.columns]).dropna()
    span = max(6.0, float(vals.abs().max()) * 1.6) if len(vals) else 10.0
    fig.update_layout(
        title=titre.upper(), barmode="group", bargap=0.42, bargroupgap=0.10,
        yaxis=dict(title=None, range=[-span, span], showgrid=False,
                   showticklabels=False, zeroline=False))
    fig.update_xaxes(showgrid=False, ticklen=10, tickcolor="rgba(0,0,0,0)",
                     tickfont=dict(family=DISPLAY, size=15, color=INK))
    show(fig, 330)
    note("Vitesse en hausse avec FC stable : vrai gain. "
         "Vitesse et FC en hausse ensemble : tu pousses plus fort, "
         "et ça se paie en fin de course longue.")


def tab_profil(store):
    """État : ce que tu vaux et où tu cours. Aucune comparaison."""
    hist = store.read(ACTIVITIES)
    bins = store.read(SLOPE_BINS)
    if hist.empty or bins.empty:
        st.info("Aucune activité. Importe ton historique depuis Réglages.")
        return

    disc = discipline_picker(hist, "disc_profil")
    label, months = period_picker("periode_profil")
    d = hist.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce", utc=True).dt.tz_localize(None)
    if disc and "discipline" in d.columns:
        d = d[d["discipline"] == disc]
    else:
        d = d[d["sport"] == "trail"]
    d = d.dropna(subset=["date"])
    if months:
        d = d[d["date"] >= d["date"].max() - pd.DateOffset(months=months)]
    if d.empty:
        st.warning(f"Aucune sortie sur « {label} » pour cette discipline.")
        return

    t = analysis.band_summary(bins, d["activity_id"])
    if t.empty:
        st.warning("Aucune bande assez fournie.")
        return

    a, b = st.columns(2)
    a.metric("Sorties trail", f"{len(d)}")
    b.metric("Distance analysée", f"{t['distance_km'].sum():.0f} km")
    a.metric("D+ cumulé", f"{d['d_plus'].sum():.0f} m")
    b.metric("Temps", f"{t['temps_h'].sum():.0f} h")

    st.subheader("Où tu cours")
    fig = go.Figure(go.Bar(
        x=t["distance_km"], y=t["bande"], orientation="h",
        marker=dict(color=[_rgba(_pastel(BAD if p > 0 else COOL, .38),
                                 .55 + .45 * min(abs(p) / .28, 1))
                           for p in t["pente"]], line=dict(width=0)),
        text=[f"{k:.0f} km · {p:.0f} %" for k, p in zip(t["distance_km"], t["part_distance"])],
        textposition="outside", textfont=dict(family=MONO, size=14, color=INK),
        hovertemplate="<b>%{y}</b><br>%{x:.0f} km<extra></extra>"))
    fig.update_layout(title="DISTANCE PAR BANDE DE PENTE", showlegend=False, bargap=0.42)
    fig.update_xaxes(title=None, range=[0, t["distance_km"].max() * 1.35])
    fig.update_yaxes(title=None, showgrid=False,
                     tickfont=dict(family=DISPLAY, size=15, color=INK))
    show(fig, max(200, 46 * len(t) + 80))
    note("Ton volume réel par type de terrain, sur la période choisie.")

    st.subheader("Ce que tu vaux")
    fig = go.Figure(go.Bar(
        x=t["vitesse_kmh"], y=t["bande"], orientation="h",
        marker=dict(color=_pastel(ACCENT, .28), line=dict(width=0)),
        text=[f"{v:.1f}" for v in t["vitesse_kmh"]], textposition="outside",
        textfont=dict(family=MONO, size=14, color=INK),
        customdata=np.stack([t["fc"], t["cadence"], t["sorties"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>%{x:.2f} km/h<br>FC %{customdata[0]:.0f}"
                       "<br>Cadence %{customdata[1]:.0f}"
                       "<br>%{customdata[2]} sorties<extra></extra>")))
    fig.update_layout(title="VITESSE MOYENNE PAR BANDE", showlegend=False, bargap=0.42)
    fig.update_xaxes(title="km/h", range=[0, t["vitesse_kmh"].max() * 1.25])
    fig.update_yaxes(title=None, showgrid=False,
                     tickfont=dict(family=DISPLAY, size=15, color=INK))
    show(fig, max(200, 46 * len(t) + 80))

    st.subheader("Courbe vitesse-durée")
    curve = efforts.build_curve(hist, months=months or 36)
    if not curve.get("ok"):
        st.info(curve["reason"])
    else:
        c = curve["courbe"]
        cols = st.columns(len(c))
        for col, (m, v) in zip(cols, sorted(c.items())):
            col.metric(f"Meilleur {m} min", f"{v:.2f} km/h",
                       f"{curve['n'][m]} sorties")
        bs = efforts.base_speed(curve, 9.0)
        if bs.get("ok"):
            note(f"Décroissance de {bs['perte_par_doublement_pct']:.1f} % par "
                 f"doublement de durée. Extrapolée à 9 h : "
                 f"{bs['v_kmh']:.2f} km/h équivalent plat."
                 + (f" {bs['note']}" if bs.get("note") else ""))
        if not curve["coherente"]:
            st.warning("La courbe ne décroît pas avec la durée — signe d'un "
                       "bloc court insuffisamment représenté.")

    st.subheader("Marcher ou courir")
    wr = analysis.walk_run_threshold(bins, d["activity_id"])
    lecture(analysis.read_walk_run(wr, noms=BAND_LABELS))
    if not wr.empty:
        fig = go.Figure()
        libelles = [BAND_LABELS.get(x, x) for x in wr["bande"]]
        fig.add_trace(go.Bar(x=libelles, y=wr["v_course"], name="Course",
                             marker_color=_pastel(ACCENT, .28)))
        fig.add_trace(go.Bar(x=libelles, y=wr["v_marche"], name="Marche",
                             marker_color=_pastel(COOL, .42)))
        fig.update_layout(title="VITESSE SELON LE MODE, PAR BANDE",
                          barmode="group", yaxis_title="km/h")
        fig.update_xaxes(showgrid=False,
                         tickfont=dict(family=DISPLAY, size=15, color=INK))
        show(fig, 280)

    st.subheader("Détail")
    aff = t[["bande", "sorties", "distance_km", "temps_h", "vitesse_kmh",
             "fc", "cadence", "cout_fc", "part_marche", "part_distance"]].copy()
    aff.columns = ["Bande", "Sorties", "km", "h", "km/h", "FC", "Cad.",
                   "Coût FC", "% marche", "% du volume"]
    aff["Sorties"] = aff["Sorties"].astype(int)
    st.dataframe(
        aff.style.format({"Sorties": "{:d}", "km": "{:.1f}", "h": "{:.1f}",
                          "km/h": "{:.2f}", "FC": "{:.0f}", "Cad.": "{:.0f}",
                          "Coût FC": "{:.1f}", "% marche": "{:.0%}",
                          "% du volume": "{:.0f} %"}),
        hide_index=True, use_container_width=True)
    note("Coût FC : battements par W/kg. Plus bas = plus économique.")


def horizon_chart(t: pd.DataFrame, titre: str, sens_positif: bool = True):
    """Le même écart sur 4 horizons. La cohérence entre colonnes EST la
    mesure de confiance."""
    if t.empty:
        return
    cols = [f"n{n}" for n in analysis.HORIZONS if f"n{n}" in t.columns]
    if not cols:
        return
    z = t[cols].to_numpy(dtype=float)
    txt = [[("" if np.isnan(v) else f"{v:+.0f}%") for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{c[1:]} dern." for c in cols],
        y=[BAND_LABELS.get(b, b) for b in t["bande"]],
        text=txt, texttemplate="%{text}",
        textfont=dict(family=MONO, size=14),
        colorscale=([[0, _pastel(BAD, .30)], [0.5, PAPER], [1, _pastel(GOOD, .30)]]
                    if sens_positif else
                    [[0, _pastel(GOOD, .30)], [0.5, PAPER], [1, _pastel(BAD, .30)]]),
        zmid=0, showscale=False, xgap=3, ygap=3,
        hovertemplate="%{y} · %{x}<br>%{z:+.1f} %<extra></extra>"))
    fig.update_layout(title=titre.upper())
    fig.update_xaxes(showgrid=False, side="top",
                     tickfont=dict(family=DISPLAY, size=14, color=SOFT))
    fig.update_yaxes(showgrid=False, autorange="reversed",
                     tickfont=dict(family=DISPLAY, size=15, color=INK))
    show(fig, max(200, 44 * len(t) + 100))
    note("Une ligne cohérente sur les quatre colonnes = progression "
         "installée. Qui change de signe = variabilité, pas tendance.")


def quadrant_chart(t: pd.DataFrame):
    """Descente en deux dimensions : vite, et à quel prix."""
    if t.empty:
        return
    x = t["ecart_pct_vitesse"]
    y = -t["ecart_pct_cout"]          # vers le haut = plus économique
    lim = max(6.0, float(max(x.abs().max(), y.abs().max())) * 1.4)
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=lim, y1=lim,
                  fillcolor=_rgba(GOOD, .10), line_width=0, layer="below")
    fig.add_shape(type="rect", x0=-lim, y0=-lim, x1=0, y1=0,
                  fillcolor=_rgba(BAD, .10), line_width=0, layer="below")
    for ax, ay, lbl, col in [(lim * .55, lim * .8, "rapide et économique", GOOD),
                             (-lim * .55, lim * .8, "lente mais économique", COOL),
                             (lim * .55, -lim * .8, "rapide mais coûteuse", ACCENT),
                             (-lim * .55, -lim * .8, "lente et coûteuse", BAD)]:
        fig.add_annotation(x=ax, y=ay, text=lbl.upper(), showarrow=False,
                           font=dict(family=DISPLAY, size=12, color=_rgba(col, .9)))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers+text",
        text=[BAND_LABELS.get(b, b).replace("Descente ", "") for b in t["bande"]],
        textposition="bottom center",
        textfont=dict(family=DISPLAY, size=13, color=INK),
        marker=dict(size=17, color=_pastel(ACCENT, .30),
                    line=dict(color=_rgba(INK, .5), width=1)),
        customdata=np.stack([t["km_recent"], t["km_ref"]], axis=-1),
        hovertemplate=("<b>%{text}</b><br>Vitesse %{x:+.1f} %"
                       "<br>Économie %{y:+.1f} %"
                       "<br>%{customdata[0]:.1f} / %{customdata[1]:.0f} km<extra></extra>")))
    fig.add_hline(y=0, line=dict(color=SOFT, width=1))
    fig.add_vline(x=0, line=dict(color=SOFT, width=1))
    fig.update_layout(title="DESCENTE — VITESSE ET ÉCONOMIE", showlegend=False)
    fig.update_xaxes(range=[-lim, lim], title="vitesse (%)", ticksuffix=" %")
    fig.update_yaxes(range=[-lim, lim], title="économie cardiaque (%)",
                     ticksuffix=" %")
    show(fig, 380)
    note("Haut-droite : tu descends plus vite en dépensant moins. "
         "Bas-droite : plus vite mais plus cher — ça se paie en fin de course.")


def tab_historique(store):
    hist = store.read(ACTIVITIES)
    if hist.empty:
        st.info("Aucune activité. Importe ton historique depuis l'onglet Réglages.")
        return

    disc = discipline_picker(hist, "disc_histo")
    label, months = period_picker("periode_histo")
    base = hist if not disc else hist[hist.get("discipline") == disc]
    sport_cible = "trail" if not disc else \
        {"vtt": "velo", "velo_route": "velo", "gravel": "velo",
         "rando": "rando"}.get(disc, "trail")
    dispo = len(analysis.split_recent(base, months, n_recent=0,
                                      sport=sport_cible)[1])
    if dispo < 7:
        st.warning(f"{dispo} sortie(s) trail sur « {label} ». Il en faut au "
                   "moins 7 pour comparer quoi que ce soit. Élargis la période.")
        return

    n_recent = recent_picker("n_recent_histo", dispo)
    recent, past = analysis.split_recent(base, months, n_recent=n_recent,
                                         sport=sport_cible)

    if len(recent) < 2 or len(past) < 4:
        st.warning(f"{len(recent)} sortie(s) récente(s) et {len(past)} en "
                   "référence. Il en faut au moins 2 et 4.")
        return

    st.caption(f"**{len(recent)} dernières sorties** "
               f"(depuis le {recent['date'].min():%d/%m}) comparées aux "
               f"**{len(past)} précédentes** sur « {label} ».")

    bins = store.read(SLOPE_BINS)
    rid, pid = recent["activity_id"], past["activity_id"]

    st.subheader("1. Montée")
    t = analysis.compare_bands(bins, rid, pid, "vitesse_kmh", uphill=True)
    bands_chart(t, "Vitesse en montée")
    lecture(analysis.read_bands(t, "montée", noms=BAND_LABELS))

    st.subheader("2. Descente")
    t = analysis.compare_bands(bins, rid, pid, "vitesse_kmh", uphill=False)
    bands_chart(t, "Vitesse en descente")
    lecture(analysis.read_bands(t, "descente", noms=BAND_LABELS))
    tc = analysis.compare_bands(bins, rid, pid, "cout_fc", uphill=False)
    if not tc.empty:
        bands_chart(tc, "Coût cardiaque en descente", "bpm/(W/kg)",
                    sens_positif=False)
        lecture(analysis.read_bands(tc, "économie de descente", "bpm/(W/kg)", sens_positif=False, noms=BAND_LABELS))
        st.caption("Descendre vite en étant crispé n'est pas progresser : "
                   "vitesse en hausse ET coût en baisse, c'est un vrai gain.")

    st.subheader("3. Relance après les montées")
    t, msg = analysis.read_transitions(recent, past, "apres_montee", "après une montée")
    transitions_chart(t, "Allure sur le plat qui suit une montée")
    lecture(msg)

    st.subheader("4. Relance après les descentes")
    t, msg = analysis.read_transitions(recent, past, "apres_descente", "après une descente")
    transitions_chart(t, "Allure sur le plat qui suit une descente")
    lecture(msg)

    st.subheader("5. Tenue sur la durée")
    lecture(analysis.read_drift(recent, past))
    d = hist.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce", utc=True).dt.tz_localize(None)
    d = d[(d["sport"] == "trail")].dropna(subset=["drift"]).sort_values("date")
    if months:
        d = d[d["date"] >= d["date"].max() - pd.DateOffset(months=months)]
    if len(d) >= 8:
        # Médiane glissante puis moyenne : une seule sortie aberrante — une
        # séance dans la chaleur, un capteur qui décroche — produisait un
        # pic qui dominait toute la lecture.
        liss = (d["drift"].rolling(7, center=True, min_periods=3).median()
                          .rolling(5, center=True, min_periods=2).mean())
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=d["date"], y=(d["drift"] - 1) * 100, name="Sorties",
            mode="markers", marker=dict(color=_rgba(MUTED, .5), size=7)))
        fig.add_trace(go.Scatter(
            x=d["date"], y=(liss - 1) * 100, name="Tendance lissée",
            mode="lines", line=dict(color=_pastel(ACCENT, .12), width=3.5,
                                    shape="spline")))
        fig.add_hrect(y0=-5, y1=5, fillcolor=_rgba(GOOD, .10), line_width=0)
        fig.add_hline(y=12, line=dict(color=_rgba(BAD, .5), width=1, dash="dot"))
        # Ordonnée bornée à ±50 % : quelques sorties dépassent, mais la
        # tendance devient illisible quand un pic à 110 % écrase l'échelle.
        fig.update_layout(title="DÉRIVE CARDIAQUE AU FIL DES SORTIES",
                          yaxis=dict(title="%", range=[-30, 50]))
        show(fig, 250)
        note("Zone verte : la fatigue ne se voit pas. "
             "Au-dessus du trait rouge (12 %), elle pèse sur une course longue.")

    st.subheader("6. Charge")
    load = sync_mod.weekly_load(hist, weeks=26 if months is None else months * 4 + 2)
    if load.empty:
        st.info("Charge non calculable : il faut la fréquence cardiaque.")
    else:
        fig = go.Figure()
        for col, color in [("trail", _pastel(ACCENT, .25)),
                           ("velo", _pastel(COOL, .40)),
                           ("rando", _pastel(MUTED, .30))]:
            if col in load.columns:
                fig.add_trace(go.Bar(x=load.index, y=load[col], name=col,
                                     marker_color=color))
        fig.update_layout(title="CHARGE HEBDOMADAIRE, TOUS SPORTS",
                          barmode="stack", yaxis_title="TRIMP")
        show(fig, 250)
        note("TRIMP : minutes d'effort pondérées exponentiellement par "
             "l'intensité cardiaque. 87 points = une heure en endurance "
             "fondamentale. C'est le seul agrégat qui permette d'additionner "
             "trail et vélo.")
        lecture(analysis.read_load(load))

    with st.expander("Données brutes"):
        cols = [c for c in ["date", "name", "sport", "distance_km", "d_plus",
                            "ke_km", "duration_h", "gap_kmh", "vam", "desc_kmh",
                            "drift", "session_type"] if c in hist.columns]
        st.dataframe(hist[cols].sort_values("date", ascending=False),
                     hide_index=True, use_container_width=True)


# ── Onglet Vélo ──────────────────────────────────────────────────────────────

W_LABELS = {"w_max_5s": "Pointe 5 s", "w15": "15 min", "w30": "30 min",
            "w60": "1 h", "w_moyen": "Moyenne"}


def tab_velo(store):
    hist = store.read(ACTIVITIES)
    if hist.empty:
        st.info("Aucune activité. Importe ton historique depuis Réglages.")
        return
    d = hist[hist["sport"] == "velo"].copy()
    if d.empty:
        st.info("Aucune sortie vélo en base. Si tu en as, la détection de "
                "sport les a classées ailleurs — réimporte avec la version "
                "actuelle.")
        return

    d["date"] = pd.to_datetime(d["date"], errors="coerce", utc=True).dt.tz_localize(None)
    d = d.dropna(subset=["date"]).sort_values("date")
    label, months = period_picker("periode_velo")
    if months:
        d = d[d["date"] >= d["date"].max() - pd.DateOffset(months=months)]
    if d.empty:
        st.warning(f"Aucune sortie vélo sur « {label} ».")
        return

    poids = st.session_state.get("poids_kg")
    avec_w = d[d["w_moyen"].notna()] if "w_moyen" in d.columns else d.iloc[0:0]

    st.subheader("Volume")
    a, b = st.columns(2)
    a.metric("Sorties", f"{len(d)}")
    b.metric("Distance", f"{d['distance_km'].sum():.0f} km")
    a.metric("Durée", f"{d['duration_h'].sum():.0f} h")
    b.metric("D+", f"{d['d_plus'].sum():.0f} m")

    if avec_w.empty:
        st.warning(
            "Aucune sortie avec puissance de capteur. Deux causes possibles : "
            "le 4iiii n'était pas encore apparié, ou l'import a été fait "
            "avant l'ajout des colonnes de puissance. Le reste de cet onglet "
            "reste vide jusque-là."
        )
        return

    st.subheader("Puissance")
    note(f"{len(avec_w)} sortie(s) avec capteur sur {len(d)}."
         + ("" if poids else " Renseigne ton poids dans Réglages pour les W/kg."))
    c1, c2 = st.columns(2)
    c1.metric("Puissance moyenne", f"{avec_w['w_moyen'].mean():.0f} W")
    c2.metric("Pointe 5 s", f"{avec_w['w_max_5s'].max():.0f} W")
    if poids and "wkg_moyen" in avec_w.columns and avec_w["wkg_moyen"].notna().any():
        c1.metric("W/kg moyen", f"{avec_w['wkg_moyen'].mean():.2f}")
        c2.metric("W/kg pointe 5 s", f"{avec_w['wkg_max_5s'].max():.2f}")

    st.subheader("Meilleurs efforts")
    cols = [c for c in ("w15", "w30", "w60") if c in avec_w.columns
            and avec_w[c].notna().any()]
    if not cols:
        st.info("Pas de sortie assez longue pour un bloc de 15 minutes.")
    else:
        best = {c: float(avec_w[c].max()) for c in cols}
        cc = st.columns(len(cols))
        for col, c in zip(cc, cols):
            sup = (f"{best[c] / poids:.2f} W/kg" if poids else
                   f"{int(avec_w[c].notna().sum())} sorties")
            col.metric(W_LABELS[c], f"{best[c]:.0f} W", sup)

        # Courbe puissance-durée : la référence du cyclisme.
        durs = [int(c[1:]) for c in cols]
        fig = go.Figure(go.Scatter(
            x=durs, y=[best[c] for c in cols], mode="lines+markers+text",
            text=[f"{best[c]:.0f} W" for c in cols], textposition="top center",
            textfont=dict(family=MONO, size=14, color=INK),
            line=dict(color=_pastel(ACCENT, .18), width=3.5, shape="spline"),
            marker=dict(size=13, color=_pastel(ACCENT, .28)),
            hovertemplate="%{x} min · %{y:.0f} W<extra></extra>"))
        fig.update_layout(title="COURBE PUISSANCE-DURÉE",
                          yaxis=dict(title="watts"),
                          xaxis=dict(title="minutes", type="log",
                                     tickvals=durs, ticktext=[f"{v}" for v in durs]))
        show(fig, 300)

    st.subheader("Tendance")
    n_recent = min(5, max(2, len(avec_w) // 3))
    recent, passe = avec_w.tail(n_recent), avec_w.iloc[:-n_recent]
    if len(passe) < 4:
        st.info(f"{len(passe)} sortie(s) en référence, il en faut 4.")
    else:
        base = ["w_moyen", "w15", "w30", "w60"]
        if poids:
            base += ["wkg_moyen", "wkg15", "wkg30", "wkg60"]
        rows = []
        for c in base:
            if c not in avec_w.columns:
                continue
            r, p = recent[c].dropna(), passe[c].dropna()
            if len(r) < 2 or len(p) < 4:
                continue
            sd = float(p.std(ddof=1))
            seuil = 1.96 * sd / np.sqrt(len(r)) if sd > 0 else np.inf
            rows.append({
                "mesure": ("W/kg " if c.startswith("wkg") else "W ")
                          + W_LABELS.get("w" + c.replace("wkg", "").lstrip("_")
                                         if c.startswith("wkg") else c, c),
                "recent": r.mean(), "reference": p.mean(),
                "ecart_pct": (r.mean() / p.mean() - 1) * 100,
                "signal": abs(r.mean() - p.mean()) > seuil,
            })
        t = pd.DataFrame(rows)
        if t.empty:
            st.info("Pas assez de mesures comparables.")
        else:
            unite = "W/kg" if poids else "W"
            fig = go.Figure(go.Bar(
                x=t["ecart_pct"], y=t["mesure"], orientation="h",
                marker=dict(color=[_rgba(_pastel(GOOD if e > 0 else BAD, .22),
                                         0.88 if sg else 0.30)
                                   for e, sg in zip(t["ecart_pct"], t["signal"])],
                            line=dict(width=0)),
                text=[f"{e:+.1f} %" for e in t["ecart_pct"]],
                textposition="outside", cliponaxis=False,
                textfont=dict(family=MONO, size=14, color=INK),
                customdata=np.stack([t["recent"], t["reference"]], axis=-1),
                hovertemplate=("<b>%{y}</b><br>%{customdata[0]:.2f} contre "
                               "%{customdata[1]:.2f}<extra></extra>")))
            fig.add_vline(x=0, line=dict(color=_rgba(SOFT, .8), width=1))
            span = max(8.0, float(t["ecart_pct"].abs().max()) * 1.55)
            fig.update_layout(title=f"{n_recent} DERNIÈRES CONTRE RÉFÉRENCE",
                              showlegend=False, bargap=0.5)
            fig.update_xaxes(range=[-span, span], showgrid=False,
                             showticklabels=False, zeroline=False)
            fig.update_yaxes(showgrid=False, ticklen=10,
                             tickcolor="rgba(0,0,0,0)",
                             tickfont=dict(family=DISPLAY, size=15, color=INK))
            show(fig, max(240, 54 * len(t) + 90))
            note("Barre dense = écart hors dispersion normale.")

    with st.expander("Détail des sorties"):
        c = [x for x in ["date", "name", "distance_km", "d_plus", "duration_h",
                         "w_moyen", "w_max_5s", "w15", "w30", "w60",
                         "wkg_moyen", "np", "tss", "trimp"] if x in d.columns]
        st.dataframe(d[c].sort_values("date", ascending=False).round(2),
                     hide_index=True, use_container_width=True)


# ── Onglet 3 : préparation de course ─────────────────────────────────────────

def tab_course(store, hr_rest, hr_max):
    hist = store.read(ACTIVITIES)
    bins = store.read(SLOPE_BINS)

    saved = store.read(COURSES)
    up = st.file_uploader("Trace de la course (GPX)", type=["gpx", "tcx", "fit"],
                          key="race")

    if not up and not saved.empty:
        # Par défaut, la dernière trace importée : pas besoin de la redéposer
        # à chaque ouverture.
        last = saved.sort_values("importe_le").iloc[-1]
        st.caption(f"Course en mémoire : **{last['nom']}** "
                   f"({last['distance_km']:.1f} km · {last['d_plus']:.0f} D+ · "
                   f"{last['ke_km']:.0f} km-effort). Dépose un fichier pour "
                   "la remplacer.")
        # StringIO obligatoire : depuis pandas 2.1, read_json traite une
        # chaîne nue comme un CHEMIN DE FICHIER et lève FileNotFoundError
        # en affichant le JSON entier comme nom de fichier.
        try:
            d = pd.read_json(io.StringIO(str(last["profil"])), orient="split")
            d["t"] = pd.to_datetime(d["t"], errors="coerce", utc=True)
        except Exception as e:
            st.error(f"Trace en mémoire illisible ({type(e).__name__}). "
                     "Dépose à nouveau le fichier de la course.")
            return
        _course_view(store, d, hist, hr_rest, hr_max, nom=last["nom"])
        return

    if not up:
        st.info("Dépose la trace officielle de ta course.")
        return

    try:
        # Mode parcours : la trace vient de l'organisateur, pas de la montre.
        d = metrics.prepare(ingest.load(up, up.name), route_mode=True)
    except Exception as e:
        st.error(f"Lecture impossible : {e}")
        return

    nom = st.text_input("Nom de la course", value=Path(up.name).stem, key="race_name")
    if st.button("Mémoriser cette course"):
        keep = d[["cum_dist", "ele_smooth", "dist", "slope", "d_plus", "d_minus",
                  "dt", "speed", "gap", "cum_time", "slope_bin", "walking",
                  "p_met", "hr_cost", "ele", "lat", "lon", "t", "hr", "cad",
                  "power"]].copy()
        store.upsert(COURSES, pd.DataFrame([{
            "course_id": re.sub(r"[^A-Za-z0-9]+", "_", nom)[:60],
            "nom": nom,
            "distance_km": float(d["dist"].sum() / 1000),
            "d_plus": float(d["d_plus"].sum()),
            "d_minus": float(d["d_minus"].sum()),
            "ke_km": predict.km_effort(d["dist"].sum() / 1000, d["d_plus"].sum()),
            "deq_km": predict.flat_equivalent_distance(d),
            "profil": keep.to_json(orient="split", date_format="iso"),
            "importe_le": pd.Timestamp.now().isoformat(),
        }]), key="course_id")
        st.success(f"« {nom} » mémorisée.")

    _course_view(store, d, hist, hr_rest, hr_max, nom=nom)


def _course_view(store, d, hist, hr_rest, hr_max, nom=""):
    bins = store.read(SLOPE_BINS)

    dist = float(d["dist"].sum() / 1000)
    dplus = float(d["d_plus"].sum())
    ke = predict.km_effort(dist, dplus)
    deq = predict.flat_equivalent_distance(d)

    a, b = st.columns(2)
    a.metric("Distance", f"{dist:.1f} km")
    b.metric("D+", f"{dplus:.0f} m")
    st.metric("Kilomètre-effort", f"{ke:.0f} km",
              help="distance + D+/100, l'unité usuelle du trail. Retenue "
                   "après comparaison de sept formules par validation "
                   "croisée : aucune ne fait significativement mieux. "
                   f"L'intégration de Minetti donnerait {deq:.0f} km, mais "
                   "elle ne capte pas le coût musculaire de la descente.")

    km = d["cum_dist"] / 1000
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=km, y=d["ele_smooth"], fill="tozeroy", name="Altitude",
                             line=dict(color=MUTED, width=1),
                             fillcolor="rgba(122,132,148,0.15)"))
    fig.update_layout(title="Profil de la course", yaxis_title="m")
    show(fig, 220)

    segs = predict.segment_course(d)
    st.subheader(f"{len(segs)} segments")

    if hist.empty:
        st.warning("Pas d'historique : impossible de prédire un temps. "
                   "Importe d'abord tes sorties passées.")
        st.dataframe(segs.round(2), use_container_width=True, hide_index=True)
        return

    hist["date"] = pd.to_datetime(hist["date"])
    runs = hist[hist["sport"].isin(["trail", "rando"])]
    # La cible pilote la plage de calibrage : on ajuste sur les sorties
    # comparables à la course visée, pas sur les footings.
    model = predict.fit_endurance_model(runs, target_deq=ke)

    if not model["ok"]:
        st.warning(f"Modèle non calibré : {model['reason']}")
        return

    c1, c2 = st.columns(2)
    c1.metric("Exposant d'endurance", f"{model['b']:.3f}",
              help="1,06 route · 1,10-1,18 trail long. Hors de [1,00 ; 1,30], "
                   "l'historique n'est pas représentatif.")
    c2.metric("R²", f"{model['r2']:.2f}", f"n = {model['n']}")
    if not model["b_plausible"]:
        st.error("Exposant hors plage plausible. Le modèle n'est pas fiable : "
                 "il te manque des sorties longues, ou des sorties d'intensité.")

    pacing = st.slider("Gestion", 0.95, 1.25, 1.05, 0.01,
                       help="1,00 = allure d'enveloppe. Au-delà, gestion prudente.")
    pred = predict.predict_time(ke, model, pacing)
    if pred["ok"]:
        st.metric("Temps estimé", predict.fmt_hours(pred["hours"]),
                  f"{predict.fmt_hours(pred['low'])} – {predict.fmt_hours(pred['high'])}")
        if pred["note"]:
            st.warning(pred["note"])

    # ── Vérification croisée du modèle ────────────────────────────────────
    curve = efforts.build_curve(runs, months=24)
    if curve.get("ok"):
        cc = efforts.cross_check(curve, model["b"], pred.get("hours", 9.0))
        if cc.get("ok"):
            c1, c2 = st.columns(2)
            c1.metric("Exposant, sorties entières", f"{cc['b_riegel']:.3f}")
            c2.metric("Exposant, meilleurs efforts", f"{cc['b_efforts']:.3f}",
                      f"écart {cc['ecart']:.3f}")
            (note if cc["accord"] else st.warning)(cc["verdict"])

    # ── Plan par segment ──────────────────────────────────────────────────
    if not bins.empty and "pente_centre" in bins.columns and pred.get("ok"):
        drift = float(runs["drift"].dropna().tail(15).median()) \
            if "drift" in runs.columns and runs["drift"].notna().any() else None
        bs = efforts.base_speed(curve, pred["hours"]) if curve.get("ok") else {}
        plan = predict.race_plan(segs, bins, runs["activity_id"],
                                 total_h=pred["hours"], drift=drift,
                                 base_kmh=bs.get("v_kmh"))
        if plan.empty:
            st.info("Segmentation vide.")
        else:
            st.subheader("Plan par segment")
            note("Les bandes de pente donnent la répartition, le modèle "
                 "d'endurance donne le total. La dérive cardiaque ne rallonge "
                 "pas la course — l'exposant d'endurance porte déjà la "
                 "fatigue — elle répartit seulement le ralentissement du "
                 "départ vers l'arrivée.")
            aff = plan[["km_debut", "km_fin", "d_plus", "pente_moy", "type",
                        "mode", "vitesse_prevue_kmh", "allure_min_km",
                        "temps_h", "temps_cumule_h", "source"]].copy()
            aff.columns = ["km début", "km fin", "D+", "pente", "terrain",
                           "mode", "km/h", "min/km", "durée h", "cumul h",
                           "source"]
            st.dataframe(
                aff.style.format({
                    "km début": "{:.1f}", "km fin": "{:.1f}", "D+": "{:.0f}",
                    "pente": "{:.1%}", "km/h": "{:.1f}", "min/km": "{:.1f}",
                    "durée h": "{:.2f}", "cumul h": "{:.2f}"}),
                use_container_width=True, hide_index=True)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plan["km_fin"], y=plan["temps_cumule_h"],
                mode="lines+markers", name="Temps cumulé",
                line=dict(color=ACCENT, width=3, shape="spline"),
                customdata=np.stack([plan["vitesse_prevue_kmh"],
                                     plan["pente_moy"] * 100], axis=-1),
                hovertemplate=("km %{x:.1f}<br>%{y:.2f} h"
                               "<br>%{customdata[0]:.1f} km/h"
                               "<br>pente %{customdata[1]:+.0f} %<extra></extra>")))
            fig.update_layout(title="TEMPS DE PASSAGE PRÉVU",
                              yaxis_title="heures", xaxis_title="km")
            show(fig, 260)
    else:
        st.info("Table des bandes vide : pas de plan par segment.")


# ── Onglet 4 : plan d'entraînement ───────────────────────────────────────────

PLAN_CSV = Path("data_plan_templiers_2026.csv")
PLAN_HTML = Path("data_plan.html")


@st.cache_data
def load_plan(path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["seance_cle"] = df["seance_cle"].astype(bool)
    return df


def charge_chart(t: pd.DataFrame, semaine_active=None):
    """
    Profil de charge, au format du plan : course en aire pleine, vélo
    empilé par-dessus, total en ligne, et le volume de chaque semaine sous
    son étiquette.
    """
    if t.empty:
        return
    x = [f"S{int(w)}" for w in t.index]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=t["course"], name="Course à pied", mode="lines",
        line=dict(color=GOOD, width=1.5, shape="spline"),
        fill="tozeroy", fillcolor=_rgba(GOOD, .42),
        hovertemplate="Course %{y:.2f} h<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=x, y=t["course"] + t["velo"] + t["renfo"], name="Vélo", mode="lines",
        line=dict(color=COOL, width=2, shape="spline"),
        fill="tonexty", fillcolor=_rgba(COOL, .16),
        hovertemplate="Total %{y:.2f} h<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=x, y=t["total"], name="Total hebdomadaire", mode="markers",
        marker=dict(size=9, color=PAPER,
                    line=dict(color=COOL, width=2)),
        hoverinfo="skip"))
    if semaine_active is not None and f"S{int(semaine_active)}" in x:
        fig.add_vline(x=f"S{int(semaine_active)}",
                      line=dict(color=_rgba(ACCENT, .45), width=2, dash="dot"))
    fig.add_annotation(x=x[-1], y=t["total"].max() * 1.05, text="18 OCT.",
                       showarrow=False, xanchor="right",
                       font=dict(family=DISPLAY, size=12, color=BAD))
    fig.update_layout(title="PROFIL DE CHARGE — 8 SEMAINES",
                      yaxis=dict(title=None, ticksuffix=" h",
                                 gridcolor=_rgba(HAIR, .6)))
    fig.update_xaxes(showgrid=False, ticklen=10, tickcolor="rgba(0,0,0,0)",
                     ticktext=[f"{a}<br><span style='font-size:11px'>"
                               f"{int(v)}h{int(round((v % 1) * 60)):02d}</span>"
                               for a, v in zip(x, t["total"])],
                     tickvals=x,
                     tickfont=dict(family=DISPLAY, size=14, color=INK))
    show(fig, 380)
    note("Heures hebdomadaires, borne basse de chaque fourchette. "
         "Le renforcement est exclu du volume.")


def tab_plan(store):
    """
    Suivi du plan : profil de charge et séances semaine par semaine, au
    format du document, mais enregistré côté serveur.

    POURQUOI PAS LE FICHIER HTML LUI-MÊME. Il enregistre dans le
    localStorage du navigateur : ce qui est coché sur l'ordinateur
    n'existe pas sur le téléphone. Reprendre la présentation en widgets
    Streamlit est le seul moyen d'écrire dans Supabase, donc d'avoir le
    même état partout.
    """
    up = st.file_uploader("Plan (HTML ou CSV)", type=["html", "csv"], key="plan")
    if up is not None:
        try:
            brut = up.read()
            p = (plan_mod.parse_plan_html(io.BytesIO(brut))
                 if up.name.lower().endswith(".html")
                 else pd.read_csv(io.BytesIO(brut), parse_dates=["date"]))
            p.to_csv(PLAN_CSV, index=False, encoding="utf-8-sig")
            st.cache_data.clear()
            st.success(f"{len(p)} séances chargées.")
        except Exception as e:
            st.error(f"Lecture impossible : {e}")
            return
    elif PLAN_CSV.exists():
        p = load_plan(str(PLAN_CSV), PLAN_CSV.stat().st_mtime)
    else:
        st.info("Dépose le plan HTML pour l'activer.")
        return

    journal = store.read(JOURNAL)
    rec = plan_mod.merge_journal(
        plan_mod.reconcile(p, store.read(ACTIVITIES)), journal)
    today = pd.Timestamp.now().normalize()

    semaines = sorted(rec["semaine"].dropna().unique())
    proche = rec.loc[(rec["date"] >= today - pd.Timedelta(days=6))
                     & (rec["date"] <= today + pd.Timedelta(days=6)), "semaine"]
    courante = int(proche.iloc[0]) if len(proche) else int(semaines[0])

    charge_chart(plan_mod.load_profile(p), courante)

    # ── Une semaine ───────────────────────────────────────────────────────
    st.subheader("Le cycle, semaine par semaine")

    def lib(w):
        sub = rec[rec["semaine"] == w]
        return (f"S{int(w)} · {sub['date'].min():%d %b}–{sub['date'].max():%d %b}"
                + (" ●" if w == courante else ""))

    w = st.selectbox("Semaine", semaines, index=semaines.index(courante),
                     format_func=lib, label_visibility="collapsed",
                     key="plan_semaine")
    sem = rec[rec["semaine"] == w].sort_values("date").reset_index(drop=True)
    r = plan_mod.week_summary(sem)

    obj = sem["objectif_semaine"].dropna()
    d1, d2 = sem["date"].min(), sem["date"].max()
    st.markdown(
        f"<div class='wk-head'><span class='wk-num'>{int(w):02d}</span>"
        f"<span class='wk-dates'>{d1:%d} – {d2:%d %B}</span>"
        f"<span class='wk-obj'>{obj.iloc[0] if len(obj) else ''}</span></div>",
        unsafe_allow_html=True)

    bouts = [f"{plan_mod.fmt_minutes(v)} {k}" for k, v in r["par_sport"].items()]
    st.markdown(
        f"<div class='wk-stats'>{r['faites']}/{r['total']} séances"
        f" · réalisé <b>{plan_mod.fmt_minutes(r['reel_min'])}</b>"
        f" · prévu {plan_mod.fmt_minutes(r['prevu_min'])}"
        + (f" · écart <b>{plan_mod.ecart_hm(r['ecart_min'])}</b>"
           if r["ecart_min"] is not None else "")
        + f" &nbsp;&nbsp; {' · '.join(bouts)}</div>",
        unsafe_allow_html=True)

    COULEUR = {"trail": GOOD, "route": GOOD, "velo": COOL,
               "renfo": ACCENT, "repos": MUTED}

    with st.form(f"sem_{int(w)}"):
        saisies = []
        for i, row in sem.iterrows():
            bord = COULEUR.get(row["sport"], MUTED)
            fait_def = bool(row["fait"]) if pd.notna(row["fait"]) \
                else row["statut"] == "réalisée"
            reel = r["reel_par_seance"][i]

            st.markdown(f"<div class='seance' style='border-left-color:{bord}'>",
                        unsafe_allow_html=True)
            c = st.columns([0.5, 1.1, 2.4, 1.3, 5])
            fait = c[0].checkbox("fait", value=fait_def, key=f"f_{w}_{i}",
                                 label_visibility="collapsed")
            c[1].markdown(f"<span class='s-date'>{row['date']:%a %d}</span>",
                          unsafe_allow_html=True)
            etoile = " <span class='s-cle'>clé</span>" if row["seance_cle"] else ""
            c[2].markdown(f"<span class='s-titre'>{row['titre']}</span>{etoile}",
                          unsafe_allow_html=True)
            c[3].markdown(
                f"<span class='s-dur'>"
                f"{_fmt_range(row['duree_min_bas'], row['duree_min_haut'])}</span>",
                unsafe_allow_html=True)
            c[4].markdown(f"<span class='s-txt'>{row['consigne'] or ''}</span>",
                          unsafe_allow_html=True)

            e = st.columns([2, 1.4, 6.3])
            temps = e[0].text_input(
                "Temps effectif",
                value=plan_mod.fmt_minutes(reel).replace("—", ""),
                key=f"t_{w}_{i}", placeholder="ex. 2h35 ou 155′")
            em = plan_mod.ecart_minutes(reel, row["duree_min_bas"],
                                        row["duree_min_haut"])
            coul = (GOOD_HEX if em is not None and em >= 0 else BAD_HEX)
            e[1].markdown(
                "<span class='lg-lab'>Écart</span><br>"
                f"<span class='lg-out' style='color:{coul}'>"
                f"{plan_mod.ecart_hm(em)}</span>", unsafe_allow_html=True)
            note_txt = e[2].text_input(
                "Commentaire", value=str(row["commentaire"] or ""),
                key=f"c_{w}_{i}",
                placeholder="Sensations, D+, genou, météo, nutrition…")
            st.markdown("</div>", unsafe_allow_html=True)
            saisies.append({"planned_key": str(row["planned_key"]),
                            "fait": bool(fait), "temps": temps,
                            "commentaire": note_txt})

        if st.form_submit_button("Enregistrer la semaine", type="primary"):
            maj = pd.DataFrame([{
                "planned_key": x["planned_key"],
                "fait": x["fait"],
                "temps_min": plan_mod.parse_temps(x["temps"]),
                "commentaire": x["commentaire"] or None,
                "maj": pd.Timestamp.now(),
            } for x in saisies]).drop_duplicates(subset=["planned_key"],
                                                 keep="last")
            store.upsert(JOURNAL, maj, key="planned_key")
            st.success("Enregistré dans Supabase — visible sur tous "
                       "tes appareils.")
            st.rerun()

    # ── Avancement du cycle ───────────────────────────────────────────────
    st.subheader("Avancement du cycle")
    c = plan_mod.compliance(rec)
    if not c.get("ok"):
        st.info(c["reason"])
    else:
        a, b = st.columns(2)
        a.metric("Séances réalisées", f"{c['n_done']}/{c['n_due']}",
                 f"{c['taux']:.0%}", delta_color="off")
        b.metric("Heures réalisées",
                 plan_mod.fmt_minutes(c["heures_realisees"] * 60),
                 f"{plan_mod.ecart_hm(c['ecart_heures'] * 60)} vs prévu "
                 f"{plan_mod.fmt_minutes(c['heures_prevues'] * 60)}",
                 delta_color="off")

        sems = c["semaines"]
        note(f"Périmètre : toutes les séances échues, semaines "
             f"{int(min(sems))} à {int(max(sems))}. La semaine en cours y "
             "entre pour ses jours déjà passés — un écart global plus grand "
             "que celui d'une semaine vient souvent de là.")

        ps = c["par_semaine"]
        if not ps.empty:
            aff = pd.DataFrame({
                "Semaine": ["S" + str(x) for x in ps["semaine"]],
                "Séances": ps["seances"],
                "Prévu": [plan_mod.fmt_minutes(x) for x in ps["prevu_min"]],
                "Réalisé": [plan_mod.fmt_minutes(x) for x in ps["reel_min"]],
                "Écart": [plan_mod.ecart_hm(x) for x in ps["ecart_min"]],
            })
            st.dataframe(aff, hide_index=True, use_container_width=True)

        for m in plan_mod.diagnose(rec):
            lecture(m)


def _fmt_range(lo, hi) -> str:
    if pd.isna(lo):
        return "—"
    f = lambda m: f"{int(m) // 60}h{int(m) % 60:02d}" if m >= 60 else f"{int(m)} min"
    return f(lo) if lo == hi else f"{f(lo)} – {f(hi)}"


# ── Onglet 5 : réglages ──────────────────────────────────────────────────────

def tab_reglages(store, hr_rest, hr_max):
    st.subheader("Profil physiologique")
    st.caption("Ces deux valeurs conditionnent tout le calcul d'intensité. "
               "Une FCmax fausse fausse tout le reste.")
    c1, c2 = st.columns(2)
    c1.number_input("FC repos", 30, 90, hr_rest, key="hr_rest")
    c2.number_input("FC max", 150, 220, hr_max, key="hr_max")

    st.subheader("Strava")
    cid = st.secrets.get("STRAVA_CLIENT_ID")
    if not cid:
        st.warning(
            "Non configuré. Crée une application sur "
            "strava.com/settings/api, puis renseigne STRAVA_CLIENT_ID, "
            "STRAVA_CLIENT_SECRET et STRAVA_REDIRECT_URI dans les secrets."
        )
    else:
        from integrations.strava import authorize_url
        url = authorize_url(cid, st.secrets["STRAVA_REDIRECT_URI"])
        st.link_button("Connecter Strava", url)
        st.caption("Polar et Bryton synchronisent déjà vers Strava. "
                   "Une seule connexion suffit pour les trois.")

    st.subheader("FTP v\u00e9lo")
    st.number_input("FTP (W)", 100, 500, st.session_state.get("ftp", 250), key="ftp")
    st.caption(bike.SINGLE_SIDED_WARNING)

    st.subheader("Rattrapage initial")
    st.markdown(
        "Créer une application Strava exige un **abonnement Strava payant** : "
        "sans lui, l'API est inaccessible, quota ou pas. La voie gratuite est "
        "l'export en masse.\n\n"
        "Strava → Paramètres → Mon compte → *Télécharger ou supprimer votre "
        "compte* → **Demander votre archive**. Tu la reçois par mail sous "
        "quelques heures.\n\n"
        "Pour 150 activités, lance le rattrapage **en local** avec "
        "`python backfill.py archive.zip` : l'archive pèse trop lourd pour un "
        "téléversement web. L'import ci-dessous ne convient qu'à un essai."
    )
    arch = st.file_uploader("Archive Strava (ZIP)", type=["zip"], key="arch")
    if arch and st.button("Importer (essai, 20 sorties)"):
        bar = st.progress(0.0, "Lecture de l'archive\u2026")
        try:
            rep = archive.import_archive(
                arch, store, hr_rest, hr_max, st.session_state.get("ftp"),
                limit=20,
                progress=lambda i, t, l: bar.progress(i / max(t, 1), f"{i}/{t} \u00b7 {l}"))
            bar.empty()
            st.success(f"{rep['imported']} import\u00e9e(s), {rep['no_gps']} sans GPS, "
                       f"{len(rep['failed'])} \u00e9chec(s).")
        except Exception as e:
            bar.empty()
            st.error(str(e))

    st.subheader("Synchronisation Strava (abonn\u00e9s)")
    if st.session_state.get("strava_token"):
        n = st.number_input("Nombre maximum d'activit\u00e9s \u00e0 importer", 10, 500, 150)
        if st.button("Lancer la synchronisation", type="primary"):
            from integrations.strava import Strava
            client = Strava(st.session_state["strava_token"],
                            st.secrets["STRAVA_CLIENT_ID"],
                            st.secrets["STRAVA_CLIENT_SECRET"])
            bar = st.progress(0.0, "Pr\u00e9paration\u2026")
            rep = sync_mod.sync(
                client, store, hr_rest, hr_max,
                ftp=st.session_state.get("ftp"), limit=int(n),
                progress=lambda i, t, lbl: bar.progress(i / max(t, 1), f"{i}/{t} \u00b7 {lbl}"))
            bar.empty()
            st.success(f"{rep['imported']} import\u00e9e(s), {rep['skipped']} ignor\u00e9e(s).")
            if rep["stopped"]:
                st.warning(rep["stopped"])
            if rep["failed"]:
                with st.expander(f"{len(rep['failed'])} \u00e9chec(s)"):
                    st.dataframe(pd.DataFrame(rep["failed"],
                                              columns=["id", "nom", "erreur"]))
    else:
        st.info("Connecte Strava ci-dessus pour synchroniser.")

    st.subheader("Disciplines")
    inc = archive.coherence_disciplines(store.read(ACTIVITIES))
    if inc.empty:
        note("Aucune incohérence entre discipline déclarée et profil.")
    else:
        st.warning(f"{len(inc)} sortie(s) dont la déclaration semble "
                   "contredire le profil. Le modèle se calibrant sur le "
                   "trail seul, chaque étiquette erronée déplace le niveau.")
        aff = inc.head(20).copy()
        if "date" in aff.columns:
            aff["date"] = pd.to_datetime(aff["date"], errors="coerce",
                                         format="mixed").dt.strftime("%d/%m/%y")
        cols = [c for c in ["date", "name", "discipline", "distance_km",
                            "d_plus", "dplus_par_km", "signal"]
                if c in aff.columns]
        st.dataframe(aff[cols].round(1), hide_index=True,
                     use_container_width=True)
        note("Pour corriger : change le type dans Strava et réexporte "
             "l'archive, ou modifie la colonne `discipline` dans Supabase.")

    st.subheader("Stockage")
    st.write(f"Backend actif : `{store.backend}`")
    if store.backend == "local":
        st.warning("Stockage local : les données sont perdues à chaque "
                   "redémarrage sur Streamlit Cloud. Configure Supabase avant "
                   "de déployer.")


def main():
    if not gate():
        return

    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Trail Lab")
    note(f"version {VERSION} · moteur {analysis.VERSION}")
    st.caption(f"version {APP_VERSION}")
    hr_rest = st.session_state.get("hr_rest", 50)
    hr_max = st.session_state.get("hr_max", 190)
    store = get_store(dict(st.secrets))

    t1, t2, t3, t4, t5, t6, t7 = st.tabs(
        ["Sortie", "Historique", "Tendance", "Vélo", "Course", "Plan",
         "Réglages"])
    with t1:
        tab_sortie(hr_rest, hr_max, store)
    with t2:
        tab_profil(store)
    with t3:
        tab_historique(store)
    with t4:
        tab_velo(store)
    with t5:
        tab_course(store, hr_rest, hr_max)
    with t6:
        tab_plan(store)
    with t7:
        tab_reglages(store, hr_rest, hr_max)


if __name__ == "__main__":
    main()
