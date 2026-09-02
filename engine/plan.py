"""
plan.py — Plan d'entraînement : du HTML aux données, puis au suivi.

Pourquoi une conversion et pas une lecture directe du HTML dans l'app :
le HTML est une mise en forme. « 4h30 – 5h » et « 1 200 – 1 600 D+ » se
lisent très bien à l'œil et ne se comparent à rien. Pour confronter le
réalisé au prévu il faut des bornes numériques, une date ISO et un sport.

Le plan Templiers 2026 s'y prête bien : chaque semaine porte un
`data-start` et chaque séance une classe `t-*` qui donne son type. La
conversion est donc déterministe, pas heuristique.

Ce que le module produit :
  - un CSV de 56 séances, une ligne par jour
  - le rapprochement automatique séance prévue / activité importée
  - un journal par séance (ressenti, réaction du genou) que TOI seul
    remplis, et que le rapprochement ne peut pas deviner
"""

from __future__ import annotations

import html
import json
import re
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Classes CSS du plan → nature de la séance
TYPE_MAP = {
    "t-renfo": ("renfo", "Renforcement"),
    "t-qual": ("trail", "Qualitative"),
    "t-ef": ("trail", "Endurance fondamentale"),
    "t-long": ("trail", "Sortie longue"),
    "t-velo": ("velo", "Vélo"),
    "t-repos": ("repos", "Repos"),
}

JOURS = {"lun": 0, "mar": 1, "mer": 2, "jeu": 3, "ven": 4, "sam": 5, "dim": 6}


def _duration_bounds(text: str) -> tuple[float | None, float | None]:
    """
    « 1h10 » → (70, 70) · « 4h30 – 5h » → (270, 300) · « 30–40′ » → (30, 40)
    « — » → (None, None)
    """
    t = html.unescape(text or "").replace("\u2032", "'").replace("\u2019", "'")
    t = t.replace("\u2013", "-").replace("\u2014", "-").replace("\u00a0", " ")
    if not re.search(r"\d", t):
        return None, None

    parts = [p.strip() for p in t.split("-") if re.search(r"\d", p)]
    vals = []
    for p in parts:
        m = re.match(r"(\d+)\s*h\s*(\d+)?", p)
        if m:
            vals.append(int(m.group(1)) * 60 + int(m.group(2) or 0))
            continue
        m = re.match(r"(\d+)", p)
        if m:
            vals.append(int(m.group(1)))

    if not vals:
        return None, None
    # « 4h30 – 5h » : la borne haute sans minutes est en heures, déjà gérée.
    # « 30–40′ » : deux minutes. « 1h – 1h10 » : deux valeurs en minutes.
    return float(min(vals)), float(max(vals))


def _dplus_bounds(text: str) -> tuple[float | None, float | None]:
    """« 1 000 – 1 400 D+ » → (1000, 1400) · « 600 – 900 D+ » → (600, 900)."""
    t = html.unescape(text or "").replace("\u2013", "-").replace("\u00a0", " ")
    m = re.search(r"([\d\s]+)\s*-\s*([\d\s]+)\s*D\s*\+", t)
    if m:
        return float(m.group(1).replace(" ", "")), float(m.group(2).replace(" ", ""))
    m = re.search(r"([\d\s]{3,})\s*D\s*\+", t)
    if m:
        v = float(m.group(1).replace(" ", ""))
        return v, v
    return None, None


def parse_plan_html(source) -> pd.DataFrame:
    """Convertit le plan HTML en table de séances."""
    raw = Path(source).read_text(encoding="utf-8") if isinstance(source, (str, Path)) \
        else source.read().decode("utf-8") if hasattr(source, "read") else str(source)

    # class="week", mais aussi "week is-peak" et "week is-race" : le motif
    # doit tolérer les modificateurs, sinon deux semaines sur huit sautent
    # silencieusement — c'est exactement ce qui s'est produit au premier essai.
    weeks = re.findall(
        r'<section class="week[^"]*"[^>]*data-start="(\d{4}-\d{2}-\d{2})"'
        r'[^>]*>(.*?)</section>', raw, flags=re.S)
    if not weeks:
        raise ValueError(
            "Aucune semaine trouvée. Le plan doit contenir des balises "
            '<section class="week" data-start="AAAA-MM-JJ">.'
        )

    rows = []
    for w_idx, (start, block) in enumerate(weeks, 1):
        w_start = pd.Timestamp(start)
        objective = _first(r'<p class="w-obj">(.*?)</p>', block)
        # `[^>]*` après la classe : la version révisée du plan ajoute des
        # attributs (data-day, data-plan, data-planhi, data-label, data-date)
        # que le motif d'origine ne tolérait pas. Il ne trouvait alors
        # AUCUNE séance et la table sortait vide.
        for li_class, attrs, li in re.findall(
                r'<li class="(day[^"]*)"([^>]*)>(.*?)</li>', block, flags=re.S):
            # Le marqueur « clé » est un span imbriqué dans le titre : on le
            # retire avant extraction, sinon on obtient « Sortie longueclé ».
            li_clean = re.sub(r'<span class="keymark".*?</span>', "", li, flags=re.S)
            date_txt = (_attr(attrs, "data-date")
                        or _first(r'<span class="d-date">(.*?)</span>', li_clean))
            title = (_attr(attrs, "data-label")
                     or _first(r'<span class="d-title">(.*?)</span>', li_clean))
            # Filet : le marqueur « clé » reste parfois accolé au titre
            # quand l'attribut data-label est absent.
            title = re.sub(r"\s*clé$", "", title).strip()
            dur_txt = _first(r'<span class="d-dur">(.*?)</span>', li_clean)
            note = _first(r'<span class="d-txt">(.*?)</span>', li_clean)

            sport, nature = "autre", title
            for css, (sp, nat) in TYPE_MAP.items():
                if css in li_class:
                    sport, nature = sp, nat
                    break

            day_key = (date_txt or "").strip().lower()[:3]
            offset = JOURS.get(day_key)
            date = w_start + timedelta(days=offset) if offset is not None else pd.NaT

            # Durées : les attributs data-plan / data-planhi les donnent en
            # minutes, sans ambiguïté. On ne retombe sur l'analyse du texte
            # (« 4h30 – 5h », « 30–40′ ») que s'ils sont absents.
            dmin, dmax = _attr_minutes(attrs, "data-plan"), _attr_minutes(attrs, "data-planhi")
            if dmin is None:
                dmin, dmax = _duration_bounds(dur_txt)
            elif dmax is None:
                dmax = dmin
            emin, emax = _dplus_bounds(note)

            rows.append({
                # data-day est unique sur les 50 séances : c'est la clé.
                # La version précédente utilisait « date_sport », qui
                # collisionne dès que deux séances du même sport tombent le
                # même jour — Postgres refusait alors l'écriture avec
                # « ON CONFLICT DO UPDATE cannot affect row a second time ».
                "planned_key": _attr(attrs, "data-day") or "",
                "date": date,
                "semaine": w_idx,
                "objectif_semaine": objective,
                "sport": sport,
                "nature": nature,
                "titre": title,
                "duree_min_bas": dmin,
                "duree_min_haut": dmax,
                "dplus_bas": emin,
                "dplus_haut": emax,
                "seance_cle": "is-key" in li_class,
                "consigne": note,
            })

    df = pd.DataFrame(rows)

    # Le jour de course n'est pas balisé comme une séance ordinaire dans le
    # plan : on l'ajoute explicitement, sinon l'objectif du cycle est absent
    # de la table qui sert à le suivre.
    race = re.search(r'data-race="(\d{4}-\d{2}-\d{2})"|jour J', raw)
    race_date = pd.Timestamp("2026-10-18")
    if race and not (df["date"] == race_date).any():
        df = pd.concat([df, pd.DataFrame([{
            "planned_key": "course", "date": race_date, "semaine": len(weeks),
            "objectif_semaine": "Semaine de course",
            "sport": "trail", "nature": "Course objectif",
            "titre": "Trail des Templiers",
            "duree_min_bas": np.nan, "duree_min_haut": np.nan,
            "dplus_bas": 3400.0, "dplus_haut": 3400.0,
            "seance_cle": True,
            "consigne": "80 km - 3 400 D+.",
        }])], ignore_index=True)

    df = df.sort_values("date").reset_index(drop=True)
    # Repli si le HTML ne porte pas data-day : date + sport + rang du jour,
    # ce qui reste unique même en cas de doublon de sport.
    manque = df["planned_key"].astype(str).str.len() == 0
    if manque.any():
        rang = df.groupby(["date", "sport"]).cumcount().astype(str)
        df.loc[manque, "planned_key"] = (
            df.loc[manque, "date"].dt.strftime("%Y-%m-%d") + "_"
            + df.loc[manque, "sport"] + "_" + rang[manque])
    assert df["planned_key"].is_unique, "clés de séance non uniques"
    return df


def _attr(attrs: str, nom: str) -> str:
    m = re.search(rf'{nom}="([^"]*)"', attrs)
    return html.unescape(m.group(1)).strip() if m else ""


def _attr_minutes(attrs: str, nom: str) -> float | None:
    v = _attr(attrs, nom)
    try:
        return float(v) if v else None
    except ValueError:
        return None


def _first(pattern: str, text: str) -> str:
    m = re.search(pattern, text, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", m.group(1)))).strip() \
        if m else ""


def weekly_targets(plan: pd.DataFrame) -> pd.DataFrame:
    """Volume cible par semaine et par sport, en heures."""
    # Le repos n'est pas du volume, même quand une durée facultative
    # (« Repos ou vélo, 1h ») figure en face.
    d = plan[(plan["sport"] != "repos")].dropna(subset=["duree_min_bas"]).copy()
    d["h_bas"] = d["duree_min_bas"] / 60
    d["h_haut"] = d["duree_min_haut"] / 60
    agg = d.groupby(["semaine", "sport"])[["h_bas", "h_haut"]].sum().reset_index()
    return agg.pivot(index="semaine", columns="sport",
                     values=["h_bas", "h_haut"]).fillna(0)


def reconcile(plan: pd.DataFrame, activities: pd.DataFrame,
              tolerance_days: int = 1) -> pd.DataFrame:
    """
    Rapproche chaque séance prévue de l'activité réalisée.

    APPARIEMENT GLOBAL, ET NON SÉQUENTIEL.

    La version précédente parcourait les séances dans l'ordre du calendrier
    et attribuait à chacune la première activité disponible dans sa fenêtre
    de tolérance. Défaut observé : une sortie de 2h51 était rattachée à une
    séance « Course EF » de 1h00–1h10 parce que celle-ci venait la veille,
    alors qu'un « Trail long » de 2h45–3h00 l'attendait le lendemain. La
    première séance servie prenait la meilleure activité, pas la bonne.

    On score donc tous les couples (séance, activité) possibles, puis on
    attribue par ordre de qualité décroissante. Le score combine l'écart de
    date et l'écart de durée : c'est la durée qui départage deux séances
    voisines dans le calendrier, et c'est bien elle qui porte l'information.

    L'écart affiché se mesure à la BORNE la plus proche de la fourchette,
    pas à son milieu : le plan écrit « 4h30 – 5h », donc 4h35 est conforme.
    """
    out = plan.copy()
    for c in ["realise_h", "realise_dplus", "ecart_duree_min", "ecart_dplus"]:
        out[c] = np.nan
    for c in ["realise_id", "realise_nom"]:
        out[c] = pd.Series([None] * len(out), dtype="object")
    out["realise_date"] = pd.Series([pd.NaT] * len(out), dtype="datetime64[ns]")
    out["statut"] = "à venir"

    today = pd.Timestamp.now().normalize()
    if activities.empty:
        echues = out["date"].notna() & (out["date"] <= today)
        out.loc[echues & out["sport"].isin(["repos", "renfo"]), "statut"] = "non suivi"
        out.loc[echues & ~out["sport"].isin(["repos", "renfo"]), "statut"] = "manquée"
        return out

    acts = activities.copy()
    acts["date"] = pd.to_datetime(acts["date"], errors="coerce",
                                  utc=True, format="mixed").dt.tz_localize(None)
    acts["jour"] = acts["date"].dt.normalize()

    # ── Scores de tous les couples plausibles ─────────────────────────────
    couples = []
    for i, r in out.iterrows():
        if pd.isna(r["date"]) or r["date"] > today:
            continue
        if r["sport"] in ("repos", "renfo"):
            continue
        cand = acts[(acts["sport"] == r["sport"])
                    & ((acts["jour"] - r["date"]).abs()
                       <= pd.Timedelta(days=tolerance_days))]
        for j, a in cand.iterrows():
            jours = abs((a["jour"] - r["date"]).days)
            h = float(a.get("duration_h", np.nan))
            if pd.isna(r["duree_min_bas"]) or pd.isna(h):
                mismatch = 0.5              # sans durée cible, neutre
            else:
                cible = (r["duree_min_bas"] + (r["duree_min_haut"]
                         if pd.notna(r["duree_min_haut"]) else r["duree_min_bas"])) / 2
                mismatch = abs(h * 60 - cible) / max(cible, 1)
            # Le jour compte, la durée départage. Un décalage d'un jour
            # équivaut à 35 % d'écart de durée : au-delà, la durée gagne.
            couples.append((jours * 0.35 + mismatch, i, j))

    couples.sort(key=lambda x: x[0])
    pris_seances, pris_acts = set(), set()
    for score, i, j in couples:
        if i in pris_seances or j in pris_acts:
            continue
        pris_seances.add(i)
        pris_acts.add(j)
        best = acts.loc[j]
        r = out.loc[i]
        h = float(best.get("duration_h", np.nan))
        out.at[i, "realise_id"] = best["activity_id"]
        out.at[i, "realise_date"] = best["jour"]
        out.at[i, "realise_nom"] = best.get("name")
        out.at[i, "realise_h"] = h
        out.at[i, "realise_dplus"] = best.get("d_plus")
        out.at[i, "ecart_duree_min"] = _gap(h * 60, r["duree_min_bas"],
                                            r["duree_min_haut"])
        out.at[i, "ecart_dplus"] = _gap(best.get("d_plus"), r["dplus_bas"],
                                        r["dplus_haut"])
        out.at[i, "statut"] = "réalisée"

    # ── Séances échues non appariées ──────────────────────────────────────
    echues = out["date"].notna() & (out["date"] <= today) & (out["statut"] != "réalisée")
    out.loc[echues & out["sport"].isin(["repos", "renfo"]), "statut"] = "non suivi"
    out.loc[echues & ~out["sport"].isin(["repos", "renfo"]), "statut"] = "manquée"
    return out


def _gap(value, low, high) -> float:
    """Écart à la fourchette. 0 si dedans, négatif si en dessous."""
    if value is None or pd.isna(value) or pd.isna(low):
        return np.nan
    if value < low:
        return float(value - low)
    if value > high:
        return float(value - high)
    return 0.0


def compliance(rec: pd.DataFrame) -> dict:
    """
    Taux de réalisation et volume, sur les séances échues.

    LE TEMPS SAISI À LA MAIN PRIME. La version précédente ne comptait que
    la durée des activités appariées : le total affichait 0,0 h alors que
    des temps effectifs avaient été renseignés. Or la saisie manuelle est
    la seule source pour tout ce qui n'a pas de fichier — renforcement,
    séance non enregistrée, montre oubliée.
    """
    rec = rec.copy()
    manuel = rec["temps_min"] / 60 if "temps_min" in rec.columns else np.nan
    rec["_heures"] = (manuel if isinstance(manuel, pd.Series)
                      else pd.Series(np.nan, index=rec.index))
    rec["_heures"] = rec["_heures"].fillna(rec.get("realise_h"))

    # Une séance cochée à la main compte comme réalisée, même sans fichier.
    if "fait" in rec.columns:
        coche = rec["fait"].fillna(False).astype(bool)
        rec.loc[coche & (rec["statut"] != "réalisée"), "statut"] = "réalisée"

    due = rec[rec["statut"].isin(["réalisée", "manquée"])]
    if due.empty:
        return {"ok": False, "reason": "Aucune séance échue."}

    done = due[due["statut"] == "réalisée"]

    # Décomposition par semaine, indispensable pour que le total soit
    # vérifiable. Un écart global de deux heures est illisible s'il ne
    # s'explique pas : il additionne ici la semaine écoulée ET les jours
    # déjà passés de la semaine en cours, encore non renseignés.
    par_sem = []
    for w, g in due.groupby("semaine"):
        f = g[g["statut"] == "réalisée"]
        prevu = float(g["duree_min_bas"].dropna().sum())
        reel = float(f["_heures"].dropna().sum() * 60)
        par_sem.append({
            "semaine": int(w),
            "seances": f"{len(f)}/{len(g)}",
            "prevu_min": prevu,
            "reel_min": reel,
            "ecart_min": reel - prevu,
        })
    keys = due[due["seance_cle"]]
    keys_done = keys[keys["statut"] == "réalisée"]

    return {
        "ok": True,
        "taux": len(done) / len(due),
        "n_due": int(len(due)),
        "n_done": int(len(done)),
        "taux_cles": len(keys_done) / len(keys) if len(keys) else np.nan,
        "n_cles": int(len(keys)),
        "ecart_duree_median": float(done["ecart_duree_min"].median())
        if not done.empty else np.nan,
        "ecart_heures": (float(done["_heures"].dropna().sum())
                         - float(due["duree_min_bas"].sum() / 60)),
        "heures_prevues": float(due["duree_min_bas"].sum() / 60),
        "heures_realisees": float(done["_heures"].dropna().sum()),
        "par_semaine": pd.DataFrame(par_sem),
        "derniere_echue": (due["date"].max() if "date" in due.columns
                           else pd.NaT),
        "semaines": sorted(due["semaine"].dropna().unique().tolist()),
    }


def diagnose(rec: pd.DataFrame) -> list[str]:
    """
    Lecture du suivi, formulée en observations et non en injonctions.

    Le plan porte déjà ses propres règles : priorité à la sortie longue,
    ne jamais augmenter durée, D+ et intensité la même semaine. Le rôle du
    module est de dire ce que montrent les données, pas de réécrire le plan.
    """
    msgs = []
    c = compliance(rec)
    if not c.get("ok"):
        return ["Pas encore de séance échue."]

    if not np.isnan(c["taux_cles"]) and c["taux_cles"] < 0.8:
        manquees = rec[(rec["seance_cle"]) & (rec["statut"] == "manquée")]
        dates = ", ".join(manquees["date"].dt.strftime("%d/%m"))
        msgs.append(
            f"{len(manquees)} sortie(s) longue(s) clé(s) manquée(s) : {dates}. "
            "Le plan les désigne comme la priorité numéro un."
        )

    ecart = c["ecart_duree_median"]
    if not np.isnan(ecart):
        if ecart < -12:
            msgs.append(
                f"Durée médiane inférieure de {abs(ecart):.0f} min à la borne "
                "basse. Le volume réel est en dessous du plan."
            )
        elif ecart > 15:
            msgs.append(
                f"Durée médiane supérieure de {ecart:.0f} min à la borne "
                "haute. Le plan prévoit une progression de charge encadrée ; "
                "dépasser systématiquement la borne haute la contourne."
            )

    velo = rec[(rec["sport"] == "velo") & rec["statut"].isin(["réalisée", "manquée"])]
    if len(velo) >= 4:
        taux_velo = (velo["statut"] == "réalisée").mean()
        if taux_velo < 0.6:
            msgs.append(
                f"Vélo réalisé à {taux_velo:.0%}. C'est le pilier du volume "
                "aérobie dans ce plan — le remplacer par de la course "
                "changerait l'exposition du genou aux descentes."
            )

    if not msgs:
        msgs.append(f"Suivi conforme : {c['n_done']}/{c['n_due']} séances "
                    f"réalisées, durées dans les fourchettes.")
    return msgs


JOURNAL_COLUMNS = ["planned_key", "fait", "temps_min", "commentaire",
                   "rpe", "genou_j1", "genou_j2", "maj"]


SUIVI_TAG = re.compile(
    r'(<script id="suivi-data"[^>]*>)(.*?)(</script>)', re.S)


def inject_state(html: str, journal: pd.DataFrame,
                 plan_nom: str = "Templiers 2026") -> str:
    """
    Injecte l'état de suivi venu de Supabase dans la balise `suivi-data`.

    LE PONT ENTRE LES APPAREILS.

    Le document HTML enregistre dans le localStorage du navigateur, donc
    rien ne circule entre l'ordinateur et le téléphone. Mais il sait aussi
    lire un état EMBARQUÉ dans le fichier, via une balise
    `<script id="suivi-data">`, et sa fonction load() retient le plus
    récent des deux en comparant les horodatages.

    On réécrit donc cette balise à chaque affichage avec ce que contient
    Supabase, en datant l'état de maintenant : le document affiche alors
    toujours la vérité serveur, sur n'importe quel appareil.

    La saisie, elle, passe par les champs Streamlit : c'est le seul moyen
    d'écrire côté serveur, un cadre HTML n'ayant aucun canal de retour vers
    Python.
    """
    if journal is None or journal.empty or "planned_key" not in journal.columns:
        data = {}
    else:
        data = {}
        for _, r in journal.iterrows():
            cle = str(r["planned_key"])
            if not cle or cle == "nan":
                continue
            rec = {}
            if "fait" in r.index and pd.notna(r["fait"]):
                rec["done"] = bool(r["fait"])
            if "temps_min" in r.index and pd.notna(r["temps_min"]):
                rec["time"] = str(int(round(float(r["temps_min"]))))
            if "commentaire" in r.index and pd.notna(r["commentaire"]) \
                    and str(r["commentaire"]).strip():
                rec["note"] = str(r["commentaire"])
            if rec:
                data[cle] = rec

    charge = json.dumps({
        "plan": plan_nom,
        "version": 99,
        "savedAt": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "data": data,
    }, ensure_ascii=False)

    if SUIVI_TAG.search(html):
        return SUIVI_TAG.sub(lambda m: m.group(1) + charge + m.group(3),
                             html, count=1)
    # Balise absente : on l'ajoute juste avant la fermeture du corps.
    balise = f'<script id="suivi-data" type="application/json">{charge}</script>'
    return html.replace("</body>", balise + "\n</body>") if "</body>" in html \
        else html + balise


def parse_temps(txt) -> float | None:
    """
    Durée saisie librement, en minutes.

    Accepte « 2h35 », « 2h », « 155 », « 155′ », « 1h05 ». Le plan HTML
    laissait ce champ libre et c'est le bon choix : imposer un format
    numérique fait perdre du temps à la saisie, alors qu'on note une durée
    comme on la lit sur sa montre.
    """
    if txt is None:
        return None
    t = str(txt).strip().lower().replace("\u2032", "").replace("'", "")
    t = t.replace(",", ".")
    if not t or t in ("-", "—"):
        return None
    m = re.match(r"^(\d+)\s*h\s*(\d+)?$", t)
    if m:
        return float(m.group(1)) * 60 + float(m.group(2) or 0)
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(?:min|m)?$", t)
    if m:
        v = float(m.group(1))
        return v * 60 if v <= 12 else v      # « 2 » se lit 2 heures
    return None


def fmt_minutes(m) -> str:
    """155 -> « 2h35 ». Zéro ou absent -> tiret."""
    if m is None or (isinstance(m, float) and np.isnan(m)) or m <= 0:
        return "—"
    m = int(round(m))
    return f"{m // 60}h{m % 60:02d}" if m >= 60 else f"{m}′"


def ecart_hm(minutes) -> str:
    """Écart signé au format du plan : « -0h10 », « +0h05 », « 0h00 »."""
    if minutes is None or (isinstance(minutes, float) and np.isnan(minutes)):
        return "—"
    m = int(round(minutes))
    signe = "-" if m < 0 else ("+" if m > 0 else "")
    m = abs(m)
    return f"{signe}{m // 60}h{m % 60:02d}"


def ecart_minutes(reel, bas, haut) -> float | None:
    """Écart en minutes à la borne la plus proche. Zéro si dans la cible."""
    if reel is None or (isinstance(reel, float) and np.isnan(reel)) or pd.isna(bas):
        return None
    if reel < bas:
        return float(reel - bas)
    if not pd.isna(haut) and reel > haut:
        return float(reel - haut)
    return 0.0


def migrate_journal_keys(plan: pd.DataFrame, journal: pd.DataFrame) -> dict:
    """
    Convertit les clés de journal de l'ancien format vers le nouveau.

    L'ancien format était « AAAA-MM-JJ_sport », remplacé par l'identifiant
    du plan (« s1-d1 ») après une collision : deux séances vélo tombaient le
    même jour. Les saisies faites avant le changement restent en base, mais
    sous des clés que le formulaire ne retrouve plus — il les affiche donc
    vides, et un enregistrement les écrase.

    On transfère chaque ancienne ligne vers la nouvelle clé, en ne
    remplaçant que les champs VIDES côté nouveau : ce qui a été saisi
    depuis prime sur ce qui vient d'avant.
    """
    if journal is None or journal.empty:
        return {"ok": False, "reason": "Journal vide."}

    ancien = re.compile(r"^\d{4}-\d{2}-\d{2}_")
    j = journal.copy()
    j["planned_key"] = j["planned_key"].astype(str)
    vieilles = j[j["planned_key"].str.match(ancien)]
    if vieilles.empty:
        return {"ok": True, "migrees": 0, "conflits": [],
                "reason": "Aucune clé à l'ancien format."}

    # Correspondance date+sport -> nouvelle clé, en écartant les jours où
    # deux séances du même sport existent : impossible de trancher.
    ref = plan.copy()
    ref["jour"] = pd.to_datetime(ref["date"]).dt.strftime("%Y-%m-%d")
    ref["ancien"] = ref["jour"] + "_" + ref["sport"]
    compte = ref["ancien"].value_counts()
    table = {a: k for a, k in zip(ref["ancien"], ref["planned_key"])
             if compte[a] == 1}

    actuel = {r["planned_key"]: r for _, r in j.iterrows()}
    maj, conflits, perdues = [], [], []
    for _, v in vieilles.iterrows():
        cle = table.get(v["planned_key"])
        if cle is None:
            (conflits if v["planned_key"] in set(ref["ancien"])
             else perdues).append(v["planned_key"])
            continue
        cible = actuel.get(cle)
        ligne = {"planned_key": cle}
        for c in ("fait", "temps_min", "commentaire", "rpe",
                  "genou_j1", "genou_j2"):
            if c not in v.index:
                continue
            neuf = cible[c] if (cible is not None and c in cible.index) else None
            vide = (neuf is None or pd.isna(neuf)
                    or (c == "fait" and not bool(neuf))
                    or (c == "commentaire" and str(neuf).strip() in ("", "nan")))
            if vide and pd.notna(v[c]):
                ligne[c] = v[c]
        if len(ligne) > 1:
            ligne["maj"] = pd.Timestamp.now()
            maj.append(ligne)

    return {"ok": True, "a_ecrire": pd.DataFrame(maj) if maj else pd.DataFrame(),
            "migrees": len(maj), "conflits": conflits, "perdues": perdues,
            "anciennes_cles": vieilles["planned_key"].tolist()}


def semaine_du_jour(rec: pd.DataFrame, today, semaines) -> int:
    """
    Semaine à afficher par défaut : celle qui CONTIENT la date du jour.

    La version précédente cherchait les séances dans une fenêtre de plus ou
    moins six jours autour d'aujourd'hui, puis prenait la première. Le
    2 septembre, cette fenêtre couvrait le 27 août au 8 septembre : elle
    attrapait donc la fin de la semaine 1 avant le début de la semaine 2, et
    proposait la semaine 1. Une fenêtre glissante ne peut pas répondre à
    « quelle semaine sommes-nous » — il faut tester l'appartenance.
    """
    if not len(semaines):
        return 0
    bornes = (rec.dropna(subset=["date", "semaine"])
              .groupby("semaine")["date"].agg(["min", "max"]))
    for w, (d1, d2) in bornes.iterrows():
        if d1 <= today <= d2:
            return int(w)
    # Entre deux semaines, ou hors du cycle : la plus proche à venir, sinon
    # la dernière écoulée.
    a_venir = bornes[bornes["min"] > today]
    if len(a_venir):
        return int(a_venir["min"].idxmin())
    return int(bornes["max"].idxmax())


def week_summary(sem: pd.DataFrame) -> dict:
    """
    Bandeau de tête d'une semaine, au format du plan.

    « 6/6 séances · réalisé 7h25 · prévu 7h55 · écart -0h30 », plus la
    répartition par sport. Le réalisé privilégie le temps saisi à la main
    et retombe sur la durée de l'activité appariée.
    """
    reel = []
    for _, r in sem.iterrows():
        t = r.get("temps_min")
        if pd.isna(t):
            h = r.get("realise_h")
            t = h * 60 if pd.notna(h) else None
        reel.append(t)
    sem = sem.copy()
    sem["_reel"] = reel

    suivies = sem[sem["sport"] != "repos"]
    faites = sem["fait"].fillna(sem["statut"].eq("réalisée")).astype(bool)
    total_reel = float(sem["_reel"].dropna().sum())
    total_prevu = float(suivies["duree_min_bas"].dropna().sum())

    par_sport = {}
    for sp in ("trail", "route", "velo", "renfo"):
        v = suivies.loc[suivies["sport"] == sp, "duree_min_bas"].dropna().sum()
        if v:
            par_sport[sp] = float(v)

    return {
        "faites": int(faites[sem["sport"] != "repos"].sum()),
        "total": int(len(suivies)),
        "reel_min": total_reel,
        "prevu_min": total_prevu,
        "ecart_min": total_reel - total_prevu if total_reel else None,
        "par_sport": par_sport,
        "reel_par_seance": reel,
    }


def load_profile(plan: pd.DataFrame) -> pd.DataFrame:
    """
    Profil de charge hebdomadaire, borne basse, séparé course et vélo.

    Reprend le graphe du plan : la course en aire pleine, le vélo empilé
    par-dessus, et le total en ligne. Le renforcement est exclu du volume,
    comme dans le document — trente minutes de gainage ne se comparent pas
    à trente minutes de côtes.
    """
    d = plan[plan["sport"] != "repos"].dropna(subset=["duree_min_bas"]).copy()
    if d.empty:
        return pd.DataFrame()
    d["famille"] = d["sport"].map(
        {"trail": "course", "route": "course", "velo": "velo",
         "renfo": "renfo"}).fillna("autre")
    t = (d.groupby(["semaine", "famille"])["duree_min_bas"].sum()
         .unstack(fill_value=0.0) / 60).sort_index()
    for c in ("course", "velo", "renfo"):
        if c not in t.columns:
            t[c] = 0.0
    t["total"] = t["course"] + t["velo"] + t["renfo"]
    return t


def ecart_fourchette(reel, bas, haut) -> str:
    """Écart à la fourchette prévue, en minutes signées. Vide si dedans."""
    if reel is None or (isinstance(reel, float) and np.isnan(reel)) or pd.isna(bas):
        return "—"
    if reel < bas:
        return f"{reel - bas:+.0f}′"
    if not pd.isna(haut) and reel > haut:
        return f"{reel - haut:+.0f}′"
    return "dans la cible"


def empty_journal() -> pd.DataFrame:
    """
    Journal de séance : ce que le rapprochement automatique ne peut pas
    deviner.

    Deux rôles distincts.

    `fait` est une validation MANUELLE. Le rapprochement automatique ne voit
    que les activités portant un fichier — il ignore donc structurellement
    le renforcement, la mobilité, et toute séance non enregistrée à la
    montre. Sans case à cocher, un tiers du plan resterait invisible.

    Les autres champs sont subjectifs. Le plan désigne deux signaux à
    surveiller : la douleur pendant la course, et la réaction du genou à
    J+1 et J+2. Aucune donnée de montre ne les remplace. Échelle 0 à 3 :
    rien, gêne, douleur, douleur limitante.
    """
    return pd.DataFrame({
        "planned_key": pd.Series(dtype="str"),
        "fait": pd.Series(dtype="bool"),
        "temps_min": pd.Series(dtype="float"),
        "rpe": pd.Series(dtype="float"),
        "genou_j1": pd.Series(dtype="float"),
        "genou_j2": pd.Series(dtype="float"),
        "commentaire": pd.Series(dtype="str"),
        "maj": pd.Series(dtype="datetime64[ns]"),
    })


def merge_journal(rec: pd.DataFrame, journal: pd.DataFrame) -> pd.DataFrame:
    """
    Superpose le journal au rapprochement automatique.

    La saisie manuelle PRIME toujours : c'est toi qui étais là. Une séance
    cochée devient « réalisée » même sans fichier ; une séance décochée
    repasse en « manquée » même si une activité a été appariée — cas réel
    quand tu as couru autre chose que ce qui était prévu ce jour-là.
    """
    out = rec.copy()
    for c in ["fait", "temps_min", "rpe", "genou_j1", "genou_j2", "commentaire"]:
        if c not in out.columns:
            out[c] = pd.Series([None] * len(out), dtype="object")

    if journal is None or journal.empty:
        return out

    j = journal.set_index("planned_key")
    for i, key in out["planned_key"].items():
        if key not in j.index:
            continue
        row = j.loc[key]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        for c in ["temps_min", "rpe", "genou_j1", "genou_j2", "commentaire"]:
            if c in row.index and pd.notna(row[c]):
                out.at[i, c] = row[c]
        if "fait" in row.index and pd.notna(row["fait"]):
            out.at[i, "fait"] = bool(row["fait"])
            if bool(row["fait"]):
                if out.at[i, "statut"] != "réalisée":
                    out.at[i, "statut"] = "réalisée"
            elif out.at[i, "date"] <= pd.Timestamp.now().normalize():
                out.at[i, "statut"] = "manquée"
    return out


def knee_trend(rec: pd.DataFrame, window: int = 6) -> dict:
    """
    Lecture de la réaction du genou sur les dernières séances renseignées.

    Ne diagnostique rien : rapporte ce que tu as noté, pour que la tendance
    soit visible avant qu'elle ne devienne un problème.
    """
    d = rec.dropna(subset=["genou_j1"]).sort_values("date").tail(window)
    if len(d) < 3:
        return {"ok": False, "reason": f"{len(d)} séance(s) renseignée(s), "
                                       "il en faut au moins 3."}
    vals = d["genou_j1"].astype(float).to_numpy()
    slope = float(np.polyfit(np.arange(len(vals)), vals, 1)[0])
    return {
        "ok": True,
        "moyenne": float(vals.mean()),
        "max": float(vals.max()),
        "tendance": slope,
        "n": len(vals),
        "sens": "en hausse" if slope > 0.15 else
                "en baisse" if slope < -0.15 else "stable",
    }
